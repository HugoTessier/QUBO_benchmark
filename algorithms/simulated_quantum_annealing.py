from algorithms.algorithm import AlgorithmQUBO, AlgorithmIsing, Algorithm
import numpy as np
import math
import random
import mpmath
from typing import Tuple, Callable
from abc import abstractmethod
from utils.schedulers import Scheduler, LinearScheduler
from utils.sampling import range_sampler


class SimulatedQuantumAnnealingCommon(Algorithm):
    """
    Based on Accelerating Simulated Quantum Annealing with GPU and Tensor Cores,
    by Yi-Hua Chung, Cheng-Jhih Shih and Shih-Hao Hung
    and An Ising computer based on simulated quantum annealing by path integral Monte Carlo method
    by Takuya Okuyama, Masato Hayashi and Masanao Yamaoka.

    We adapted some calculations to fit QUBO solutions in {0, 1} instead of {-1, 1}.
    """

    def __init__(self,
                 monte_carlo_steps: int,
                 n_trotters: int,
                 temperature_scheduler: Scheduler = LinearScheduler(start=4.0, end=1 / 32),
                 transverse_field_scheduler: Scheduler = LinearScheduler(start=2.0, end=0.),
                 sampler: Callable = range_sampler):
        """
        :param monte_carlo_steps: Number of Monte-Carlo steps, i.e. the outermost loop of the algorithm.
        :param n_trotters: Number of replicas that are optimized in parallel and eventually fused together.
        :param start_transverse_field: Starting value of the transverse field, that gives the strength of the coupling
        between trotters.
        :param end_transverse_field: Final value of the transverse field.
        :param start_temperature: Starting value of the temperature, that gives the chance of accepting changes.
        :param end_temperature: Final value of the temperature.
        """
        self.monte_carlo_steps = monte_carlo_steps
        self.n_trotters = n_trotters
        self.temperature_scheduler = temperature_scheduler
        self.transverse_field_scheduler = transverse_field_scheduler
        self.sampler = sampler

    def _compute_trotters_mean_energy(self, x: np.ndarray, *args, **kwargs) -> float:
        # Since, for most of the process, each trotter is more or less independent, we choose to report the mean of
        # the energy of each trotter
        energies = []
        for m in range(self.n_trotters):
            energies.append(self.compute_energy(x[:, m], *args, **kwargs))
        return np.mean(np.array(energies))

    def _compute_replica_coupling_penalty(self,
                                          x: np.ndarray,
                                          temperature: float,
                                          transverse_field: float,
                                          i: int,
                                          m: int) -> float:
        # Weird formula to compute the coupling strength, that increases during training to force trotters to
        # become the same.
        replicas_coupling_strength = (temperature / 2) * math.log(
            mpmath.coth(transverse_field / (self.n_trotters * temperature)))
        # The coupling depends on the m-1 and m+1 trotters
        replica_coupling = 0
        if m + 1 != self.n_trotters:
            replica_coupling += x[i, m + 1]
        if m != 0:
            replica_coupling += x[i, m - 1]
        return replicas_coupling_strength * replica_coupling

    @abstractmethod
    def _compute_energy_delta(self,
                              x: np.ndarray,
                              local_field: np.ndarray,
                              replica_coupling_penalty,
                              i: int,
                              m: int) -> float:
        raise NotImplementedError

    def _select_final_answer(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        energies = []
        for solution in x.T:
            energies.append(self.compute_energy(solution, *args, **kwargs))
        return x.T[np.argmin(np.array(energies))]

    def __call__(self, *args, **kwargs) -> Tuple:
        """
        Solves a QUBO problem using Simulated Quantum Annealing (SQA).

        :param qubo: The matrix of the coupling coefficients of the QUBO problem.
        :param offset: Offset value of the energy.
        :return: The solution and a history of the optimization process.
        """

        length = self.get_length(*args, **kwargs)

        # SQA is based on the principle of running n_trotter copies in parallel, that each try to solve the problem
        # and which are progressively applied a penalty to force them to converge toward a common solution at the end.
        # These copies are called "Trotters".
        x = np.stack([self.generate_random_solution(length) for _ in range(self.n_trotters)]).T

        # Since the hamiltonian is purely a sum, the energy delta induced by changing one element of x at a time
        # can be calculated using only a "local" energy. It is then more efficient to compute said "local energy" field
        # beforehand and to update it progressively.
        local_energy_field = self._initialize_local_energy_field(x, *args, **kwargs)

        history = []
        for t in range(self.monte_carlo_steps):
            history.append([t, self._compute_trotters_mean_energy(x, *args, **kwargs)])
            temperature = self.temperature_scheduler.update(t, self.monte_carlo_steps)  # Update temperature
            transverse_field = self.transverse_field_scheduler.update(t,
                                                                      self.monte_carlo_steps)  # update transverse_field
            for i in self.sampler(length):
                for m in self.sampler(self.n_trotters):
                    replica_coupling_penalty = self._compute_replica_coupling_penalty(x, temperature,
                                                                                      transverse_field, i, m)
                    delta_energy = self._compute_energy_delta(x, local_energy_field, replica_coupling_penalty, i, m)

                    # Metropolis algorithm: the random number is in [0,1]; if the tested change in x reduces the
                    # energy, then exp(-delta_energy / temperature) > 1 and the change is accepted; otherwise, we have
                    # exp(-delta_energy / temperature) in [0,1] too and then, the change is accepted in a probabilistic
                    # way, with decreasing chances as the energy increase becomes larger.
                    if math.exp(-delta_energy / temperature) > random.random():
                        # We flip the i,m element in x
                        self._flip_bit(x, i, m)

                        local_energy_field = self._update_local_energy_field(x, local_energy_field, i, m,
                                                                             *args, **kwargs)

        x = self._select_final_answer(x, *args, **kwargs)
        history.append([self.monte_carlo_steps, self.compute_energy(x, *args, **kwargs)])
        return x, history

    @staticmethod
    @abstractmethod
    def _initialize_local_energy_field(x, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _update_local_energy_field(x, local_energy_field, i, m, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _flip_bit(x, i, m):
        raise NotImplementedError


class SimulatedQuantumAnnealingQUBO(AlgorithmQUBO, SimulatedQuantumAnnealingCommon):
    """
    Based on Accelerating Simulated Quantum Annealing with GPU and Tensor Cores,
    by Yi-Hua Chung, Cheng-Jhih Shih and Shih-Hao Hung
    and An Ising computer based on simulated quantum annealing by path integral Monte Carlo method
    by Takuya Okuyama, Masato Hayashi and Masanao Yamaoka.

    We adapted some calculations to fit QUBO solutions in {0, 1} instead of {-1, 1}.
    """

    def __init__(self,
                 monte_carlo_steps: int,
                 n_trotters: int,
                 temperature_scheduler: Scheduler = LinearScheduler(start=4.0, end=1 / 32),
                 transverse_field_scheduler: Scheduler = LinearScheduler(start=2.0, end=0.),
                 sampler: Callable = range_sampler):
        """
        :param monte_carlo_steps: Number of Monte-Carlo steps, i.e. the outermost loop of the algorithm.
        :param n_trotters: Number of replicas that are optimized in parallel and eventually fused together.
        :param start_transverse_field: Starting value of the transverse field, that gives the strength of the coupling
        between trotters.
        :param end_transverse_field: Final value of the transverse field.
        :param start_temperature: Starting value of the temperature, that gives the chance of accepting changes.
        :param end_temperature: Final value of the temperature.
        """
        super().__init__(monte_carlo_steps, n_trotters, temperature_scheduler, transverse_field_scheduler, sampler)

    def _compute_energy_delta(self,
                              x: np.ndarray,
                              local_field: np.ndarray,
                              replica_coupling_penalty,
                              i: int,
                              m: int) -> float:
        # We compute the difference in energy, with a sign that depends on whether the element is set to 0 or 1, and
        # therefore removed or added.
        delta_energy = local_field[i, m] - replica_coupling_penalty
        flip_direction = 1 if x[i, m] == 0 else -1
        return delta_energy * flip_direction

    @staticmethod
    def _update_local_energy_field(x, local_energy_field, i, m, qubo, offset):
        # If x[i, m] == 1, then was 0, and we add it; otherwise, now 0 and we remove
        flip_direction = 1 if x[i, m] == 1 else -1
        local_energy_field[:, m] += (flip_direction * qubo[i, :])
        return local_energy_field

    @staticmethod
    def _flip_bit(x, i, m):
        x[i, m] = int(x[i, m] == 0)
        return x

    @staticmethod
    def _initialize_local_energy_field(x, qubo, offset):
        local_energy_field = np.dot(qubo, x)
        return local_energy_field

    def __call__(self, qubo: np.ndarray, offset: float) -> Tuple:
        return super().__call__(qubo, offset)


class SimulatedQuantumAnnealingIsing(AlgorithmIsing, SimulatedQuantumAnnealingCommon):
    """
    Based on Accelerating Simulated Quantum Annealing with GPU and Tensor Cores,
    by Yi-Hua Chung, Cheng-Jhih Shih and Shih-Hao Hung
    and An Ising computer based on simulated quantum annealing by path integral Monte Carlo method
    by Takuya Okuyama, Masato Hayashi and Masanao Yamaoka.

    We adapted some calculations to fit QUBO solutions in {0, 1} instead of {-1, 1}.
    """

    def __init__(self,
                 monte_carlo_steps: int,
                 n_trotters: int,
                 temperature_scheduler: Scheduler = LinearScheduler(start=4.0, end=1 / 32),
                 transverse_field_scheduler: Scheduler = LinearScheduler(start=2.0, end=0.),
                 sampler: Callable = range_sampler):
        """
        :param monte_carlo_steps: Number of Monte-Carlo steps, i.e. the outermost loop of the algorithm.
        :param n_trotters: Number of replicas that are optimized in parallel and eventually fused together.
        :param start_transverse_field: Starting value of the transverse field, that gives the strength of the coupling
        between trotters.
        :param end_transverse_field: Final value of the transverse field.
        :param start_temperature: Starting value of the temperature, that gives the chance of accepting changes.
        :param end_temperature: Final value of the temperature.
        """
        super().__init__(monte_carlo_steps, n_trotters, temperature_scheduler, transverse_field_scheduler, sampler)

    def _compute_energy_delta(self,
                              x: np.ndarray,
                              local_field: np.ndarray,
                              replica_coupling_penalty,
                              i: int,
                              m: int) -> float:
        return - x[i, m] * (local_field[i, m] - replica_coupling_penalty)

    @staticmethod
    def _update_local_energy_field(x, local_energy_field, i, m, linear: np.ndarray,
                                   quadratic: np.ndarray, offset: float):
        local_energy_field[:, m] += 2 * x[i, m] * quadratic[i, :]
        return local_energy_field

    @staticmethod
    def _flip_bit(x, i, m):
        x[i, m] = -x[i, m]
        return x

    @staticmethod
    def _initialize_local_energy_field(x, linear: np.ndarray,
                                       quadratic: np.ndarray, offset: float):
        local_energy_field = np.dot(quadratic, x) + linear[:, None].repeat(x.shape[1], axis=1)
        return local_energy_field

    def __call__(self, linear: np.ndarray,
                 quadratic: np.ndarray, offset: float) -> Tuple:
        return super().__call__(linear, quadratic, offset)
