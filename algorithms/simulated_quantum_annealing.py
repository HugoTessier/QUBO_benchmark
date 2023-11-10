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
        :param temperature_scheduler: Controls the evolution of the temperature during annealing.
        :param transverse_field_scheduler: Controls the evolution of the transverse field during annealing.
        :param sampler: Controls in which order to test changes in the value of elements in the solution.
        """
        self.monte_carlo_steps = monte_carlo_steps
        self.n_trotters = n_trotters
        self.temperature_scheduler = temperature_scheduler
        self.transverse_field_scheduler = transverse_field_scheduler
        self.sampler = sampler

    def _select_best_energy_among_trotters(self, x: np.ndarray, *args, **kwargs) -> float:
        # Since each trotter is a candidate solution, we report the energy of the best one.
        # Indeed, at the end of the annealing, we return the trotter of least energy.
        return min([self.compute_energy(solution, *args, **kwargs) for solution in x.T])

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

    def _select_final_answer(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        # Gives the trotter with the best energy. We do that, instead of giving an average, because, not only does
        # it lead to non-acceptable values (e.g. zeros for Ising models), but also because the average of the solutions
        # is not guaranteed to give the average of their energies.
        energies = []
        for solution in x.T:
            energies.append(self.compute_energy(solution, *args, **kwargs))
        return x.T[np.argmin(np.array(energies))]

    def __call__(self, *args, **kwargs) -> Tuple:
        # args and kwargs are here to be replaced with qubo + offset
        # or linear + quadratic + offset depending on whether it is implemented for QUBO or Ising model.

        length = self.get_length(*args, **kwargs)

        # SQA is based on the principle of running n_trotter copies in parallel, that each try to solve the problem
        # and which are progressively applied a penalty to force them to converge toward a common solution at the end.
        # These copies are called "Trotters".
        x = np.stack([self.generate_random_solution(length) for _ in range(self.n_trotters)]).T

        # Since the hamiltonian is purely a sum, the energy delta induced by changing one element of x at a time
        # can be calculated using only a "local" energy. It is then more efficient to compute said "local energy" field
        # beforehand and to update it progressively.
        local_energy_field = self._initialize_local_energy_field(x, *args, **kwargs)

        history = []  # We monitor the energy evolution at each Monte-Carlo step
        for t in range(self.monte_carlo_steps):
            history.append([t, self._select_best_energy_among_trotters(x, *args, **kwargs)])
            temperature = self.temperature_scheduler.update(t, self.monte_carlo_steps)
            transverse_field = self.transverse_field_scheduler.update(t, self.monte_carlo_steps)

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
                        # We accept the change in the i,m element of x
                        self._flip_element(x, i, m)

                        local_energy_field = self._update_local_energy_field(local_energy_field, x, i, m, *args,
                                                                             **kwargs)

        x = self._select_final_answer(x, *args, **kwargs)
        history.append([self.monte_carlo_steps, self.compute_energy(x, *args, **kwargs)])
        return x, history

    @abstractmethod
    def _compute_energy_delta(self,
                              x: np.ndarray,
                              local_energy_field: np.ndarray,
                              replica_coupling_penalty: float,
                              i: int,
                              m: int) -> float:
        """
        Gives the variation in energy when changing the [i,m] element of x.

        :param x: The candidate solution whose variations in energy we want to measure when changing the [i,m] element.
        :param local_energy_field: Energy associated to each element of x, in order to simplify calculations.
        :param replica_coupling_penalty: Strength of the penalty that pushes the trotters toward a same solution.
        :param i: Element i of any trotter of x.
        :param m: Trotter m of x.

        :return: The variation of energy when changing the [i,m] element of x.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _initialize_local_energy_field(x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Initializes the local energy field of x.
        :param x: The solution whose energy field to compute.
        :return: The local energy field.
        """
        # args and kwargs are here to be replaced with qubo + offset
        # or linear + quadratic + offset depending on whether it is implemented for QUBO or Ising model.
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _update_local_energy_field(local_energy_field: np.ndarray,
                                   x: np.ndarray,
                                   i: int,
                                   m: int,
                                   *args,
                                   **kwargs) -> np.ndarray:
        """
        Updates the local energy field.

        :param local_energy_field: The energy field to update.
        :param x: The candidate solution that has changed and whose energy field we must update.
        :param i: Element i of any trotter of x.
        :param m: Trotter m of x.

        :return: The updated local energy field.
        """
        # args and kwargs are here to be replaced with qubo + offset
        # or linear + quadratic + offset depending on whether it is implemented for QUBO or Ising model.
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _flip_element(x: np.ndarray, i: int, m: int) -> np.ndarray:
        """
        Flips the [i,m] of x.

        :param x: The candidate solution whose element [i,m] is to flip.
        :param i: Element i of any trotter of x.
        :param m: Trotter m of x.

        :return: The modified x.
        """
        raise NotImplementedError


class SimulatedQuantumAnnealingQUBO(AlgorithmQUBO, SimulatedQuantumAnnealingCommon):
    """
    QUBO version of SQA.
    We adapted some calculations to fit QUBO solutions in {0, 1} instead of {-1, 1}.
    """

    def _compute_energy_delta(self,
                              x: np.ndarray,
                              local_energy_field: np.ndarray,
                              replica_coupling_penalty: float,
                              i: int,
                              m: int) -> float:
        # We compute the difference in energy, with a sign that depends on whether the element is set to 0 or 1, and
        # therefore removed or added.
        delta_energy = local_energy_field[i, m] - replica_coupling_penalty
        flip_direction = 1 if x[i, m] == 0 else -1
        return delta_energy * flip_direction

    @staticmethod
    def _update_local_energy_field(local_energy_field: np.ndarray,
                                   x: np.ndarray,
                                   i: int,
                                   m: int,
                                   qubo: np.ndarray,
                                   offset: float) -> np.ndarray:
        # If x[i, m] == 1, then was 0, and we add it; otherwise, now 0 and we remove
        flip_direction = 1 if x[i, m] == 1 else -1
        local_energy_field[:, m] += (flip_direction * qubo[i, :])
        return local_energy_field

    @staticmethod
    def _flip_element(x: np.ndarray, i: int, m: int) -> np.ndarray:
        x[i, m] = int(x[i, m] == 0)
        return x

    @staticmethod
    def _initialize_local_energy_field(x: np.ndarray, qubo: np.ndarray, offset: float) -> np.ndarray:
        local_energy_field = np.dot(qubo, x)
        return local_energy_field

    def __call__(self, qubo: np.ndarray, offset: float) -> Tuple:
        """
        :param qubo: Coupling coefficients of the QUBO problem.*
        :param offset: Energy offset of the QUBO problem.
        """
        return super().__call__(qubo, offset)


class SimulatedQuantumAnnealingIsing(AlgorithmIsing, SimulatedQuantumAnnealingCommon):
    """Ising model version of SQA."""

    def _compute_energy_delta(self,
                              x: np.ndarray,
                              local_energy_field: np.ndarray,
                              replica_coupling_penalty,
                              i: int,
                              m: int) -> float:
        return - x[i, m] * (local_energy_field[i, m] - replica_coupling_penalty)

    @staticmethod
    def _update_local_energy_field(local_energy_field: np.ndarray,
                                   x: np.ndarray,
                                   i: int,
                                   m: int,
                                   linear: np.ndarray,
                                   quadratic: np.ndarray,
                                   offset: float) -> np.ndarray:
        local_energy_field[:, m] += 2 * x[i, m] * quadratic[i, :]
        # The coefficient 2 is here because flipping the spin does multiply by -1, so removing the previous energy
        # and adding the new one equates adding twice the new one.
        return local_energy_field

    @staticmethod
    def _flip_element(x: np.ndarray, i: int, m: int) -> np.ndarray:
        x[i, m] = -x[i, m]
        return x

    @staticmethod
    def _initialize_local_energy_field(x, linear: np.ndarray, quadratic: np.ndarray, offset: float) -> np.ndarray:
        local_energy_field = np.dot(quadratic, x) + linear[:, None].repeat(x.shape[1], axis=1)
        return local_energy_field

    def __call__(self, linear: np.ndarray, quadratic: np.ndarray, offset: float) -> Tuple:
        """
        :param linear: Local magnetic field of the Ising model.
        :param quadratic: Coupling coefficients of the Ising model.
        :param offset: Energy offset of the QUBO problem.
        """
        return super().__call__(linear, quadratic, offset)
