from algorithms.algorithm import AlgorithmQUBO, AlgorithmIsing, Algorithm
import numpy as np
import math
import random
import mpmath
from typing import Tuple, Callable
from abc import abstractmethod
from utils.schedulers import Scheduler, LinearScheduler, HyperbolicScheduler
from utils.sampling import range_sampler


class SimulatedQuantumAnnealingCommon(Algorithm):
    """
    Based on Accelerating Simulated Quantum Annealing with GPU and Tensor Cores,
    by Yi-Hua Chung, Cheng-Jhih Shih and Shih-Hao Hung
    """

    def __init__(self,
                 monte_carlo_steps: int,
                 n_trotters: int,
                 temperature_scheduler: Scheduler = HyperbolicScheduler(start=1e9, end=8 / 7),
                 transverse_field_scheduler: Scheduler = LinearScheduler(start=8.0, end=0.),
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

    def _compute_replicas_coupling_strength(self, temperature: float, transverse_field: float) -> float:
        # Weird formula to compute the coupling strength, that increases during training to force trotters to
        # become the same.
        return (temperature / 2) * math.log(mpmath.coth(transverse_field / (self.n_trotters * temperature)))

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

        # Computation trick: it is possible to compute the delta_energy more simply by computing first a "local energy"
        # that is only updated if the change is accepted.
        local_energy = self._initialize_local_energy(x, *args, **kwargs)

        history = []  # We monitor the energy evolution at each Monte-Carlo step
        for t in range(self.monte_carlo_steps):
            history.append([t, self._select_best_energy_among_trotters(x, *args, **kwargs)])
            temperature = self.temperature_scheduler.update(t, self.monte_carlo_steps)
            transverse_field = self.transverse_field_scheduler.update(t, self.monte_carlo_steps)
            replicas_coupling_strength = self._compute_replicas_coupling_strength(temperature, transverse_field)

            for i in self.sampler(length):
                for m in self.sampler(self.n_trotters):
                    replica_coupling_penalty = self._compute_replica_coupling_penalty(x, replicas_coupling_strength,
                                                                                      i, m)
                    delta_energy = self._compute_energy_delta(x, local_energy, i, m, *args, **kwargs)
                    delta_energy = delta_energy + replica_coupling_penalty

                    # Metropolis algorithm: the random number is in [0,1]; if the tested change in x reduces the
                    # energy, then exp(-delta_energy / temperature) > 1 and the change is accepted; otherwise, we have
                    # exp(-delta_energy / temperature) in [0,1] too and then, the change is accepted in a probabilistic
                    # way, with decreasing chances as the energy increase becomes larger.
                    if math.exp(min(-delta_energy / temperature, 1)) > random.random():  # min to avoid math range error
                        # We accept the change in the i,m element of x
                        self._flip_element(x, i, m)
                        local_energy = self._update_local_energy(local_energy, x, i, m, *args, **kwargs)

        x = self._select_final_answer(x, *args, **kwargs)
        history.append([self.monte_carlo_steps, self.compute_energy(x, *args, **kwargs)])
        return x, history

    @abstractmethod
    def _compute_energy_delta(self,
                              x: np.ndarray,
                              local_energy: np.ndarray,
                              i: int,
                              m: int,
                              *args, **kwargs) -> float:
        """
        Gives the variation in energy when changing the [i,m] element of x.

        :param x: The candidate solution whose variations in energy we want to measure when changing the [i,m] element.
        :param local_energy: Energy associated to each element of x, in order to simplify calculations.
        :param i: Element i of any trotter of x.
        :param m: Trotter m of x.

        :return: The variation of energy when changing the [i,m] element of x.
        """
        # args and kwargs are here to be replaced with qubo + offset
        # or linear + quadratic + offset depending on whether it is implemented for QUBO or Ising model.
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _initialize_local_energy(x: np.ndarray, *args, **kwargs) -> np.ndarray:
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
    def _update_local_energy(local_energy: np.ndarray,
                             x: np.ndarray,
                             i: int,
                             m: int,
                             *args,
                             **kwargs) -> np.ndarray:
        """
        Updates the local energy field.

        :param local_energy: The energy field to update.
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

    @abstractmethod
    def _compute_replica_coupling_penalty(self,
                                          x: np.ndarray,
                                          replicas_coupling_strength: float,
                                          i: int,
                                          m: int) -> float:
        """
        Computes the penalty that enforces the trotters to converge toward a common solution.

        :param x: The ensemble of trotters.
        :param replicas_coupling_strength: A coefficient.
        :param i: Index of the element in each trotter.
        :param m: Index of the trotter.
        :return: The computed penalty.
        """
        raise NotImplementedError


class SimulatedQuantumAnnealingQUBO(AlgorithmQUBO, SimulatedQuantumAnnealingCommon):
    """
    QUBO version of SQA.
    We adapted some calculations to fit QUBO solutions in {0, 1} instead of {-1, 1}.
    """

    def _compute_energy_delta(self,
                              x: np.ndarray,
                              local_energy: np.ndarray,
                              i: int,
                              m: int,
                              qubo: np.ndarray,
                              offset: float) -> float:
        flip_direction = 1 if x[i, m] == 0 else -1  # x_after - x_before
        return flip_direction * (local_energy[i, m] + (qubo[i, i] * (1 - (2 * x[i, m]))))

    @staticmethod
    def _update_local_energy(local_energy: np.ndarray,
                             x: np.ndarray,
                             i: int,
                             m: int,
                             qubo: np.ndarray,
                             offset: float) -> np.ndarray:
        flip_direction = -1 if x[i, m] == 0 else 1
        local_energy[:, m] += flip_direction * (qubo[:, i] + qubo[i, :])
        return local_energy

    @staticmethod
    def _flip_element(x: np.ndarray, i: int, m: int) -> np.ndarray:
        x[i, m] = int(x[i, m] == 0)
        return x

    @staticmethod
    def _initialize_local_energy(x: np.ndarray, qubo: np.ndarray, offset: float) -> np.ndarray:
        return np.dot(qubo, x) + np.dot(qubo.T, x)

    def _compute_replica_coupling_penalty(self,
                                          x: np.ndarray,
                                          replicas_coupling_strength: float,
                                          i: int,
                                          m: int) -> float:
        # The coupling depends on the m-1 and m+1 trotters
        # It is supposed to penalize when the current spin takes a value different from the others, which
        # requires some modifications to work properly when using 0 or 1
        replica_coupling = 0
        if m + 1 != self.n_trotters:
            replica_coupling += x[i, m + 1] * 2 - 1  # Converts from {0,1} to {-1,1}
        if m != 0:
            replica_coupling += x[i, m - 1] * 2 - 1  # Converts from {0,1} to {-1,1}
        flip_direction = 2 if x[i, m] == 0 else -2  # 2 and not 1 to have the same magnitude as with Ising models
        return - replicas_coupling_strength * flip_direction * replica_coupling

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
                              local_energy: np.ndarray,
                              i: int,
                              m: int,
                              linear: np.ndarray,
                              quadratic: np.ndarray,
                              offset: float) -> float:
        # No quadratic[i,i] term because in Ising model the diagonal is 0
        return (-2 * x[i, m]) * (local_energy[i, m] + linear[i])

    @staticmethod
    def _update_local_energy(local_energy: np.ndarray,
                             x: np.ndarray,
                             i: int,
                             m: int,
                             linear: np.ndarray,
                             quadratic: np.ndarray,
                             offset: float) -> np.ndarray:
        local_energy[:, m] += 2 * x[i, m] * (quadratic[:, i] + quadratic[i, :])
        return local_energy

    @staticmethod
    def _flip_element(x: np.ndarray, i: int, m: int) -> np.ndarray:
        x[i, m] = -x[i, m]
        return x

    @staticmethod
    def _initialize_local_energy(x, linear: np.ndarray, quadratic: np.ndarray, offset: float) -> np.ndarray:
        return np.dot(quadratic, x) + np.dot(quadratic.T, x)

    def _compute_replica_coupling_penalty(self,
                                          x: np.ndarray,
                                          replicas_coupling_strength: float,
                                          i: int,
                                          m: int) -> float:
        # The coupling depends on the m-1 and m+1 trotters
        replica_coupling = 0
        if m + 1 != self.n_trotters:
            replica_coupling += x[i, m + 1]
        if m != 0:
            replica_coupling += x[i, m - 1]
        flip_direction = - 2 * x[i, m]
        return - replicas_coupling_strength * flip_direction * replica_coupling

    def __call__(self, linear: np.ndarray, quadratic: np.ndarray, offset: float) -> Tuple:
        """
        :param linear: Local magnetic field of the Ising model.
        :param quadratic: Coupling coefficients of the Ising model.
        :param offset: Energy offset of the QUBO problem.
        """
        return super().__call__(linear, quadratic, offset)
