from algorithms.algorithm import QAlgorithm, IAlgorithm, Algorithm
import numpy as np
import math
import random
import mpmath
from typing import Tuple, Callable
from abc import abstractmethod
from utils.schedulers import Scheduler, LinearScheduler, HyperbolicScheduler
from utils.sampling import range_sampler
from utils.data_struct import *
from utils.history import *


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
        super().__init__()
        self.monte_carlo_steps = monte_carlo_steps
        self.n_trotters = n_trotters
        self.temperature_scheduler = temperature_scheduler
        self.transverse_field_scheduler = transverse_field_scheduler
        self.sampler = sampler

    def _select_best_energy_among_trotters(self, x: np.ndarray, problem: ProblemData) -> float:
        # Since each trotter is a candidate solution, we report the energy of the best one.
        # Indeed, at the end of the annealing, we return the trotter of least energy.
        return min([self.compute_energy(solution, problem) for solution in x.T])

    def _compute_replicas_coupling_strength(self, temperature: float, transverse_field: float) -> float:
        # Weird formula to compute the coupling strength, that increases during training to force trotters to
        # become the same.
        return (temperature / 2) * math.log(mpmath.coth(transverse_field / (self.n_trotters * temperature)))

    def _select_final_answer(self, x: np.ndarray, problem: ProblemData) -> np.ndarray:
        # Gives the trotter with the best energy. We do that, instead of giving an average, because, not only does
        # it lead to non-acceptable values (e.g. zeros for Ising models), but also because the average of the solutions
        # is not guaranteed to give the average of their energies.
        energies = []
        for solution in x.T:
            energies.append(self.compute_energy(solution, problem))
        return x.T[np.argmin(np.array(energies))]

    @abstractmethod
    def _compute_energy_delta(self, x: np.ndarray, local_energy: np.ndarray, i: int, m: int,
                              problem: ProblemData) -> float:
        raise NotImplementedError

    @abstractmethod
    def _initialize_local_energy(self, x: np.ndarray, problem: ProblemData) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _update_local_energy(self, local_energy: np.ndarray, x: np.ndarray, i: int, m: int,
                             problem: ProblemData) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _flip_element(self, x: np.ndarray, i: int, m: int) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _compute_replica_coupling_penalty(self, x: np.ndarray, replicas_coupling_strength: float, i: int,
                                          m: int) -> float:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _preprocess_problem(problem: ProblemData) -> ProblemData:
        raise NotImplementedError

    def metropolis_test(self, delta_energy, temperature):
        # Metropolis algorithm: the random number is in [0,1]; if the tested change in x reduces the
        # energy, then exp(-delta_energy / temperature) > 1 and the change is accepted; otherwise, we have
        # exp(-delta_energy / temperature) in [0,1] too and then, the change is accepted in a probabilistic
        # way, with decreasing chances as the energy increase becomes larger.

        self.oprec.float_sign_flip()  # -delta_energy
        self.oprec.float_division()  # / temperature
        self.oprec.float_exp()  # math.exp
        self.oprec.random_number_generation(1)  # random.random()
        self.oprec.float_comparison()  # >

        return math.exp(min(-delta_energy / temperature, 1)) > random.random()

    def __call__(self, problem: ProblemData) -> Tuple[np.ndarray, History]:
        self.initialize_history_and_opset()
        problem = self._preprocess_problem(problem)  # Allows computation optimization tricks
        length = self.get_length(problem)

        # SQA is based on the principle of running n_trotter copies in parallel, that each try to solve the problem
        # and which are progressively applied a penalty to force them to converge toward a common solution at the end.
        # These copies are called "Trotters".
        self.oprec.random_number_generation(length * self.n_trotters)
        x = self.generate_random_solution(length, self.n_trotters)

        # Computation trick: it is possible to compute the delta_energy more simply by computing first a "local energy"
        # that is only updated if the change is accepted.
        local_energy = self._initialize_local_energy(x, problem)

        for t in range(self.monte_carlo_steps):
            self.history.record(ENERGY, self._select_best_energy_among_trotters(x, problem))
            self.history.record(OLS)
            temperature = self.temperature_scheduler.update(t, self.monte_carlo_steps)
            transverse_field = self.transverse_field_scheduler.update(t, self.monte_carlo_steps)
            replicas_coupling_strength = self._compute_replicas_coupling_strength(temperature, transverse_field)

            for i in self.sampler(length):
                for m in self.sampler(self.n_trotters):
                    self.history.record(ILS)
                    replica_coupling_penalty = self._compute_replica_coupling_penalty(x, replicas_coupling_strength,
                                                                                      i, m)
                    delta_energy = self._compute_energy_delta(x, local_energy, i, m, problem)
                    delta_energy = delta_energy + replica_coupling_penalty

                    if self.metropolis_test(delta_energy, temperature):  # min to avoid math range error
                        # We accept the change in the i,m element of x
                        self._flip_element(x, i, m)
                        local_energy = self._update_local_energy(local_energy, x, i, m, problem)

        x = self._select_final_answer(x, problem)
        self.history.record(ENERGY, self.compute_energy(x, problem))
        return x, self.history


class QSimulatedQuantumAnnealing(QAlgorithm, SimulatedQuantumAnnealingCommon):
    """
    QUBO version of SQA.
    We adapted some calculations to fit QUBO solutions in {0, 1} instead of {-1, 1}.
    """

    def _compute_energy_delta(self, x: np.ndarray, local_energy: np.ndarray, i: int, m: int,
                              problem: QUBOData) -> float:
        self.oprec.value_check()  # 1 if x[i] == 0 else -1
        flip_direction = 1 if x[i, m] == 0 else -1  # x_after - x_before

        self.oprec.float_multiplication()  # 2 * x[i, m]
        self.oprec.float_sign_flip()  # - (2 * x[i, m])
        self.oprec.float_addition()  # 1 - ...
        self.oprec.float_multiplication()  # problem.Q[i, i] * ...
        self.oprec.float_addition()  # local_energy[i, m] + ...
        self.oprec.float_multiplication()  # flip_direction *
        return flip_direction * (local_energy[i, m] + (problem.Q[i, i] * (1 - (2 * x[i, m]))))

    def _update_local_energy(self, local_energy: np.ndarray, x: np.ndarray, i: int, m: int,
                             problem: QUBOData) -> np.ndarray:
        self.oprec.value_check()  # - 1 if x[i, m] == 0 else 1
        flip_direction = -1 if x[i, m] == 0 else 1

        self.oprec.float_multiplication(problem.extra['symmetric_Q'][:, i].size)  # flip_direction * ...
        self.oprec.float_addition(local_energy[:, m].size)  # +=
        local_energy[:, m] += flip_direction * problem.extra['symmetric_Q'][:, i]
        return local_energy

    def _flip_element(self, x: np.ndarray, i: int, m: int) -> np.ndarray:
        self.oprec.spin_flip()
        x[i, m] = int(x[i, m] == 0)
        return x

    def _initialize_local_energy(self, x: np.ndarray, problem: QUBOData) -> np.ndarray:
        self.oprec.dot_product(problem.extra['symmetric_Q'], x)
        return np.dot(problem.extra['symmetric_Q'], x)

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
            self.oprec.float_addition()
            replica_coupling += x[i, m + 1] * 2 - 1  # Converts from {0,1} to {-1,1}
        if m != 0:
            self.oprec.float_addition()
            replica_coupling += x[i, m - 1] * 2 - 1  # Converts from {0,1} to {-1,1}
        self.oprec.value_check()
        flip_direction = 2 if x[i, m] == 0 else -2  # 2 and not 1 to have the same magnitude as with Ising models
        self.oprec.float_multiplication(2)
        self.oprec.float_sign_flip()
        return - replicas_coupling_strength * flip_direction * replica_coupling

    @staticmethod
    def _preprocess_problem(problem: QUBOData) -> QUBOData:
        problem.extra = {"symmetric_Q": problem.Q + problem.Q.T}
        return problem

    def __call__(self, problem: QUBOData) -> Tuple[np.ndarray, History]:
        return super().__call__(problem)


class ISimulatedQuantumAnnealing(IAlgorithm, SimulatedQuantumAnnealingCommon):
    """Ising model version of SQA."""

    def _compute_energy_delta(self, x: np.ndarray, local_energy: np.ndarray, i: int, m: int,
                              problem: IsingData) -> float:
        # No quadratic[i,i] term because in Ising model the diagonal is 0
        self.oprec.float_addition()
        self.oprec.float_multiplication(2)
        return -2 * x[i, m] * (local_energy[i, m] + problem.h[i])

    def _update_local_energy(self, local_energy: np.ndarray, x: np.ndarray, i: int, m: int,
                             problem: IsingData) -> np.ndarray:
        self.oprec.float_multiplication()  # 2 * x[i, m]
        self.oprec.float_multiplication(problem.extra['symmetric_J'][:, i].size)  # ... * ...
        self.oprec.float_addition(local_energy[:, m].size)  # +=
        local_energy[:, m] += 2 * x[i, m] * problem.extra['symmetric_J'][:, i]
        return local_energy

    def _flip_element(self, x: np.ndarray, i: int, m: int) -> np.ndarray:
        self.oprec.spin_flip()
        x[i, m] = -x[i, m]
        return x

    def _initialize_local_energy(self, x, problem: IsingData) -> np.ndarray:
        self.oprec.dot_product(problem.extra['symmetric_J'], x)
        return np.dot(problem.extra['symmetric_J'], x)

    def _compute_replica_coupling_penalty(self,
                                          x: np.ndarray,
                                          replicas_coupling_strength: float,
                                          i: int,
                                          m: int) -> float:
        # The coupling depends on the m-1 and m+1 trotters
        replica_coupling = 0
        if m + 1 != self.n_trotters:
            self.oprec.float_addition()
            replica_coupling += x[i, m + 1]
        if m != 0:
            self.oprec.float_addition()
            replica_coupling += x[i, m - 1]
        self.oprec.float_multiplication()
        flip_direction = 2 * x[i, m]
        self.oprec.float_multiplication(2)
        return replicas_coupling_strength * flip_direction * replica_coupling

    @staticmethod
    def _preprocess_problem(problem: IsingData) -> IsingData:
        problem.extra = {"symmetric_J": problem.J + problem.J.T}
        return problem

    def __call__(self, problem: IsingData) -> Tuple[np.ndarray, History]:
        return super().__call__(problem)
