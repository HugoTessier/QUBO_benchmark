from algorithms.algorithm import QAlgorithm, IAlgorithm, Algorithm
import numpy as np
import random
import math
from abc import abstractmethod
from typing import Tuple
from utils.schedulers import Scheduler, GeometricScheduler
from utils.sampling import Sampler, RangeSampler
from utils.data_struct import *
from utils.history import *


class SimulatedAnnealingCommon(Algorithm):
    """
    Based on "Simulated annealing: From basics to applications"
    by Daniel Delahaye, Supatcha Chaimatanan and Marcel Mongeau.
    """

    def __init__(self,
                 monte_carlo_steps: int,
                 temperature_scheduler: Scheduler = GeometricScheduler(start=20, multiplier=0.9),
                 sampler: Sampler = RangeSampler()):
        """
        :param monte_carlo_steps: Number of Monte-Carlo steps, i.e. the outermost loop of the algorithm.
        :param temperature_scheduler: Controls the evolution of the temperature during annealing.
        :param sampler: Controls in which order to test changes in the value of elements in the solution.
        """
        super().__init__()
        self.monte_carlo_steps = monte_carlo_steps
        self.temperature_scheduler = temperature_scheduler
        self.sampler = sampler

    @abstractmethod
    def _flip_element(self, x: np.ndarray, i: int) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _compute_energy_delta(self, x: np.ndarray, i: int, local_energy: np.ndarray,
                              problem: ProblemData) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _initialize_local_energy(self, x: np.ndarray, problem: ProblemData) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _update_local_energy(self, local_energy: np.ndarray, x: np.ndarray, i: int, problem: ProblemData) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _preprocess_problem(problem: ProblemData) -> ProblemData:
        raise NotImplementedError

    def _metropolis_test(self, delta_energy, temperature):
        # Metropolis algorithm: the random number is in [0,1]; if the tested change in x reduces the
        # energy, then exp(-delta_energy / temperature) > 1 and the change is accepted; otherwise, we have
        # exp(-delta_energy / temperature) in [0,1] too and then, the change is accepted in a probabilistic
        # way, with decreasing chances as the energy increase becomes larger.

        self.oprec.sign_flip()  # -delta_energy
        self.oprec.division()  # / temperature
        self.oprec.exp()  # math.exp
        probability = math.exp(min(-delta_energy / temperature, 1))  # min() to avoid math range error

        self.oprec.random_number_generation(1)  # random.random()
        self.oprec.comparison()  # >
        return probability > random.random()

    def __call__(self, problem: ProblemData) -> Tuple[np.ndarray, History]:
        self.initialize_history_and_opset()
        problem = self._preprocess_problem(problem)  # Allows computation optimization tricks

        length = self.get_length(problem)

        self.oprec.random_number_generation(length)  # self.generate_random_solution(length)
        x = self.generate_random_solution(length)

        # Computation trick: it is possible to compute the delta_energy more simply by computing first a "local energy"
        # that is only updated if the change is accepted.
        local_energy = self._initialize_local_energy(x, problem)

        for step in range(self.monte_carlo_steps):
            self.history.record(ENERGY, self.compute_energy(x, problem))
            self.history.record(MAIN_LOOP)

            temperature = self.temperature_scheduler.update(step, self.monte_carlo_steps)
            for i in self.sampler(length):
                self.history.record(SEQUENCE)
                delta_energy = self._compute_energy_delta(x, i, local_energy, problem)

                if self._metropolis_test(delta_energy, temperature):  # min() to avoid math range error
                    # We accept the change in the i-th element of x
                    x = self._flip_element(x, i)
                    local_energy = self._update_local_energy(local_energy, x, i, problem)

        self.history.record(ENERGY, self.compute_energy(x, problem))
        return x, self.history


class QSimulatedAnnealing(QAlgorithm, SimulatedAnnealingCommon):
    """QUBO version of SA."""

    def _flip_element(self, x: np.ndarray, i: int) -> np.ndarray:
        self.oprec.spin_flip()
        x[i] = int(x[i] == 0)
        return x

    def _compute_energy_delta(self, x: np.ndarray, i: int, local_energy: np.ndarray, problem: QUBOData) -> np.ndarray:
        self.oprec.value_check()  # 1 if x[i] == 0 else -1
        flip_direction = 1 if x[i] == 0 else -1  # x_after - x_before

        self.oprec.multiplication()  # 2 * x[i]
        self.oprec.subtraction()  # 1 -  (2 * x[i])
        self.oprec.multiplication()  # problem.Q[i, i] * ...
        diagonal_term = problem.Q[i, i] * (1 - (2 * x[i]))

        self.oprec.addition()  # local_energy[i] + ...
        energy_delta = local_energy[i] + diagonal_term

        self.oprec.multiplication()  # flip_direction *
        return flip_direction * energy_delta

    def _initialize_local_energy(self, x: np.ndarray, problem: QUBOData) -> np.ndarray:
        self.oprec.dot_product(problem.extra['symmetric_Q'], x)
        return np.dot(problem.extra['symmetric_Q'], x)

    def _update_local_energy(self, local_energy: np.ndarray, x: np.ndarray, i: int, problem: QUBOData) -> np.ndarray:
        self.oprec.value_check()  # -1 if x[i] == 0 else 1
        flip_direction = -1 if x[i] == 0 else 1

        self.oprec.multiplication(problem.extra['symmetric_Q'][:, i].size)  # flip_direction * ...
        self.oprec.addition(problem.extra['symmetric_Q'][:, i].size)  # +=
        local_energy += flip_direction * problem.extra['symmetric_Q'][:, i]
        return local_energy

    @staticmethod
    def _preprocess_problem(problem: QUBOData) -> QUBOData:
        problem.extra = {"symmetric_Q": problem.Q + problem.Q.T}
        return problem

    def __call__(self, problem: QUBOData) -> Tuple[np.ndarray, History]:
        return super().__call__(problem)


class ISimulatedAnnealing(IAlgorithm, SimulatedAnnealingCommon):
    """Ising model version of SA."""

    def _flip_element(self, x: np.ndarray, i: int) -> np.ndarray:
        self.oprec.spin_flip()  # Could also say flip_sign but spin flip is more generic
        x[i] = -x[i]
        return x

    def _compute_energy_delta(self, x: np.ndarray, i: int, local_energy: np.ndarray, problem: IsingData) -> np.ndarray:
        self.oprec.addition()  # local_energy[i] + problem.h[i]
        self.oprec.multiplication()  # -2 * x[i]
        self.oprec.multiplication()  # ... * ...

        # No problem.J[i,i] term because in Ising model the diagonal is 0
        return (-2 * x[i]) * (local_energy[i] + problem.h[i])

    def _initialize_local_energy(self, x: np.ndarray, problem: IsingData) -> np.ndarray:
        self.oprec.dot_product(problem.extra['symmetric_J'], x)
        return np.dot(problem.extra['symmetric_J'], x)

    def _update_local_energy(self, local_energy: np.ndarray, x: np.ndarray, i: int, problem: IsingData) -> np.ndarray:
        self.oprec.multiplication()  # 2 * x[i]
        self.oprec.multiplication(problem.extra['symmetric_J'][:, i].size)  # 2 * x[i] * ...
        self.oprec.addition(problem.extra['symmetric_J'][:, i].size)  # +=

        local_energy += 2 * x[i] * problem.extra['symmetric_J'][:, i]
        return local_energy

    @staticmethod
    def _preprocess_problem(problem: IsingData) -> IsingData:
        problem.extra = {"symmetric_J": problem.J + problem.J.T}
        return problem

    def __call__(self, problem: IsingData) -> Tuple[np.ndarray, History]:
        return super().__call__(problem)
