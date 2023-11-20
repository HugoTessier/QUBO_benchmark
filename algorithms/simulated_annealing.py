from algorithms.algorithm import QAlgorithm, IAlgorithm, Algorithm
import numpy as np
import random
import math
from abc import abstractmethod
from typing import Tuple, Callable
from utils.schedulers import Scheduler, GeometricScheduler
from utils.sampling import range_sampler
from utils.data_struct import *


class SimulatedAnnealingCommon(Algorithm):
    """
    Based on "Simulated annealing: From basics to applications"
    by Daniel Delahaye, Supatcha Chaimatanan and Marcel Mongeau.
    """

    def __init__(self,
                 monte_carlo_steps: int,
                 temperature_scheduler: Scheduler = GeometricScheduler(start=20, multiplier=0.9),
                 sampler: Callable = range_sampler):
        """
        :param monte_carlo_steps: Number of Monte-Carlo steps, i.e. the outermost loop of the algorithm.
        :param temperature_scheduler: Controls the evolution of the temperature during annealing.
        :param sampler: Controls in which order to test changes in the value of elements in the solution.
        """
        self.monte_carlo_steps = monte_carlo_steps
        self.temperature_scheduler = temperature_scheduler
        self.sampler = sampler

    @staticmethod
    @abstractmethod
    def _flip_element(x: np.ndarray, i: int) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _compute_energy_delta(x: np.ndarray, i: int, local_energy: np.ndarray, problem: ProblemData) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _initialize_local_energy(x: np.ndarray, problem: ProblemData) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _update_local_energy(local_energy: np.ndarray, x: np.ndarray, i: int, problem: ProblemData) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _preprocess_problem(problem: ProblemData) -> ProblemData:
        raise NotImplementedError

    def __call__(self, problem: ProblemData) -> Tuple:
        problem = self._preprocess_problem(problem)  # Allows computation optimization tricks

        length = self.get_length(problem)
        x = self.generate_random_solution(length)
        # Computation trick: it is possible to compute the delta_energy more simply by computing first a "local energy"
        # that is only updated if the change is accepted.
        local_energy = self._initialize_local_energy(x, problem)

        history = []  # We monitor the energy evolution at each Monte-Carlo step
        for step in range(self.monte_carlo_steps):
            history.append([step, self.compute_energy(x, problem)])

            temperature = self.temperature_scheduler.update(step, self.monte_carlo_steps)
            for i in self.sampler(length):
                delta_energy = self._compute_energy_delta(x, i, local_energy, problem)

                # Metropolis algorithm: the random number is in [0,1]; if the tested change in x reduces the
                # energy, then exp(-delta_energy / temperature) > 1 and the change is accepted; otherwise, we have
                # exp(-delta_energy / temperature) in [0,1] too and then, the change is accepted in a probabilistic
                # way, with decreasing chances as the energy increase becomes larger.
                if math.exp(min(-delta_energy / temperature, 1)) > random.random():  # min() to avoid math range error
                    # We accept the change in the i-th element of x
                    x = self._flip_element(x, i)
                    local_energy = self._update_local_energy(local_energy, x, i, problem)

        history.append([self.monte_carlo_steps, self.compute_energy(x, problem)])
        return x, history


class QSimulatedAnnealing(QAlgorithm, SimulatedAnnealingCommon):
    """QUBO version of SA."""

    @staticmethod
    def _flip_element(x: np.ndarray, i: int) -> np.ndarray:
        x[i] = int(x[i] == 0)
        return x

    @staticmethod
    def _compute_energy_delta(x: np.ndarray, i: int, local_energy: np.ndarray, problem: QUBOData) -> np.ndarray:
        flip_direction = 1 if x[i] == 0 else -1  # x_after - x_before
        return flip_direction * (local_energy[i] + (problem.Q[i, i] * (1 - (2 * x[i]))))

    @staticmethod
    def _initialize_local_energy(x: np.ndarray, problem: QUBOData) -> np.ndarray:
        return np.dot(problem.extra['symmetric_Q'], x)

    @staticmethod
    def _update_local_energy(local_energy: np.ndarray, x: np.ndarray, i: int, problem: QUBOData) -> np.ndarray:
        flip_direction = -1 if x[i] == 0 else 1
        local_energy += flip_direction * problem.extra['symmetric_Q'][:, i]
        return local_energy

    @staticmethod
    def _preprocess_problem(problem: QUBOData) -> QUBOData:
        problem.extra = {"symmetric_Q": problem.Q + problem.Q.T}
        return problem

    def __call__(self, problem: QUBOData) -> Tuple:
        return super().__call__(problem)


class ISimulatedAnnealing(IAlgorithm, SimulatedAnnealingCommon):
    """Ising model version of SA."""

    @staticmethod
    def _flip_element(x: np.ndarray, i: int) -> np.ndarray:
        x[i] = -x[i]
        return x

    @staticmethod
    def _compute_energy_delta(x: np.ndarray, i: int, local_energy: np.ndarray, problem: IsingData) -> np.ndarray:
        # No problem.J[i,i] term because in Ising model the diagonal is 0
        return (-2 * x[i]) * (local_energy[i] + problem.h[i])

    @staticmethod
    def _initialize_local_energy(x: np.ndarray, problem: IsingData) -> np.ndarray:
        return np.dot(problem.extra['symmetric_J'], x)

    @staticmethod
    def _update_local_energy(local_energy: np.ndarray, x: np.ndarray, i: int, problem: IsingData) -> np.ndarray:
        local_energy += 2 * x[i] * problem.extra['symmetric_J'][:, i]
        return local_energy

    @staticmethod
    def _preprocess_problem(problem: IsingData) -> IsingData:
        problem.extra = {"symmetric_J": problem.J + problem.J.T}
        return problem

    def __call__(self, problem: IsingData) -> Tuple:
        return super().__call__(problem)
