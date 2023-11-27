from algorithms.algorithm import IAlgorithm
import numpy as np
from typing import Tuple
import math
from abc import abstractmethod
from utils.schedulers import Scheduler, LinearScheduler
from utils.data_struct import *
from utils.history import *


class ISimulatedBifurcation(IAlgorithm):
    """
    Based on High-performance combinatorial optimization based on classical mechanics
    by Hayato Goto, Kotaro Endo, Masaru Suzuki, Yoshisato Sakai, Taro Kanao, Yohei Hamakawa, Ryo Hidaka, Masaya Yamasaki
    and Kosuke Tatsumura.
    """

    def __init__(self,
                 euler_steps: int,
                 a0: float = 1.,
                 a_scheduler: Scheduler = LinearScheduler(start=0., end=1.),
                 delta_t: float = 1.25,
                 initialization_range: float = 0.1):
        """
        :param euler_steps: Number of steps of the Euler method, i.e. the outermost loop of the algorithm.
        :param a0: Final value of the control parameter a.
        :param a_scheduler: Controls the evolution of the control parameter a
        :param delta_t: Time step of the Euler method.
        :param initialization_range: Positions and momenta are initialized around a middle state, with a random range.
        """
        super().__init__()
        self.euler_steps = euler_steps
        self.a0 = a0
        self.delta_t = delta_t
        self.initialization_range = initialization_range
        self.a_scheduler = a_scheduler

    def _compute_position_variation(self, momenta: np.ndarray) -> np.ndarray:
        self.oprec.float_multiplication(momenta.size * 2)
        return self.a0 * momenta * self.delta_t

    @abstractmethod
    def _compute_momenta_variation(self, x: np.ndarray, a: float, c0: float, problem) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _compute_c0(problem) -> float:
        # It is unclear in the paper how we should generalize this initialization when there is a linear term.
        n = problem.extra['symmetric_J'].shape[0]
        return 0.5 / (math.sqrt((problem.extra['symmetric_J'] ** 2).sum() / (n * (n - 1))) * math.sqrt(n))

    def _binarize(self, x: np.ndarray) -> np.ndarray:
        self.oprec.float_sign(x.size)
        return (x >= 0.) * 2. - 1.

    def _clip(self, x: np.ndarray, momenta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.oprec.float_comparison(x.size * 2)  # x > 1, x < -1
        self.oprec.bitwise_or(x.size)  # np.bitwise_or
        self.oprec.conditional_fill(momenta.size)
        momenta[np.where(np.bitwise_or(x > 1, x < -1))] = 0
        self.oprec.clip(x.size)
        x = np.clip(x, -1, 1)
        return x, momenta

    def initialize_vector(self, length: int) -> np.ndarray:
        self.oprec.random_number_generation(length)
        return (np.random.random(length) - 0.5) * self.initialization_range

    @staticmethod
    def _preprocess_problem(problem: IsingData) -> IsingData:
        problem.extra = {"symmetric_J": -2 * (problem.J + problem.J.T)}
        return problem

    def __call__(self, problem: IsingData) -> Tuple[np.ndarray, History]:
        self.initialize_history_and_opset()
        problem = self._preprocess_problem(problem)  # Allows computation optimization tricks

        length = self.get_length(problem)
        c0 = self._compute_c0(problem)

        x = self.initialize_vector(length)  # Positions of the element of the solution.
        momenta = self.initialize_vector(length)  # Momentum of each element of x within the search space.

        for t in range(self.euler_steps):
            self.history.record(ENERGY, self.compute_energy(self._binarize(x), problem))
            self.history.record(OLS)
            self.history.record(ILS)

            a = self.a_scheduler.update(t, self.euler_steps)

            # Euler's method: solving differential equations iteratively.
            momenta += self._compute_momenta_variation(x, a, c0, problem)
            self.oprec.float_addition(momenta.size)
            x += self._compute_position_variation(momenta)
            self.oprec.float_addition(x.size)
            x, momenta = self._clip(x, momenta)  # "Inelastic walls" to reduce the "analog errors"

        x = self._binarize(x)  # We turn the real-values into a valid combinatorial solution.
        self.history.record(ENERGY, self.compute_energy(x, problem))
        return x, self.history


class IDiscreteSimulatedBifurcation(ISimulatedBifurcation):
    """Ising model version of DSB."""

    def _compute_momenta_variation(self, x: np.ndarray, a: float, c0: float, problem: IsingData) -> np.ndarray:
        self.oprec.dot_product(problem.extra['symmetric_J'], x)
        self.oprec.float_addition(x.size)  # ... + problem.h
        self.oprec.float_multiplication(x.size)  # c0 * ...
        self.oprec.float_subtraction(1)  # a - self.a0
        self.oprec.float_multiplication(x.size)  # ((a - self.a0) * x)
        self.oprec.float_addition(x.size)  # ... + ...
        self.oprec.float_multiplication(x.size)  # self.delta_t * ...
        return self.delta_t * (((a - self.a0) * x) + (
                c0 * (np.dot(problem.extra['symmetric_J'], self._binarize(x)) + problem.h)))


class IBallisticSimulatedBifurcation(ISimulatedBifurcation):
    """Ising model version of BSB."""

    def _compute_momenta_variation(self, x: np.ndarray, a: float, c0: float, problem: IsingData) -> np.ndarray:
        self.oprec.dot_product(problem.extra['symmetric_J'], x)
        self.oprec.float_addition(x.size)  # ... + problem.h
        self.oprec.float_multiplication(x.size)  # c0 * ...
        self.oprec.float_subtraction(1)  # a - self.a0
        self.oprec.float_multiplication(x.size)  # ((a - self.a0) * x)
        self.oprec.float_addition(x.size)  # ... + ...
        self.oprec.float_multiplication(x.size)  # self.delta_t * ...
        return self.delta_t * (
                -((self.a0 - a) * x) + (c0 * (np.dot(problem.extra['symmetric_J'], x) + problem.h)))
