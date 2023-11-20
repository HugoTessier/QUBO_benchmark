from algorithms.algorithm import IAlgorithm
import numpy as np
from typing import Tuple
import math
from abc import abstractmethod
from utils.schedulers import Scheduler, LinearScheduler


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
        self.euler_steps = euler_steps
        self.a0 = a0
        self.delta_t = delta_t
        self.initialization_range = initialization_range
        self.a_scheduler = a_scheduler

    def _compute_position_variation(self, momenta: np.ndarray) -> np.ndarray:
        return self.a0 * momenta * self.delta_t

    @abstractmethod
    def _compute_momenta_variation(self,
                                   x: np.ndarray,
                                   a: float,
                                   c0: float,
                                   linear: np.ndarray,
                                   quadratic: np.ndarray,
                                   offset: float) -> np.ndarray:
        """
        :param linear: Local magnetic field of the Ising model.
        :param quadratic: Coupling coefficients of the Ising model.
        :param offset: Energy offset of the QUBO problem.
        """
        raise NotImplementedError

    @staticmethod
    def _compute_c0(linear: np.ndarray, quadratic: np.ndarray, offset: float) -> float:
        # It is unclear in the paper how we should generalize this initialization when there is a linear term.
        n = quadratic.shape[0]
        return 0.5 / (math.sqrt((quadratic ** 2).sum() / (n * (n - 1))) * math.sqrt(n))

    @staticmethod
    def _binarize(x: np.ndarray) -> np.ndarray:
        return np.sign(x)

    @staticmethod
    def _clip(x: np.ndarray, momenta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        momenta[np.where(np.bitwise_or(x > 1, x < -1))] = 0
        x = np.clip(x, -1, 1)
        return x, momenta

    def initialize_vector(self, length: int) -> np.ndarray:
        return (np.random.random(length) - 0.5) * self.initialization_range

    def __call__(self, linear: np.ndarray, quadratic: np.ndarray, offset: float) -> Tuple:
        """
        :param linear: Local magnetic field of the Ising model.
        :param quadratic: Coupling coefficients of the Ising model.
        :param offset: Energy offset of the QUBO problem.
        """
        quad = -2 * (quadratic + quadratic.T)
        length = self.get_length(linear, quadratic, offset)
        c0 = self._compute_c0(linear, quad, offset)

        x = self.initialize_vector(length)  # Positions of the element of the solution.
        momenta = self.initialize_vector(length)  # Momentum of each element of x within the search space.

        history = []  # We monitor the energy evolution at each Euler step
        for t in range(self.euler_steps):
            history.append([t, self.compute_energy(self._binarize(x), linear, quadratic, offset)])

            a = self.a_scheduler.update(t, self.euler_steps)

            # Euler's method: solving differential equations iteratively.
            momenta += self._compute_momenta_variation(x, a, c0, linear, quad, offset)
            x += self._compute_position_variation(momenta)
            x, momenta = self._clip(x, momenta)  # "Inelastic walls" to reduce the "analog errors"

        x = self._binarize(x)  # We turn the real-values into a valid combinatorial solution.
        history.append([self.euler_steps, self.compute_energy(x, linear, quadratic, offset)])
        return x, history


class IDiscreteSimulatedBifurcation(ISimulatedBifurcation):
    """Ising model version of DSB."""

    def _compute_momenta_variation(self,
                                   x: np.ndarray,
                                   a: float,
                                   c0: float,
                                   linear: np.ndarray,
                                   quadratic: np.ndarray,
                                   offset: float) -> np.ndarray:
        return self.delta_t * (-((self.a0 - a) * x) + (c0 * ((quadratic * self._binarize(x)).sum(axis=1) + linear)))


class IBallisticSimulatedBifurcation(ISimulatedBifurcation):
    """Ising model version of BSB."""

    def _compute_momenta_variation(self,
                                   x: np.ndarray,
                                   a: float,
                                   c0: float,
                                   linear: np.ndarray,
                                   quadratic: np.ndarray,
                                   offset: float) -> np.ndarray:
        return self.delta_t * (-((self.a0 - a) * x) + (c0 * ((quadratic * x).sum(axis=1) + linear)))
