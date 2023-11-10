from algorithms.algorithm import AlgorithmQUBO, AlgorithmIsing, Algorithm
import numpy as np
from typing import Tuple
import math
from abc import abstractmethod
from utils.schedulers import Scheduler, LinearScheduler


class SimulatedBifurcationCommon(Algorithm):
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

    @staticmethod
    @abstractmethod
    def _compute_c0(*args, **kwargs) -> float:
        """
        Creation of the c0 parameter.
        
        :return: The initialized c0.
        """
        # args and kwargs are here to be replaced with qubo + offset
        # or linear + quadratic + offset depending on whether it is implemented for QUBO or Ising model.
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _binarize(x: np.ndarray) -> np.ndarray:
        """
        Takes the real-value x and converts it into a solution of a valid form.

        :param x: The vector to binarize.
        :return: The binarized x.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _clip(x: np.ndarray, momenta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        When the real-value elements of x exceed given thresholds, they are clipped and their corresponding momentum
        is set to 0. This is the 'inelastic walls' mentioned in the paper.

        :param x: The positions.
        :param momenta: The momenta.
        :return: The updated x and momenta.
        """
        raise NotImplementedError

    @abstractmethod
    def initialize_vector(self, length: int) -> np.ndarray:
        """
        Initializes a vector of given length, around a certain value, with a certain range.

        :param length: Length of the vector.
        :return: The initialized vector.
        """
        raise NotImplementedError

    def _compute_position_variation(self, momenta: np.ndarray) -> np.ndarray:
        return self.a0 * momenta * self.delta_t

    @abstractmethod
    def _compute_momenta_variation(self, x: np.ndarray, a: float, c0: float, *args, **kwargs) -> np.ndarray:
        """
        Computes the next Euler step for the variation of momenta.

        :param x: The positions.
        :param a: Control parameter a.
        :param c0: The c0 positive constant parameter.

        :return: The variation of momenta with which to update them.
        """
        # args and kwargs are here to be replaced with qubo + offset
        # or linear + quadratic + offset depending on whether it is implemented for QUBO or Ising model.
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Tuple:
        # args and kwargs are here to be replaced with qubo + offset
        # or linear + quadratic + offset depending on whether it is implemented for QUBO or Ising model.

        length = self.get_length(*args, **kwargs)
        c0 = self._compute_c0(*args, **kwargs)

        x = self.initialize_vector(length)  # Positions of the element of the solution.
        momenta = self.initialize_vector(length)  # Momentum of each element of x within the search space.

        history = []  # We monitor the energy evolution at each Euler step
        for t in range(self.euler_steps):
            history.append([t, self.compute_energy(self._binarize(x), *args, **kwargs)])

            a = self.a_scheduler.update(t, self.euler_steps)

            # Euler's method: solving differential equations iteratively.
            momenta += self._compute_momenta_variation(x, a, c0, *args, **kwargs)
            x += self._compute_position_variation(momenta)
            x, momenta = self._clip(x, momenta)  # "Inelastic walls" to reduce the "analog errors"

        x = self._binarize(x)  # We turn the real-values into a valid combinatorial solution.
        history.append([self.euler_steps, self.compute_energy(x, *args, **kwargs)])
        return x, history


class SimulatedBifurcationQUBO(AlgorithmQUBO, SimulatedBifurcationCommon):
    """QUBO version of SB."""

    @staticmethod
    def _compute_c0(qubo: np.ndarray, offset: float) -> float:
        n = qubo.shape[0]
        return 0.5 / (math.sqrt((qubo ** 2).sum() / (n * (n - 1))) * math.sqrt(n))

    @staticmethod
    def _binarize(x: np.ndarray) -> np.ndarray:
        return (x > 0.5).astype(float)

    @staticmethod
    def _clip(x: np.ndarray, momenta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        momenta[np.where(np.bitwise_or(x > 1, x < 0))] = 0
        x = np.clip(x, 0, 1)
        return x, momenta

    def initialize_vector(self, length: int) -> np.ndarray:
        return (np.random.random(length) - 0.5) * self.initialization_range + 0.5

    @abstractmethod
    def _compute_momenta_variation(self,
                                   x: np.ndarray,
                                   a: float,
                                   c0: float,
                                   qubo: np.ndarray,
                                   offset: np.ndarray) -> np.ndarray:
        """
        :param qubo: Coupling coefficients of the QUBO problem.*
        :param offset: Energy offset of the QUBO problem.
        """
        raise NotImplementedError

    def __call__(self, qubo: np.ndarray, offset: float) -> Tuple:
        """
        :param qubo: Coupling coefficients of the QUBO problem.*
        :param offset: Energy offset of the QUBO problem.
        """
        return super().__call__(qubo, offset)


class SimulatedBifurcationIsing(AlgorithmIsing, SimulatedBifurcationCommon):
    """Ising model version of SB."""

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

    def __call__(self, linear: np.ndarray, quadratic: np.ndarray, offset: float) -> Tuple:
        """
        :param linear: Local magnetic field of the Ising model.
        :param quadratic: Coupling coefficients of the Ising model.
        :param offset: Energy offset of the QUBO problem.
        """
        return super().__call__(linear, quadratic, offset)


class DiscreteSimulatedBifurcationQUBO(SimulatedBifurcationQUBO):
    """QUBO version of DSB."""

    def _compute_momenta_variation(self,
                                   x: np.ndarray,
                                   a: float,
                                   c0: float,
                                   qubo: np.ndarray,
                                   offset: np.ndarray) -> np.ndarray:
        return self.delta_t * (-((self.a0 - a) * (x - 0.5)) + (c0 * (-qubo * self._binarize(x)).sum(axis=1)))


class BallisticSimulatedBifurcationQUBO(SimulatedBifurcationQUBO):
    """QUBO version of BSB."""

    def _compute_momenta_variation(self,
                                   x: np.ndarray,
                                   a: float,
                                   c0: float,
                                   qubo: np.ndarray,
                                   offset: np.ndarray) -> np.ndarray:
        return self.delta_t * (-((self.a0 - a) * (x - 0.5)) + (c0 * (-qubo * x).sum(axis=1)))


class DiscreteSimulatedBifurcationIsing(SimulatedBifurcationIsing):
    """Ising model version of DSB."""

    def _compute_momenta_variation(self,
                                   x: np.ndarray,
                                   a: float,
                                   c0: float,
                                   linear: np.ndarray,
                                   quadratic: np.ndarray,
                                   offset: float) -> np.ndarray:
        return self.delta_t * (-((self.a0 - a) * x) + (c0 * ((quadratic * self._binarize(x)).sum(axis=1) + linear)))


class BallisticSimulatedBifurcationIsing(SimulatedBifurcationIsing):
    """Ising model version of BSB."""

    def _compute_momenta_variation(self,
                                   x: np.ndarray,
                                   a: float,
                                   c0: float,
                                   linear: np.ndarray,
                                   quadratic: np.ndarray,
                                   offset: float) -> np.ndarray:
        return self.delta_t * (-((self.a0 - a) * x) + (c0 * ((quadratic * x).sum(axis=1) + linear)))
