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
                 n_step: int,
                 a0: float = 1.,
                 a_scheduler: Scheduler = LinearScheduler(start=0., end=1.),
                 delta_t: float = 1.25,
                 initialization_range: float = 0.1):
        self.n_step = n_step
        self.a0 = a0
        self.delta_t = delta_t
        self.initialization_range = initialization_range
        self.a_scheduler = a_scheduler

    @staticmethod
    @abstractmethod
    def _compute_c0(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _binarize(x):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _clip(x, momenta):
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Tuple:
        length = self.get_length(*args, **kwargs)
        c0 = self._compute_c0(*args, **kwargs)
        history = []

        x = self.initialize_vector(length)
        momenta = self.initialize_vector(length)

        for t in range(self.n_step):
            history.append([t, self.compute_energy(self._binarize(x), *args, **kwargs)])
            a = self.a_scheduler.update(t, self.n_step)

            momenta += self._update_momenta(x, a, c0, *args, **kwargs)
            x += self._update_position(momenta)
            x, momenta = self._clip(x, momenta)

        x = self._binarize(x)
        history.append([self.n_step, self.compute_energy(x, *args, **kwargs)])
        return x, history

    @abstractmethod
    def initialize_vector(self, length):
        raise NotImplementedError

    def _update_position(self, momenta):
        return self.a0 * momenta * self.delta_t

    @abstractmethod
    def _update_momenta(self, x, a, c0, *args, **kwargs):
        raise NotImplementedError


class SimulatedBifurcationQUBO(AlgorithmQUBO, SimulatedBifurcationCommon):
    """
    Based on High-performance combinatorial optimization based on classical mechanics
    by Hayato Goto, Kotaro Endo, Masaru Suzuki, Yoshisato Sakai, Taro Kanao, Yohei Hamakawa, Ryo Hidaka, Masaya Yamasaki
    and Kosuke Tatsumura.
    """

    def __init__(self,
                 n_step: int,
                 a0: float = 1.,
                 a_scheduler: Scheduler = LinearScheduler(start=0., end=1.),
                 delta_t: float = 1.25,
                 initialization_range: float = 0.1):
        super().__init__(n_step, a0, a_scheduler, delta_t, initialization_range)

    @staticmethod
    def _compute_c0(qubo, offset):
        n = qubo.shape[0]
        return 0.5 / (math.sqrt((qubo ** 2).sum() / (n * (n - 1))) * math.sqrt(n))

    @staticmethod
    def _binarize(x):
        return (x > 0.5).astype(float)

    @staticmethod
    def _clip(x, momenta):
        momenta[np.where(np.bitwise_or(x > 1, x < 0))] = 0
        x = np.clip(x, 0, 1)
        return x, momenta

    def initialize_vector(self, length):
        return (np.random.random(length) - 0.5) * self.initialization_range + 0.5

    @abstractmethod
    def _update_momenta(self, x, a, c0, qubo, offset):
        raise NotImplementedError

    def __call__(self, qubo: np.ndarray, offset: float) -> Tuple:
        return super().__call__(qubo, offset)


class SimulatedBifurcationIsing(AlgorithmIsing, SimulatedBifurcationCommon):
    """
    Based on High-performance combinatorial optimization based on classical mechanics
    by Hayato Goto, Kotaro Endo, Masaru Suzuki, Yoshisato Sakai, Taro Kanao, Yohei Hamakawa, Ryo Hidaka, Masaya Yamasaki
    and Kosuke Tatsumura.
    """

    def __init__(self,
                 n_step: int,
                 a0: float = 1.,
                 a_scheduler: Scheduler = LinearScheduler(start=0., end=1.),
                 delta_t: float = 1.25,
                 initialization_range: float = 0.1):
        super().__init__(n_step, a0, a_scheduler, delta_t, initialization_range)

    @staticmethod
    def _compute_c0(linear, quadratic, offset):
        # What to do of the linear ???
        n = quadratic.shape[0]
        return 0.5 / (math.sqrt((quadratic ** 2).sum() / (n * (n - 1))) * math.sqrt(n))

    @staticmethod
    def _binarize(x):
        return np.sign(x)

    @staticmethod
    def _clip(x, momenta):
        momenta[np.where(np.bitwise_or(x > 1, x < -1))] = 0
        x = np.clip(x, -1, 1)
        return x, momenta

    def initialize_vector(self, length):
        return (np.random.random(length) - 0.5) * self.initialization_range

    @abstractmethod
    def _update_momenta(self, x, a, c0, linear, quadratic, offset):
        raise NotImplementedError

    def __call__(self, linear: np.ndarray, quadratic: np.ndarray, offset: float) -> Tuple:
        return super().__call__(linear, quadratic, offset)


class DiscreteSimulatedBifurcationQUBO(SimulatedBifurcationQUBO):
    """
    Based on High-performance combinatorial optimization based on classical mechanics
    by Hayato Goto, Kotaro Endo, Masaru Suzuki, Yoshisato Sakai, Taro Kanao, Yohei Hamakawa, Ryo Hidaka, Masaya Yamasaki
    and Kosuke Tatsumura.
    """

    def __init__(self,
                 n_step: int,
                 a0: float = 1.,
                 a_scheduler: Scheduler = LinearScheduler(start=0., end=1.),
                 delta_t: float = 1.25,
                 initialization_range: float = 0.1):
        super().__init__(n_step, a0, a_scheduler, delta_t, initialization_range)

    def _update_momenta(self, x, a, c0, qubo, offset):
        return self.delta_t * (-((self.a0 - a) * (x - 0.5)) + (c0 * (-qubo * self._binarize(x)).sum(axis=1)))


class BallisticSimulatedBifurcationQUBO(SimulatedBifurcationQUBO):
    """
    Based on High-performance combinatorial optimization based on classical mechanics
    by Hayato Goto, Kotaro Endo, Masaru Suzuki, Yoshisato Sakai, Taro Kanao, Yohei Hamakawa, Ryo Hidaka, Masaya Yamasaki
    and Kosuke Tatsumura.
    """

    def __init__(self,
                 n_step: int,
                 a0: float = 1.,
                 a_scheduler: Scheduler = LinearScheduler(start=0., end=1.),
                 delta_t: float = 1.25,
                 initialization_range: float = 0.1):
        super().__init__(n_step, a0, a_scheduler, delta_t, initialization_range)

    def _update_momenta(self, x, a, c0, qubo, offset):
        return self.delta_t * (-((self.a0 - a) * (x - 0.5)) + (c0 * (-qubo * x).sum(axis=1)))


class DiscreteSimulatedBifurcationIsing(SimulatedBifurcationIsing):
    """
    Based on High-performance combinatorial optimization based on classical mechanics
    by Hayato Goto, Kotaro Endo, Masaru Suzuki, Yoshisato Sakai, Taro Kanao, Yohei Hamakawa, Ryo Hidaka, Masaya Yamasaki
    and Kosuke Tatsumura.
    """

    def __init__(self,
                 n_step: int,
                 a0: float = 1.,
                 a_scheduler: Scheduler = LinearScheduler(start=0., end=1.),
                 delta_t: float = 1.25,
                 initialization_range: float = 0.1):
        super().__init__(n_step, a0, a_scheduler, delta_t, initialization_range)

    def _update_momenta(self, x, a, c0, linear, quadratic, offset):
        return self.delta_t * (-((self.a0 - a) * x) + (c0 * ((quadratic * self._binarize(x)).sum(axis=1) + linear)))


class BallisticSimulatedBifurcationIsing(SimulatedBifurcationIsing):
    """
    Based on High-performance combinatorial optimization based on classical mechanics
    by Hayato Goto, Kotaro Endo, Masaru Suzuki, Yoshisato Sakai, Taro Kanao, Yohei Hamakawa, Ryo Hidaka, Masaya Yamasaki
    and Kosuke Tatsumura.
    """

    def __init__(self,
                 n_step: int,
                 a0: float = 1.,
                 a_scheduler: Scheduler = LinearScheduler(start=0., end=1.),
                 delta_t: float = 1.25,
                 initialization_range: float = 0.1):
        super().__init__(n_step, a0, a_scheduler, delta_t, initialization_range)

    def _update_momenta(self, x, a, c0, linear, quadratic, offset):
        return self.delta_t * (-((self.a0 - a) * x) + (c0 * ((quadratic * x).sum(axis=1) + linear)))
