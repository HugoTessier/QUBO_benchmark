from abc import ABC, abstractmethod
import math


class Scheduler(ABC):
    """
    Multiple Combinatorial Problem solvers require to vary control parameters during their process.
    The scheduler allows to control the strategy with which to make these parameters vary.
    """

    @abstractmethod
    def update(self, i: int, n: int) -> float:
        """
        Provides the next value for the parameter.

        :param i: The i-th iteration in the loop during which to make the parameter vary.
        :param n: The total number of iterations to do.

        :return: The new value of the parameter.
        """
        pass


class LinearScheduler(Scheduler):
    """Linear interpolation."""

    def __init__(self, start: float, end: float):
        """
        :param start: Value at which the parameter starts.
        :param end: Value at which the parameter ends.
        """
        self.start = start
        self.end = end

    def update(self, i: int, n: int) -> float:
        return self.start + ((self.end - self.start) * (i / n))


class GeometricScheduler(Scheduler):
    """Geometric evolution."""

    def __init__(self, start: float, decay_rate: float):
        """
        :param start: Value at which the parameter starts.
        :param decay_rate: Proportion by which each previous value is decayed.
        """
        self.start = start
        self.decay_rate = decay_rate

    def update(self, i: int, n: int) -> float:
        return self.start * math.pow(self.decay_rate, i)
