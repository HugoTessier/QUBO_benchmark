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
        raise NotImplementedError


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


class HyperbolicScheduler(Scheduler):
    """Hyperbolic interpolation."""

    def __init__(self, start: float, end: float):
        """
        :param start: Value at which the parameter starts.
        :param end: Value at which the parameter ends.
        """
        self.start = 1 / start
        self.end = 1 / end

    def update(self, i: int, n: int) -> float:
        return 1 / (self.start + ((self.end - self.start) * (i / n)))


class GeometricScheduler(Scheduler):
    """Geometric evolution."""

    def __init__(self, start: float, multiplier: float, max_value: float = None, min_value: float = None):
        """
        :param start: Value at which the parameter starts.
        :param multiplier: Coefficient by which each previous value is multiplied.
        :param max_value: Optional, provides a maximum value above which not to go.
        :param min_value: Optional, provides a minimum value below which not to go.
        """
        self.start = start
        self.multiplier = multiplier
        self.max_value = max_value
        self.min_value = min_value

    def update(self, i: int, n: int) -> float:
        value = self.start * math.pow(self.multiplier, i)
        if self.min_value is not None:
            value = max(self.min_value, value)
        if self.max_value is not None:
            value = min(self.max_value, value)
        return value


class DiscreteScheduler(Scheduler):
    def __init__(self, scheduler: Scheduler, n_plateau: int):
        self.scheduler = scheduler
        self.n_plateau = n_plateau

    def update(self, i: int, n: int) -> float:
        if i >= (n // self.n_plateau) * self.n_plateau:
            i = (n // self.n_plateau) * self.n_plateau - 1
        return self.scheduler.update(i // (n // self.n_plateau), n // self.n_plateau)


class WarmRestartScheduler(Scheduler):
    def __init__(self, scheduler: Scheduler, n_restarts: int):
        self.scheduler = scheduler
        self.n_restarts = n_restarts

    def update(self, i: int, n: int) -> float:
        if i >= (n // self.n_restarts) * self.n_restarts:
            i = (n // self.n_restarts) * self.n_restarts - 1
        return self.scheduler.update(i % (n // self.n_restarts), n // self.n_restarts)
