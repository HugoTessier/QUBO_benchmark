from abc import ABC, abstractmethod
import math


class Scheduler(ABC):
    @abstractmethod
    def update(self, i: int, n: int) -> float:
        pass


class LinearScheduler(Scheduler):
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end

    def update(self, i: int, n: int) -> float:
        return self.start + ((self.end - self.start) * (i / n))


class GeometricScheduler(Scheduler):
    def __init__(self, start: float, decay_rate: float):
        self.start = start
        self.decay_rate = decay_rate

    def update(self, i: int, n: int) -> float:
        return self.start * math.pow(self.decay_rate, i)
