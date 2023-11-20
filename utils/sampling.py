from abc import ABC, abstractmethod
import random
from typing import Self


class SamplerIterator(ABC):
    """
    Combinatorial Problem solver is usually iterative; and each iteration often requires iterating over each element
    of the candidate solution.
    The sampler allows to define the strategy with which to iterate over these elements.
    """

    @abstractmethod
    def __iter__(self) -> Self:
        raise NotImplementedError

    @abstractmethod
    def __next__(self) -> int:
        raise NotImplementedError


def range_sampler(n: int) -> SamplerIterator:
    """Linear iteration over n elements."""

    class RangeSampler(SamplerIterator):
        def __iter__(self) -> Self:
            self.n = n
            self.i = 0
            return self

        def __next__(self) -> int:
            if self.i == self.n:
                raise StopIteration
            i = self.i
            self.i += 1
            return i

    return iter(RangeSampler())


def shuffle_sampler(n: int) -> SamplerIterator:
    """Random iterator over n elements."""

    class ShuffleSampler(SamplerIterator):
        def __iter__(self) -> Self:
            self.values = [i for i in range(n)]
            random.shuffle(self.values)
            self.i = 0
            return self

        def __next__(self) -> int:
            if self.i == len(self.values):
                raise StopIteration
            i = self.values[self.i]
            self.i += 1
            return i

    return iter(ShuffleSampler())
