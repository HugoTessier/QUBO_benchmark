from abc import ABC, abstractmethod
import random


class SamplerIterator(ABC):
    """
    Combinatorial Problem solver is usually iterative; and each iteration often requires iterating over each element
    of the candidate solution.
    The sampler allows to define the strategy with which to iterate over these elements.
    """

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass


def range_sampler(n):
    """Linear iteration."""

    class RangeSampler(SamplerIterator):
        def __iter__(self):
            self.n = n
            self.i = 0
            return self

        def __next__(self):
            if self.i == self.n:
                raise StopIteration
            i = self.i
            self.i += 1
            return i

    return iter(RangeSampler())


def shuffle_sampler(n):
    """Random iterator."""

    class ShuffleSampler(SamplerIterator):
        def __iter__(self):
            self.values = [i for i in range(n)]
            random.shuffle(self.values)
            self.i = 0
            return self

        def __next__(self):
            if self.i == len(self.values):
                raise StopIteration
            i = self.values[self.i]
            self.i += 1
            return i

    return iter(ShuffleSampler())
