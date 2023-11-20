from abc import ABC, abstractmethod
from utils.data_struct import *
import numpy as np


class Problem(ABC):
    """
    Any combinatorial optimisation problem that can be turned into QUBO.
    """

    @abstractmethod
    def qubo(self, seed: int = 0) -> QUBOData:
        """
        Generate an instance of the problem, depending on a seed, and returns its corresponding QUBO.

        :param seed: Seed to generate the problem.
        :return: Instance of the problem as a QUBO, corresponding to the given seed, under the form of a QUBOData
        object containing the 'Q' matrix as a numpy array and the energy 'offset' as a float.
        """
        raise NotImplementedError

    @abstractmethod
    def ising(self, seed: int = 0) -> IsingData:
        """
        Generate an instance of the problem, depending on a seed, and returns its corresponding Ising model.

        :param seed: Seed to generate the problem.
        :return: Instance of the problem as an Ising model, corresponding to the given seed, under the form of a
        IsingData object containing the 'h' and 'J' matrices as numpy arrays and the energy 'offset' as a float.
        """
        raise NotImplementedError

    @abstractmethod
    def visualize(self, seed: int, x: np.ndarray) -> None:
        """
        Gives a visualization of the solution.

        :param seed: Seed to recreate the problem.
        :param x: The solution to visualize.
        """
        raise NotImplementedError
