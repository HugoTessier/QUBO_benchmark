from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple
from utils.data_struct import *


class Algorithm(ABC):
    """Any algorithm that is able to solve a combinatorial optimization problem."""

    @staticmethod
    @abstractmethod
    def generate_random_solution(*args: int) -> np.ndarray:
        """
        Generates a random solution, for a CO problem, of given length.

        :param args: Dimensions of the vector to generate.
        :return: The generated vector.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def compute_energy(x, problem: ProblemData) -> float:
        """
        Computes the Hamiltonian energy of the CO problem.

        :param x: The solution whose energy to evaluate.
        :param problem: Object containing the data of the problem to solve.
        :return: The evaluated Hamiltonian energy.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_length(problem: ProblemData) -> int:
        """
        Gets the length of the solution to provide to the CO problem.
        :param problem: Object containing the data of the problem to solve.
        :return: The length.
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(self, problem: ProblemData) -> Tuple:
        """
        Solves a CO problem using a given algorithm.
        :param problem: Object containing the data of the problem to solve.
        :return: The solution and a history of the optimization process.
        """
        raise NotImplementedError


class QAlgorithm(Algorithm):
    """
    Any algorithm that is able to solve a QUBO problem.
    """

    @staticmethod
    def generate_random_solution(*args: int) -> np.ndarray:
        return np.random.randint(size=args, low=0, high=2)

    @staticmethod
    def compute_energy(x: np.ndarray, problem: QUBOData) -> float:
        return np.dot(x.T, np.dot(problem.Q, x)) + problem.offset

    @staticmethod
    def get_length(problem: QUBOData) -> int:
        return problem.Q.shape[0]

    @abstractmethod
    def __call__(self, problem: QUBOData) -> Tuple:
        return super().__call__(problem)


class IAlgorithm(Algorithm):
    """Any algorithm that is able to solve an Ising problem."""

    @staticmethod
    def generate_random_solution(*args) -> np.ndarray:
        return np.random.randint(size=args, low=0, high=2) * 2 - 1

    @staticmethod
    def compute_energy(x: np.ndarray, problem: IsingData) -> float:
        return np.dot(problem.h, x) + np.dot(x.T, np.dot(problem.J, x)) + problem.offset

    @staticmethod
    def get_length(problem: IsingData) -> int:
        return problem.J.shape[0]

    @abstractmethod
    def __call__(self, problem: IsingData) -> Tuple:
        return super().__call__(problem)
