from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple


class Algorithm(ABC):
    @staticmethod
    def generate_random_solution(length: int) -> np.ndarray:
        """
        Generates a random vector of bits in {0,1} of a given length.

        :param length: Length of the vector to generate.
        :return: The generated vector.
        """
        x = np.zeros(length)
        x[np.random.choice(length, length // 2, replace=False)] = 1.0
        return x

    @staticmethod
    def compute_energy(x: np.ndarray, qubo: np.ndarray, offset: float) -> float:
        """
        Computes the Hamiltonian energy of the QUBO problem.

        :param x: The solution whose energy to evaluate.
        :param qubo: The matrix of the coupling coefficients of the QUBO problem.
        :param offset: Offset value of the energy.
        :return: The evaluated Hamiltonian energy.
        """
        return np.dot(x.T, np.dot(qubo, x)) + offset

    @abstractmethod
    def __call__(self, qubo: np.ndarray, offset: float) -> Tuple:
        """
        Solves a QUBO problem using a given algorithm.

        :param qubo: The matrix of the coupling coefficients of the QUBO problem.
        :param offset: Offset value of the energy.
        :return: The solution and a history of the optimization process.
        """
        pass
