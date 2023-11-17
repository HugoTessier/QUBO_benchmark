from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple


class Algorithm(ABC):
    """
    Any algorithm that is able to solve a combinatorial optimization problem.
    """

    @staticmethod
    @abstractmethod
    def generate_random_solution(length: int) -> np.ndarray:
        """
        Generates a random solution, for a CO problem, of given length.

        :param length: Length of the vector to generate.
        :return: The generated vector.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def compute_energy(x, *args, **kwargs) -> float:
        """
        Computes the Hamiltonian energy of the CO problem.

        :param x: The solution whose energy to evaluate.
        :return: The evaluated Hamiltonian energy.
        """
        # args and kwargs are here to be replaced with qubo + offset
        # or linear + quadratic + offset depending on whether it is implemented for QUBO or Ising model.
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_length(*args, **kwargs) -> int:
        """
        Gets the length of the solution to provide to the CO problem.
        :return: The length.
        """
        # args and kwargs are here to be replaced with qubo + offset
        # or linear + quadratic + offset depending on whether it is implemented for QUBO or Ising model.
        raise NotImplementedError

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Tuple:
        """
        Solves a CO problem using a given algorithm.

        :return: The solution and a history of the optimization process.
        """
        # args and kwargs are here to be replaced with qubo + offset
        # or linear + quadratic + offset depending on whether it is implemented for QUBO or Ising model.
        raise NotImplementedError


class QAlgorithm(Algorithm):
    """
    Any algorithm that is able to solve a QUBO problem.
    """

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

    @staticmethod
    def get_length(qubo: np.ndarray, offset: float) -> int:
        return qubo.shape[0]

    @abstractmethod
    def __call__(self, qubo: np.ndarray, offset: float) -> Tuple:
        """
        Solves a QUBO problem using a given algorithm.

        :param qubo: The matrix of the coupling coefficients of the QUBO problem.
        :param offset: Offset value of the energy.
        :return: The solution and a history of the optimization process.
        """
        return super().__call__(qubo, offset)


class IAlgorithm(Algorithm):
    """
    Any algorithm that is able to solve an Ising problem.
    """

    @staticmethod
    def generate_random_solution(length: int) -> np.ndarray:
        """
        Generates a random vector of bits in {-1,1} of a given length.

        :param length: Length of the vector to generate.
        :return: The generated vector.
        """
        x = np.ones(length)
        x[np.random.choice(length, length // 2, replace=False)] = -1.0
        return x

    @staticmethod
    def compute_energy(x: np.ndarray, linear: np.ndarray, quadratic: np.ndarray, offset: float) -> float:
        """
        Computes the Hamiltonian energy of the Ising problem.

        :param x: The solution whose energy to evaluate.
        :param linear: The vector of the local magnetic field coefficients of the Ising problem.
        :param quadratic: The matrix of the coupling coefficients of the Ising problem.
        :param offset: Offset value of the energy.
        :return: The evaluated Hamiltonian energy.
        """
        return np.dot(linear, x) + np.dot(x.T, np.dot(quadratic, x)) + offset

    @staticmethod
    def get_length(linear: np.ndarray, quadratic: np.ndarray, offset: float) -> int:
        return quadratic.shape[0]

    @abstractmethod
    def __call__(self, linear: np.ndarray, quadratic: np.ndarray, offset: float) -> Tuple:
        """
        Solves a QUBO problem using a given algorithm.

        :param linear: The vector of the local magnetic field coefficients of the Ising problem.
        :param quadratic: The matrix of the coupling coefficients of the Ising problem.
        :param offset: Offset value of the energy.
        :return: The solution and a history of the optimization process.
        """
        return super().__call__(linear, quadratic, offset)
