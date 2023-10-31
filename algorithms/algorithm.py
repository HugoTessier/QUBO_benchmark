from abc import ABC, abstractmethod
import numpy as np


class Algorithm(ABC):
    @staticmethod
    def generate_random_qubo_solution(length: int) -> np.ndarray:
        x = np.zeros(length)
        x[np.random.choice(length, length // 2, replace=False)] = 1.0
        return x

    @staticmethod
    def generate_random_ising_solution(length: int) -> np.ndarray:
        x = np.ones(length)
        x[np.random.choice(length, length // 2, replace=False)] = -1.0
        return x

    @staticmethod
    def flip_bit(x: np.ndarray, index: int) -> np.ndarray:
        y = np.copy(x)
        if y[index] == 0.:
            y[index] = 1.
            return y
        else:
            y[index] = 0.
            return y

    @staticmethod
    def flip_spin(x: np.ndarray, index: int) -> np.ndarray:
        y = np.copy(x)
        if y[index] == 1.:
            y[index] = -1.
            return y
        else:
            y[index] = 1.
            return y

    @staticmethod
    def compute_qubo(x: np.ndarray, qubo: np.ndarray, offset: float) -> float:
        return np.dot(x.T, np.dot(qubo, x)) + offset

    @staticmethod
    def compute_ising(x: np.ndarray, linear: np.ndarray, quadratic: np.ndarray, offset: float) -> float:
        return np.dot(x.T, np.dot(quadratic, x)) + np.dot(linear, x) + offset

    @abstractmethod
    def solve_qubo(self, qubo: np.ndarray, offset: float):
        pass

    @abstractmethod
    def solve_ising(self, linear: np.ndarray, quadratic: np.ndarray, offset: float):
        pass
