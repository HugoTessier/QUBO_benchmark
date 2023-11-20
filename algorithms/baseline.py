from algorithms.algorithm import QAlgorithm, IAlgorithm
import numpy as np
from utils.data_struct import *


def qubo_baseline(n_trials: int, problem: QUBOData) -> float:
    """
    Computes a baseline energy using only random solutions, for a QUBO problem.

    :param n_trials: Number of tested random solutions, whose energy to average.
    :param problem: Object containing the data of the QUBO model.
    :return: The average energy.
    """
    energies = []
    length = QAlgorithm.get_length(problem)
    for _ in range(n_trials):
        x = QAlgorithm.generate_random_solution(length)
        energies.append(QAlgorithm.compute_energy(x, problem))
    return np.mean(np.array(energies))


def ising_baseline(n_trials: int, problem: IsingData) -> float:
    """
    Computes a baseline energy using only random solutions, for an Ising model.

    :param n_trials: Number of tested random solutions, whose energy to average.
    :param problem: Object containing the data of the Ising model.
    :return: The average energy.
    """
    energies = []
    length = IAlgorithm.get_length(problem)
    for _ in range(n_trials):
        x = IAlgorithm.generate_random_solution(length)
        energies.append(IAlgorithm.compute_energy(x, problem))
    return np.mean(np.array(energies))
