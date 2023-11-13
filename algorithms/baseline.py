from algorithms.algorithm import AlgorithmQUBO, AlgorithmIsing
import numpy as np


def qubo_baseline(n_trials: int, qubo: np.ndarray, offset: float) -> float:
    """
    Computes a baseline energy using only random solutions, for a QUBO problem.

    :param n_trials: Number of tested random solutions, whose energy to average.
    :param qubo: Coupling coefficients of the QUBO problem.
    :param offset: Energy offset of the QUBO problem.
    :return: The average energy.
    """
    energies = []
    length = AlgorithmQUBO.get_length(qubo, offset)
    for _ in range(n_trials):
        x = AlgorithmQUBO.generate_random_solution(length)
        energies.append(AlgorithmQUBO.compute_energy(x, qubo, offset))
    return np.mean(np.array(energies))


def ising_baseline(n_trials: int, linear: np.ndarray, quadratic: np.ndarray, offset: float) -> float:
    """
    Computes a baseline energy using only random solutions, for an Ising model.

    :param n_trials: Number of tested random solutions, whose energy to average.
    :param linear: Local magnetic field of the ising model.
    :param quadratic: Coupling coefficients of the Ising model.
    :param offset: Energy offset of the Ising model.
    :return: The average energy.
    """
    energies = []
    length = AlgorithmIsing.get_length(linear, quadratic, offset)
    for _ in range(n_trials):
        x = AlgorithmIsing.generate_random_solution(length)
        energies.append(AlgorithmIsing.compute_energy(x, linear, quadratic, offset))
    return np.mean(np.array(energies))
