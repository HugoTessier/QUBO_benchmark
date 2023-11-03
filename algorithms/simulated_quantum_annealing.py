from algorithms.algorithm import Algorithm
import numpy as np
import math
import random
import mpmath
from typing import Tuple


class SimulatedQuantumAnnealing(Algorithm):
    """
    Based on Accelerating Simulated Quantum Annealing with GPU and Tensor Cores,
    by Yi-Hua Chung, Cheng-Jhih Shih and Shih-Hao Hung
    and An Ising computer based on simulated quantum annealing by path integral Monte Carlo method
    by Takuya Okuyama, Masato Hayashi and Masanao Yamaoka.

    We adapted some calculations to fit QUBO solutions in {0, 1} instead of {-1, 1}.
    """

    def __init__(self,
                 monte_carlo_steps: int,
                 n_trotters: int,
                 start_transverse_field: float = 2.,
                 end_transverse_field: float = 0.,
                 start_temperature: float = 4.0,
                 end_temperature: float = 1 / 32):
        """
        :param monte_carlo_steps: Number of Monte-Carlo steps, i.e. the outermost loop of the algorithm.
        :param n_trotters: Number of replicas that are optimized in parallel and eventually fused together.
        :param start_transverse_field: Starting value of the transverse field, that gives the strength of the coupling
        between trotters.
        :param end_transverse_field: Final value of the transverse field.
        :param start_temperature: Starting value of the temperature, that gives the chance of accepting changes.
        :param end_temperature: Final value of the temperature.
        """
        self.monte_carlo_steps = monte_carlo_steps
        self.n_trotters = n_trotters
        self.start_transverse_field = start_transverse_field
        self.end_transverse_field = end_transverse_field
        self.start_temperature = start_temperature
        self.end_temperature = end_temperature

    def _compute_trotters_mean_energy(self, x: np.ndarray, qubo: np.ndarray, offset: float) -> float:
        # Since, for most of the process, each trotter is more or less independent, we choose to report the mean of
        # the energy of each trotter
        energies = []
        for m in range(self.n_trotters):
            energies.append(self.compute_energy(x[:, m], qubo, offset))
        return np.mean(np.array(energies))

    def _update_temperature(self, t: int) -> float:
        # Linear interpolation; in order to avoid 0 values, we do not take t+1 or monte_carlo_steps-1
        temperature = self.start_temperature + (
                (t / self.monte_carlo_steps) * (self.end_temperature - self.start_temperature))
        return temperature

    def _update_transverse_field(self, t: int) -> float:
        # Linear interpolation; in order to avoid 0 values, we do not take t+1 or monte_carlo_steps-1
        transverse_field = self.start_transverse_field + (
                (t / self.monte_carlo_steps) * (self.end_transverse_field - self.start_transverse_field))
        return transverse_field

    def _compute_energy_delta(self,
                              x: np.ndarray,
                              local_field: np.ndarray,
                              temperature: float,
                              transverse_field: float,
                              i: int,
                              m: int) -> float:
        # Weird formula to compute the coupling strength, that increases during training to force trotters to
        # become the same.
        replicas_coupling_strength = (temperature / 2) * math.log(
            mpmath.coth(transverse_field / (self.n_trotters * temperature)))
        # The coupling depends on the m-1 and m+1 trotters
        replica_coupling = 0
        if m + 1 != self.n_trotters:
            replica_coupling += x[i, m + 1]
        if m != 0:
            replica_coupling += x[i, m - 1]
        # We compute the difference in energy, with a sign that depends on whether the element is set to 0 or 1, and
        # therefore removed or added.
        delta_energy = local_field[i, m] - replicas_coupling_strength * replica_coupling
        flip_direction = 1 if x[i, m] == 0 else -1
        return delta_energy * flip_direction

    @staticmethod
    def _fuse_trotters(x: np.ndarray) -> np.ndarray:
        return (x.T.mean(axis=0) >= 0.5).astype(int)  # Majority judgment over which bits should be 0 or 1

    def __call__(self, qubo: np.ndarray, offset: float) -> Tuple:
        """
        Solves a QUBO problem using Simulated Quantum Annealing (SQA).

        :param qubo: The matrix of the coupling coefficients of the QUBO problem.
        :param offset: Offset value of the energy.
        :return: The solution and a history of the optimization process.
        """

        length = qubo.shape[0]

        # SQA is based on the principle of running n_trotter copies in parallel, that each try to solve the problem
        # and which are progressively applied a penalty to force them to converge toward a common solution at the end.
        # These copies are called "Trotters".
        x = np.stack([self.generate_random_solution(length) for _ in range(self.n_trotters)]).T

        # Since the hamiltonian is purely a sum, the energy delta induced by changing one element of x at a time
        # can be calculated using only a "local" energy. It is then more efficient to compute said "local energy" field
        # beforehand and to update it progressively.
        local_energy_field = np.dot(qubo, x)

        history = []
        for t in range(self.monte_carlo_steps):
            history.append([t, self._compute_trotters_mean_energy(x, qubo, offset)])
            temperature = self._update_temperature(t)  # Update temperature
            transverse_field = self._update_transverse_field(t)  # update transverse_field
            for i in range(length):
                for m in range(self.n_trotters):
                    delta_energy = self._compute_energy_delta(x, local_energy_field, temperature, transverse_field, i,
                                                              m)

                    # Metropolis algorithm: the random number is in [0,1]; if the tested change in x reduces the
                    # energy, then exp(-delta_energy / temperature) > 1 and the change is accepted; otherwise, we have
                    # exp(-delta_energy / temperature) in [0,1] too and then, the change is accepted in a probabilistic
                    # way, with decreasing chances as the energy increase becomes larger.
                    if math.exp(-delta_energy / temperature) > random.random():
                        # We flip the i,m element in x
                        if x[i, m] == 1:
                            x[i, m] = 0
                        else:
                            x[i, m] = 1

                        # If x[i, m] == 1, then was 0, and we add it; otherwise, now 0 and we remove
                        flip_direction = 1 if x[i, m] == 1 else -1
                        local_energy_field[:, m] += (flip_direction * qubo[i, :])

        x = self._fuse_trotters(x)
        history.append([self.monte_carlo_steps, self.compute_energy(x, qubo, offset)])
        return x, history
