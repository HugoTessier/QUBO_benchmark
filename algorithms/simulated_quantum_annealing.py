from algorithms.algorithm import Algorithm
import numpy as np
import math
import random
import mpmath


class SimulatedQuantumAnnealing(Algorithm):
    def __init__(self,
                 monte_carlo_steps: int,
                 n_trotters: int,
                 start_transverse_field: float = 2.,
                 end_transverse_field: float = 0.,
                 start_temperature: float = 4.0,
                 end_temperature: float = 1 / 32):
        self.monte_carlo_steps = monte_carlo_steps
        self.n_trotters = n_trotters
        self.start_transverse_field = start_transverse_field
        self.end_transverse_field = end_transverse_field
        self.start_temperature = start_temperature
        self.end_temperature = end_temperature

    def solve_qubo(self, qubo: np.ndarray, offset: float):
        """
        Based on Accelerating Simulated Quantum Annealing with GPU and Tensor Cores,
        by Yi-Hua Chung, Cheng-Jhih Shih and Shih-Hao Hung
        """
        length = qubo.shape[0]
        x = np.stack([self.generate_random_qubo_solution(length) for _ in range(self.n_trotters)]).T
        local_field = self.initialize_local_field_energy(length, qubo, x)
        history = []
        for t in range(self.monte_carlo_steps):
            history.append([t, self.compute_qubo((x.T.mean(axis=0) >= 0.5).astype(int), qubo, offset)])
            temperature = self.update_temperature(t)  # Update temperature
            transverse_field = self.update_transverse_field(t)  # update transverse_field
            for i in range(length):
                for m in range(self.n_trotters):
                    delta_energy = self.compute_energy_delta(x, local_field, temperature, transverse_field, i, m)

                    if math.exp(-delta_energy / temperature) > random.random():  # Judge to flip
                        if x[i, m] == 1:
                            x[i, m] = 0
                        else:
                            x[i, m] = 1

                        for j in range(length):  # Update local-field energy
                            # Original paper propose local_field += 2*qubo[i, j]*x[i,m], but that's because it was
                            # meant for Ising models with spins equal to -1 or 1, whereas for QUBO values are 0 or 1.
                            if x[i, m] == 1:
                                local_field[j, m] += qubo[i, j]
                            else:
                                local_field[j, m] -= qubo[i, j]
            # print(x)
        history.append([self.monte_carlo_steps, self.compute_qubo((x.T.mean(axis=0) >= 0.5).astype(int), qubo, offset)])
        return (x.T.mean(axis=0) >= 0.5).astype(int), history

    def compute_energy_delta(self, x, local_field, temperature, transverse_field, i, m):
        replicas_coupling_strength = (temperature / 2) * math.log(
            mpmath.coth(transverse_field / (self.n_trotters * temperature)))
        if m != 0 and m != self.n_trotters - 1:
            replica_coupling = x[i, m + 1] + x[i, m - 1]
        elif m == 0:
            replica_coupling = x[i, m + 1]
        else:
            replica_coupling = x[i, m - 1]
        delta_energy = local_field[i, m] - replicas_coupling_strength * replica_coupling
        if x[i, m] == 1:
            return -delta_energy  # Because flipping it would remove the term
        else:
            return delta_energy  # Because, if x[i,m] == 0, flipping it would add the term

    def initialize_local_field_energy(self, length, qubo, x):
        local_field = np.zeros((length, self.n_trotters))
        for m in range(self.n_trotters):
            for i in range(length):
                for j in range(length):
                    # print(local_field.shape, qubo.shape, x.shape, i, j, m)
                    local_field[i, m] += qubo[i, j] * x[j, m]
        return local_field

    def update_temperature(self, t):
        # Linear interpolation
        temperature = self.start_temperature + (
                (t / self.monte_carlo_steps) * (self.end_temperature - self.start_temperature))
        return temperature

    def update_transverse_field(self, t):
        # Linear interpolation
        transverse_field = self.start_transverse_field + (
                (t / self.monte_carlo_steps) * (self.end_transverse_field - self.start_transverse_field))
        return transverse_field
