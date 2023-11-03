from algorithms.algorithm import Algorithm
import numpy as np
import random
import math


class SimulatedAnnealing(Algorithm):
    def __init__(self, outer_loop_steps: int, temperature_decay_rate: float = 0.9, acceptance_rate: float = 0.99):
        self.outer_loop_steps = outer_loop_steps
        self.temperature_decay_rate = temperature_decay_rate
        self.acceptance_rate = acceptance_rate

    @staticmethod
    def probability(energy: float, new_energy: float, temperature: float):
        return math.exp((energy - new_energy) / temperature)

    def update_temperature(self, temperature):
        return temperature * self.temperature_decay_rate

    def compute_initial_temperature(self, energy_deltas):
        energy_deltas = np.array(energy_deltas)
        energy_increase = energy_deltas[energy_deltas <= 0]
        energy_decrease = energy_deltas[energy_deltas > 0]
        m1 = energy_decrease.shape[0]
        m2 = energy_increase.shape[0]
        increase_mean = np.abs(energy_increase).mean()
        return increase_mean / math.log(m2 / ((m2 * self.acceptance_rate) - (m1 * (1 - self.acceptance_rate))))

    def solve_qubo(self, qubo: np.ndarray, offset: float):
        """
        Based on "Simulated annealing: From basics to applications" by Daniel Delahaye, Supatcha Chaimatanan
        and Marcel Mongeau.
        """
        history = []

        temperature = 0  # Initial temperature is computed after a first iteration
        n = qubo.shape[0]
        x = self.generate_random_qubo_solution(n)
        energy = self.compute_qubo(x, qubo, offset)
        energy_deltas = []
        for step in range(self.outer_loop_steps):
            for index in np.random.choice(n, n, replace=False):
                x_modified = self.flip_bit(x, index)
                new_energy = self.compute_qubo(x_modified, qubo, offset)
                if step == 0:
                    energy_deltas.append(energy - new_energy)
                    # First iterations acts as if the probability equals 100%
                    x = x_modified
                    energy = new_energy
                else:
                    if new_energy < energy:
                        x = x_modified
                        energy = new_energy
                    else:
                        if random.uniform(0, 1) <= self.probability(energy, new_energy, temperature):
                            x = x_modified
                            energy = new_energy
            history.append([step, energy])
            if step == 0:
                temperature = self.compute_initial_temperature(energy_deltas)
            temperature = self.update_temperature(temperature)
        return x, history
