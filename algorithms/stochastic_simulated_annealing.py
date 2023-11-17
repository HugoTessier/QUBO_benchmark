from algorithms.algorithm import IAlgorithm
import numpy as np
from typing import Tuple
from utils.schedulers import Scheduler, GeometricScheduler


class IStochasticSimulatedAnnealing(IAlgorithm):
    """
    Based on "Fast-Converging Simulated Annealing for Ising Models Based on Integral Stochastic Computing"
    by Naoya Onizawa, Kota Katsuki, Duckgyu Shin, Warren J. Gross and Takahiro Hanyu
    and "STOCHASTIC SIMULATED QUANTUM ANNEALING FOR FAST SOLUTION OF COMBINATORIAL OPTIMIZATION PROBLEMS"
    by Naoya Onizawa, Ryoma Sasaki, Duckgyu Shin and Warren J. Gross
    and "Memory-Efficient FPGA Implementation of Stochastic Simulated Annealing"
    by Duckgyu Shin, Naoya Onizawa, and Warren J. Gross.
    """

    def __init__(self,
                 monte_carlo_steps: int,
                 temperature_scheduler: Scheduler = GeometricScheduler(start=5, multiplier=1.),
                 noise_magnitude: float = 2.,
                 alpha: float = 1.):
        """
        :param monte_carlo_steps: Number of Monte-Carlo steps, i.e. the outermost loop of the algorithm.
        :param temperature_scheduler: Controls the evolution of the temperature during annealing.
        :param noise_magnitude: Magnitude of the stochastic part of the integral method.
        :param alpha: Magic value, please refer to the original paper because I have no idea why.
        """
        self.monte_carlo_steps = monte_carlo_steps
        self.temperature_scheduler = temperature_scheduler
        self.noise_magnitude = noise_magnitude
        self.alpha = alpha

    def __call__(self, linear, quadratic, offset) -> Tuple:
        """
        :param linear: Local magnetic field of the Ising model.
        :param quadratic: Coupling coefficients of the Ising model.
        :param offset: Energy offset of the QUBO problem.
        """
        quad = -2 * (quadratic + quadratic.T)  # Algorithm originally designed for symmetric Ising models
        length = self.get_length(linear, quadratic, offset)

        # Same initialization as the original implementation
        x = self.generate_random_solution(length)
        signal = np.copy(linear)

        history = []
        for step in range(self.monte_carlo_steps):
            history.append([step, self.compute_energy(x, linear, quadratic, offset)])

            temperature = self.temperature_scheduler.update(step, self.monte_carlo_steps)

            # Stochastic integral computing
            noise = self.noise_magnitude * (np.random.randint(size=signal.shape, low=0, high=2) * 2 - 1)
            signal += noise + np.dot(quad, x) - linear

            # Approximation of the tanh
            signal[signal >= temperature] = temperature - self.alpha
            signal[signal < -temperature] = -temperature

            # Sign function without 0
            x = (signal >= 0.) * 2. - 1.

        history.append([self.monte_carlo_steps, self.compute_energy(x, linear, quadratic, offset)])
        return x, history
