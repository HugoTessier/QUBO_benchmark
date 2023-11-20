from algorithms.algorithm import IAlgorithm
import numpy as np
from typing import Tuple
from utils.schedulers import Scheduler, GeometricScheduler
from utils.data_struct import *


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

    @staticmethod
    def _preprocess_problem(problem: IsingData) -> IsingData:
        problem.extra = {"symmetric_J": -2 * (problem.J + problem.J.T)}
        return problem

    def __call__(self, problem: IsingData) -> Tuple:
        problem = self._preprocess_problem(problem)  # Allows computation optimization tricks
        length = self.get_length(problem)

        # Same initialization as the original implementation
        x = self.generate_random_solution(length)
        signal = np.copy(problem.h)

        history = []
        for step in range(self.monte_carlo_steps):
            history.append([step, self.compute_energy(x, problem)])

            temperature = self.temperature_scheduler.update(step, self.monte_carlo_steps)

            # Stochastic integral computing
            noise = self.noise_magnitude * (np.random.randint(size=signal.shape, low=0, high=2) * 2 - 1)
            signal += noise + np.dot(problem.extra['symmetric_J'], x) - problem.h

            # Approximation of the tanh
            signal[signal >= temperature] = temperature - self.alpha
            signal[signal < -temperature] = -temperature

            # Sign function without 0
            x = (signal >= 0.) * 2. - 1.

        history.append([self.monte_carlo_steps, self.compute_energy(x, problem)])
        return x, history
