from algorithms.algorithm import IAlgorithm
import numpy as np
from typing import Tuple
from utils.schedulers import Scheduler, GeometricScheduler
from utils.data_struct import *
from utils.history import *


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
        super().__init__()
        self.monte_carlo_steps = monte_carlo_steps
        self.temperature_scheduler = temperature_scheduler
        self.noise_magnitude = noise_magnitude
        self.alpha = alpha

    @staticmethod
    def _preprocess_problem(problem: IsingData) -> IsingData:
        problem.extra = {"symmetric_J": -2 * (problem.J + problem.J.T)}
        return problem

    def tanh(self, signal: np.ndarray, temperature: float) -> np.ndarray:
        # Approximation of the tanh

        self.oprec.subtraction()
        upper_value = temperature - self.alpha

        self.oprec.comparison(signal.size)  # signal >= temperature
        self.oprec.conditional_fill(signal.size)
        signal[signal >= temperature] = upper_value

        self.oprec.sign_flip()  # - temperature
        lower_value = -temperature

        self.oprec.comparison(signal.size)  # signal < -temperature
        self.oprec.conditional_fill(signal.size)
        signal[signal < -temperature] = lower_value

        return signal

    def update_signal(self, noise: np.ndarray, problem: IsingData, signal: np.ndarray, x: np.ndarray) -> np.ndarray:
        self.oprec.dot_product(problem.extra['symmetric_J'], x)
        self.oprec.subtraction(problem.h.size)  # - problem.h
        energy_term = np.dot(problem.extra['symmetric_J'], x) - problem.h

        self.oprec.addition(x.size)  # noise + ...
        self.oprec.addition(signal.size)  # signal +=
        signal += noise + energy_term
        return signal

    def generate_noise(self, signal: np.ndarray) -> np.ndarray:
        self.oprec.random_number_generation(signal.size)
        return self.noise_magnitude * (np.random.randint(size=signal.shape, low=0, high=2) * 2 - 1)

    def __call__(self, problem: IsingData) -> Tuple[np.ndarray, History]:
        self.initialize_history_and_opset()
        problem = self._preprocess_problem(problem)  # Allows computation optimization tricks
        length = self.get_length(problem)

        # Same initialization as the original implementation
        self.oprec.random_number_generation(length)
        x = self.generate_random_solution(length)
        signal = np.copy(problem.h)  # Same initialization as original implementation

        for step in range(self.monte_carlo_steps):
            self.history.record(ENERGY, self.compute_energy(x, problem))
            self.history.record(MAIN_LOOP)
            self.history.record(SEQUENCE)

            temperature = self.temperature_scheduler.update(step, self.monte_carlo_steps)

            # Stochastic integral computing
            noise = self.generate_noise(signal)
            signal = self.update_signal(noise, problem, signal, x)
            signal = self.tanh(signal, temperature)

            # Sign function without 0
            self.oprec.sign(signal.size)
            x = (signal >= 0.) * 2. - 1.

        self.history.record(ENERGY, self.compute_energy(x, problem))
        return x, self.history
