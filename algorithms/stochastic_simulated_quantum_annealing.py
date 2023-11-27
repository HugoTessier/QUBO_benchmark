from algorithms.algorithm import IAlgorithm
import numpy as np
from typing import Tuple
from utils.schedulers import Scheduler, GeometricScheduler, LinearScheduler
from utils.data_struct import *
from utils.history import *


class IStochasticSimulatedQuantumAnnealing(IAlgorithm):
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
                 n_trotters: int,
                 temperature_scheduler: Scheduler = GeometricScheduler(start=5, multiplier=1.),
                 coupling_scheduler: Scheduler = LinearScheduler(start=0, end=0.5),
                 noise_magnitude: float = 1,
                 alpha: float = 1.0,
                 delay_cycle: int = 1):
        """
        :param monte_carlo_steps: Number of Monte-Carlo steps, i.e. the outermost loop of the algorithm.
        :param n_trotters: Number of replicas that are optimized in parallel and eventually fused together.
        :param temperature_scheduler: Controls the evolution of the temperature during annealing.
        :param temperature_scheduler: Controls the evolution of the trotters coupling strength during annealing.
        :param noise_magnitude: Magnitude of the stochastic part of the integral method.
        :param alpha: Magic value, please refer to the original paper because I have no idea why.
        :param delay_cycle: Trotters are coupled, but with a delay in the MC steps.
        """
        super().__init__()
        self.monte_carlo_steps = monte_carlo_steps
        self.n_trotters = n_trotters
        self.temperature_scheduler = temperature_scheduler
        self.coupling_scheduler = coupling_scheduler
        self.noise_magnitude = noise_magnitude
        self.alpha = alpha
        self.delay_cycle = delay_cycle

    def _select_best_energy_among_trotters(self, x: np.ndarray, problem: IsingData) -> float:
        # Since each trotter is a candidate solution, we report the energy of the best one.
        # Indeed, at the end of the annealing, we return the trotter of least energy.
        return min([self.compute_energy(solution, problem) for solution in x.T])

    def _select_final_answer(self, x: np.ndarray, problem: IsingData) -> np.ndarray:
        # Gives the trotter with the best energy. We do that, instead of giving an average, because, not only does
        # it lead to non-acceptable values (e.g. zeros for Ising models), but also because the average of the solutions
        # is not guaranteed to give the average of their energies.
        energies = []
        for solution in x.T:
            energies.append(self.compute_energy(solution, problem))
        return x.T[np.argmin(np.array(energies))]

    def _compute_coupling(self, previous_x, coupling_strength):
        self.oprec.float_multiplication(previous_x.size)  # coupling_strength *
        return coupling_strength * np.hstack([previous_x[:, 1:], np.zeros((previous_x.shape[0], 1))])

    @staticmethod
    def _preprocess_problem(problem: IsingData) -> IsingData:
        problem.extra = {"symmetric_J": -2 * (problem.J + problem.J.T)}
        return problem

    def tanh(self, signal, temperature):
        # Approximation of the tanh
        self.oprec.float_comparison(signal.size)  # signal >= temperature
        self.oprec.conditional_fill(signal.size)
        self.oprec.float_sign_flip(1)  # - self.alpha
        self.oprec.float_addition(1)  # temperature - self.alpha

        signal[signal >= temperature] = temperature - self.alpha

        self.oprec.float_comparison(signal.size)  # signal < -temperature
        self.oprec.conditional_fill(signal.size)
        self.oprec.float_sign_flip(1)  # - temperature

        signal[signal < -temperature] = -temperature
        return signal

    def update_signal(self, coupling, noise, problem, signal, x):
        self.oprec.dot_product(problem.extra['symmetric_J'], x)
        self.oprec.float_addition(noise.size)  # + noise
        self.oprec.float_addition(coupling.size)  # + coupling
        self.oprec.float_sign_flip(noise.size)  # - problem.h
        self.oprec.float_addition(signal.size)  # signal +=

        signal += np.dot(problem.extra['symmetric_J'], x) - problem.h[:, None] + noise + coupling
        return signal

    def generate_noise(self, x):
        self.oprec.random_number_generation(x.size)
        self.oprec.float_multiplication(x.size)  # * 2
        self.oprec.float_addition(x.size)  # - 1
        self.oprec.float_multiplication(x.size)  # self.noise_magnitude * ...
        return self.noise_magnitude * (np.random.randint(size=x.shape, low=0, high=2) * 2 - 1)

    def __call__(self, problem: IsingData) -> Tuple[np.ndarray, History]:
        self.initialize_history_and_opset()
        problem = self._preprocess_problem(problem)  # Allows computation optimization tricks
        length = self.get_length(problem)

        self.oprec.random_number_generation(length * self.n_trotters)
        x = self.generate_random_solution(length, self.n_trotters)
        signal = np.zeros((length, self.n_trotters))

        delay = [x]

        for step in range(self.monte_carlo_steps):
            self.history.record(ENERGY, self._select_best_energy_among_trotters(x, problem))
            self.history.record(OLS)
            self.history.record(ILS)
            temperature = self.temperature_scheduler.update(step, self.monte_carlo_steps)
            coupling_strength = self.coupling_scheduler.update(step, self.monte_carlo_steps)

            # Delayed trotters coupling
            if len(delay) == self.delay_cycle:
                coupling = self._compute_coupling(delay[0], coupling_strength)
                delay.pop(0)
            else:
                coupling = 0

            # Stochastic integral computing
            noise = self.generate_noise(x)
            signal = self.update_signal(coupling, noise, problem, signal, x)
            signal = self.tanh(signal, temperature)

            # Sign function without 0
            self.oprec.float_sign(signal.size)
            x = (signal >= 0.) * 2. - 1.

            delay.append(x)

        x = self._select_final_answer(x, problem)
        self.history.record(ENERGY, self.compute_energy(x, problem))
        return x, self.history
