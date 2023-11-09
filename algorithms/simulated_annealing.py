from algorithms.algorithm import AlgorithmQUBO, AlgorithmIsing, Algorithm
import numpy as np
import random
import math
from abc import abstractmethod
from typing import Tuple, Callable
from utils.schedulers import Scheduler, GeometricScheduler
from utils.sampling import range_sampler


class SimulatedAnnealingCommon(Algorithm):
    """
    Based on "Simulated annealing: From basics to applications" by Daniel Delahaye, Supatcha Chaimatanan
    and Marcel Mongeau.
    """

    def __init__(self, outer_loop_steps: int,
                 temperature_scheduler: Scheduler = GeometricScheduler(start=20, decay_rate=0.9),
                 sampler: Callable = range_sampler):
        self.outer_loop_steps = outer_loop_steps
        self.temperature_scheduler = temperature_scheduler
        self.sampler = sampler

    @staticmethod
    @abstractmethod
    def flip_bit(x: np.ndarray, index: int) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def compute_energy_delta(x, local_energy_field, index) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def initialize_local_energy_field(x, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def update_local_energy_field(x, local_energy_field, index, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        history = []

        length = self.get_length(*args, **kwargs)
        x = self.generate_random_solution(length)
        local_energy_field = self.initialize_local_energy_field(x, *args, **kwargs)
        for step in range(self.outer_loop_steps):
            temperature = self.temperature_scheduler.update(step, self.outer_loop_steps)
            for index in self.sampler(length):
                delta_energy = self.compute_energy_delta(x, local_energy_field, index)

                if math.exp(min(-delta_energy / temperature, 1)) > random.random():  # Avoid math range error
                    x = self.flip_bit(x, index)
                    local_energy_field = self.update_local_energy_field(x, local_energy_field, index,
                                                                        *args, **kwargs)
            history.append([step, self.compute_energy(x, *args, **kwargs)])
        return x, history


class SimulatedAnnealingQUBO(AlgorithmQUBO, SimulatedAnnealingCommon):
    """
    Based on "Simulated annealing: From basics to applications" by Daniel Delahaye, Supatcha Chaimatanan
    and Marcel Mongeau.
    """

    def __init__(self,
                 outer_loop_steps: int,
                 temperature_scheduler: Scheduler = GeometricScheduler(start=20, decay_rate=0.9),
                 sampler: Callable = range_sampler):
        super().__init__(outer_loop_steps, temperature_scheduler, sampler)

    @staticmethod
    def flip_bit(x: np.ndarray, index: int) -> np.ndarray:
        x[index] = int(x[index] == 0)
        return x

    @staticmethod
    def compute_energy_delta(x, local_energy_field, index) -> np.ndarray:
        flip_direction = 1 if x[index] == 0 else -1
        return local_energy_field[index] * flip_direction

    @staticmethod
    def initialize_local_energy_field(x, qubo, offset) -> np.ndarray:
        return np.dot(qubo, x)

    @staticmethod
    def update_local_energy_field(x, local_energy_field, index, qubo, offset) -> np.ndarray:
        flip_direction = 1 if x[index] == 1 else -1
        local_energy_field += (flip_direction * qubo[index, :])
        return local_energy_field

    def __call__(self, qubo: np.ndarray, offset: float) -> Tuple:
        return super().__call__(qubo, offset)


class SimulatedAnnealingIsing(AlgorithmIsing, SimulatedAnnealingCommon):
    """
    Based on "Simulated annealing: From basics to applications" by Daniel Delahaye, Supatcha Chaimatanan
    and Marcel Mongeau.
    """

    def __init__(self,
                 outer_loop_steps: int,
                 temperature_scheduler: Scheduler = GeometricScheduler(start=20, decay_rate=0.9),
                 sampler: Callable = range_sampler):
        super().__init__(outer_loop_steps, temperature_scheduler, sampler)

    @staticmethod
    def flip_bit(x: np.ndarray, index: int) -> np.ndarray:
        x[index] = -x[index]
        return x

    @staticmethod
    def compute_energy_delta(x, local_energy_field, index) -> np.ndarray:
        return - 2 * x[index] * local_energy_field[index]  # - because before spin update

    @staticmethod
    def initialize_local_energy_field(x, linear, quadratic, offset) -> np.ndarray:
        return np.dot(quadratic, x) + linear

    @staticmethod
    def update_local_energy_field(x, local_energy_field, index, linear, quadratic, offset) -> np.ndarray:
        local_energy_field += 2 * quadratic[index, :] * x[index]  # + because after spin update
        return local_energy_field

    def __call__(self, linear: np.ndarray, quadratic: np.ndarray, offset: float) -> Tuple:
        return super().__call__(linear, quadratic, offset)
