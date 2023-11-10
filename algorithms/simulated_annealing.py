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
    Based on "Simulated annealing: From basics to applications"
    by Daniel Delahaye, Supatcha Chaimatanan and Marcel Mongeau.
    """

    def __init__(self,
                 monte_carlo_steps: int,
                 temperature_scheduler: Scheduler = GeometricScheduler(start=20, decay_rate=0.9),
                 sampler: Callable = range_sampler):
        """
        :param monte_carlo_steps: Number of Monte-Carlo steps, i.e. the outermost loop of the algorithm.
        :param temperature_scheduler: Controls the evolution of the temperature during annealing.
        :param sampler: Controls in which order to test changes in the value of elements in the solution.
        """
        self.monte_carlo_steps = monte_carlo_steps
        self.temperature_scheduler = temperature_scheduler
        self.sampler = sampler

    @staticmethod
    @abstractmethod
    def _flip_element(x: np.ndarray, i: int) -> np.ndarray:
        """
        Flips the [i,m] of x.

        :param x: The candidate solution whose element [i,m] is to flip.
        :param i: Index of an element of x.

        :return: The modified x.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _compute_energy_delta(x: np.ndarray, local_energy_field: np.ndarray, i: int) -> np.ndarray:
        """
        Gives the variation in energy when changing the element i of x.

        :param x: The candidate solution whose variations in energy we want to measure when changing the element i.
        :param local_energy_field: Energy associated to each element of x, in order to simplify calculations.
        :param i: Index of an element of x.

        :return: The variation of energy when changing the element i of x.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _initialize_local_energy_field(x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Initializes the local energy field of x.
        :param x: The solution whose energy field to compute.
        :return: The local energy field.
        """
        # args and kwargs are here to be replaced with qubo + offset
        # or linear + quadratic + offset depending on whether it is implemented for QUBO or Ising model.
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _update_local_energy_field(local_energy_field, x, i, *args, **kwargs) -> np.ndarray:
        """
        Updates the local energy field.

        :param local_energy_field: The energy field to update.
        :param x: The candidate solution that has changed and whose energy field we must update.
        :param i: Element i of any trotter of x.

        :return: The updated local energy field.
        """
        # args and kwargs are here to be replaced with qubo + offset
        # or linear + quadratic + offset depending on whether it is implemented for QUBO or Ising model.
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Tuple:
        # args and kwargs are here to be replaced with qubo + offset
        # or linear + quadratic + offset depending on whether it is implemented for QUBO or Ising model.

        length = self.get_length(*args, **kwargs)
        x = self.generate_random_solution(length)

        # Since the hamiltonian is purely a sum, the energy delta induced by changing one element of x at a time
        # can be calculated using only a "local" energy. It is then more efficient to compute said "local energy" field
        # beforehand and to update it progressively.
        local_energy_field = self._initialize_local_energy_field(x, *args, **kwargs)

        history = []  # We monitor the energy evolution at each Monte-Carlo step
        for step in range(self.monte_carlo_steps):
            history.append([step, self.compute_energy(x, *args, **kwargs)])

            temperature = self.temperature_scheduler.update(step, self.monte_carlo_steps)
            for index in self.sampler(length):
                delta_energy = self._compute_energy_delta(x, local_energy_field, index)

                # Metropolis algorithm: the random number is in [0,1]; if the tested change in x reduces the
                # energy, then exp(-delta_energy / temperature) > 1 and the change is accepted; otherwise, we have
                # exp(-delta_energy / temperature) in [0,1] too and then, the change is accepted in a probabilistic
                # way, with decreasing chances as the energy increase becomes larger.
                if math.exp(min(-delta_energy / temperature, 1)) > random.random():  # min() to avoid math range error
                    x = self._flip_element(x, index)
                    local_energy_field = self._update_local_energy_field(local_energy_field, x, index,
                                                                         *args, **kwargs)

        history.append([self.monte_carlo_steps, self.compute_energy(x, *args, **kwargs)])
        return x, history


class SimulatedAnnealingQUBO(AlgorithmQUBO, SimulatedAnnealingCommon):
    """QUBO version of SA."""

    @staticmethod
    def _flip_element(x: np.ndarray, i: int) -> np.ndarray:
        x[i] = int(x[i] == 0)
        return x

    @staticmethod
    def _compute_energy_delta(x, local_energy_field, i) -> np.ndarray:
        flip_direction = 1 if x[i] == 0 else -1
        return local_energy_field[i] * flip_direction

    @staticmethod
    def _initialize_local_energy_field(x, qubo, offset) -> np.ndarray:
        return np.dot(qubo, x)

    @staticmethod
    def _update_local_energy_field(local_energy_field, x, i, qubo, offset) -> np.ndarray:
        flip_direction = 1 if x[i] == 1 else -1
        local_energy_field += (flip_direction * qubo[i, :])
        return local_energy_field

    def __call__(self, qubo: np.ndarray, offset: float) -> Tuple:
        """
        :param qubo: Coupling coefficients of the QUBO problem.*
        :param offset: Energy offset of the QUBO problem.
        """
        return super().__call__(qubo, offset)


class SimulatedAnnealingIsing(AlgorithmIsing, SimulatedAnnealingCommon):
    """Ising model version of SA."""

    @staticmethod
    def _flip_element(x: np.ndarray, i: int) -> np.ndarray:
        x[i] = -x[i]
        return x

    @staticmethod
    def _compute_energy_delta(x, local_energy_field, i) -> np.ndarray:
        return - 2 * x[i] * local_energy_field[i]  # - because before spin update

    @staticmethod
    def _initialize_local_energy_field(x, linear, quadratic, offset) -> np.ndarray:
        return np.dot(quadratic, x) + linear

    @staticmethod
    def _update_local_energy_field(local_energy_field, x, i, linear, quadratic, offset) -> np.ndarray:
        local_energy_field += 2 * quadratic[i, :] * x[i]  # + because after spin update
        return local_energy_field

    def __call__(self, linear: np.ndarray, quadratic: np.ndarray, offset: float) -> Tuple:
        """
        :param linear: Local magnetic field of the Ising model.
        :param quadratic: Coupling coefficients of the Ising model.
        :param offset: Energy offset of the QUBO problem.
        """
        return super().__call__(linear, quadratic, offset)