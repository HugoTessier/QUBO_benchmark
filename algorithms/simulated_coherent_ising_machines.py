from algorithms.algorithm import IAlgorithm
from utils.data_struct import *
from utils.history import *
from typing import Tuple
import numpy as np
from utils.schedulers import TanHScheduler, Scheduler


class SimulatedCoherentIsingMachines(IAlgorithm):
    """
    SimCIM
    based on "Annealing by simulating the coherent Ising machine"
    by EGOR S. TIUNOV, ALEXANDER E. ULANOV, AND A. I. LVOV SKY

    and complemented using the supplementary materials from
    High-performance combinatorial optimization based on classical mechanics
    by Hayato Goto, Kotaro Endo, Masaru Suzuki, Yoshisato Sakai, Taro Kanao, Yohei Hamakawa, Ryo Hidaka, Masaya Yamasaki
    and Kosuke Tatsumura
    as well as the following GitHub repo:
    https://github.com/PMosharev/SimCIM-simple-implementation
    """

    def __init__(self,
                 n_steps: int,
                 nu_scheduler: Scheduler = TanHScheduler(multiplier=4, steepness=3, offset=0),
                 zeta: float = 10.,
                 noise_magnitude: float = 1.):
        """
        :param n_steps: Number of main loop steps.
        :param nu_scheduler: Controls the evolution of the hyperparameter nu (lasting term).
        :param zeta: Constant hyperparameter zeta (energy term).
        :param noise_magnitude: Constant magnitude of the noise term.
        """
        super().__init__()
        self.n_steps = n_steps
        self.nu_scheduler = nu_scheduler
        self.zeta = zeta
        self.noise_magnitude = noise_magnitude

    @staticmethod
    def _preprocess_problem(problem: IsingData) -> IsingData:
        problem.extra = {"symmetric_J": -2 * (problem.J + problem.J.T)}
        return problem

    def _binarize(self, x: np.ndarray) -> np.ndarray:
        self.oprec.sign(x.size)
        return (x >= 0.) * 2. - 1.  # Sign function

    def __call__(self, problem: IsingData) -> Tuple[np.ndarray, History]:
        self.initialize_history_and_opset()
        problem = self._preprocess_problem(problem)  # Allows computation optimization tricks
        length = self.get_length(problem)

        self.oprec.random_number_generation(length)
        x = self.generate_random_solution(length).astype(float)

        for step in range(self.n_steps):
            self.history.record(ENERGY, self.compute_energy(self._binarize(x), problem))
            self.history.record(MAIN_LOOP)
            self.history.record(SEQUENCE)

            nu = self.nu_scheduler.update(step, self.n_steps)

            self.oprec.random_number_generation(x.size)
            self.oprec.multiplication(x.size)  # * self.noise_magnitude
            f = np.random.normal(size=x.shape) * self.noise_magnitude

            self.oprec.dot_product(problem.extra['symmetric_J'], x)
            energy_term = np.dot(problem.extra['symmetric_J'], x)

            self.oprec.multiplication(x.size)
            lasting_term = nu * x

            self.oprec.multiplication(x.size)  # self.zeta * energy_term
            self.oprec.addition(x.size)  # lasting_term + ...
            self.oprec.addition(x.size)  # + f
            x += lasting_term + (self.zeta * energy_term) + f

            self.oprec.clip(x.size)
            x = np.clip(x, a_min=-1, a_max=1)

        x = self._binarize(x)
        self.history.record(ENERGY, self.compute_energy(x, problem))
        return x, self.history
