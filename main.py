from algorithms import *
from problems import maxcut_gset
import matplotlib.pyplot as plt
import numpy as np
from utils import schedulers as sch
from utils import sampling as sp

if __name__ == '__main__':
    nsteps = 150
    seed = 10
    maxcut = maxcut_gset.MaxCutGSET()
    qubo, qubo_offset = maxcut.qubo(seed)
    linear, quadratic, ising_offset = maxcut.ising(seed)
    name, size, best = maxcut.get_gset_info(seed)

    baseline_qubo = qubo_baseline(100, qubo, qubo_offset)
    baseline_ising = ising_baseline(100, linear, quadratic, ising_offset)

    qubo_solvers = {
        'Simulated Annealing QUBO': SimulatedAnnealingQUBO(monte_carlo_steps=nsteps),
        'Simulated Quantum Annealing QUBO': SimulatedQuantumAnnealingQUBO(monte_carlo_steps=nsteps,
                                                                          n_trotters=10),
    }

    ising_solvers = {
        'Simulated Annealing Ising': SimulatedAnnealingIsing(monte_carlo_steps=nsteps),
        'Simulated Quantum Annealing Ising': SimulatedQuantumAnnealingIsing(monte_carlo_steps=nsteps,
                                                                            n_trotters=10),
        'Discrete Simulated Bifurcation Ising': DiscreteSimulatedBifurcationIsing(euler_steps=nsteps),
        'Ballistic Simulated Bifurcation Ising': BallisticSimulatedBifurcationIsing(euler_steps=nsteps),
        'Stochastic Simulated Annealing Ising': StochasticSimulatedAnnealingIsing(
            monte_carlo_steps=nsteps,
            # temperature_scheduler=sch.WarmRestartScheduler(
            #     sch.DiscreteScheduler(
            #         sch.GeometricScheduler(
            #             start=1.,
            #             multiplier=2.,
            #             max_value=32.
            #         ),
            #         n_plateau=6),
            #     n_restarts=150)
        ),
        'Stochastic Simulated Quantum Annealing Ising': StochasticSimulatedQuantumAnnealingIsing(
            monte_carlo_steps=nsteps, n_trotters=10,
        ),
    }

    for k, v in qubo_solvers.items():
        print(k)
        _, history_sa = v(qubo, qubo_offset)
        history_sa = np.array(history_sa)
        plt.plot(history_sa[:, 0], history_sa[:, 1], label=k)
    for k, v in ising_solvers.items():
        print(k)
        _, history_sa = v(linear, quadratic, ising_offset)
        history_sa = np.array(history_sa)
        plt.plot(history_sa[:, 0], history_sa[:, 1], label=k)

    # plt.axhline(baseline_qubo, label='Baseline QUBO')
    plt.axhline(baseline_ising, label='Baseline Ising')
    plt.axhline(-best, label='Best known')

    plt.xlabel('Number of Monte Carlo steps')
    plt.ylabel('Energy')
    plt.legend()
    plt.title(f'MaxCut GSET {name} ({size} nodes)')
    plt.show()
