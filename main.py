from algorithms import *
from problems import maxcut_gset
import matplotlib.pyplot as plt
import numpy as np
from utils import schedulers as sch
from utils import sampling as sp

if __name__ == '__main__':
    nsteps = 100
    seed = 0
    maxcut = maxcut_gset.MaxCutGSET()
    qubo, qubo_offset = maxcut.qubo(seed)
    linear, quadratic, ising_offset = maxcut.ising(seed)
    name, size, best = maxcut.get_gset_info(0)

    baseline_qubo = qubo_baseline(100, qubo, qubo_offset)
    baseline_ising = ising_baseline(100, linear, quadratic, ising_offset)

    qubo_solvers = {
        'Simulated Annealing QUBO': SimulatedAnnealingQUBO(monte_carlo_steps=nsteps),
        # 'Simulated Quantum Annealing QUBO': SimulatedQuantumAnnealingQUBO(monte_carlo_steps=nsteps, n_trotters=10),
        # 'Discrete Simulated Bifurcation QUBO': DiscreteSimulatedBifurcationQUBO(euler_steps=nsteps),
        # 'Ballistic Simulated Bifurcation QUBO': BallisticSimulatedBifurcationQUBO(euler_steps=nsteps),
    }

    ising_solvers = {
        'Simulated Annealing Ising': SimulatedAnnealingIsing(monte_carlo_steps=nsteps),
    #     'Simulated Quantum Annealing Ising': SimulatedQuantumAnnealingIsing(monte_carlo_steps=nsteps, n_trotters=10),
    #     'Discrete Simulated Bifurcation Ising': DiscreteSimulatedBifurcationIsing(euler_steps=nsteps),
    #     'Ballistic Simulated Bifurcation Ising': BallisticSimulatedBifurcationIsing(euler_steps=nsteps),
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

    plt.axhline(baseline_qubo, label='Baseline QUBO')
    plt.axhline(baseline_ising, label='Baseline Ising')
    plt.axhline(-best, label='Best known')

    plt.xlabel('Number of Monte Carlo steps')
    plt.ylabel('Energy')
    plt.legend()
    plt.title(f'MaxCut GSET {name} ({size} nodes)')
    plt.show()
