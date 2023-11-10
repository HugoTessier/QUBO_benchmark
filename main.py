from algorithms import *
from problems import MaxCut
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    nsteps = 100
    maxcut_size = 40
    seed = 0
    maxcut = MaxCut(maxcut_size, 0.1)
    qubo, qubo_offset = maxcut.qubo(seed)
    linear, quadratic, ising_offset = maxcut.ising(seed)

    qubo_solvers = {
        'Simulated Annealing QUBO': SimulatedAnnealingQUBO(monte_carlo_steps=nsteps),
        'Simulated Quantum Annealing QUBO': SimulatedQuantumAnnealingQUBO(monte_carlo_steps=nsteps, n_trotters=10),
        'Discrete Simulated Bifurcation QUBO': DiscreteSimulatedBifurcationQUBO(euler_steps=nsteps),
        'Ballistic Simulated Bifurcation QUBO': BallisticSimulatedBifurcationQUBO(euler_steps=nsteps),
    }

    ising_solvers = {
        'Simulated Annealing Ising': SimulatedAnnealingIsing(monte_carlo_steps=nsteps),
        'Simulated Quantum Annealing Ising': SimulatedQuantumAnnealingIsing(monte_carlo_steps=nsteps, n_trotters=10),
        'Discrete Simulated Bifurcation Ising': DiscreteSimulatedBifurcationIsing(euler_steps=nsteps),
        'Ballistic Simulated Bifurcation Ising': BallisticSimulatedBifurcationIsing(euler_steps=nsteps),
    }

    for k, v in qubo_solvers.items():
        _, history_sa = v(qubo, qubo_offset)
        history_sa = np.array(history_sa)
        plt.plot(history_sa[:, 0], history_sa[:, 1], label=k)
    for k, v in ising_solvers.items():
        _, history_sa = v(linear, quadratic, ising_offset)
        history_sa = np.array(history_sa)
        plt.plot(history_sa[:, 0], history_sa[:, 1], label=k)

    plt.xlabel('Number of Monte Carlo steps')
    plt.ylabel('Energy')
    plt.legend()
    plt.title(f'MaxCut with {maxcut_size} nodes')
    plt.show()
