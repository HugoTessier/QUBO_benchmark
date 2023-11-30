from algorithms import *
from problems import maxcut_gset
import matplotlib.pyplot as plt
import numpy as np
from utils import schedulers as sch
from utils import sampling as sp
from utils.history import *
from utils.operations import *

if __name__ == '__main__':
    nsteps = 150
    seed = 10
    maxcut = maxcut_gset.MaxCutGSET()
    qubo_data = maxcut.qubo(seed)
    ising_data = maxcut.ising(seed)
    name, size, best = maxcut.get_gset_info(seed)

    baseline_ising = ising_baseline(100, ising_data)

    qubo_solvers = {
        'Simulated Annealing QUBO': QSimulatedAnnealing(monte_carlo_steps=nsteps),
        'Simulated Quantum Annealing QUBO': QSimulatedQuantumAnnealing(monte_carlo_steps=nsteps,
                                                                       n_trotters=10),
    }

    ising_solvers = {
        'Simulated Annealing Ising': ISimulatedAnnealing(monte_carlo_steps=nsteps),
        'Simulated Quantum Annealing Ising': ISimulatedQuantumAnnealing(monte_carlo_steps=nsteps,
                                                                        n_trotters=10),
        'Discrete Simulated Bifurcation Ising': IDiscreteSimulatedBifurcation(euler_steps=nsteps),
        'Ballistic Simulated Bifurcation Ising': IBallisticSimulatedBifurcation(euler_steps=nsteps),
        'Stochastic Simulated Annealing Ising': IStochasticSimulatedAnnealing(
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
        'Stochastic Simulated Quantum Annealing Ising': IStochasticSimulatedQuantumAnnealing(
            monte_carlo_steps=nsteps, n_trotters=10,
        ),
        'Simulated Coherent Ising Machines': SimulatedCoherentIsingMachines(n_steps=nsteps),
    }

    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312)
    ax3 = plt.subplot(313)
    axes = [ax1, ax2, ax3]
    for k, v in qubo_solvers.items():
        print(k)
        _, history_sa = v(qubo_data)
        x, y = history_sa.plot(x_key=[MAIN_LOOP], y_key=[ENERGY], x_mode=SUM, y_mode=INSTANT)
        ax1.plot(x, y, label=k)
        x, y = history_sa.plot(x_key=[SEQUENCE], y_key=[ENERGY], x_mode=SUM, y_mode=INSTANT)
        ax2.plot(x, y, label=k)
        x, y = history_sa.plot(x_key=[MULTIPLICATION, ADDITION], y_key=[ENERGY], x_mode=SUM, y_mode=INSTANT)
        ax3.plot(x, y, label=k)
    for k, v in ising_solvers.items():
        print(k)
        _, history_sa = v(ising_data)
        x, y = history_sa.plot(x_key=[MAIN_LOOP], y_key=[ENERGY], x_mode=SUM, y_mode=INSTANT)
        ax1.plot(x, y, label=k)
        x, y = history_sa.plot(x_key=[SEQUENCE], y_key=[ENERGY], x_mode=SUM, y_mode=INSTANT)
        ax2.plot(x, y, label=k)
        x, y = history_sa.plot(x_key=[MULTIPLICATION, ADDITION], y_key=[ENERGY], x_mode=SUM, y_mode=INSTANT)
        ax3.plot(x, y, label=k)
    for ax in axes:
        ax.axhline(baseline_ising, label='Baseline', linestyle='dashed', color='red')
        ax.axhline(-best, label='Best known', linestyle='dashed', color='blue')
        ax.legend()

    ax1.set_xlabel('Number of outermost loop steps')
    ax1.set_ylabel('Energy')
    ax2.set_xlabel('Number of sequential innermost loop steps')
    ax2.set_ylabel('Energy')
    ax3.set_xlabel('Mass of Operations')
    ax3.set_ylabel('Energy')
    plt.suptitle(f'MaxCut GSET {name} ({size} nodes)')
    plt.show()
