from problems.problem import Problem
import networkx as nx
from itertools import combinations, groupby
import random
import pyqubo
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt


def gnp_random_connected_graph(n: int, p: float) -> nx.Graph:
    """
    Generates a random undirected graph, similarly to an Erdős-Rényi
    graph, but enforcing that the resulting graph is connected
    Source: https://stackoverflow.com/questions/61958360/how-to-create-random-graph-where-each-node-has-at-least-1-edge-using-networkx

    :param n: Number of nodes in the graph.
    :param p: Probability of generating an edge (after each node has been given at least one).
    :return: The graph as a networkx object.
    """
    edges = combinations(range(n), 2)
    g = nx.Graph()
    g.add_nodes_from(range(n))
    if p <= 0:
        return g
    if p >= 1:
        return nx.complete_graph(n, create_using=g)
    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        g.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < p:
                g.add_edge(*e)
    return g


class MaxCut(Problem):
    """
    Cf. Wikipedia: "For a graph, a maximum cut is a cut whose size is at least the size of any other cut. That is,
    it is a partition of the graph's vertices into two complementary sets S and T, such that the number of edges
    between S and T is as large as possible."
    """

    def __init__(self, n_nodes: int, edge_prob: float):
        """
        Prepares a generator of MaxCut problems.

        :param n_nodes:
        :param edge_prob:
        """
        self.n_nodes = n_nodes
        self.edge_prob = edge_prob

    def _create_model(self, seed) -> pyqubo.Model:
        random.seed(seed)
        g = gnp_random_connected_graph(self.n_nodes, self.edge_prob)

        # Cf. "A Tutorial on Formulating and Using QUBO Models" by Fred Glover, Gary Kochenberger, Yu Du
        binaries = [pyqubo.Binary(f'{i}') for i in range(self.n_nodes)]
        hamiltonian = None
        for e in nx.edges(g):
            if hamiltonian is None:
                hamiltonian = (2 * binaries[e[0]] * binaries[e[1]]) - binaries[e[0]] - binaries[e[1]]
            else:
                hamiltonian += (2 * binaries[e[0]] * binaries[e[1]]) - binaries[e[0]] - binaries[e[1]]
        model = hamiltonian.compile()
        return model

    def qubo(self, seed: int = 0) -> Tuple[np.ndarray, float]:
        """
        Generate an instance of the MaxCut problem, depending on a seed, and returns its corresponding QUBO.

        :param seed: Seed to generate the problem.
        :return: Instance of the problem as a QUBO, corresponding to the given seed, under the form of a tuple
        containing the Q matrix as a numpy array and the energy offset as a float.
        """
        model = self._create_model(seed)
        qubo, offset = model.to_qubo()
        qubo_array = np.zeros((self.n_nodes, self.n_nodes))
        for k, v in qubo.items():
            qubo_array[int(k[0]), int(k[1])] = v
        return qubo_array, offset

    def ising(self, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Generate an instance of the MaxCut problem, depending on a seed, and returns its corresponding Ising model.

        :param seed: Seed to generate the problem.
        :return: Instance of the problem as an Ising model, corresponding to the given seed, under the form of a tuple
        containing the h and J matrices as numpy arrays and the energy offset as a float.
        """
        model = self._create_model(seed)
        linear, quadratic, offset = model.to_ising()
        quadratic_array = np.zeros((self.n_nodes, self.n_nodes))
        for k, v in quadratic.items():
            quadratic_array[int(k[0]), int(k[1])] = v
        linear = np.zeros(self.n_nodes)  # Linear part in MaxCut is zero
        return linear, quadratic_array, offset

    def visualize(self, seed: int, x: np.ndarray) -> None:
        """
        Gives a visualization of the solution.

        :param seed: Seed to recreate the problem.
        :param x: The solution to visualize.
        """
        random.seed(seed)
        g = gnp_random_connected_graph(self.n_nodes, self.edge_prob)
        color_map = []
        for i in x:
            if i == 1:
                color_map.append('blue')
            else:
                color_map.append('green')
        nx.draw_networkx(g, node_color=color_map)
        ax = plt.gca()
        ax.margins(0.20)
        plt.axis("off")
        plt.show()
