import os
from typing import Tuple
import numpy as np
from problems.problem import Problem
import pyqubo


class MaxCutGSET(Problem):
    """
    GSET MaxCut problems, from https://web.stanford.edu/~yyye/yyye/Gset/
    Best solutions baselines from Breakout Local Search for the Max-Cutproblem by Una Benlic and Jin-Kao Hao
    """

    def _parse_gset_content(self):
        files_list = os.listdir(os.path.join(self.path, 'graphs'))
        indices = [int(f.replace('G', '')) for f in files_list]
        files_list = np.array(files_list)[np.argsort(np.array(indices))].tolist()
        data = {}
        for file in files_list:
            if file != 'GSET_known_best_energies.txt':
                with open(os.path.join(self.path, 'graphs', file), 'r') as f:
                    data[file] = f.readlines()
        gset_content = {}
        for k, v in data.items():
            gset_content[k] = {}
            gset_content[k]['n_nodes'] = int(v[0].split(' ')[0])
            v.pop(0)

            content = []
            for line in v:
                line = line.replace('\n', '').split(' ')
                line_data = {'node1': int(line[0]),
                             'node2': int(line[1]),
                             'weight': int(line[2])}
                content.append(line_data)

            gset_content[k]['content'] = content
        with open(os.path.join(self.path, 'GSET_known_best_energies.txt'), 'r') as f:
            for line in f.readlines():
                l_split = line.split(' ')
                gset_content[l_split[0]]['best'] = int(l_split[1])
        return gset_content

    def __init__(self):
        self.path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gset')
        self.gset_content = self._parse_gset_content()

    def _create_model(self, index: int) -> pyqubo.Model:
        if index >= len(self.gset_content) or index < 0:
            raise IndexError

        key = list(self.gset_content.keys())[index]
        n_nodes = self.gset_content[key]['n_nodes']
        binaries = [pyqubo.Binary(f'{i}') for i in range(n_nodes)]
        hamiltonian = None
        for e in self.gset_content[key]['content']:
            node1 = e['node1'] - 1
            node2 = e['node2'] - 1
            weight = e['weight']
            if hamiltonian is None:
                hamiltonian = weight * ((2 * binaries[node1] * binaries[node2]) - binaries[node1] - binaries[node2])
            else:
                hamiltonian += weight * ((2 * binaries[node1] * binaries[node2]) - binaries[node1] - binaries[node2])
        model = hamiltonian.compile()
        return model, n_nodes

    def qubo(self, seed: int = 0) -> Tuple[np.ndarray, float]:
        """
        Generate an instance from the GSET MaxCut problem list, and returns its corresponding QUBO.

        :param seed: Index of the problem in the list of all GSET problems.
        :return: Instance of the problem as a QUBO, corresponding to the given seed, under the form of a tuple
        containing the Q matrix as a numpy array and the energy offset as a float.
        """
        model, n_nodes = self._create_model(seed)
        qubo, offset = model.to_qubo()
        qubo_array = np.zeros((n_nodes, n_nodes))
        for k, v in qubo.items():
            qubo_array[int(k[0]), int(k[1])] = v
        return qubo_array, offset

    def ising(self, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Generate an instance from the GSET MaxCut problem list, and returns its corresponding Ising model.

        :param seed: Index of the problem in the list of all GSET problems.
        :return: Instance of the problem as an Ising model, corresponding to the given seed, under the form of a tuple
        containing the h and J matrices as numpy arrays and the energy offset as a float.
        """
        model, n_nodes = self._create_model(seed)
        linear, quadratic, offset = model.to_ising()
        quadratic_array = np.zeros((n_nodes, n_nodes))
        for k, v in quadratic.items():
            quadratic_array[int(k[0]), int(k[1])] = v
        linear = np.zeros(n_nodes)  # Linear part in MaxCut is zero
        return linear, quadratic_array, offset

    def visualize(self, seed: int, x: np.ndarray) -> None:
        raise NotImplementedError

    def get_gset_info(self, seed: int = 0) -> Tuple:
        """
        Returns info related to GSET problem of given index.
        :param seed: Index of said problem in the list of GSET MaxCut problems.
        :return: A tuple containing the name, the number of nodes and the est known score.
        """
        key = list(self.gset_content.keys())[seed]
        return key, self.gset_content[key]['n_nodes'], self.gset_content[key]['best']


"""
Summary of the content of the 71 GSET MaxCut problems:

Name / Number of nodes / Number of edges / Best known score
G1 800 19176 11624
G2 800 19176 11620
G3 800 19176 11622
G4 800 19176 11646
G5 800 19176 11631
G6 800 19176 2178
G7 800 19176 2006
G8 800 19176 2005
G9 800 19176 2054
G10 800 19176 2000
G11 800 1600 564
G12 800 1600 556
G13 800 1600 582
G14 800 4694 3064
G15 800 4661 3050
G16 800 4672 3052
G17 800 4667 3047
G18 800 4694 992
G19 800 4661 906
G20 800 4672 941
G21 800 4667 931
G22 2000 19990 13359
G23 2000 19990 13344
G24 2000 19990 13337
G25 2000 19990 13340
G26 2000 19990 13328
G27 2000 19990 3341
G28 2000 19990 3298
G29 2000 19990 3405
G30 2000 19990 3412
G31 2000 19990 3309
G32 2000 4000 1410
G33 2000 4000 1382
G34 2000 4000 1384
G35 2000 11778 7684
G36 2000 11766 7678
G37 2000 11785 7689
G38 2000 11779 7687
G39 2000 11778 2408
G40 2000 11766 2400
G41 2000 11785 2405
G42 2000 11779 2481
G43 1000 9990 6660
G44 1000 9990 6650
G45 1000 9990 6654
G46 1000 9990 6649
G47 1000 9990 6657
G48 3000 6000 6000
G49 3000 6000 6000
G50 3000 6000 5880
G51 1000 5909 3848
G52 1000 5916 3851
G53 1000 5914 3850
G54 1000 5916 3852
G55 5000 12498 10294
G56 5000 12498 4012
G57 5000 10000 3492
G58 5000 29570 19263
G59 5000 29570 6078
G60 7000 17148 14176
G61 7000 17148 5789
G62 7000 14000 4868
G63 7000 41459 26997
G64 7000 41459 8735
G65 8000 16000 5558
G66 9000 18000 6360
G67 10000 20000 6940
G70 10000 9999 9541
G72 10000 20000 6998
G77 14000 28000 9926
G81 20000 40000 14030
"""
