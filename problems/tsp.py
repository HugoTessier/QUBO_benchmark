from problem import Problem
import numpy as np
import pyqubo


class TSP(Problem):
    """
    Inspired by https://github.com/recruit-communications/pyqubo/blob/master/notebooks/TSP.ipynb
    """

    def __init__(self, number_of_cities: int, map_height: float = 1., map_width: float = 1.):
        self.n_cities = number_of_cities
        self.map_height = map_height
        self.map_width = map_width

    def generate_cities(self, seed):
        np.random.seed(seed)
        x = np.random.random(self.n_cities) * self.map_width
        y = np.random.random(self.n_cities) * self.map_height
        return np.stack([x, y]).T

    def dist(self, i, j, cities):
        pos_i = cities[i]
        pos_j = cities[j]
        return np.sqrt((pos_i[0] - pos_j[0]) ** 2 + (pos_i[1] - pos_j[1]) ** 2)

    def generate_model(self, cities):
        x = pyqubo.Array.create('c', (self.n_cities, self.n_cities), 'BINARY')

        # Constraint not to visit more than two cities at the same time.
        time_const = 0.0
        for i in range(self.n_cities):
            # If you wrap the hamiltonian by Const(...), this part is recognized as constraint
            time_const += pyqubo.Constraint((sum(x[i, j] for j in range(self.n_cities)) - 1) ** 2,
                                            label="time{}".format(i))

        # Constraint not to visit the same city more than twice.
        city_const = 0.0
        for j in range(self.n_cities):
            city_const += pyqubo.Constraint((sum(x[i, j] for i in range(self.n_cities)) - 1) ** 2,
                                            label="city{}".format(j))

        # distance of route
        distance = 0.0
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                for k in range(self.n_cities):
                    d_ij = self.dist(i, j, cities)
                    distance += d_ij * x[k, i] * x[(k + 1) % self.n_cities, j]

        # Construct hamiltonian
        constraint = pyqubo.Placeholder("A")
        hamiltonian = distance + constraint * (time_const + city_const)

        # Compile model
        model = hamiltonian.compile()
        return model

    def qubo(self, seed: int = 0, constraint_strength: float = 1.0):
        cities = self.generate_cities(seed)
        model = self.generate_model(cities)
        feed_dict = {'A': constraint_strength}
        return model.to_qubo(feed_dict=feed_dict)

    def ising(self, seed: int = 0, constraint_strength: float = 1.0):
        cities = self.generate_cities(seed)
        model = self.generate_model(cities)
        feed_dict = {'A': constraint_strength}
        return model.to_ising(feed_dict=feed_dict)


if __name__ == "__main__":
    tsp = TSP(10)
    qubo, offset = tsp.qubo(0)
    for k, v in qubo.items():
        print(k, v)
