from algorithms.algorithm import AlgorithmQUBO
import numpy as np
import torch
import torch.nn.functional as F
import math


class QuboNetwork(torch.nn.Module):
    def __init__(self, x):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.unsqueeze(torch.tensor(x).type(torch.float), 0))

    def forward(self, qubo, offset, binarized=False):
        weights = torch.sigmoid(self.weights)
        if binarized:
            weights = (weights >= 0.5).type(torch.float)
            out = F.linear(input=qubo, weight=weights, bias=None)
            out = F.linear(input=torch.unsqueeze(weights, 1), weight=torch.squeeze(out, 2), bias=None)
            return out + offset
        else:
            out = F.linear(input=qubo, weight=weights, bias=None)
            out = F.linear(input=torch.unsqueeze(weights, 1), weight=torch.squeeze(out, 2), bias=None)
            return out + offset


class GradientDescent(AlgorithmQUBO):
    def __init__(self,
                 n_steps,
                 learning_rate=0.1,
                 momentum=0.9,
                 weight_decay=0.,
                 noise_rate=0.,
                 binarization_lambda=1.,
                 batch_size=1):
        self.n_steps = n_steps
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.noise_rate = noise_rate
        self.binarization_lambda = binarization_lambda
        self.batch_size = batch_size

    def loss(self, energy, x, i):
        weights = torch.sigmoid(x.weights)
        binarization_term = torch.sum(0.25 - (weights - 0.5).pow(2))
        penalty = 1  # ((i + 1) / self.n_steps)**2  #sinus_scheduling((i + 1) / self.n_steps)
        # print(energy, binarization_term, penalty)
        return torch.sum(energy) + (self.binarization_lambda * penalty * binarization_term)

    def noised_qubo(self, qubo, i):
        qubo = torch.unsqueeze(qubo, 0)
        qubo = qubo.repeat(self.batch_size, 1, 1)
        if self.noise_rate == 0.:
            return qubo
        noise_rate = self.noise_rate * (1 - ((i + 1) / self.n_steps) ** 2)
        return qubo * (((torch.rand(qubo.shape) - 0.5) * (noise_rate * 2)) + 1)

    def __call__(self, qubo: np.ndarray, offset: float):
        history = []

        x = self.generate_random_solution(qubo.shape[0])
        x[x == 0.] = -1.
        x = QuboNetwork(x)
        qubo = torch.Tensor(qubo)
        opti = torch.optim.SGD(x.parameters(),
                               lr=self.learning_rate,
                               weight_decay=self.weight_decay,
                               momentum=self.momentum)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opti,
                                                               T_max=self.n_steps,
                                                               eta_min=0)
        energy_calculations = 0
        for i in range(self.n_steps):
            x.train()
            opti.zero_grad()
            energy = x(self.noised_qubo(qubo, i), offset)
            energy_calculations += self.batch_size
            loss = self.loss(energy, x, i)
            loss.backward()
            opti.step()
            scheduler.step()
            with torch.no_grad():
                x.eval()
                history.append([energy_calculations, x(torch.unsqueeze(qubo, 0), offset, binarized=True).item()])

        return (torch.sigmoid(x.weights) >= 0.5).type(torch.float), history

    def solve_ising(self, linear: np.ndarray, quadratic: np.ndarray, offset: float):
        pass
