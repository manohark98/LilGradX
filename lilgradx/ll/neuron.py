from lilgradx.tensor import Tensor
import random

class Neuron:
    def __init__(self, nin):
        self.w = [Tensor(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Tensor(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.leaky_relu()
        return out

    def parameters(self):
        return self.w + [self.b]
