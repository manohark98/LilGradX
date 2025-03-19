from lilgradx.ll.layer import Layer
from lilgradx.ll.activations import SoftmaxLayer

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]
        self.softmax = SoftmaxLayer()

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.probs = self.softmax(x)
        return self.probs



    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0
