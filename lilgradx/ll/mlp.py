import numpy as np
from lilgradx.ll.layer import Layer
from lilgradx.ll.activations import SoftmaxLayer

class MLP:
    def __init__(self, nin, nouts):
        # Build the MLP architecture.
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
        self.softmax = SoftmaxLayer()

    def __call__(self, x):
        # Forward pass: sequentially pass the input through all layers.
        for layer in self.layers:
            x = layer(x)
        self.probs = self.softmax(x)
        return self.probs

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        # Reset gradients to zero for all parameters.
        for p in self.parameters():
            if p.requires_grad:
                import numpy as np
                p.grad = np.zeros_like(p.data, dtype=np.float32)
