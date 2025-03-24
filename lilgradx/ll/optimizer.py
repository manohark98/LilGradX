import numpy as np
from lilgradx.tensor import Tensor

class Adam:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        # Initialize first and second moment vectors.
        self.m = [Tensor(0.0, requires_grad=False) for _ in parameters]
        self.v = [Tensor(0.0, requires_grad=False) for _ in parameters]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.parameters):
            # Wrap the gradient in a Tensor (without tracking).
            grad_tensor = Tensor(p.grad, requires_grad=False)
            # Update biased first moment estimate.
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad_tensor
            # Update biased second moment estimate.
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * Tensor(p.grad * p.grad, requires_grad=False)
            # Compute bias-corrected estimates.
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            # Update parameters.
            p.data -= self.lr * m_hat.data / (np.sqrt(v_hat.data) + self.epsilon)
