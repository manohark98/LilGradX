import numpy as np
import random
import math

class Tensor:
    def __init__(self, data, _children=(), _op='', label='', requires_grad=True):
        # Store data as a NumPy array to enable vectorized operations and broadcasting.
        self.data = np.array(data, dtype=np.float32)
        # If gradient tracking is enabled, initialize grad; otherwise, set to None.
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data, dtype=np.float32) if requires_grad else None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Tensor(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out_requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data + other.data, (self, other), '+', requires_grad=out_requires_grad)
        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    self.grad += out.grad
                if other.requires_grad:
                    other.grad += out.grad
            out._backward = _backward
        else:
            out._backward = lambda: None
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out_requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data * other.data, (self, other), '*', requires_grad=out_requires_grad)
        if out.requires_grad:
            def _backward():
                if self.requires_grad:
                    self.grad += other.data * out.grad
                if other.requires_grad:
                    other.grad += self.data * out.grad
            out._backward = _backward
        else:
            out._backward = lambda: None
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Only supporting int/float powers for now."
        out_requires_grad = self.requires_grad
        out = Tensor(self.data ** other, (self,), f'**{other}', requires_grad=out_requires_grad)
        if out.requires_grad:
            def _backward():
                self.grad += other * (self.data ** (other - 1)) * out.grad
            out._backward = _backward
        else:
            out._backward = lambda: None
        return out

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return self * -1

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        return self * (other ** -1)

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), 'tanh', requires_grad=self.requires_grad)
        if out.requires_grad:
            def _backward():
                self.grad += (1 - t ** 2) * out.grad
            out._backward = _backward
        else:
            out._backward = lambda: None
        return out

    def exp(self):
        out_val = np.exp(self.data)
        out = Tensor(out_val, (self,), 'exp', requires_grad=self.requires_grad)
        if out.requires_grad:
            def _backward():
                self.grad += out_val * out.grad
            out._backward = _backward
        else:
            out._backward = lambda: None
        return out

    def log(self):
        x = np.maximum(self.data, 1e-9)  # Avoid log(0)
        out_val = np.log(x)
        out = Tensor(out_val, (self,), 'log', requires_grad=self.requires_grad)
        if out.requires_grad:
            def _backward():
                self.grad += (1 / x) * out.grad
            out._backward = _backward
        else:
            out._backward = lambda: None
        return out

    def leaky_relu(self, alpha=0.01):
        out_val = np.where(self.data > 0, self.data, self.data * alpha)
        out = Tensor(out_val, (self,), 'leaky_relu', requires_grad=self.requires_grad)
        if out.requires_grad:
            def _backward():
                grad_val = np.where(self.data > 0, 1, alpha)
                self.grad += grad_val * out.grad
            out._backward = _backward
        else:
            out._backward = lambda: None
        return out

    def backward(self):
        # Only run backward if this tensor is tracking gradients.
        if not self.requires_grad:
            return
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        # Initialize gradient for the output tensor.
        self.grad = np.ones_like(self.data, dtype=np.float32)
        for node in reversed(topo):
            node._backward()

    def detach(self):
        """Return a new Tensor with the same data but with gradient tracking disabled."""
        return Tensor(self.data, requires_grad=False)

    def set_requires_grad(self, flag):
        """Set gradient tracking on or off. If turned off, grad is set to None."""
        self.requires_grad = flag
        if not flag:
            self.grad = None
        else:
            if self.grad is None:
                self.grad = np.zeros_like(self.data, dtype=np.float32)
