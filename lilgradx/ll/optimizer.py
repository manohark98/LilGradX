from lilgradx.tensor import Value

class Adam:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [Value(0.0) for _ in parameters]  # Initialize first moment vector
        self.v = [Value(0.0) for _ in parameters]  # Initialize second moment vector
        self.t = 0  # Initialize time step

    def step(self):
        self.t += 1
        for i, p in enumerate(self.parameters):
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            # Update biased second moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad * p.grad)

            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            # Compute bias-corrected second moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            #print("Adam Sqrt",v_hat**0.5,self.epsilon)
            # Update parameters
            p.data -= self.lr * m_hat.data / ((v_hat**0.5).data + self.epsilon)
