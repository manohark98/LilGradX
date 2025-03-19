from lilgradx.tensor import Value

class SoftmaxLayer:
    def __init__(self):
        pass


    def __call__(self, logits):
        # Stabilize softmax by subtracting the max logit
        
        max_logit = max(logit.data for logit in logits)  # Find the maximum logit value
        counts = [(logit - max_logit).exp() for logit in logits]  # Subtract max_logit and compute exp

        denominator = sum(counts)
        self.probs = [c / denominator for c in counts]  # Normalize to get probabilities
        return self.probs
