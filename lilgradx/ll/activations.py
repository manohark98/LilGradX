from lilgradx.tensor import Tensor

class SoftmaxLayer:
    def __init__(self):
        pass

    def __call__(self, logits):
        # For numerical stability, subtract the maximum logit.
        max_logit = max(logit.data for logit in logits)
        # Compute the exponential of adjusted logits.
        counts = [(logit - max_logit).exp() for logit in logits]
        # Sum the exponentials.
        denominator = counts[0] if len(counts) == 1 else sum(counts[1:], counts[0])
        # Normalize to obtain probabilities.
        self.probs = [c / denominator for c in counts]
        return self.probs
