import numpy as np
from lilgradx.tensor import Value

class CrossEntropyLoss:
    def __init__(self, eps=1e-9):
        self.eps = eps  

    def __call__(self, predictions, targets):
        batch_size = len(targets)  # Number of samples in batch

        # Convert predictions to list of raw probabilities
        softmax_probs = [p.data for p in predictions]  # Extract `.data`
        
        # Convert to numpy array
        softmax_probs = np.array(softmax_probs)

        # Apply softmax normalization
        softmax_probs = np.exp(softmax_probs) / np.sum(np.exp(softmax_probs))

        # Extract probabilities corresponding to the target class
        probs = softmax_probs[targets[0]]  # Indexing the correct class probability

        # Compute log loss
        log_probs = -np.log(probs + self.eps)

        # Return loss as a Value object
        return Value(log_probs) 
