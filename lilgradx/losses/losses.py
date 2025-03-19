from lilgradx.tensor import Value

def nll_loss(probs, target_index):
    """
    Calculates the Negative Log Likelihood loss.

    Args:
        probs: A list of Value objects representing softmax probabilities.
        target_index: The index of the correct class (integer).

    Returns:
        A Value object representing the NLL loss.
    """
    log_prob = probs[target_index].log()  # Log probability of the target class
    return -log_prob  # Negative log likelihood

def mse_loss(outputs, targets):
    """
    Mean Squared Error Loss

    Args:
        outputs: A list of Value objects (predictions).
        targets: A list of Value objects (actual values).

    Returns:
        A Value object representing the MSE loss.
    """
    loss = sum((o - t) ** 2 for o, t in zip(outputs, targets)) / len(outputs)
    return loss
