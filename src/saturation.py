import numpy as np

def hill_saturation(x, alpha, gamma, eps=1e-8):
    """
    Apply Hill function to model diminishing returns.

    Parameters
    ----------
    x : array-like
        Adstocked spend or exposure.
    alpha : float
        Steepness of the curve.
    gamma : float
        Half-saturation point.
    eps : float, optional
        Small constant to avoid division by zero.

    Returns
    -------
    numpy.ndarray
        Saturated values scaled between 0 and 1.
    """
    x = np.maximum(x, 0)  # no negatives
    return np.power(x, alpha) / (np.power(x, alpha) + np.power(gamma, alpha) + eps)
