import numpy as np

def mixed_noise_state(rho, p):
    """Make a noisy state from rho, with noise parameter p
    """
    d = rho.shape[0]
    return p*rho + (1-p) * np.eye(d)/d
