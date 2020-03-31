"""Contains functions related to noise on states
"""

import numpy as np
from numba import njit


def mixed_noise_state(rho, p):
    """Make a noisy state from rho, with noise parameter p
    """
    d = rho.shape[0]
    return p * rho + (1 - p) * np.eye(d) / d


@njit
def add_crosstalk(rho, sigma, amplitude):
    """Add crosstalk between the modes of rho, parameterised by sigma and amplitude
    """
    if sigma == 0:
        return rho.copy() / np.trace(rho)
    dim = int(np.rint(np.sqrt(rho.shape[0])))
    d_lim = dim // 2
    new_arr = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                dist_sq = (i - k)**2 + (j - k)**2
                new_arr[i, j] += rho[k*dim+k, k*dim+k] * np.exp(-dist_sq / (2 * sigma**2))
    new_rho = rho.copy()
    for i in range(dim**2):
        new_rho[i, i] += new_arr[i//dim, i%dim] * amplitude
    return new_rho / np.trace(new_rho)
