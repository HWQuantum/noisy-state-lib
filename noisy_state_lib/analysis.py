"""Contains analysis functions for noisy MUBs
"""

import numpy as np


def quantum_contrast(rho):
    """Calculate the quantum contrast from the diagonals
    of a coincidence matrix.
    The quantum contrast is the average of the diagonal divided by the average of the off-diagonal
    """
    off_diag_avg = (rho.sum() - rho.diagonal().sum()) / (rho.shape[0] *
                                                         (rho.shape[1] - 1))
    return np.average(rho.diagonal()) / off_diag_avg


def visibility(rho):
    """Calculate the visibility of the state. 
    The visibility is the sum of the diagonal divided by the sum over the whole state
    """
    return rho.diagonal().sum() / rho.sum()
