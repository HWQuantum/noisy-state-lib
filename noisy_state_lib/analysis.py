"""Contains analysis functions for noisy MUBs
"""

import numpy as np


def quantum_contrast(mat):
    """Calculate the quantum contrast from the diagonals
    of a coincidence matrix.
    The quantum contrast is the average of the diagonal divided by the average of the off-diagonal

    The input should be a coincidence matrix
    """
    off_diag_avg = (rho.sum() - rho.diagonal().sum()) / (rho.shape[0] *
                                                         (rho.shape[1] - 1))
    return np.average(rho.diagonal()) / off_diag_avg


def visibility(mat):
    """Calculate the visibility of the state. 
    The visibility is the sum of the diagonal divided by the sum over the whole coincidence matrix

    The input should be a coincidence matrix
    """
    return rho.diagonal().sum() / rho.sum()


def all_mub_fidelity_bound(data):
    """Calculate the fidelity bound on a set of data given the coincidence matrices in all MUBs
    
    data should be an iterable over the coincidence matrices in each MUB
    """
    d = data[0].shape[0]
    return sum([i.diagonal().sum() for i in data]) / d - 1 / d


def conditional_entropy(data):
    """Calculate the conditional entropy of the diagonals of a coincidence matrix
    """
    p_x = data.sum(axis=1).reshape((data.shape[0], 1))
    d = np.log2(data / p_x)
    non_zero = np.where((np.invert(np.isnan(d))) & (np.invert(np.isinf(d))))
    return -(data[non_zero] * d[non_zero]).sum()
