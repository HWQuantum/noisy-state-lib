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


def v_j(v, a, d):
    """The vth basis vector with measurement setting a in dimension d
    """
    return 1 / np.sqrt(d) * np.array(
        [np.exp(2 * np.pi * j * 1j / d * (v + a)) for j in range(d)])


def w_j(w, b, d):
    """The wth basis vector with measurement setting b in dimension d
    """
    return v_j(-w, b, d)


def p_single(k, l, a, b, d):
    """Generate the operator P(A=k, B=l) in dimension d
    """
    prod = np.kron(v_j(k, a, d), w_j(l, b, d))
    return np.tensordot(prod, prod.conj(), 0)


def p_mult(a, b, k, d):
    """Generate the operator P(A=B+k) in dimension d
    """
    return np.sum([p_single(i + k, i, a, b, d) for i in range(d)], axis=0)


def s(a_0, a_1, b_0, b_1, d):
    """Bell operator for dimension d with measurement settings a_0, a_1, b_0, b_1
    """
    return np.sum([(1 - 2 * k / (d - 1)) *
                   (p_mult(a_0, b_0, k, d) + p_mult(a_1, b_0, -k - 1, d) +
                    p_mult(a_1, b_1, k, d) + p_mult(a_0, b_1, -k, d) -
                    (p_mult(a_0, b_0, -k - 1, d) + p_mult(a_1, b_0, k, d) +
                     p_mult(a_1, b_1, -k - 1, d) + p_mult(a_0, b_1, k + 1, d)))
                   for k in range(int(d / 2))],
                  axis=0)
