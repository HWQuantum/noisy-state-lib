"""Contains analysis functions for noisy MUBs
"""

import numpy as np
from typing import Sequence, Tuple


def quantum_contrast(matrix: np.ndarray) -> float:
    """Calculate the quantum contrast from the diagonals
    of a coincidence matrix.
    The quantum contrast is the average of the diagonal divided by the average of the off-diagonal

    The input should be a coincidence matrix
    """
    off_diag_avg = (matrix.sum() -
                    matrix.diagonal().sum()) / (matrix.shape[0] *
                                                (matrix.shape[1] - 1))
    return np.average(matrix.diagonal()) / off_diag_avg


def visibility(matrix: np.ndarray) -> float:
    """Calculate the visibility of the state. 
    The visibility is the sum of the diagonal divided by the sum over the whole coincidence matrix

    The input should be a coincidence matrix
    """
    return matrix.diagonal().sum() / matrix.sum()


def all_mub_fidelity_bound(data: Sequence[np.ndarray]) -> float:
    """Calculate the fidelity bound on a set of data given the coincidence matrices in all MUBs
    
    data should be an iterable over the coincidence matrices in each MUB
    """
    d = data[0].shape[0]
    return sum([i.diagonal().sum() for i in data]) / d - 1 / d


def conditional_entropy(data: np.ndarray) -> float:
    """Calculate the conditional entropy of the diagonals of a coincidence matrix
    """
    p_x = data.sum(axis=1).reshape((data.shape[0], 1))
    d = np.log2(data / p_x)
    non_zero = np.where((np.invert(np.isnan(d))) & (np.invert(np.isinf(d))))
    return -(data[non_zero] * d[non_zero]).sum()


def v_j(v: int, a: int, d: int) -> np.ndarray:
    """The vth basis vector with measurement setting a in dimension d
    """
    return 1 / np.sqrt(d) * np.array(
        [np.exp(2 * np.pi * j * 1j / d * (v + a)) for j in range(d)])


def w_j(w: int, b: int, d: int) -> np.ndarray:
    """The wth basis vector with measurement setting b in dimension d
    """
    return v_j(-w, b, d)


def p_single(k: int, l: int, a: int, b: int, d: int) -> np.ndarray:
    """Generate the operator P(A=k, B=l) in dimension d
    """
    prod = np.kron(v_j(k, a, d), w_j(l, b, d))
    return np.tensordot(prod, prod.conj(), 0)


def p_mult(a: int, b: int, k: int, d: int) -> np.ndarray:
    """Generate the operator P(A=B+k) in dimension d
    """
    return np.sum(np.array([p_single(i + k, i, a, b, d) for i in range(d)]),
                  axis=0)


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


def bloch_sphere_representation(state: np.ndarray) -> Tuple[float, float]:
    """Get the (θ, φ) in the bloch sphere of the given 2-dimensional state
    
    Args:
        state: The 2-dimensional state
    
    Returns:
        (float, float): A tuple of  (θ, φ)
    """
    theta = 2 * np.arccos(state[0])
    s_t = np.sin(theta / 2)
    phi = 0.0 if np.isclose(s_t, 0) else np.angle(state[1] / s_t)
    return (np.real(theta), np.real(phi))
