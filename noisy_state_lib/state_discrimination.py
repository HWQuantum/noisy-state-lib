"""Contains functions to do with state discrimination
"""

import numpy as np
from typing import Tuple


def symmetric_states(theta: float, d: int) -> np.ndarray:
    """Generate d symmetric overlap states in d dimensions, with parameter theta controlling the overlap

    Args:
        theta: The angle between the states
        d: The dimension to generate the states in

    Returns:
        np.ndarray: An array with indices ``(state, state_coefficients)``
    """
    v_1 = np.zeros(d, dtype=np.complex128)
    v_1[0] = 1
    vecs = [v_1]
    for i in range(1, d):
        v = np.zeros(d, dtype=np.complex128)
        for j, v_j in enumerate(vecs):
            v[j] = (-1 / (d - 1) - v[:j].dot(v_j[:j].conj())) / v_j[j]
        if i < d - 1:
            v[i] = np.sqrt(1 - v.dot(v.conj()))
        vecs.append(v)
    last_dim = np.zeros(d, dtype=np.complex128)
    last_dim[-1] = 1
    for i in range(d):
        vecs[i] = vecs[i] * np.sin(theta) + last_dim * np.cos(theta)
    return np.array(vecs)


def orthogonal_states(states: np.ndarray) -> np.ndarray:
    """Generate the set of states |ψi⊥> which are orthogonal to all vectors apart from |ψi>
    
    Args:
        states: The input set of states. Can come from the ``symmetric_states`` function.

    Returns:
        np.ndarray: A new set of states which are orthogonal to the input states
    """
    vecs = []
    for i, s in enumerate(states):
        _, _, u = np.linalg.svd(np.delete(states, i, 0), full_matrices=False)
        v = s.copy()
        for u_v in u:
            v -= v @ u_v.conj() * u_v
        v = v / np.sqrt(np.abs(v @ v.conj()))
        vecs.append(v)
    return np.array(vecs)


def measurement_states(orthogonal_states: np.ndarray) -> np.ndarray:
    """Given a set of d states which are orthogonal to the symmetric states, create the d+1 measurement states
    in dimension d+1
    
    Args:
        orthogonal_states: The set of orthogonal states
    
    Returns: 
        np.ndarray: The states that should be projected onto 
    """
    d = len(orthogonal_states[0])
    new_states = np.empty((d + 1, d + 1), dtype=np.complex128)
    new_states[:-1, :-1] = np.array(orthogonal_states)
    new_states[:, -1] = np.sqrt(
        -orthogonal_states[0] @ orthogonal_states[1].conj())
    _, _, u = np.linalg.svd(new_states[:-1], full_matrices=False)
    unknown_v = np.ones(d + 1, dtype=np.complex128)
    for v in u:
        unknown_v -= unknown_v @ v.conj() * v
    new_states[-1, :] = unknown_v
    for i, v in enumerate(new_states):
        new_states[i, :] /= np.sqrt(v @ v.conj())
    return new_states


def overlaps(theta: float, d: int) -> np.ndarray:
    """Create the overlap matrix <ψi|Dj><Dj|ψi> for a given theta and dimension

    Args:
        theta: The angle between the states
        d: The dimension to create the states in.

    Returns:
        np.ndarray: The overlap matrix between the states.
    """
    s = symmetric_states(theta, d)
    o = orthogonal_states(s)
    m = measurement_states(o)
    s_expanded = np.zeros((d, d + 1), dtype=np.complex128)
    s_expanded[:, :-1] = np.array(s)
    cross_mat = np.zeros((d, d + 1))
    for i, v in enumerate(s_expanded):
        for j, u in enumerate(m):
            u = u / np.sqrt(u @ u.conj())
            cross_mat[i, j] = np.abs(v @ u.conj())**2
    return cross_mat


def get_transformation_states(theta: float,
                              d: int) -> Tuple[np.ndarray, np.ndarray]:
    """Given a dimension d and a theta, return the output states 
    to be transformed into as a matrix

    Args:
        theta: The angle between the states
        d: The dimension of the states

    Returns:
        A tuple of the expanded set of states to transform into and the overlap matrix.
    """
    s = symmetric_states(theta, d)
    o = orthogonal_states(s)
    m = measurement_states(o)
    s_expanded = np.zeros((d, d + 1), dtype=np.complex128)
    s_expanded[:, :-1] = np.array(s)
    cross_mat = np.zeros((d, d + 1), dtype=np.complex128)
    for i, v in enumerate(s_expanded):
        for j, u in enumerate(m):
            u = u / np.sqrt(u @ u.conj())
            cross_mat[i, j] = v @ u.conj()
    assert np.all(np.isclose(
        (np.abs(cross_mat)**2).sum(axis=1),
        np.ones(d))), "Overlap isn't correct. Try a smaller angle"
    return s_expanded, cross_mat
