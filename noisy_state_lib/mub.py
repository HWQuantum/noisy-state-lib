"""Contains functions to do with MUBs, like generating bases and taking measurements
"""

import numpy as np
from .noise import mixed_noise_state
from .spiral_state import n_particle_spiral_rho_reduced


def basis(dim, a, n):
    """Generate the basis vectors of the mutually unbiased bases in dim = 2j+1
    dimensions
    The index a ∈ (0, 2j+1) (dim+1 bases) denotes which MUB the vector is drawn from
    a=0 gives the computational basis
    The index n ∈ (0, 2j) denotes which vector is chosen
    Taken from the paper: https://arxiv.org/pdf/quant-ph/0601092.pdf
    """
    if a == 0:
        v = np.zeros(dim, dtype=np.complex128)
        v[n % dim] = 1
        return v
    else:
        j = (dim - 1) / 2
        q = np.exp(1j * 2 * np.pi / dim)
        return 1 / np.sqrt(dim) * np.array([
            np.power(q, 0.5 * (j + m) * (j - m + 1) * (a - 1) + (j + m) * n)
            for m in np.linspace(-j, j, dim)
        ],
                                           dtype=np.complex128)


def transform_matrix(dim, mub):
    """Generate the transformation matrix from computational MUB 
    to MUB=mub
    """
    return np.array([basis(dim, mub, i) for i in range(dim)]).T


def dual_transform(dim, mub):
    """Generate the transformation matrix to move two particles from the computational
    MUB to MUB mub
    """
    m = transform_matrix(dim, mub)
    return np.kron(m, m.conj())


def coincidences_vec(d, width, p):
    """Get the coincidence matrices in MUB 0 and MUB 1 for the values
    of d, width and werner state p
    returns a matrix [width [p [MUB0, MUB1]]]
    """

    t = dual_transform(d, 1)
    mub_t = (t, t.conj().T)
    if type(width) != np.ndarray:
        if type(width) == list:
            width = np.array(width)
        else:
            width = np.array([width])
    if type(p) != np.ndarray:
        if type(p) == list:
            p = np.array(p)
        else:
            p = np.array([p])

    coincidences = np.zeros((len(width), len(p), 2, d, d), dtype=np.complex128)

    for i, w in enumerate(width):
        for j, prob in enumerate(p):
            comp = mixed_noise_state(n_particle_spiral_rho_reduced(d, w), prob)
            coincidences[i, j, 0] = comp.diagonal().reshape((d, d))
            coincidences[i, j,
                         1] = (mub_t[0] @ comp @ mub_t[1]).diagonal().reshape(
                             (d, d))

    return np.abs(coincidences)
