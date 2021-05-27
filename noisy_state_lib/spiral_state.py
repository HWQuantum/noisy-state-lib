"""Functions to generate spiral states
"""

import numpy as np
from typing import Union, Callable, Sequence


def spiral_rho(d: int, sigma: float) -> np.ndarray:
    """Generate a normalised spiral density matrix with gaussian width sigma and dimension d
    """
    d_lim = d // 2
    rho = np.diag(np.exp(-(np.linspace(-d_lim, d_lim, d))**2 / (2 * sigma**2)))
    return rho / rho.diagonal().sum()


def _n_particle_func(reducing_func: Callable[[np.ndarray, np.ndarray],
                                             np.ndarray],
                     d: int,
                     sigma: Union[Sequence[float], float],
                     N=2) -> np.ndarray:
    """Helper function for the n_particle functions
    helps with code reuse
    """
    if isinstance(sigma, Sequence):
        if len(sigma) > 0:
            rho = spiral_rho(d, sigma[0])
            for s in sigma[1:]:
                rho = reducing_func(rho, spiral_rho(d, s))
            return rho
        else:
            raise ValueError("Sigma doesn't have any elements")
    else:
        rho = spiral_rho(d, sigma)
        for _ in range(N - 1):
            rho = reducing_func(rho, spiral_rho(d, sigma))
        return rho


def n_particle_spiral_rho(d: int,
                          sigma: Union[Sequence[float], float],
                          N=2) -> np.ndarray:
    """Generate an n-particle density matrix.
    d is the dimension of each particle
    if sigma is a number then it generates a two particle state where each particle 
    has the bandwidth sigma
    
    if sigma is a list then a state is generated which has the same number of particles
    as elements of sigma, each with the given width

    The order of indices is (p_1_bra, p_1_ket, p_2_bra, p_2_ket, ...)

    N gives the number of particles that should be in the state if sigma is just a number
    """
    return _n_particle_func(lambda d, s: np.tensordot(d, s, axes=0), d, sigma,
                            N)


def n_particle_spiral_rho_reduced(d: int,
                                  sigma: Union[Sequence[float], float],
                                  N: int = 2) -> np.ndarray:
    """Generate an n-particle density matrix.
    d is the dimension of each particle
    if sigma is a number then it generates a two particle state where each particle 
    has the bandwidth sigma
    
    if sigma is a list then a state is generated which has the same number of particles
    as elements of sigma, each with the given width

    N gives the number of particles that should be in the state if sigma is just a number

    This function produces a final density matrix with two indices.
    """
    rho = n_particle_spiral_rho(d, sigma, N)
    return rho.reshape((d**N, d**N))
