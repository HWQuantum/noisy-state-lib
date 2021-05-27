"""Contains functions to help displaying states
"""

import numpy as np
from matplotlib.colors import hsv_to_rgb


def colourise(a: np.ndarray) -> np.ndarray:
    """Turn an array a with dimensions d into an HSV array

    Args:
        a (np.ndarray): The input array

    Returns:
        np.ndarray: The coloured array, ready to display with matplotlib
    """
    def map_value(v, min_v, max_v, min_o, max_o):
        """Map the values v from the initial min and max vs to min and max os
        """
        ratio = (max_o - min_o) / (max_v - min_v)
        return (v - min_v) * ratio + min_o

    abs_a = np.abs(a).astype(np.float64)

    return hsv_to_rgb(
        np.moveaxis(
            np.array([
                map_value(np.angle(a), -np.pi, np.pi, 0, 1),
                np.ones_like(a),
                map_value(abs_a, 0, np.max(abs_a), 0, 1)
            ]), 0, -1).astype(np.float64))
