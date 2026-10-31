"""
Метрики якості відновлення зображення.
"""

import math

import numpy as np


def mse(first: np.ndarray, second: np.ndarray) -> float:
    """Обчислює середньоквадратичну помилку."""

    if first.shape != second.shape:
        raise ValueError("Images must have identical dimensions.")

    difference = first.astype(np.float64) - second.astype(np.float64)

    return float(np.mean(difference ** 2))


def psnr(first: np.ndarray, second: np.ndarray) -> float:
    """Обчислює PSNR у децибелах."""

    error = mse(first, second)

    if error == 0.0:
        return math.inf

    return 10.0 * math.log10(1.0 / error)