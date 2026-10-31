"""
Гаусів фільтр для відновлення зображення.

Фільтрація виконується у два проходи:
спочатку горизонтально, потім вертикально.
"""

import math

import numpy as np


def gaussian_kernel(sigma: float) -> np.ndarray:
    """Створює нормоване ядро Гауса."""

    if not math.isfinite(sigma) or sigma < 0.0 or sigma > 8.0:
        raise ValueError("Gaussian sigma must be in [0, 8].")

    if sigma <= 1e-6:
        return np.array([1.0], dtype=np.float64)

    radius = int(4.0 * sigma + 0.5)

    offsets = np.arange(
        -radius,
        radius + 1,
        dtype=np.float64,
    )

    kernel = np.exp(
        -(offsets ** 2) / (2.0 * sigma ** 2)
    )

    kernel /= np.sum(kernel)

    return kernel


def gaussian_blur(
    image: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Застосовує гаусів фільтр до зображення."""

    kernel = gaussian_kernel(sigma)

    if kernel.size == 1:
        return image.copy()

    radius = kernel.size // 2
    height, width = image.shape

    horizontal_source = np.pad(
        image,
        ((0, 0), (radius, radius)),
        mode="edge",
    )

    horizontal = np.zeros_like(
        image,
        dtype=np.float64,
    )

    for index, weight in enumerate(kernel):
        horizontal += (
            weight
            * horizontal_source[
                :,
                index:index + width,
            ]
        )

    vertical_source = np.pad(
        horizontal,
        ((radius, radius), (0, 0)),
        mode="edge",
    )

    result = np.zeros_like(
        image,
        dtype=np.float64,
    )

    for index, weight in enumerate(kernel):
        result += (
            weight
            * vertical_source[
                index:index + height,
                :,
            ]
        )

    return result