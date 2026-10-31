"""
Відтворюваний генератор випадкових чисел.

Для нормального розподілу використовується
перетворення Бокса — Мюллера.
"""

import math

import numpy as np


UINT64_MASK = (1 << 64) - 1


class RandomGenerator:
    def __init__(self, seed: int):
        self.state = seed & UINT64_MASK

    def uniform(self) -> float:
        """Повертає рівномірне число з інтервалу [0, 1)."""

        self.state = (
            self.state * 6364136223846793005
            + 1442695040888963407
        ) & UINT64_MASK

        return (self.state >> 11) / 9007199254740992.0

    def normal(self) -> float:
        """Повертає число з нормального розподілу N(0, 1)."""

        u1 = max(self.uniform(), 1e-300)
        u2 = self.uniform()

        return (
            math.sqrt(-2.0 * math.log(u1))
            * math.cos(2.0 * math.pi * u2)
        )


def add_gaussian_noise(
    image: np.ndarray,
    sigma: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Додає гаусів шум.

    Повертає:
    raw — значення до обрізання;
    noisy — значення в межах [0, 1];
    outside_count — кількість значень поза [0, 1].
    """

    generator = RandomGenerator(seed)

    noise = np.array(
        [generator.normal() for _ in range(image.size)],
        dtype=np.float64,
    ).reshape(image.shape)

    raw = image + sigma * noise
    outside_count = int(np.count_nonzero((raw < 0.0) | (raw > 1.0)))

    noisy = np.clip(raw, 0.0, 1.0)

    return raw, noisy, outside_count