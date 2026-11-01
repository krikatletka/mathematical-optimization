
import math

import numpy as np


def generate_clean_image(size: int = 128) -> np.ndarray:
    """Програмно будує еталонне чисте зображення."""

    image = np.zeros((size, size), dtype=np.float64)

    for i in range(size):
        for j in range(size):
            u = j / (size - 1)
            v = i / (size - 1)

            # 1. Фон: градієнт і м'яка хвиля від центра.
            value = 0.25 + 0.35 * (0.6 * u + 0.4 * v)
            value += 0.08 * math.cos(
                2.0
                * math.pi
                * math.sqrt((u - 0.5) ** 2 + (v - 0.5) ** 2)
            )

            # 2. Світле коло.
            if (u - 0.32) ** 2 + (v - 0.34) ** 2 <= 0.16 ** 2:
                value = 0.92

            # 3. Темний квадрат.
            if abs(u - 0.70) <= 0.15 and abs(v - 0.30) <= 0.15:
                value = 0.10

            # 4. Трикутник.
            if 0.60 < v < 0.90 and abs(u - 0.30) < 0.9 * (v - 0.60):
                value = 0.68

            # 5. Шахівниця.
            if 0.58 < u < 0.92 and 0.60 < v < 0.90:
                cell = max(3, size // 12)
                value = (
                    0.85
                    if ((j // cell) + (i // cell)) % 2 == 0
                    else 0.20
                )

            image[i, j] = np.clip(value, 0.0, 1.0)

    # 6. Чотири білі смуги.
    for index in range(4):
        thickness = max(2, (4 - index) * max(1, size // 64))
        row = int(0.04 * size) + index * max(3, size // 22)

        if row + thickness < size:
            image[
                row : row + thickness,
                int(0.05 * size) : int(0.95 * size),
            ] = 0.98

    return image