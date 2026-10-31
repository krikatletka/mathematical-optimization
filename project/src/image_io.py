"""
Завантаження і  збереження зображень.

Зображення представляється як NumPy-масив із числами від 0 до 1.
"""

from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: str) -> np.ndarray:
    """Завантажує зображення у відтінках сірого."""

    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Cannot find image: {path}")

    with Image.open(file_path) as image:
        grayscale = image.convert("L")
        pixels = np.asarray(grayscale, dtype=np.float64)

    return pixels / 255.0


def save_image(image: np.ndarray, path: str) -> None:
    """Зберігає зображення у форматі PNG."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    clipped = np.clip(image, 0.0, 1.0)
    pixels = np.rint(clipped * 255.0).astype(np.uint8)

    Image.fromarray(pixels, mode="L").save(file_path)