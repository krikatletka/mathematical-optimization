"""
Частина 0 — підготовка проєкту.

Завантажує початкове зображення, додає шум варіанта 8,
зберігає результати та обчислює MSE і PSNR.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.image_io import load_image, save_image
from src.metrics import mse, psnr
from src.rng import RandomGenerator, add_gaussian_noise
from src.variant import (
    EPSILON,
    IMAGE_SIZE,
    LAMBDA0,
    NOISE_SIGMA,
    SEED,
    SIMPLEX_EDGE,
    SVENN_DELTA0,
    SVENN_X0,
    VARIANT_NUMBER,
)


@dataclass
class ProjectData:
    clean: np.ndarray
    raw_noisy: np.ndarray
    noisy: np.ndarray


def prepare_project() -> ProjectData:
    """Виконує нульову частину проєкту."""

    print("IMAGE RESTORATION PROJECT")
    print(f"Variant = {VARIANT_NUMBER}")
    print(f"Seed = {SEED}")
    print(f"Noise sigma = {NOISE_SIGMA:.6f}")
    print(f"Svenn x0 = {SVENN_X0:.6f}")
    print(f"Svenn delta0 = {SVENN_DELTA0:.6f}")
    print(f"Epsilon = {EPSILON:.6f}")
    print(f"Lambda0 = {LAMBDA0:.6f}")
    print(f"Simplex edge = {SIMPLEX_EDGE:.6f}")

    Path("outputs").mkdir(exist_ok=True)

    clean = load_image("input/00_clean.png")

    if clean.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(
            f"Expected {IMAGE_SIZE} x {IMAGE_SIZE} image, "
            f"received {clean.shape[1]} x {clean.shape[0]}."
        )

    save_image(
        clean,
        "outputs/00_clean_copy.png",
    )

    copied_image = load_image(
        "outputs/00_clean_copy.png"
    )

    if not np.array_equal(clean, copied_image):
        raise RuntimeError(
            "Saved image pixels do not match source pixels."
        )

    print("\nSource image = input/00_clean.png")
    print(
        f"Image size = {clean.shape[1]} x {clean.shape[0]}"
    )
    print(f"Pixels = {clean.size}")
    print("Saved copy = outputs/00_clean_copy.png")
    print("Pixel check: PASSED")

    check_generator = RandomGenerator(SEED)

    first_values = [
        check_generator.normal()
        for _ in range(6)
    ]

    print("\nFirst six normal values:")

    print(
        " ".join(
            f"{value:.6f}"
            for value in first_values
        )
    )

    raw_noisy, noisy, outside_count = add_gaussian_noise(
        clean,
        NOISE_SIGMA,
        SEED,
    )

    save_image(
        noisy,
        "outputs/01_noisy.png",
    )

    print("\nNOISE RESULTS")
    print(f"MSE = {mse(noisy, clean):.9f}")
    print(f"PSNR = {psnr(noisy, clean):.6f} dB")
    print(
        f"Raw pixels outside [0,1] = {outside_count}"
    )
    print("Saved image = outputs/01_noisy.png")

    return ProjectData(
        clean=clean,
        raw_noisy=raw_noisy,
        noisy=noisy,
    )