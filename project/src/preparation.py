
from src.clean_image import generate_clean_image
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

    clean = generate_clean_image(IMAGE_SIZE)

    save_image(
        clean,
        "outputs/00_clean.png",
    )

    print("\nClean image 5x5 check:")

    for i in range(60, 65):
        print(
            " ".join(
                f"{clean[i, j]:.4f}"
                for j in range(60, 65)
            )
        )

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

    max_save_error = float(
        np.max(np.abs(clean - copied_image))
    )

    if max_save_error > 1.0 / 255.0:
        raise RuntimeError(
            "Saved image differs from the generated image too much: "
            f"{max_save_error:.9e}"
        )

    print(
        "PNG round-trip check: PASSED, "
        f"maximum rounding error = {max_save_error:.9e}"
    )

    print("\nSource image = input/00_clean.png")
    print(
        
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