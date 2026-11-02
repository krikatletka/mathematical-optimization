"""Додаткове завдання ЛР4: згладжена повна варіація (TV).

Покладіть цей файл поряд із main.py та запустіть: python lab4_tv.py
Параметри й модель відповідають додатковому завданню перед пунктом 4.12.
"""

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.image_io import save_image
from src.filters import gaussian_blur
from src.lab1 import create_objective, svenn
from src.lab2 import golden_section
from src.lab3 import HI, LO, Objective2, bilateral_filter, nelder_mead
from src.lab4 import (
    _golden_lambda, check_operators, divergence, grad_xy, tikhonov_exact,
)
from src.metrics import mse, psnr
from src.oracle import Oracle
from src.preparation import prepare_project
from src.variant import (
    EPSILON, LAMBDA0, NOISE_SIGMA, SEED, SIMPLEX_EDGE, SVENN_DELTA0, SVENN_X0,
)


TV_SMOOTHING = 5e-3


def tv_energy(image, noisy, lambda_tv, smoothing=TV_SMOOTHING):
    """E_TV = 0.5*||I-noisy||^2 + lambda_tv*sum(sqrt(|grad I|^2+eps^2))."""
    gx, gy = grad_xy(image)
    magnitude = np.sqrt(gx * gx + gy * gy + smoothing * smoothing)
    return (0.5 * float(np.sum((image - noisy) ** 2))
            + lambda_tv * float(np.sum(magnitude)))


def tv_gradient(image, noisy, lambda_tv, smoothing=TV_SMOOTHING):
    """Градієнт тієї самої згладженої TV-енергії."""
    gx, gy = grad_xy(image)
    magnitude = np.sqrt(gx * gx + gy * gy + smoothing * smoothing)
    return (image - noisy) - lambda_tv * divergence(gx / magnitude, gy / magnitude)


def nesterov_tv(noisy, lambda_tv, smoothing=TV_SMOOTHING,
                max_iterations=3000, tolerance=1e-6):
    """Нестеров для сильно опуклої TV-моделі, mu=1.

    Журнал і результат містять точки запиту y_k. Тому норма градієнта
    стосується саме повернутого зображення. k — число виконаних переходів;
    перевірки початкового й кінцевого стану також враховані в лічильниках.
    """
    noisy = np.asarray(noisy, dtype=float)
    if noisy.ndim != 2 or min(noisy.shape) < 2 or not np.all(np.isfinite(noisy)):
        raise ValueError("Expected a finite 2D image with both dimensions >= 2.")
    if not math.isfinite(lambda_tv) or lambda_tv < 0:
        raise ValueError("lambda_tv must be finite and nonnegative.")
    if not math.isfinite(smoothing) or smoothing <= 0:
        raise ValueError("TV smoothing must be finite and positive.")
    if not isinstance(max_iterations, int) or max_iterations < 0:
        raise ValueError("max_iterations must be a nonnegative integer.")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive.")

    lipschitz = 1.0 + 8.0 * lambda_tv / smoothing
    step = 1.0 / lipschitz
    root_l = math.sqrt(lipschitz)
    beta = (root_l - 1.0) / (root_l + 1.0)
    x = noisy.copy()
    y = x.copy()
    rows = []
    n_f = n_grad = 0
    converged = False

    for k in range(max_iterations + 1):
        gradient = tv_gradient(y, noisy, lambda_tv, smoothing)
        n_grad += 1
        gradient_norm = float(np.linalg.norm(gradient))
        current_energy = tv_energy(y, noisy, lambda_tv, smoothing)
        n_f += 1
        if not math.isfinite(current_energy) or not math.isfinite(gradient_norm):
            raise RuntimeError("TV iteration produced a non-finite value.")
        rows.append([k, current_energy, gradient_norm, step, n_f, n_grad])

        if gradient_norm <= tolerance:
            converged = True
            break
        if k == max_iterations:
            break

        new_x = y - step * gradient
        y = new_x + beta * (new_x - x)
        x = new_x

    return {
        "image": y.copy(), "iterations": rows[-1][0],
        "energy": rows[-1][1], "gradient_norm": rows[-1][2],
        "n_f": n_f, "n_grad": n_grad, "converged": converged,
        "lipschitz": lipschitz, "step": step, "beta": beta, "rows": rows,
    }


def check_tv_gradient(noisy, lambda_tv, smoothing=TV_SMOOTHING):
    """Скінченні різниці: випадкове, майже плоске та спостережене зображення.

    Діагностика окрема від розв'язувача: 6 запитів E і 3 запити градієнта.
    """
    rng = np.random.default_rng(SEED)
    crop = np.asarray(noisy[:16, :16], dtype=float)
    cases = [
        ("random", rng.random(crop.shape)),
        ("nearly flat", 0.5 + 1e-5 * rng.normal(size=crop.shape)),
        ("observation", crop.copy()),
    ]
    rows = []
    for name, image in cases:
        direction = rng.normal(size=image.shape)
        direction /= np.linalg.norm(direction)
        h = 1e-6
        numeric = (
            tv_energy(image + h * direction, crop, lambda_tv, smoothing)
            - tv_energy(image - h * direction, crop, lambda_tv, smoothing)
        ) / (2.0 * h)
        analytic = float(np.sum(tv_gradient(image, crop, lambda_tv, smoothing) * direction))
        error = abs(numeric - analytic) / max(1.0, abs(numeric), abs(analytic))
        rows.append([name, numeric, analytic, error])
    return rows


def write_csv(path, headers, rows):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(headers)
        writer.writerows(rows)


def find_lambda_star(clean, noisy):
    oracle = Oracle(lambda lam: mse(tikhonov_exact(noisy, lam), clean))
    lam = _golden_lambda(oracle.value, start=LAMBDA0, eps=1e-3)
    return lam, oracle.calls


def comparison_images(clean, noisy, lambda_star, tv_image):
    """Повторює базові налаштування фільтрів без підміни їхніх параметрів."""
    objective = create_objective(noisy, clean)
    bracket_oracle = Oracle(objective)
    bracket = svenn(bracket_oracle, SVENN_X0, SVENN_DELTA0)
    search_oracle = Oracle(objective)
    gaussian_result = golden_section(search_oracle, bracket.a, bracket.b, EPSILON)
    gaussian_image = gaussian_blur(noisy, gaussian_result.x)

    x0 = np.array([
        float(np.clip(2.0 * gaussian_result.x, 0.5, 4.0)),
        float(np.clip(0.15 + 0.1 * NOISE_SIGMA, 0.02, 0.9)),
    ])
    objective_2d = Objective2(noisy, clean)
    nm = nelder_mead(objective_2d, x0, SIMPLEX_EDGE)
    theta = np.clip(nm["x"], LO, HI)
    bilateral_image = bilateral_filter(noisy, theta[0], theta[1])
    print(f"Baseline fitting (separate): Svenn Nf={bracket_oracle.calls}; "
          f"golden Nf={search_oracle.calls}; Nelder-Mead Nf={objective_2d.calls}; "
          f"NM converged={nm['converged']}")

    return [
        ("Noisy", noisy, "observation"),
        ("Gaussian", gaussian_image, f"sigma={gaussian_result.x:.9f}"),
        ("Bilateral (NM)", bilateral_image,
         f"sigma_s={theta[0]:.9f}, sigma_r={theta[1]:.9f}"),
        ("Tikhonov", tikhonov_exact(noisy, lambda_star), f"lambda={lambda_star:.9f}"),
        ("Smoothed TV", tv_image, "lambda_tv=max(0.02,0.15*lambda*)"),
    ], [["Svenn", bracket_oracle.calls], ["Golden", search_oracle.calls],
         ["Nelder-Mead", objective_2d.calls]]


def run_tv(clean, noisy, lambda_star=None):
    """Запускає додаткове завдання; lambda_star можна передати з run_lab4()."""
    Path("outputs").mkdir(exist_ok=True)
    print("\nLAB 4 - ADDITIONAL TASK: SMOOTHED TV")
    search_calls = 0
    if lambda_star is None:
        lambda_star, search_calls = find_lambda_star(clean, noisy)
        print(f"Lambda search calls = {search_calls} (separate)")
    else:
        lambda_star = float(lambda_star)
        print("Lambda supplied from the main Lab 4 run; no repeated search.")
    if not math.isfinite(lambda_star) or lambda_star <= 0:
        raise ValueError("lambda_star must be finite and positive.")
    lambda_tv = max(0.02, 0.15 * lambda_star)
    print(f"lambda* = {lambda_star:.9f}")
    print(f"lambda_TV = {lambda_tv:.9f}; smoothing = {TV_SMOOTHING:.6f}")

    adjoint_error = check_operators(noisy.shape, seed=SEED)
    checks = check_tv_gradient(noisy, lambda_tv)
    max_error = max(row[3] for row in checks)
    write_csv("outputs/lab4_tv_gradient_check.csv",
              ["Case", "Numeric", "Analytic", "Relative_error"], checks)
    print(f"Adjoint check error = {adjoint_error:.3e}")
    print(f"TV gradient check error = {max_error:.3e} (diagnostic Nf=6, Ngrad=3)")
    if not math.isfinite(adjoint_error) or adjoint_error > 1e-12:
        raise RuntimeError("Adjoint operator check failed.")
    if any(not math.isfinite(row[3]) or row[3] > 1e-6 for row in checks):
        raise RuntimeError("TV finite-difference gradient check failed.")

    result = nesterov_tv(noisy, lambda_tv)
    image = result["image"]
    result.update(lambda_star=lambda_star, lambda_tv=lambda_tv,
                  smoothing=TV_SMOOTHING, lambda_search_calls=search_calls,
                  psnr=psnr(image, clean), mse=mse(image, clean))
    print(f"L bound = {result['lipschitz']:.9f}; mu = 1; beta = {result['beta']:.9f}")
    print(f"Step = {result['step']:.9f}")
    print(f"Nesterov iterations = {result['iterations']}; Nf = {result['n_f']}; "
          f"Ngrad = {result['n_grad']}; total = {result['n_f'] + result['n_grad']}")
    print(f"Final gradient norm = {result['gradient_norm']:.9e}")
    print(f"Converged = {result['converged']}")
    print(f"TV PSNR = {result['psnr']:.6f} dB")
    print(f"Pixel range before PNG encoding = [{image.min():.9f}; {image.max():.9f}]")

    write_csv("outputs/lab4_tv_trace.csv",
              ["k", "E_y", "Gradient_norm_y", "Step", "Nf", "Ngrad"], result["rows"])
    summary = [[key, result[key]] for key in [
        "lambda_star", "lambda_tv", "smoothing", "lipschitz", "step", "beta",
        "iterations", "n_f", "n_grad", "energy", "gradient_norm", "converged", "mse", "psnr",
    ]]
    write_csv("outputs/lab4_tv_summary.csv", ["Parameter", "Value"], summary)
    save_image(image, "outputs/06_tv.png")

    records, costs = comparison_images(clean, noisy, lambda_star, image)
    write_csv("outputs/lab4_tv_preparation_costs.csv", ["Stage", "Nf"],
              [["lambda search", search_calls], *costs])
    comparisons = [[name, mse(data, clean), psnr(data, clean), parameters]
                   for name, data, parameters in records]
    write_csv("outputs/lab4_tv_comparison.csv", ["Method", "MSE", "PSNR_dB", "Parameters"],
              comparisons)
    for name, error, quality, _ in comparisons:
        print(f"{name}: MSE={error:.9f}, PSNR={quality:.6f} dB")
    tikhonov_psnr = next(row[2] for row in comparisons if row[0] == "Tikhonov")
    print(f"TV improvement over Tikhonov = {result['psnr'] - tikhonov_psnr:.6f} dB")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    trace = result["rows"]
    axes[0].plot([r[0] for r in trace], [r[1] for r in trace])
    axes[0].set(xlabel="iteration k", ylabel="E_TV(y_k)", title="TV energy")
    axes[1].semilogy([r[0] for r in trace], [max(r[2], 1e-16) for r in trace])
    axes[1].axhline(1e-6, color="red", linestyle="--", label="tolerance")
    axes[1].set(xlabel="iteration k", ylabel="gradient norm at y_k", title="TV stopping criterion")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("outputs/lab4_tv_convergence.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    panels = [("Clean", clean)] + [(f"{name}\nPSNR={psnr(data, clean):.4f} dB", data)
                                         for name, data, _ in records]
    for ax, (name, data) in zip(axes.flat, panels):
        ax.imshow(data, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(name)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig("outputs/lab4_tv_images.png", dpi=150)
    plt.close(fig)
    print("Saved image = outputs/06_tv.png")
    print("Saved plots = outputs/lab4_tv_convergence.png, outputs/lab4_tv_images.png")
    print("TV TASK COMPLETE" if result["converged"] else "TV RUN FINISHED: iteration limit reached")
    return result


def main():
    project = prepare_project()
    run_tv(project.clean, project.noisy)


if __name__ == "__main__":
    main()
