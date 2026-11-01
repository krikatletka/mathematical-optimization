"""Обов'язкові експерименти до лабораторної роботи №4.

Запускає три досліди:
1. Порівняння методів при великому числі обумовленості.
2. Демонстрація розбіжності при завеликому кроці.
3. Дослідження впливу lambda на PSNR.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.lab4 import (
    _golden_lambda,
    _mse,
    _psnr,
    energy,
    energy_grad,
    gradient_method,
    tikhonov_exact,
)
from src.preparation import prepare_project


LAMBDA_START = 1.4817


def write_csv(path: str, header: list[str], rows: list[list[object]]) -> None:
    """Зберігає таблицю результатів у CSV-файл."""
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def print_table(headers: list[str], rows: list[list[object]]) -> None:
    """Виводить таблицю з рамкою в консоль."""
    values = [[str(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]

    for row in values:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def print_row(row: list[str]) -> None:
        print(
            "| "
            + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
            + " |"
        )

    print(border)
    print_row(headers)
    print(border)
    for row in values:
        print_row(row)
    print(border)


def experiment_large_condition_number(
    clean: np.ndarray,
    noisy: np.ndarray,
) -> list[list[object]]:
    """Дослід 1: запускає три методи при lambda=6 та kappa=49."""
    print("\nEXPERIMENT 1 - LARGE CONDITION NUMBER")

    lam = 6.0
    mu = 1.0
    lipschitz = 1.0 + 8.0 * lam
    kappa = lipschitz / mu
    reference = tikhonov_exact(noisy, lam)

    root_l = math.sqrt(lipschitz)
    beta_hb = ((root_l - 1.0) / (root_l + 1.0)) ** 2
    beta_nag = (root_l - 1.0) / (root_l + 1.0)

    methods = [
        (
            "Gradient descent",
            gradient_method(
                noisy,
                lam,
                lipschitz,
                kind="gd",
                max_iter=3000,
                tol=1e-6,
                reference=reference,
            ),
        ),
        (
            "Heavy ball optimal",
            gradient_method(
                noisy,
                lam,
                lipschitz,
                kind="hb",
                beta=beta_hb,
                max_iter=3000,
                tol=1e-6,
                reference=reference,
            ),
        ),
        (
            "Nesterov strongly convex",
            gradient_method(
                noisy,
                lam,
                lipschitz,
                kind="nag",
                beta=beta_nag,
                max_iter=3000,
                tol=1e-6,
                reference=reference,
            ),
        ),
    ]

    rows: list[list[object]] = []
    for name, result in methods:
        rows.append([
            name,
            result["iterations"],
            result["n_f"],
            result["n_grad"],
            f"{result['energy']:.9f}",
            f"{_psnr(result['image'], clean):.6f}",
            "yes" if result["converged"] else "no",
        ])

    headers = ["Method", "Iterations", "Nf", "Ngrad", "Energy", "PSNR", "Converged"]
    print(f"lambda = {lam:.1f}")
    print(f"L = {lipschitz:.1f}, mu = {mu:.1f}, kappa = {kappa:.1f}")
    print(f"beta heavy ball = {beta_hb:.6f}")
    print(f"beta Nesterov = {beta_nag:.6f}")
    print_table(headers, rows)

    write_csv("outputs/lab4_experiment_conditioning.csv", headers, rows)
    return rows


def unstable_gradient_descent(
    noisy: np.ndarray,
    lam: float,
    iterations: int = 50,
) -> tuple[list[list[object]], float, float]:
    """Запускає спуск із нестійким кроком h=2.1/L без ранньої зупинки."""
    lipschitz = 1.0 + 8.0 * lam
    stability_limit = 2.0 / lipschitz
    step = 2.1 / lipschitz
    image = noisy.copy()

    selected = {0, 10, 20, 30, 40, 50}
    rows: list[list[object]] = []

    for iteration in range(iterations + 1):
        current_energy = energy(image, noisy, lam)
        gradient_norm = float(np.linalg.norm(energy_grad(image, noisy, lam)))

        if iteration in selected:
            rows.append([
                iteration,
                f"{current_energy:.9e}",
                f"{gradient_norm:.9e}",
            ])

        if iteration < iterations:
            image = image - step * energy_grad(image, noisy, lam)

    return rows, step, stability_limit


def experiment_unstable_step(
    noisy: np.ndarray,
    optimal_lambda: float,
) -> list[list[object]]:
    """Дослід 2: показує розбіжність при h=2.1/L."""
    print("\nEXPERIMENT 2 - STEP ABOVE THE STABILITY LIMIT")

    rows, step, stability_limit = unstable_gradient_descent(
        noisy,
        optimal_lambda,
        iterations=50,
    )

    headers = ["Iteration", "Energy", "Gradient norm"]
    print(f"lambda* = {optimal_lambda:.9f}")
    print(f"h = 2.1/L = {step:.9f}")
    print(f"stability limit 2/L = {stability_limit:.9f}")
    print_table(headers, rows)
    write_csv("outputs/lab4_experiment_unstable_step.csv", headers, rows)

    iterations = [int(row[0]) for row in rows]
    energies = [float(row[1]) for row in rows]

    plt.figure(figsize=(8, 5))
    plt.semilogy(iterations, energies, "o-", color="crimson")
    plt.xlabel("iteration k")
    plt.ylabel("E(I_k)")
    plt.title("Lab 4 experiment: divergence for h = 2.1/L")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/lab4_experiment_unstable_step.png", dpi=150)
    plt.close()

    return rows


def find_optimal_lambda(
    clean: np.ndarray,
    noisy: np.ndarray,
) -> tuple[float, int]:
    """Повторює пошук оптимального lambda з основної лабораторної."""
    calls = 0

    def objective(lam: float) -> float:
        nonlocal calls
        calls += 1
        return _mse(tikhonov_exact(noisy, lam), clean)

    value = _golden_lambda(objective, start=LAMBDA_START, eps=1e-4)
    return value, calls


def experiment_lambda_influence(
    clean: np.ndarray,
    noisy: np.ndarray,
    optimal_lambda: float,
) -> list[list[object]]:
    """Дослід 3: порівнює PSNR для заданих значень lambda."""
    print("\nEXPERIMENT 3 - INFLUENCE OF LAMBDA")

    values = [0.05, 0.24, optimal_lambda, 1.0, 5.0]
    rows: list[list[object]] = []

    for lam in values:
        restored = tikhonov_exact(noisy, lam)
        rows.append([
            f"{lam:.9f}",
            f"{_mse(restored, clean):.9f}",
            f"{_psnr(restored, clean):.6f}",
            "optimal" if abs(lam - optimal_lambda) < 1e-12 else "test",
        ])

    headers = ["lambda", "MSE", "PSNR", "Type"]
    print_table(headers, rows)
    write_csv("outputs/lab4_experiment_lambda.csv", headers, rows)

    lambda_grid = np.geomspace(0.01, 10.0, 100)
    psnr_grid = [
        _psnr(tikhonov_exact(noisy, lam), clean)
        for lam in lambda_grid
    ]

    plt.figure(figsize=(8, 5))
    plt.semilogx(lambda_grid, psnr_grid, label="PSNR(lambda)")
    plt.axvline(
        optimal_lambda,
        color="red",
        linestyle="--",
        label=f"lambda*={optimal_lambda:.6f}",
    )
    plt.scatter(
        values,
        [
            _psnr(tikhonov_exact(noisy, lam), clean)
            for lam in values
        ],
        color="black",
        zorder=3,
        label="experiment values",
    )
    plt.xlabel("lambda")
    plt.ylabel("PSNR, dB")
    plt.title("Lab 4 experiment: influence of lambda")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/lab4_experiment_lambda.png", dpi=150)
    plt.close()

    return rows


def main() -> None:
    """Запускає всі обов'язкові експерименти лабораторної №4."""
    Path("outputs").mkdir(exist_ok=True)
    project = prepare_project()

    print("\nLAB 4 - REQUIRED EXPERIMENTS")

    optimal_lambda, lambda_calls = find_optimal_lambda(
        project.clean,
        project.noisy,
    )
    print(f"Optimal lambda = {optimal_lambda:.9f}")
    print(f"Lambda search calls = {lambda_calls}")

    experiment_large_condition_number(project.clean, project.noisy)
    experiment_unstable_step(project.noisy, optimal_lambda)
    experiment_lambda_influence(
        project.clean,
        project.noisy,
        optimal_lambda,
    )

    print("\nLAB 4 EXPERIMENTS COMPLETE")
    print("Saved files:")
    print("  outputs/lab4_experiment_conditioning.csv")
    print("  outputs/lab4_experiment_unstable_step.csv")
    print("  outputs/lab4_experiment_unstable_step.png")
    print("  outputs/lab4_experiment_lambda.csv")
    print("  outputs/lab4_experiment_lambda.png")


if __name__ == "__main__":
    main()
