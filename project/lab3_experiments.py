"""Обов'язкові експерименти до лабораторної роботи №3.

Файл не змінює основний алгоритм лабораторної. Він повторно використовує
реалізації білатерального фільтра, Нелдера—Міда та Пауелла з src/lab3.py.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from src.lab3 import (
    HI,
    LO,
    Objective2,
    bilateral_filter,
    nelder_mead,
    powell,
)
from src.preparation import prepare_project


GAUSSIAN_SIGMA = 0.668953998
SIMPLEX_EDGE = 0.6976
NOISE_SIGMA = 0.1449


def image_psnr(image: np.ndarray, reference: np.ndarray) -> float:
    """Обчислює PSNR для зображень із яскравостями у діапазоні [0, 1]."""
    error = float(np.mean((image - reference) ** 2))
    return float("inf") if error == 0.0 else 10.0 * math.log10(1.0 / error)


def initial_point() -> np.ndarray:
    """Повертає ту саму початкову точку, що використовується в run_lab3()."""
    sigma_s = float(np.clip(2.0 * GAUSSIAN_SIGMA, 0.5, 4.0))
    sigma_r = float(np.clip(0.15 + 0.1 * NOISE_SIGMA, 0.02, 0.9))
    return np.array([sigma_s, sigma_r], dtype=float)


def write_csv(path: str, header: list[str], rows: list[list[object]]) -> None:
    """Зберігає результати одного досліду у CSV."""
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def print_table(headers: list[str], rows: list[list[object]]) -> None:
    """Друкує невелику текстову таблицю з рамкою."""
    text_rows = [[str(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]

    for row in text_rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def print_row(values: list[str]) -> None:
        print(
            "| "
            + " | ".join(value.ljust(widths[i]) for i, value in enumerate(values))
            + " |"
        )

    print(border)
    print_row(headers)
    print(border)
    for row in text_rows:
        print_row(row)
    print(border)


def run_base_methods(clean: np.ndarray, noisy: np.ndarray) -> tuple[dict, dict]:
    """Запускає базові варіанти обох методів для подальших порівнянь."""
    x0 = initial_point()

    objective_nm = Objective2(noisy, clean)
    nm = nelder_mead(objective_nm, x0, SIMPLEX_EDGE)
    nm["nf"] = objective_nm.calls

    objective_pw = Objective2(noisy, clean)
    pw = powell(objective_pw, x0, tol=1e-4)
    pw["nf"] = objective_pw.calls

    return nm, pw


def experiment_simplex_size(clean: np.ndarray, noisy: np.ndarray) -> list[list[object]]:
    """Дослід 1: порівнює три розміри початкового трикутника."""
    print("\nEXPERIMENT 1 - INITIAL SIMPLEX SIZE")

    rows: list[list[object]] = []
    cases = [
        ("s / 3", SIMPLEX_EDGE / 3.0),
        ("s", SIMPLEX_EDGE),
        ("3s", SIMPLEX_EDGE * 3.0),
    ]

    for label, edge in cases:
        objective = Objective2(noisy, clean)
        result = nelder_mead(objective, initial_point(), edge)
        theta = np.clip(result["x"], LO, HI)
        restored = bilateral_filter(noisy, theta[0], theta[1])

        rows.append([
            label,
            f"{edge:.6f}",
            f"{theta[0]:.6f}",
            f"{theta[1]:.6f}",
            result["iterations"],
            objective.calls,
            f"{image_psnr(restored, clean):.6f}",
        ])

    headers = ["Case", "Edge", "sigma_s", "sigma_r", "Iterations", "Nf", "PSNR"]
    print_table(headers, rows)
    write_csv("outputs/lab3_experiment_simplex.csv", headers, rows)
    return rows


def golden_line_with_counts(
    function: Objective2,
    x: np.ndarray,
    direction: np.ndarray,
    eps: float = 1e-4,
) -> tuple[float, int, int]:
    """Лінійний пошук із окремим підрахунком звернень Свенна і золотого перерізу."""
    svenn_calls = 0
    golden_calls = 0

    def phi_svenn(alpha: float) -> float:
        nonlocal svenn_calls
        svenn_calls += 1
        return function(x + alpha * direction)

    def phi_golden(alpha: float) -> float:
        nonlocal golden_calls
        golden_calls += 1
        return function(x + alpha * direction)

    alpha0, step = 0.0, 0.1
    f0 = phi_svenn(alpha0)
    fp = phi_svenn(step)
    fm = phi_svenn(-step)

    if fp >= f0 and fm >= f0:
        left, right = -step, step
    else:
        if fm < fp:
            step = -step
            previous, current = 0.0, step
            f_current = fm
        else:
            previous, current = 0.0, step
            f_current = fp

        for power in range(1, 30):
            candidate = current + step * (2**power)
            f_candidate = phi_svenn(candidate)
            if f_candidate >= f_current:
                left, right = sorted((previous, candidate))
                break
            previous, current, f_current = current, candidate, f_candidate
        else:
            left, right = sorted((previous, current))

    tau = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = left, right
    x1 = a + (1.0 - tau) * (b - a)
    x2 = a + tau * (b - a)
    f1 = phi_golden(x1)
    f2 = phi_golden(x2)

    while b - a > eps:
        if f1 <= f2:
            b, x2, f2 = x2, x1, f1
            x1 = a + (1.0 - tau) * (b - a)
            f1 = phi_golden(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + tau * (b - a)
            f2 = phi_golden(x2)

    return (a + b) / 2.0, svenn_calls, golden_calls


def experiment_powell_cycle(clean: np.ndarray, noisy: np.ndarray) -> list[list[object]]:
    """Дослід 2: розкладає ціну першого циклу Пауелла на складові."""
    print("\nEXPERIMENT 2 - COST OF THE FIRST POWELL CYCLE")

    objective = Objective2(noisy, clean)
    x = initial_point()
    directions = np.eye(2)
    rows: list[list[object]] = []

    objective(x)  # перевірка значення на початку циклу
    check_calls = 1
    total_svenn = 0
    total_golden = 0

    for number, direction in enumerate(directions, start=1):
        objective(x)
        check_calls += 1

        alpha, svenn_calls, golden_calls = golden_line_with_counts(
            objective, x, direction
        )
        x = x + alpha * direction

        objective(x)
        check_calls += 1
        total_svenn += svenn_calls
        total_golden += golden_calls

        rows.append([
            f"Direction {number}",
            svenn_calls,
            golden_calls,
            2,
            svenn_calls + golden_calls + 2,
        ])

    objective(x)  # перевірка значення наприкінці циклу
    check_calls += 1

    rows.append(["Cycle start/end", 0, 0, 2, 2])
    rows.append([
        "TOTAL",
        total_svenn,
        total_golden,
        check_calls,
        objective.calls,
    ])

    headers = ["Part", "Svenn", "Golden", "Checks", "Total"]
    print_table(headers, rows)
    print(f"Point after first cycle = ({x[0]:.6f}; {x[1]:.6f})")
    write_csv("outputs/lab3_experiment_powell_cycle.csv", headers, rows)
    return rows


def experiment_powell_tolerance(
    clean: np.ndarray,
    noisy: np.ndarray,
    base_powell: dict,
) -> list[list[object]]:
    """Дослід 3: порівнює критерії зупинки Пауелла 1e-4 та 1e-7."""
    print("\nEXPERIMENT 3 - POWELL STOPPING TOLERANCE")

    objective = Objective2(noisy, clean)
    strict = powell(objective, initial_point(), tol=1e-7)
    strict["nf"] = objective.calls

    rows = [
        [
            "1e-4",
            f"{base_powell['x'][0]:.6f}",
            f"{base_powell['x'][1]:.6f}",
            base_powell["cycles"],
            base_powell["nf"],
            f"{base_powell['f']:.9f}",
        ],
        [
            "1e-7",
            f"{strict['x'][0]:.6f}",
            f"{strict['x'][1]:.6f}",
            strict["cycles"],
            strict["nf"],
            f"{strict['f']:.9f}",
        ],
    ]

    headers = ["Tolerance", "sigma_s", "sigma_r", "Cycles", "Nf", "MSE"]
    print_table(headers, rows)
    write_csv("outputs/lab3_experiment_tolerance.csv", headers, rows)
    return rows


def optimize_sigma_s(
    function: Objective2,
    sigma_r: float,
    eps: float = 1e-4,
) -> tuple[np.ndarray, float]:
    """Мінімізує функцію лише за sigma_s при зафіксованому sigma_r."""
    x = np.array([initial_point()[0], sigma_r], dtype=float)
    direction = np.array([1.0, 0.0], dtype=float)
    alpha, _, _ = golden_line_with_counts(function, x, direction, eps)
    point = np.clip(x + alpha * direction, LO, HI)
    return point, function(point)


def experiment_one_parameter(
    clean: np.ndarray,
    noisy: np.ndarray,
    best_2d: dict,
) -> list[list[object]]:
    """Дослід 4: фіксує найкраще sigma_r та оптимізує тільки sigma_s."""
    print("\nEXPERIMENT 4 - ONE PARAMETER AGAINST TWO")

    best_point = np.clip(best_2d["x"], LO, HI)
    fixed_sigma_r = float(best_point[1])

    objective = Objective2(noisy, clean)
    point_1d, value_1d = optimize_sigma_s(objective, fixed_sigma_r)

    restored_1d = bilateral_filter(noisy, point_1d[0], point_1d[1])
    restored_2d = bilateral_filter(noisy, best_point[0], best_point[1])

    rows = [
        [
            "One parameter",
            f"{point_1d[0]:.6f}",
            f"{point_1d[1]:.6f}",
            f"{value_1d:.9f}",
            objective.calls,
            f"{image_psnr(restored_1d, clean):.6f}",
        ],
        [
            "Two parameters",
            f"{best_point[0]:.6f}",
            f"{best_point[1]:.6f}",
            f"{best_2d['f']:.9f}",
            best_2d["nf"],
            f"{image_psnr(restored_2d, clean):.6f}",
        ],
    ]

    headers = ["Problem", "sigma_s", "sigma_r", "MSE", "Nf", "PSNR"]
    print_table(headers, rows)
    write_csv("outputs/lab3_experiment_one_parameter.csv", headers, rows)
    return rows


def main() -> None:
    """Запускає всі чотири обов'язкові досліди лабораторної №3."""
    Path("outputs").mkdir(exist_ok=True)
    project = prepare_project()

    print("\nLAB 3 - REQUIRED EXPERIMENTS")
    print(f"Initial point = ({initial_point()[0]:.6f}; {initial_point()[1]:.6f})")
    print(f"Base simplex edge = {SIMPLEX_EDGE:.6f}")

    base_nm, base_powell = run_base_methods(project.clean, project.noisy)
    base_nm["nf"] = base_nm.get("nf", 0)
    base_powell["nf"] = base_powell.get("nf", 0)

    best_2d = base_nm if base_nm["f"] <= base_powell["f"] else base_powell

    experiment_simplex_size(project.clean, project.noisy)
    experiment_powell_cycle(project.clean, project.noisy)
    experiment_powell_tolerance(project.clean, project.noisy, base_powell)
    experiment_one_parameter(project.clean, project.noisy, best_2d)

    print("\nLAB 3 EXPERIMENTS COMPLETE")
    print("Saved files:")
    print("  outputs/lab3_experiment_simplex.csv")
    print("  outputs/lab3_experiment_powell_cycle.csv")
    print("  outputs/lab3_experiment_tolerance.csv")
    print("  outputs/lab3_experiment_one_parameter.csv")


if __name__ == "__main__":
    main()
