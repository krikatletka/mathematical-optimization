"""
Лабораторна робота №1.

Пошук інтервалу локалізації мінімуму
за допомогою методу Свенна.
"""

import csv
MAX_FILTER_SIGMA = 8.0
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from src.filters import gaussian_blur
from src.metrics import mse
from src.oracle import Oracle
from src.variant import (
    MIN_FILTER_SIGMA,
    SVENN_DELTA0,
    SVENN_X0,
)


@dataclass
class SvennRow:
    call: int
    k: int
    sigma: float
    function_value: float
    delta: float
    direction: int
    action: str


@dataclass
class SvennResult:
    a: float
    b: float
    evaluations: int
    iterations: int
    direction: int
    rows: list[SvennRow]


def create_objective(
    noisy: np.ndarray,
    clean: np.ndarray,
):

    def objective(sigma: float) -> float:
        sigma = float(sigma)

        valid_sigma = float(
            np.clip(
                sigma,
                MIN_FILTER_SIGMA,
                MAX_FILTER_SIGMA,
            )
        )

        restored = gaussian_blur(
            noisy,
            valid_sigma,
        )

        penalty = 1000.0 * (sigma - valid_sigma) ** 2

        return mse(restored, clean) + penalty

    return objective


def _evaluate(
    oracle: Oracle,
    rows: list[SvennRow],
    k: int,
    sigma: float,
    delta: float,
    direction: int,
    action: str,
) -> tuple[float, float]:
    sigma = float( np.clip( sigma, MIN_FILTER_SIGMA, MAX_FILTER_SIGMA,))
    value = oracle.value(sigma)

    rows.append(
        SvennRow(
            call=oracle.calls,
            k=k,
            sigma=sigma,
            function_value=value,
            delta=delta,
            direction=direction,
            action=action,
        )
    )

    return sigma, value


def svenn(
    oracle: Oracle,
    x0: float,
    delta0: float,
) -> SvennResult:
    """Знаходить інтервал, який містить мінімум."""

    if delta0 <= 0.0:
        raise ValueError("Delta0 must be positive.")

    rows: list[SvennRow] = []

    x0, f0 = _evaluate(
        oracle,
        rows,
        0,
        x0,
        delta0,
        0,
        "initial point",
    )

    right, f_right = _evaluate(
        oracle,
        rows,
        0,
        x0 + delta0,
        delta0,
        0,
        "probe right",
    )

    left, f_left = _evaluate(
        oracle,
        rows,
        0,
        x0 - delta0,
        -delta0,
        1,
        "probe left",
    )

    if f_left >= f0 <= f_right:
        return SvennResult(
            a=left,
            b=right,
            evaluations=oracle.calls,
            iterations=0,
            direction=0,
            rows=rows,
        )

    if f_right < f_left:
        direction = 0
        delta = delta0
        previous = x0
        current = right
        current_value = f_right
    else:
        direction = 1
        delta = -delta0
        previous = x0
        current = left
        current_value = f_left

    iterations = 0

    while iterations < 100:
        next_delta = 2.0 * delta

        next_point = float(
            np.clip(
                current + next_delta,
                MIN_FILTER_SIGMA,
                MAX_FILTER_SIGMA,
            )
        )

        if next_point == current:
            rows[-1].action += "; boundary reached: STOP"

            return SvennResult(
                a=min(previous, current),
                b=max(previous, current),
                evaluations=oracle.calls,
                iterations=iterations,
                direction=direction,
                rows=rows,
            )

        iterations += 1
        delta = next_delta

        next_value = oracle.value(next_point)

        if next_value >= current_value:
            rows.append(
                SvennRow(
                    call=oracle.calls,
                    k=iterations,
                    sigma=next_point,
                    function_value=next_value,
                    delta=delta,
                    direction=direction,
                    action="f did not decrease: STOP",
                )
            )

            return SvennResult(
                a=min(previous, next_point),
                b=max(previous, next_point),
                evaluations=oracle.calls,
                iterations=iterations,
                direction=direction,
                rows=rows,
            )

        rows.append(
            SvennRow(
                call=oracle.calls,
                k=iterations,
                sigma=next_point,
                function_value=next_value,
                delta=delta,
                direction=direction,
                action="expand",
            )
        )

        previous = current
        current = next_point
        current_value = next_value

    raise RuntimeError(
        "Svenn method exceeded iteration limit."
    )


def print_svenn_table(result: SvennResult) -> None:
    """Виводить таблицю методу Свенна."""

    print("\nLAB 1 - SVENN")
    print(
        "+------+-----+------------+---------------+"
        "------------+---+--------------------------+"
    )
    print(
        "| Call | k   | sigma      | f(sigma)      |"
        " Delta      | p | Action                   |"
    )
    print(
        "+------+-----+------------+---------------+"
        "------------+---+--------------------------+"
    )

    for row in result.rows:
        print(
            f"| {row.call:<4} "
            f"| {row.k:<3} "
            f"| {row.sigma:<10.6f} "
            f"| {row.function_value:<13.9f} "
            f"| {row.delta:<10.6f} "
            f"| {row.direction:<1} "
            f"| {row.action:<24} |"
        )

    print(
        "+------+-----+------------+---------------+"
        "------------+---+--------------------------+"
    )

    print(
        f"Interval = [{result.a:.6f}; {result.b:.6f}]"
    )
    print(f"Length = {result.b - result.a:.6f}")
    print(
        f"Expansion iterations = {result.iterations}"
    )
    print(f"Direction p = {result.direction}")
    print(f"Nf = {result.evaluations}")
    print("Status = bracket found")


def save_lab1_results(
    result: SvennResult,
    oracle: Oracle,
) -> None:
    """Зберігає таблицю та графік Lab 1."""

    with open(
        "outputs/lab1_svenn.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file, delimiter=";")

        writer.writerow(
            [
                "call",
                "k",
                "sigma",
                "f_sigma",
                "delta",
                "direction",
                "action",
            ]
        )

        for row in result.rows:
            writer.writerow(
                [
                    row.call,
                    row.k,
                    row.sigma,
                    row.function_value,
                    row.delta,
                    row.direction,
                    row.action,
                ]
            )

    sigmas = np.linspace(
        MIN_FILTER_SIGMA,
        MAX_FILTER_SIGMA,
        200,
    )

    values = [
        oracle.peek(float(sigma))
        for sigma in sigmas
    ]

    plt.figure(figsize=(9, 5))

    plt.plot(
        sigmas,
        values,
        color="royalblue",
        label="MSE(sigma)",
    )

    tested_sigma = [
        row.sigma
        for row in result.rows
    ]

    tested_values = [
        row.function_value
        for row in result.rows
    ]

    plt.scatter(
        tested_sigma,
        tested_values,
        color="crimson",
        zorder=3,
        label="Svenn points",
    )

    for row in result.rows:
        plt.annotate(
            str(row.call),
            (row.sigma, row.function_value),
            xytext=(4, 5),
            textcoords="offset points",
        )

    plt.axvspan(
        result.a,
        result.b,
        color="green",
        alpha=0.12,
        label="Final interval",
    )

    plt.xlabel("sigma")
    plt.ylabel("MSE")
    plt.title("Lab 1 — Svenn Method")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "outputs/lab1_function.png",
        dpi=160,
    )

    plt.close()


def run_lab1(
    clean: np.ndarray,
    noisy: np.ndarray,
) -> SvennResult:
    """Запускає першу лабораторну роботу."""

    oracle = Oracle(
        create_objective(noisy, clean)
    )

    result = svenn(
        oracle,
        SVENN_X0,
        SVENN_DELTA0,
    )

    print_svenn_table(result)
    save_lab1_results(result, oracle)

    print("Saved table = outputs/lab1_svenn.csv")
    print("Saved plot = outputs/lab1_function.png")
    print("\nLAB 1 COMPLETE")

    return result