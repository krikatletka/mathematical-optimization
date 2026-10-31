"""
Лабораторна робота №2.

Уточнення оптимального sigma методами
дихотомії та золотого перерізу.
"""

import csv
import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from src.filters import gaussian_blur
from src.image_io import save_image
from src.lab1 import create_objective
from src.metrics import mse, psnr
from src.oracle import Oracle
from src.variant import EPSILON


TAU = (math.sqrt(5.0) - 1.0) / 2.0


@dataclass
class SearchRow:
    k: int
    a: float
    b: float
    x1: float
    x2: float
    f1: float
    f2: float
    length: float
    decision: str


@dataclass
class SearchResult:
    x: float
    fx: float
    a: float
    b: float
    iterations: int
    evaluations: int
    rows: list[SearchRow]


def dichotomy(
    oracle: Oracle,
    a: float,
    b: float,
    epsilon: float,
) -> SearchResult:
    """Метод дихотомії."""

    delta = epsilon / 3.0
    calls_before = oracle.calls
    rows: list[SearchRow] = []
    iteration = 0

    while b - a > epsilon:
        old_a = a
        old_b = b

        middle = (a + b) / 2.0

        x1 = middle - delta / 2.0
        x2 = middle + delta / 2.0

        f1 = oracle.value(x1)
        f2 = oracle.value(x2)

        if f1 <= f2:
            b = x2
            decision = "f1 <= f2: b = x2"
        else:
            a = x1
            decision = "f1 > f2: a = x1"

        iteration += 1

        rows.append(
            SearchRow(
                k=iteration,
                a=old_a,
                b=old_b,
                x1=x1,
                x2=x2,
                f1=f1,
                f2=f2,
                length=old_b - old_a,
                decision=decision,
            )
        )

    x_star = (a + b) / 2.0
    fx = oracle.value(x_star)

    return SearchResult(
        x=x_star,
        fx=fx,
        a=a,
        b=b,
        iterations=iteration,
        evaluations=oracle.calls - calls_before,
        rows=rows,
    )


def golden_section(
    oracle: Oracle,
    a: float,
    b: float,
    epsilon: float,
) -> SearchResult:
    """Метод золотого перерізу."""

    calls_before = oracle.calls
    rows: list[SearchRow] = []

    x1 = a + (1.0 - TAU) * (b - a)
    x2 = a + TAU * (b - a)

    f1 = oracle.value(x1)
    f2 = oracle.value(x2)

    iteration = 0

    while b - a > epsilon:
        old_a = a
        old_b = b
        old_x1 = x1
        old_x2 = x2
        old_f1 = f1
        old_f2 = f2

        if f1 <= f2:
            b = x2

            x2 = x1
            f2 = f1

            x1 = (
                a
                + (1.0 - TAU)
                * (b - a)
            )

            f1 = oracle.value(x1)

            decision = (
                "f1 <= f2: b = x2, inherit x1"
            )
        else:
            a = x1

            x1 = x2
            f1 = f2

            x2 = (
                a
                + TAU
                * (b - a)
            )

            f2 = oracle.value(x2)

            decision = (
                "f1 > f2: a = x1, inherit x2"
            )

        iteration += 1

        rows.append(
            SearchRow(
                k=iteration,
                a=old_a,
                b=old_b,
                x1=old_x1,
                x2=old_x2,
                f1=old_f1,
                f2=old_f2,
                length=old_b - old_a,
                decision=decision,
            )
        )

    x_star = (a + b) / 2.0
    fx = oracle.value(x_star)

    return SearchResult(
        x=x_star,
        fx=fx,
        a=a,
        b=b,
        iterations=iteration,
        evaluations=oracle.calls - calls_before,
        rows=rows,
    )


def print_search_table(
    title: str,
    result: SearchResult,
) -> None:
    """Виводить таблицю ітерацій."""

    print(f"\n{title}")

    print(
        "+----+----------+----------+----------+----------+"
        "-------------+-------------+----------+"
        "----------------------------------------+"
    )

    print(
        "| k  | a_k      | b_k      | x1       | x2       |"
        " f(x1)       | f(x2)       | L_k      |"
        " Decision                               |"
    )

    print(
        "+----+----------+----------+----------+----------+"
        "-------------+-------------+----------+"
        "----------------------------------------+"
    )

    for row in result.rows:
        print(
            f"| {row.k:<2} "
            f"| {row.a:<8.5f} "
            f"| {row.b:<8.5f} "
            f"| {row.x1:<8.5f} "
            f"| {row.x2:<8.5f} "
            f"| {row.f1:<11.8f} "
            f"| {row.f2:<11.8f} "
            f"| {row.length:<8.5f} "
            f"| {row.decision:<38} |"
        )

    print(
        "+----+----------+----------+----------+----------+"
        "-------------+-------------+----------+"
        "----------------------------------------+"
    )

    print(f"sigma* = {result.x:.9f}")
    print(f"f(sigma*) = {result.fx:.9f}")
    print(
        f"Final interval = [{result.a:.9f}; "
        f"{result.b:.9f}]"
    )
    print(
        f"Final length = {result.b - result.a:.9f}"
    )
    print(f"Iterations = {result.iterations}")
    print(f"Nf = {result.evaluations}")


def save_search_csv(
    path: str,
    result: SearchResult,
) -> None:
    """Зберігає таблицю методу в CSV."""

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file, delimiter=";")

        writer.writerow(
            [
                "k",
                "a",
                "b",
                "x1",
                "x2",
                "f1",
                "f2",
                "length",
                "decision",
            ]
        )

        for row in result.rows:
            writer.writerow(
                [
                    row.k,
                    row.a,
                    row.b,
                    row.x1,
                    row.x2,
                    row.f1,
                    row.f2,
                    row.length,
                    row.decision,
                ]
            )


def run_lab2(
    clean: np.ndarray,
    noisy: np.ndarray,
    initial_a: float,
    initial_b: float,
) -> SearchResult:
    """Запускає другу лабораторну роботу."""

    print("\nLAB 2 - DICHOTOMY AND GOLDEN SECTION")
    print(
        f"Initial interval = "
        f"[{initial_a:.6f}; {initial_b:.6f}]"
    )
    print(f"epsilon = {EPSILON:.6f}")
    print(f"delta = {EPSILON / 3.0:.9f}")
    print(f"tau = {TAU:.10f}")

    objective = create_objective(
        noisy,
        clean,
    )

    dichotomy_oracle = Oracle(objective)

    dichotomy_result = dichotomy(
        dichotomy_oracle,
        initial_a,
        initial_b,
        EPSILON,
    )

    print_search_table(
        "DICHOTOMY",
        dichotomy_result,
    )

    golden_oracle = Oracle(objective)

    golden_result = golden_section(
        golden_oracle,
        initial_a,
        initial_b,
        EPSILON,
    )

    print_search_table(
        "GOLDEN SECTION",
        golden_result,
    )

    save_search_csv(
        "outputs/lab2_dichotomy.csv",
        dichotomy_result,
    )

    save_search_csv(
        "outputs/lab2_golden.csv",
        golden_result,
    )

    print("\nLAB 2 - METHOD COMPARISON")

    print(
        "+----------------+-------------+-------------+"
        "------------+------+"
    )
    print(
        "| Method         | sigma*      | f(sigma*)   |"
        " Iterations | Nf   |"
    )
    print(
        "+----------------+-------------+-------------+"
        "------------+------+"
    )

    print(
        f"| {'Dichotomy':<14} "
        f"| {dichotomy_result.x:<11.8f} "
        f"| {dichotomy_result.fx:<11.8f} "
        f"| {dichotomy_result.iterations:<10} "
        f"| {dichotomy_result.evaluations:<4} |"
    )

    print(
        f"| {'Golden section':<14} "
        f"| {golden_result.x:<11.8f} "
        f"| {golden_result.fx:<11.8f} "
        f"| {golden_result.iterations:<10} "
        f"| {golden_result.evaluations:<4} |"
    )

    print(
        "+----------------+-------------+-------------+"
        "------------+------+"
    )

    restored = gaussian_blur(
        noisy,
        golden_result.x,
    )

    save_image(
        restored,
        "outputs/02_gaussian.png",
    )

    before_mse = mse(noisy, clean)
    after_mse = mse(restored, clean)

    before_psnr = psnr(noisy, clean)
    after_psnr = psnr(restored, clean)

    print("\nLAB 2 - IMAGE RESTORATION")
    print(f"Before MSE = {before_mse:.9f}")
    print(f"Before PSNR = {before_psnr:.6f} dB")
    print(f"Optimal sigma = {golden_result.x:.9f}")
    print(f"After MSE = {after_mse:.9f}")
    print(f"After PSNR = {after_psnr:.6f} dB")
    print("Saved image = outputs/02_gaussian.png")

    sigmas = np.linspace(
        initial_a,
        initial_b,
        250,
    )

    values = [
        objective(float(sigma))
        for sigma in sigmas
    ]

    plt.figure(figsize=(9, 5))

    plt.plot(
        sigmas,
        values,
        color="royalblue",
        label="MSE(sigma)",
    )

    plt.scatter(
        [dichotomy_result.x],
        [dichotomy_result.fx],
        color="crimson",
        s=60,
        label="Dichotomy",
        zorder=3,
    )

    plt.scatter(
        [golden_result.x],
        [golden_result.fx],
        color="green",
        s=60,
        label="Golden section",
        zorder=3,
    )

    plt.xlabel("sigma")
    plt.ylabel("MSE")
    plt.title("Lab 2 — One-Dimensional Search")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "outputs/lab2_search.png",
        dpi=160,
    )

    plt.close()

    print("Saved plot = outputs/lab2_search.png")
    print("\nLAB 2 COMPLETE")

    return golden_result