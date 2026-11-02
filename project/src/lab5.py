"""
Лабораторна робота №5.

Умовна оптимізація зображення:
проєкція градієнта, FISTA та метод зовнішніх штрафів.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.image_io import save_image
from src.lab4 import energy, energy_grad, tikhonov_exact
from src.metrics import psnr


def project(image, lower=0.0, upper=1.0):
    """Проєкція зображення на допустимий інтервал [0, 1]."""

    return np.clip(image, lower, upper)


def violation(image, lower=0.0, upper=1.0):
    """Повертає найбільше порушення та кількість недопустимих пікселів."""

    below = np.maximum(0.0, lower - image)
    above = np.maximum(0.0, image - upper)

    return {
        "max": float(max(below.max(), above.max())),
        "count": int(
            np.count_nonzero(
                (image < lower) | (image > upper)
            )
        ),
    }


def projected_gradient(
    raw,
    lam,
    lipschitz,
    accelerated=False,
    max_iterations=200,
    tolerance=1e-6,
):
    """Метод проєкції градієнта або його прискорений варіант FISTA."""

    step = 1.0 / lipschitz

    image = project(raw.copy())
    previous = image.copy()
    extrapolated = image.copy()

    function_calls = 0
    gradient_calls = 0
    rows = []
    converged = False

    for iteration in range(1, max_iterations + 2):
        point = extrapolated if accelerated else image

        gradient = energy_grad(
            point,
            raw,
            lam,
        )

        gradient_calls += 1

        new_image = project(
            point - step * gradient
        )

        gradient_mapping = float(
            np.linalg.norm(new_image - point)
            / step
        )

        active = int(
            np.count_nonzero(
                (image <= 1e-12)
                | (image >= 1.0 - 1e-12)
            )
        )

        current_violation = violation(image)

        current_energy = energy(
            image,
            raw,
            lam,
        )

        function_calls += 1

        rows.append(
            {
                "k": iteration - 1,
                "E": current_energy,
                "G": gradient_mapping,
                "active": active,
                "violation": current_violation["max"],
                "n_f": function_calls,
                "n_grad": gradient_calls,
            }
        )

        if gradient_mapping <= tolerance:
            converged = True
            break

        if iteration - 1 == max_iterations:
            break

        if accelerated:
            beta = (iteration - 1) / (iteration + 2)

            extrapolated = (
                new_image
                + beta * (new_image - previous)
            )

            previous = new_image

        image = new_image

    return {
        "image": image,
        "iterations": len(rows) - 1,
        "n_f": function_calls,
        "n_grad": gradient_calls,
        "energy": rows[-1]["E"],
        "gradient_mapping": rows[-1]["G"],
        "converged": converged,
        "rows": rows,
    }


def penalty_value(image, lower=0.0, upper=1.0):
    """Сума квадратів усіх порушень обмежень."""

    below = np.maximum(0.0, lower - image)
    above = np.maximum(0.0, image - upper)

    return float(
        np.sum(below**2 + above**2)
    )


def penalty_gradient(image, lower=0.0, upper=1.0):
    """Градієнт квадратичного штрафу."""

    return (
        -2.0 * np.maximum(0.0, lower - image)
        + 2.0 * np.maximum(0.0, image - upper)
    )


def penalty_method(
    raw,
    lam,
    lipschitz,
    initial_penalty=1.0,
    growth=10.0,
    outer_iterations=7,
    inner_iterations=100,
    inner_tolerance=1e-6,
    tolerance=1e-8,
):
    """Метод зовнішніх квадратичних штрафів."""

    image = raw.copy()
    penalty = initial_penalty

    function_calls = 0
    gradient_calls = 0
    rows = []

    for outer in range(outer_iterations):
        penalized_lipschitz = (
            lipschitz + 2.0 * penalty
        )

        step = 1.0 / penalized_lipschitz
        previous = image.copy()

        completed_inner = 0

        for inner in range(1, inner_iterations + 1):
            beta = (inner - 1) / (inner + 2)

            extrapolated = (
                image
                + beta * (image - previous)
            )

            gradient = (
                energy_grad(
                    extrapolated,
                    raw,
                    lam,
                )
                + penalty
                * penalty_gradient(extrapolated)
            )

            gradient_calls += 1

            new_image = (
                extrapolated - step * gradient
            )

            completed_inner = inner

            if float(np.linalg.norm(gradient)) <= inner_tolerance:
                previous = image
                image = new_image
                break

            previous = image
            image = new_image

        value = penalty_value(image)
        function_calls += 1

        current_violation = violation(image)

        rows.append(
            {
                "s": outer,
                "r": penalty,
                "L_r": penalized_lipschitz,
                "step": step,
                "inner": completed_inner,
                "E": energy(image, raw, lam),
                "P": value,
                "rP": penalty * value,
                "violation": current_violation["max"],
                "outside": current_violation["count"],
                "n_f": function_calls,
                "n_grad": gradient_calls,
            }
        )

        if penalty * value <= tolerance:
            break

        penalty *= growth

    return {
        "image": image,
        "outer_iterations": len(rows),
        "n_f": function_calls,
        "n_grad": gradient_calls,
        "energy": rows[-1]["E"],
        "rows": rows,
    }


def gradient_norm_experiment(
    raw,
    lam,
    lipschitz,
    iterations=200,
):
    """Дослід із неправильним критерієм зупинки за нормою градієнта."""

    step = 1.0 / lipschitz
    image = project(raw.copy())

    for _ in range(iterations):
        gradient = energy_grad(
            image,
            raw,
            lam,
        )

        image = project(
            image - step * gradient
        )

    gradient = energy_grad(
        image,
        raw,
        lam,
    )

    new_image = project(
        image - step * gradient
    )

    return {
        "gradient_norm": float(
            np.linalg.norm(gradient)
        ),
        "gradient_mapping": float(
            np.linalg.norm(new_image - image)
            / step
        ),
    }


def save_projected_table(path, result):
    """Зберігає таблицю проєкційного методу."""

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(
            file,
            delimiter=";",
        )

        writer.writerow(
            [
                "k",
                "E",
                "G",
                "active",
                "violation",
                "Nf",
                "Ngrad",
            ]
        )

        for row in result["rows"]:
            writer.writerow(
                [
                    row["k"],
                    row["E"],
                    row["G"],
                    row["active"],
                    row["violation"],
                    row["n_f"],
                    row["n_grad"],
                ]
            )


def save_penalty_table(path, result):
    """Зберігає таблицю зовнішніх ітерацій штрафного методу."""

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(
            file,
            delimiter=";",
        )

        writer.writerow(
            [
                "s",
                "r",
                "L_r",
                "step",
                "inner",
                "E",
                "P",
                "rP",
                "violation",
                "outside",
                "Nf",
                "Ngrad",
            ]
        )

        for row in result["rows"]:
            writer.writerow(
                [
                    row["s"],
                    row["r"],
                    row["L_r"],
                    row["step"],
                    row["inner"],
                    row["E"],
                    row["P"],
                    row["rP"],
                    row["violation"],
                    row["outside"],
                    row["n_f"],
                    row["n_grad"],
                ]
            )


def save_histogram(raw, unconstrained):
    """Будує гістограму яскравостей."""

    plt.figure(figsize=(9, 4))

    plt.hist(
        raw.ravel(),
        bins=120,
        alpha=0.55,
        label="raw observation",
    )

    plt.hist(
        unconstrained.ravel(),
        bins=120,
        alpha=0.55,
        label="unconstrained solution",
    )

    plt.axvline(
        0.0,
        color="red",
        linestyle="--",
        label="feasible interval",
    )

    plt.axvline(
        1.0,
        color="red",
        linestyle="--",
    )

    plt.xlabel("pixel intensity")
    plt.ylabel("number of pixels")
    plt.title("Lab 5: intensity histogram")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "outputs/lab5_histogram.png",
        dpi=150,
    )

    plt.close()


def save_convergence_plot(
    projected_result,
    fista_result,
    exact_energy,
):
    """Будує графік збіжності проєкційних методів."""

    plt.figure(figsize=(9, 5))

    for name, result in (
        ("Projected gradient", projected_result),
        ("FISTA", fista_result),
    ):
        values = [
            max(row["E"] - exact_energy, 1e-16)
            for row in result["rows"]
        ]

        plt.semilogy(
            range(len(values)),
            values,
            label=name,
        )

    plt.xlabel("iteration k")
    plt.ylabel("E(I_k) - E*")
    plt.title("Lab 5: projected methods convergence")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "outputs/lab5_convergence.png",
        dpi=150,
    )

    plt.close()


def save_active_map(image):
    """Зберігає карту активних обмежень."""

    active_map = np.full_like(
        image,
        0.5,
        dtype=float,
    )

    active_map[image <= 1e-12] = 0.0
    active_map[image >= 1.0 - 1e-12] = 1.0

    save_image(
        active_map,
        "outputs/lab5_active_constraints.png",
    )


def print_comparison(
    projected_result,
    fista_result,
    penalty_result,
):
    """Виводить порівняльну таблицю методів."""

    projected_violation = violation(
        projected_result["image"]
    )

    fista_violation = violation(
        fista_result["image"]
    )

    penalty_violation = violation(
        penalty_result["image"]
    )

    print("\nLAB 5 - METHOD COMPARISON")

    print(
        "+----------------------+------------+"
        "-------------+---------+---------+-------+"
    )

    print(
        "| Method               | Iterations |"
        " Violation   | Outside | Calls   | Valid |"
    )

    print(
        "+----------------------+------------+"
        "-------------+---------+---------+-------+"
    )

    print(
        f"| {'Projected gradient':<20} "
        f"| {projected_result['iterations']:<10} "
        f"| {projected_violation['max']:<11.3e} "
        f"| {projected_violation['count']:<7} "
        f"| {projected_result['n_f'] + projected_result['n_grad']:<7} "
        f"| {'yes':<5} |"
    )

    print(
        f"| {'FISTA':<20} "
        f"| {fista_result['iterations']:<10} "
        f"| {fista_violation['max']:<11.3e} "
        f"| {fista_violation['count']:<7} "
        f"| {fista_result['n_f'] + fista_result['n_grad']:<7} "
        f"| {'yes':<5} |"
    )

    print(
        f"| {'Quadratic penalty':<20} "
        f"| {penalty_result['outer_iterations']:<10} "
        f"| {penalty_violation['max']:<11.3e} "
        f"| {penalty_violation['count']:<7} "
        f"| {penalty_result['n_f'] + penalty_result['n_grad']:<7} "
        f"| {'no':<5} |"
    )

    print(
        "+----------------------+------------+"
        "-------------+---------+---------+-------+"
    )


def run_lab5(clean, raw_noisy, lambda_value):
    """Запускає п'яту лабораторну роботу."""

    Path("outputs").mkdir(
        exist_ok=True
    )

    lipschitz = 1.0 + 8.0 * lambda_value

    raw_violation = violation(raw_noisy)

    unconstrained = tikhonov_exact(
        raw_noisy,
        lambda_value,
    )

    unconstrained_violation = violation(
        unconstrained
    )

    save_histogram(
        raw_noisy,
        unconstrained,
    )

    projected_result = projected_gradient(
        raw=raw_noisy,
        lam=lambda_value,
        lipschitz=lipschitz,
        accelerated=False,
    )

    fista_result = projected_gradient(
        raw=raw_noisy,
        lam=lambda_value,
        lipschitz=lipschitz,
        accelerated=True,
    )

    penalty_result = penalty_method(
        raw=raw_noisy,
        lam=lambda_value,
        lipschitz=lipschitz,
    )

    exact_energy = projected_result["energy"]

    save_projected_table(
        "outputs/lab5_projected.csv",
        projected_result,
    )

    save_projected_table(
        "outputs/lab5_fista.csv",
        fista_result,
    )

    save_penalty_table(
        "outputs/lab5_penalty.csv",
        penalty_result,
    )

    save_convergence_plot(
        projected_result,
        fista_result,
        exact_energy,
    )

    save_active_map(
        projected_result["image"]
    )

    save_image(
        projected_result["image"],
        "outputs/05_constrained.png",
    )

    wrong_stop = gradient_norm_experiment(
        raw_noisy,
        lambda_value,
        lipschitz,
    )

    growth_2 = penalty_method(
        raw_noisy,
        lambda_value,
        lipschitz,
        growth=2.0,
    )

    growth_100 = penalty_method(
        raw_noisy,
        lambda_value,
        lipschitz,
        growth=100.0,
    )

    restored_psnr = psnr(
        projected_result["image"],
        clean,
    )

    print("\nLAB 5 - CONSTRAINED OPTIMIZATION")
    print(f"lambda = {lambda_value:.9f}")
    print(f"L = {lipschitz:.9f}")

    print("\nRAW OBSERVATION")
    print(
        f"Pixels outside [0, 1] = "
        f"{raw_violation['count']}"
    )
    print(
        f"Maximum violation = "
        f"{raw_violation['max']:.9e}"
    )
    print(
        f"Minimum value = "
        f"{raw_noisy.min():.9f}"
    )
    print(
        f"Maximum value = "
        f"{raw_noisy.max():.9f}"
    )

    print("\nUNCONSTRAINED SOLUTION")
    print(
        f"Pixels outside [0, 1] = "
        f"{unconstrained_violation['count']}"
    )
    print(
        f"Maximum violation = "
        f"{unconstrained_violation['max']:.9e}"
    )
    print(
        f"Energy = "
        f"{energy(unconstrained, raw_noisy, lambda_value):.9f}"
    )

    print_comparison(
        projected_result,
        fista_result,
        penalty_result,
    )

    print("\nEXPERIMENT 1 - WRONG STOPPING CRITERION")
    print(
        f"Gradient norm after 200 iterations = "
        f"{wrong_stop['gradient_norm']:.9e}"
    )
    print(
        f"Gradient mapping after 200 iterations = "
        f"{wrong_stop['gradient_mapping']:.9e}"
    )

    print("\nEXPERIMENT 2 - PENALTY GROWTH")
    for growth, result in (
        (2, growth_2),
        (10, penalty_result),
        (100, growth_100),
    ):
        current_violation = violation(result["image"])
        calls = result["n_f"] + result["n_grad"]

        print(
            f"growth = {growth}: "
            f"outer iterations = {result['outer_iterations']}, "
            f"outside = {current_violation['count']}, "
            f"violation = {current_violation['max']:.9e}, "
            f"calls = {calls}"
        )

    print("\nLAB 5 - FINAL RESULT")
    print(
        f"Restored PSNR = "
        f"{restored_psnr:.6f} dB"
    )
    print(
        "Saved image = "
        "outputs/05_constrained.png"
    )
    print(
        "Saved histogram = "
        "outputs/lab5_histogram.png"
    )
    print(
        "Saved convergence plot = "
        "outputs/lab5_convergence.png"
    )
    print(
        "Saved active map = "
        "outputs/lab5_active_constraints.png"
    )
    print("\nLAB 5 COMPLETE")

    return {
        "projected": projected_result,
        "fista": fista_result,
        "penalty": penalty_result,
        "image": projected_result["image"],
        "psnr": restored_psnr,
    }
