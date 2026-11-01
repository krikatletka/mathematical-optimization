
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


EPS = 0.01
LINE_EPS = 1e-6
SVENN_STEP = 0.01
RESTART_PERIOD = 2
OUTPUT_DIR = Path("outputs")


Vector = np.ndarray
Function = Callable[[Vector], float]
Gradient = Callable[[Vector], Vector]


def objective(x: Vector) -> float:
    x1, x2 = x
    return float(x1**2 + 3.0 * x2**2 + 2.0 * x1 * x2 - 9.0 * x1 - 17.0 * x2)


def objective_gradient(x: Vector) -> Vector:
    x1, x2 = x
    return np.array([2.0 * x1 + 2.0 * x2 - 9.0,
                     2.0 * x1 + 6.0 * x2 - 17.0])


HESSIAN = np.array([[2.0, 2.0], [2.0, 6.0]])


def rosenbrock(x: Vector) -> float:
    x1, x2 = x
    return float((1.0 - x1) ** 2 + 100.0 * (x2 - x1**2) ** 2)


def rosenbrock_gradient(x: Vector) -> Vector:
    x1, x2 = x
    return np.array([
        -2.0 * (1.0 - x1) - 400.0 * x1 * (x2 - x1**2),
        200.0 * (x2 - x1**2),
    ])


@dataclass
class Counter:
    function_calls: int = 0
    gradient_calls: int = 0


def counted_function(function: Function, counter: Counter) -> Function:
    def wrapped(x: Vector) -> float:
        counter.function_calls += 1
        return function(x)

    return wrapped


def counted_gradient(gradient: Gradient, counter: Counter) -> Gradient:
    def wrapped(x: Vector) -> Vector:
        counter.gradient_calls += 1
        return gradient(x)

    return wrapped


def svenn(
    phi: Callable[[float], float],
    x0: float = 0.0,
    delta: float = SVENN_STEP,
    max_steps: int = 60,
) -> tuple[float, float]:
    f0 = phi(x0)
    f_right = phi(x0 + delta)
    f_left = phi(x0 - delta)

    if f_left >= f0 and f_right >= f0:
        return x0 - delta, x0 + delta

    if f_left < f_right:
        delta = -delta
        previous = x0
        current = x0 + delta
        current_value = f_left
    else:
        previous = x0
        current = x0 + delta
        current_value = f_right

    for power in range(1, max_steps + 1):
        candidate = current + delta * (2**power)
        candidate_value = phi(candidate)

        if candidate_value >= current_value:
            return tuple(sorted((previous, candidate)))

        previous = current
        current = candidate
        current_value = candidate_value

    raise RuntimeError("Svenn method could not bracket the minimum.")


def golden_section(
    phi: Callable[[float], float],
    left: float,
    right: float,
    eps: float = LINE_EPS,
) -> float:
    tau = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = left + (1.0 - tau) * (right - left)
    x2 = left + tau * (right - left)
    f1 = phi(x1)
    f2 = phi(x2)

    while right - left > eps:
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = left + (1.0 - tau) * (right - left)
            f1 = phi(x1)
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + tau * (right - left)
            f2 = phi(x2)

    return (left + right) / 2.0


def line_search(function: Function, x: Vector, direction: Vector) -> float:
    phi = lambda h: function(x + h * direction)
    left, right = svenn(phi)
    return golden_section(phi, left, right)


def exact_quadratic_step(gradient: Vector, direction: Vector) -> float:
    denominator = float(direction @ HESSIAN @ direction)
    return -float(gradient @ direction) / denominator


def fletcher_reeves(
    function: Function,
    gradient: Gradient,
    x0: Vector,
    eps: float = EPS,
    restart_period: int | None = RESTART_PERIOD,
    max_iterations: int = 5000,
    hessian: Vector | None = None,
) -> dict:
    counter = Counter()
    f = counted_function(function, counter)
    grad = counted_gradient(gradient, counter)

    x = np.asarray(x0, dtype=float).copy()
    g = grad(x)
    direction = -g
    trajectory = [x.copy()]
    rows: list[dict] = []

    for k in range(max_iterations + 1):
        grad_norm = float(np.linalg.norm(g))
        f_x = f(x)

        if grad_norm <= eps:
            rows.append({
                "k": k,
                "x1": x[0],
                "x2": x[1],
                "F": f_x,
                "g1": g[0],
                "g2": g[1],
                "grad_norm": grad_norm,
                "d1": direction[0],
                "d2": direction[1],
                "h": math.nan,
                "beta": math.nan,
                "orthogonality": math.nan,
                "conjugacy": math.nan,
                "descent": math.nan,
            })
            break

        old_direction = direction.copy()
        h = line_search(f, x, old_direction)
        x_new = x + h * old_direction
        g_new = grad(x_new)

        beta_raw = float(g_new @ g_new) / float(g @ g)
        restart = restart_period is not None and (k + 1) % restart_period == 0
        beta = 0.0 if restart else beta_raw
        direction_new = -g_new + beta * old_direction

        orthogonality = float(g_new @ old_direction)
        conjugacy = (
            float(direction_new @ hessian @ old_direction)
            if hessian is not None
            else math.nan
        )
        descent = float(g_new @ direction_new)

        rows.append({
            "k": k,
            "x1": x[0],
            "x2": x[1],
            "F": f_x,
            "g1": g[0],
            "g2": g[1],
            "grad_norm": grad_norm,
            "d1": old_direction[0],
            "d2": old_direction[1],
            "h": h,
            "beta": beta,
            "orthogonality": orthogonality,
            "conjugacy": conjugacy,
            "descent": descent,
        })

        x, g, direction = x_new, g_new, direction_new
        trajectory.append(x.copy())
    else:
        raise RuntimeError("Fletcher-Reeves reached the iteration limit.")

    return {
        "x": x,
        "f": f(x),
        "iterations": len(trajectory) - 1,
        "n_f": counter.function_calls,
        "n_grad": counter.gradient_calls,
        "rows": rows,
        "trajectory": np.asarray(trajectory),
    }


def steepest_descent_exact(
    function: Function,
    gradient: Gradient,
    x0: Vector,
    eps: float = EPS,
    max_iterations: int = 5000,
) -> dict:
    counter = Counter()
    f = counted_function(function, counter)
    grad = counted_gradient(gradient, counter)
    x = np.asarray(x0, dtype=float).copy()
    trajectory = [x.copy()]
    rows = []

    for k in range(max_iterations + 1):
        g = grad(x)
        norm = float(np.linalg.norm(g))
        rows.append((k, x[0], x[1], f(x), norm))
        if norm <= eps:
            break
        direction = -g
        h = line_search(f, x, direction)
        x = x + h * direction
        trajectory.append(x.copy())
    else:
        raise RuntimeError("Steepest descent reached the iteration limit.")

    return {
        "x": x,
        "f": f(x),
        "iterations": len(trajectory) - 1,
        "n_f": counter.function_calls,
        "n_grad": counter.gradient_calls,
        "rows": rows,
        "trajectory": np.asarray(trajectory),
    }


def gradient_descent_backtracking(
    function: Function,
    gradient: Gradient,
    x0: Vector,
    eps: float = EPS,
    max_iterations: int = 5000,
) -> dict:
    counter = Counter()
    f = counted_function(function, counter)
    grad = counted_gradient(gradient, counter)
    x = np.asarray(x0, dtype=float).copy()
    trajectory = [x.copy()]

    for _ in range(max_iterations):
        g = grad(x)
        if np.linalg.norm(g) <= eps:
            break
        step = 1.0
        current = f(x)
        while f(x - step * g) > current - 1e-4 * step * float(g @ g):
            step *= 0.5
        x = x - step * g
        trajectory.append(x.copy())

    return {
        "x": x,
        "f": f(x),
        "iterations": len(trajectory) - 1,
        "n_f": counter.function_calls,
        "n_grad": counter.gradient_calls,
        "trajectory": np.asarray(trajectory),
    }


def write_iteration_table(path: Path, rows: list[dict]) -> None:
    headers = [
        "k", "x1", "x2", "F", "g1", "g2", "grad_norm",
        "d1", "d2", "h", "beta", "orthogonality", "conjugacy", "descent",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def print_fletcher_table(rows: list[dict]) -> None:
    print("+---+-----------------------+-------------+-------------+-------------+----------+----------+")
    print("| k | x_k                   | F(x_k)      | ||grad F||  | h_k         | beta_k   | restart  |")
    print("+---+-----------------------+-------------+-------------+-------------+----------+----------+")
    for row in rows:
        h = "-" if math.isnan(row["h"]) else f"{row['h']:.6f}"
        beta = "-" if math.isnan(row["beta"]) else f"{row['beta']:.6f}"
        restart = "-" if math.isnan(row["beta"]) else ("yes" if row["beta"] == 0.0 else "no")
        print(
            f"| {row['k']:<1} | ({row['x1']:>9.6f}; {row['x2']:>9.6f}) "
            f"| {row['F']:>11.6f} | {row['grad_norm']:>11.6f} "
            f"| {h:>11} | {beta:>8} | {restart:<8} |"
        )
    print("+---+-----------------------+-------------+-------------+-------------+----------+----------+")


def plot_quadratic_trajectories(results: dict[str, dict]) -> None:
    x1 = np.linspace(-3.0, 4.5, 350)
    x2 = np.linspace(-1.0, 7.0, 350)
    xx, yy = np.meshgrid(x1, x2)
    zz = xx**2 + 3.0 * yy**2 + 2.0 * xx * yy - 9.0 * xx - 17.0 * yy

    plt.figure(figsize=(9, 7))
    plt.contour(xx, yy, zz, levels=24, colors="#8aa6d8", linewidths=0.8)

    styles = {
        "Gradient with backtracking": ("#2563eb", "o-"),
        "Antigradient + exact search": ("#7c3aed", "s-"),
        "Fletcher-Reeves": ("#15803d", "o-"),
    }

    for name, result in results.items():
        trajectory = result["trajectory"]
        color, style = styles[name]
        plt.plot(
            trajectory[:, 0], trajectory[:, 1], style,
            color=color, markersize=4, linewidth=1.7, label=name,
        )

    optimum = np.linalg.solve(HESSIAN, np.array([9.0, 17.0]))
    plt.scatter(optimum[0], optimum[1], marker="*", s=180, color="red", label="Minimum")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Lab 9: optimization trajectories, variant 8")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lab9_trajectories.png", dpi=170)
    plt.close()


def plot_rosenbrock(results: dict[str, dict]) -> None:
    """Будує траєкторії Флетчера—Рівса на функції Розенброка."""
    x1 = np.linspace(-1.7, 1.4, 350)
    x2 = np.linspace(-0.4, 2.1, 350)
    xx, yy = np.meshgrid(x1, x2)
    zz = (1.0 - xx) ** 2 + 100.0 * (yy - xx**2) ** 2

    plt.figure(figsize=(9, 7))
    levels = np.geomspace(1e-2, 1e3, 25)
    plt.contour(xx, yy, zz, levels=levels, norm="log", colors="#9ab0d2", linewidths=0.75)

    for name, result in results.items():
        trajectory = result["trajectory"]
        plt.plot(trajectory[:, 0], trajectory[:, 1], "o-", markersize=2.5, label=name)

    plt.scatter(1.0, 1.0, marker="*", s=180, color="red", label="Minimum (1, 1)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Fletcher-Reeves on the Rosenbrock function")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lab9_rosenbrock.png", dpi=170)
    plt.close()


def print_summary(results: dict[str, dict]) -> None:
    print("\nMETHOD COMPARISON")
    print("+------------------------------+------------+------+-------+---------------------------+")
    print("| Method                       | Iterations | Nf   | Ngrad | x*                        |")
    print("+------------------------------+------------+------+-------+---------------------------+")
    for name, result in results.items():
        print(
            f"| {name:<28} | {result['iterations']:<10} | {result['n_f']:<4} "
            f"| {result['n_grad']:<5} | ({result['x'][0]:>9.6f}; {result['x'][1]:>9.6f}) |"
        )
    print("+------------------------------+------------+------+-------+---------------------------+")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    start = np.array([-2.0, 6.0])

    eigenvalues = np.linalg.eigvalsh(HESSIAN)
    optimum = np.linalg.solve(HESSIAN, np.array([9.0, 17.0]))

    print("LABORATORY WORK 9 - VARIANT 8")
    print("F(x1,x2) = x1^2 + 3*x2^2 + 2*x1*x2 - 9*x1 - 17*x2")
    print(f"Initial point = ({start[0]:.6f}; {start[1]:.6f})")
    print(f"epsilon = {EPS:.6f}")
    print("Gradient = (2*x1 + 2*x2 - 9; 2*x1 + 6*x2 - 17)")
    print("Hessian = [[2, 2], [2, 6]]")
    print(f"Hessian eigenvalues = ({eigenvalues[0]:.6f}; {eigenvalues[1]:.6f})")
    print("Hessian is positive definite: yes")
    print(f"Analytic minimum = ({optimum[0]:.6f}; {optimum[1]:.6f})")
    print(f"F(minimum) = {objective(optimum):.6f}")

    first_gradient = objective_gradient(start)
    first_direction = -first_gradient
    analytic_h = exact_quadratic_step(first_gradient, first_direction)
    counter = Counter()
    numerical_h = line_search(counted_function(objective, counter), start, first_direction)

    print("\nFIRST LINE SEARCH CHECK")
    print(f"gradient(x0) = ({first_gradient[0]:.6f}; {first_gradient[1]:.6f})")
    print(f"direction d0 = ({first_direction[0]:.6f}; {first_direction[1]:.6f})")
    print(f"analytic h* = {analytic_h:.9f}")
    print(f"Svenn + golden h* = {numerical_h:.9f}")
    print(f"absolute difference = {abs(analytic_h - numerical_h):.3e}")

    fr = fletcher_reeves(
        objective,
        objective_gradient,
        start,
        restart_period=RESTART_PERIOD,
        hessian=HESSIAN,
    )
    antigradient = steepest_descent_exact(objective, objective_gradient, start)
    backtracking = gradient_descent_backtracking(objective, objective_gradient, start)

    print("\nFLETCHER-REEVES ITERATIONS")
    print_fletcher_table(fr["rows"])

    print("\nTHREE REQUIRED CHECKS")
    for row in fr["rows"]:
        if math.isnan(row["orthogonality"]):
            continue
        print(
            f"k={row['k']}: <g_next,d_k>={row['orthogonality']:.3e}, "
            f"d_next^T H d_k={row['conjugacy']:.3e}, "
            f"<g_next,d_next>={row['descent']:.3e}"
        )

    quadratic_results = {
        "Gradient with backtracking": backtracking,
        "Antigradient + exact search": antigradient,
        "Fletcher-Reeves": fr,
    }
    print_summary(quadratic_results)
    plot_quadratic_trajectories(quadratic_results)
    write_iteration_table(OUTPUT_DIR / "lab9_fletcher_reeves.csv", fr["rows"])

    print("\nROSENBROCK EXPERIMENT")
    rosen_start = np.array([-1.2, 1.0])
    rosen_restart = fletcher_reeves(
        rosenbrock,
        rosenbrock_gradient,
        rosen_start,
        eps=EPS,
        restart_period=2,
        max_iterations=5000,
    )
    rosen_without_restart = fletcher_reeves(
        rosenbrock,
        rosenbrock_gradient,
        rosen_start,
        eps=EPS,
        restart_period=None,
        max_iterations=5000,
    )

    rosen_results = {
        "Fletcher-Reeves, restart 2": rosen_restart,
        "Fletcher-Reeves, no restart": rosen_without_restart,
    }
    print_summary(rosen_results)
    plot_rosenbrock(rosen_results)

    print("\nFINAL RESULT")
    print(f"x* = ({fr['x'][0]:.9f}; {fr['x'][1]:.9f})")
    print(f"F(x*) = {fr['f']:.12f}")
    print(f"iterations = {fr['iterations']}")
    print(f"Nf = {fr['n_f']}, Ngrad = {fr['n_grad']}")
    print("Saved table = outputs/lab9_fletcher_reeves.csv")
    print("Saved plot = outputs/lab9_trajectories.png")
    print("Saved Rosenbrock plot = outputs/lab9_rosenbrock.png")


if __name__ == "__main__":
    main()