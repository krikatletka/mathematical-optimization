
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np


START = np.array([-2.0, 6.0])
EXACT_MINIMUM = np.array([2.5, 2.0])
EPSILON = 0.01
OUTPUT_DIR = Path("outputs")


def objective(x: np.ndarray) -> float:
    """Цільова функція варіанта 8."""
    x1, x2 = np.asarray(x, dtype=float)
    return float(x1**2 + 3.0 * x2**2 + 2.0 * x1 * x2 - 9.0 * x1 - 17.0 * x2)


def gradient(x: np.ndarray) -> np.ndarray:
    x1, x2 = np.asarray(x, dtype=float)
    return np.array([2.0 * x1 + 2.0 * x2 - 9.0,
                     2.0 * x1 + 6.0 * x2 - 17.0])


HESSIAN = np.array([[2.0, 2.0], [2.0, 6.0]])


def nonsmooth(x: np.ndarray) -> float:
    x1, x2 = np.asarray(x, dtype=float)
    return float(2.0 * abs(x1 - 1.0) + 3.0 * abs(x2 + 2.0)
                 + 0.5 * (x1 - 1.0) ** 2)


def rosenbrock(x: np.ndarray) -> float:
    x1, x2 = np.asarray(x, dtype=float)
    return float((1.0 - x1) ** 2 + 100.0 * (x2 - x1**2) ** 2)


@dataclass
class EvaluationCounter:
    function: Callable[[np.ndarray], float]
    calls: int = 0

    def __call__(self, x: np.ndarray) -> float:
        self.calls += 1
        return float(self.function(np.asarray(x, dtype=float)))


@dataclass
class ProtocolRow:
    iteration: int
    action: str
    point: np.ndarray
    value: float
    delta: float
    decision: str
    nf: int


@dataclass
class HookeJeevesResult:
    x: np.ndarray
    fx: float
    iterations: int
    evaluations: int
    final_delta: float
    converged: bool
    rows: list[ProtocolRow] = field(default_factory=list)
    trajectory: list[np.ndarray] = field(default_factory=list)


def explore(
    counted: EvaluationCounter,
    point: np.ndarray,
    value: float,
    delta: float,
    iteration: int,
    rows: list[ProtocolRow],
) -> tuple[np.ndarray, float]:
    current = np.asarray(point, dtype=float).copy()
    current_value = float(value)

    for coordinate in range(len(current)):
        plus = current.copy()
        plus[coordinate] += delta
        plus_value = counted(plus)
        if plus_value < current_value:
            current, current_value = plus, plus_value
            rows.append(ProtocolRow(iteration, f"explore +x{coordinate + 1}",
                                    current.copy(), current_value, delta,
                                    "accepted", counted.calls))
            continue

        rows.append(ProtocolRow(iteration, f"explore +x{coordinate + 1}",
                                plus, plus_value, delta, "rejected", counted.calls))
        minus = current.copy()
        minus[coordinate] -= delta
        minus_value = counted(minus)
        if minus_value < current_value:
            current, current_value = minus, minus_value
            decision = "accepted"
        else:
            decision = "rejected"
        rows.append(ProtocolRow(iteration, f"explore -x{coordinate + 1}",
                                minus, minus_value, delta, decision, counted.calls))

    return current, current_value


def hooke_jeeves(
    function: Callable[[np.ndarray], float],
    x0: Iterable[float],
    delta0: float = 1.0,
    epsilon: float = EPSILON,
    reduction: float = 2.0,
    max_iterations: int = 10000,
    max_evaluations: int = 200000,
) -> HookeJeevesResult:
    if delta0 <= 0 or epsilon <= 0:
        raise ValueError("delta0 and epsilon must be positive")
    if reduction <= 1:
        raise ValueError("reduction must be greater than 1")

    counted = EvaluationCounter(function)
    base = np.asarray(tuple(x0), dtype=float)
    base_value = counted(base)
    delta = float(delta0)
    rows: list[ProtocolRow] = [
        ProtocolRow(0, "initial base", base.copy(), base_value,
                    delta, "start", counted.calls)
    ]
    trajectory = [base.copy()]
    iteration = 0

    while delta >= epsilon and iteration < max_iterations:
        if counted.calls >= max_evaluations:
            break
        iteration += 1
        old_base = base.copy()
        old_value = base_value
        candidate, candidate_value = explore(
            counted, old_base, old_value, delta, iteration, rows
        )

        if candidate_value < old_value:
            direction = candidate - old_base
            pattern = candidate + direction
            pattern_value = counted(pattern)
            rows.append(ProtocolRow(iteration, "pattern", pattern.copy(),
                                    pattern_value, delta,
                                    "probe", counted.calls))

            pattern_candidate, pattern_candidate_value = explore(
                counted, pattern, pattern_value, delta, iteration, rows
            )
            if pattern_candidate_value < candidate_value:
                base, base_value = pattern_candidate, pattern_candidate_value
                decision = "pattern accepted"
            else:
                base, base_value = candidate, candidate_value
                decision = "return to exploratory point"
            rows.append(ProtocolRow(iteration, "new base", base.copy(),
                                    base_value, delta, decision, counted.calls))
            trajectory.append(base.copy())
        else:
            delta /= reduction
            rows.append(ProtocolRow(iteration, "reduce step", base.copy(),
                                    base_value, delta,
                                    "delta = delta / 2", counted.calls))

    return HookeJeevesResult(
        x=base.copy(),
        fx=float(base_value),
        iterations=iteration,
        evaluations=counted.calls,
        final_delta=delta,
        converged=delta < epsilon,
        rows=rows,
        trajectory=trajectory,
    )


def gradient_backtracking(x0: np.ndarray, epsilon: float = EPSILON) -> list[np.ndarray]:
    x = np.asarray(x0, dtype=float).copy()
    points = [x.copy()]
    for _ in range(1000):
        g = gradient(x)
        if np.linalg.norm(g) <= epsilon:
            break
        h = 1.0
        fx = objective(x)
        while objective(x - h * g) > fx - 1e-4 * h * float(g @ g):
            h /= 2.0
        x = x - h * g
        points.append(x.copy())
    return points


def fletcher_reeves_quadratic(x0: np.ndarray, epsilon: float = EPSILON) -> list[np.ndarray]:
    x = np.asarray(x0, dtype=float).copy()
    g = gradient(x)
    d = -g
    points = [x.copy()]
    for k in range(20):
        if np.linalg.norm(g) <= epsilon:
            break
        h = -float(g @ d) / float(d @ HESSIAN @ d)
        x = x + h * d
        points.append(x.copy())
        new_g = gradient(x)
        beta = float(new_g @ new_g) / float(g @ g)
        if (k + 1) % 2 == 0:
            beta = 0.0
        d = -new_g + beta * d
        g = new_g
    return points


def ascii_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))
    border = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    line = lambda row: "|" + "|".join(
        f" {value:<{widths[i]}} " for i, value in enumerate(row)
    ) + "|"
    return "\n".join([border, line(headers), border,
                      *(line(row) for row in rows), border])


def format_point(point: np.ndarray) -> str:
    return f"({point[0]:.6f}; {point[1]:.6f})"


def print_result(title: str, result: HookeJeevesResult, full: bool = False) -> None:
    print(f"\n{title}")
    if full:
        table_rows = [
            [str(row.iteration), row.action, format_point(row.point),
             f"{row.value:.9f}", f"{row.delta:.6f}", row.decision, str(row.nf)]
            for row in result.rows
        ]
        print(ascii_table(
            ["k", "Action", "Point", "F", "Delta", "Decision", "Nf"],
            table_rows,
        ))
    print(f"x* = {format_point(result.x)}")
    print(f"F(x*) = {result.fx:.12f}")
    print(f"Iterations = {result.iterations}")
    print(f"Nf = {result.evaluations}")
    print(f"Final delta = {result.final_delta:.9f}")
    print(f"Status = {'CONVERGED' if result.converged else 'LIMIT REACHED'}")


def save_protocol(path: Path, result: HookeJeevesResult) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["iteration", "action", "x1", "x2", "F",
                         "delta", "decision", "Nf"])
        for row in result.rows:
            writer.writerow([row.iteration, row.action, *row.point,
                             row.value, row.delta, row.decision, row.nf])


def contour_plot(result: HookeJeevesResult, path: Path) -> None:
    x1 = np.linspace(-3.0, 5.5, 450)
    x2 = np.linspace(-1.0, 7.0, 450)
    xx, yy = np.meshgrid(x1, x2)
    zz = xx**2 + 3 * yy**2 + 2 * xx * yy - 9 * xx - 17 * yy
    fig, ax = plt.subplots(figsize=(9, 7))
    levels = np.linspace(-28.0, 90.0, 18)
    ax.contour(xx, yy, zz, levels=levels, colors="#87a6dd", linewidths=0.8)

    trajectories = [
        (gradient_backtracking(START), "Gradient + backtracking", "#356ac3"),
        (fletcher_reeves_quadratic(START), "Fletcher-Reeves", "#159447"),
        (result.trajectory, "Hooke-Jeeves", "#e45b2d"),
    ]
    for points, label, color in trajectories:
        data = np.asarray(points)
        ax.plot(data[:, 0], data[:, 1], "o-", ms=3.5, lw=1.5,
                color=color, label=label)
    ax.scatter(*EXACT_MINIMUM, marker="*", s=150, color="gold",
               edgecolor="black", label="Exact minimum")
    ax.set(xlabel="$x_1$", ylabel="$x_2$",
           title="Laboratory 10: level lines and method trajectories")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def generic_plot(function: Callable[[np.ndarray], float], result: HookeJeevesResult,
                 bounds: tuple[tuple[float, float], tuple[float, float]],
                 title: str, path: Path) -> None:
    x1 = np.linspace(*bounds[0], 350)
    x2 = np.linspace(*bounds[1], 350)
    xx, yy = np.meshgrid(x1, x2)
    zz = np.empty_like(xx)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            zz[i, j] = function(np.array([xx[i, j], yy[i, j]]))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contour(xx, yy, zz, levels=20, cmap="viridis")
    trajectory = np.asarray(result.trajectory)
    ax.plot(trajectory[:, 0], trajectory[:, 1], "o-", color="#e45b2d", ms=3)
    ax.scatter(*result.x, marker="*", s=130, color="gold", edgecolor="black")
    ax.set(xlabel="$x_1$", ylabel="$x_2$", title=title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("LABORATORY WORK 10 - VARIANT 8")
    print("HOOKE-JEEVES DIRECT SEARCH METHOD")
    print("F(x1,x2) = x1^2 + 3*x2^2 + 2*x1*x2 - 9*x1 - 17*x2")
    print(f"Initial point = {format_point(START)}")
    print(f"epsilon = {EPSILON:.6f}")
    print(f"Analytic minimum = {format_point(EXACT_MINIMUM)}")
    print(f"F(minimum) = {objective(EXACT_MINIMUM):.6f}")
    print("Derivatives are not used by Hooke-Jeeves.")

    base = hooke_jeeves(objective, START, delta0=1.0, epsilon=EPSILON)
    smaller = hooke_jeeves(objective, START, delta0=0.7, epsilon=EPSILON)
    sharp = hooke_jeeves(nonsmooth, (-5.0, -3.0), delta0=1.0, epsilon=EPSILON)
    rose_loose = hooke_jeeves(rosenbrock, (-1.2, 1.0),
                              delta0=0.1, epsilon=0.01,
                              max_iterations=10000)
    rose_tight = hooke_jeeves(rosenbrock, (-1.2, 1.0),
                              delta0=0.1, epsilon=0.0001,
                              max_iterations=20000)

    print_result("RUN 1 - delta0 = 1.0 (FULL PROTOCOL)", base, full=True)
    print_result("RUN 2 - delta0 = 0.7", smaller)
    print_result("NONSMOOTH FUNCTION", sharp)
    print_result("ROSENBROCK - epsilon = 0.01", rose_loose)
    print_result("ROSENBROCK - epsilon = 0.0001", rose_tight)

    comparison = [
        ["Variant 8, delta0=1.0", str(base.iterations), str(base.evaluations),
         format_point(base.x), f"{base.fx:.9f}"],
        ["Variant 8, delta0=0.7", str(smaller.iterations), str(smaller.evaluations),
         format_point(smaller.x), f"{smaller.fx:.9f}"],
        ["Nonsmooth", str(sharp.iterations), str(sharp.evaluations),
         format_point(sharp.x), f"{sharp.fx:.9f}"],
        ["Rosenbrock eps=1e-2", str(rose_loose.iterations), str(rose_loose.evaluations),
         format_point(rose_loose.x), f"{rose_loose.fx:.9e}"],
        ["Rosenbrock eps=1e-4", str(rose_tight.iterations), str(rose_tight.evaluations),
         format_point(rose_tight.x), f"{rose_tight.fx:.9e}"],
    ]
    print("\nFINAL COMPARISON")
    print(ascii_table(["Experiment", "Iterations", "Nf", "x*", "F(x*)"], comparison))

    save_protocol(OUTPUT_DIR / "lab10_protocol_delta1.csv", base)
    save_protocol(OUTPUT_DIR / "lab10_protocol_delta07.csv", smaller)
    save_protocol(OUTPUT_DIR / "lab10_nonsmooth.csv", sharp)
    save_protocol(OUTPUT_DIR / "lab10_rosenbrock_1e-2.csv", rose_loose)
    save_protocol(OUTPUT_DIR / "lab10_rosenbrock_1e-4.csv", rose_tight)
    contour_plot(base, OUTPUT_DIR / "lab10_trajectories.png")
    generic_plot(nonsmooth, sharp, ((-6, 5), (-6, 3)),
                 "Hooke-Jeeves on a nonsmooth function",
                 OUTPUT_DIR / "lab10_nonsmooth.png")
    generic_plot(rosenbrock, rose_tight, ((-1.6, 1.4), (-0.5, 2.0)),
                 "Hooke-Jeeves on the Rosenbrock function",
                 OUTPUT_DIR / "lab10_rosenbrock.png")

    error = float(np.linalg.norm(base.x - EXACT_MINIMUM))
    print("\nLAB 10 COMPLETE")
    print(f"Absolute point error = {error:.9e}")
    print("Saved tables = outputs/lab10_*.csv")
    print("Saved plots = outputs/lab10_*.png")


if __name__ == "__main__":
    main()