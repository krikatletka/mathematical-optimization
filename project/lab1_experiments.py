"""ЛР1: малий/великий крок Свенна та інша початкова точка."""

from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiment_utils import report, write_csv
from src.lab1 import MAX_FILTER_SIGMA, create_objective, svenn
from src.oracle import Oracle
from src.preparation import prepare_project
from src.variant import MIN_FILTER_SIGMA, SVENN_DELTA0, SVENN_X0


def run_experiments(project):
    Path("outputs").mkdir(exist_ok=True)
    print("\nLAB 1 - REQUIRED EXPERIMENTS")
    objective = create_objective(project.noisy, project.clean)
    cases = [
        ("base", SVENN_X0, SVENN_DELTA0),
        ("delta / 10", SVENN_X0, SVENN_DELTA0 / 10.0),
        ("delta * 10", SVENN_X0, SVENN_DELTA0 * 10.0),
        ("x0 = 3", 3.0, SVENN_DELTA0),
    ]
    results, summary, details = [], [], []
    for name, x0, delta in cases:
        oracle = Oracle(objective)
        result = svenn(oracle, x0, delta)
        if result.evaluations != oracle.calls or len(result.rows) != oracle.calls:
            raise RuntimeError("Svenn call counter and trace disagree.")
        results.append((name, result))
        summary.append([
            name, x0, delta, result.a, result.b, result.b - result.a,
            result.direction, result.iterations, result.evaluations,
        ])
        for row in result.rows:
            details.append([name, *asdict(row).values()])

    report("outputs/lab1_experiment_summary.csv", [
        "Case", "x0", "delta0", "a", "b", "Length", "p", "Expansions", "Nf",
    ], summary)
    write_csv("outputs/lab1_experiment_trace.csv", [
        "Case", "call", "k", "sigma", "f_sigma", "delta", "direction", "action",
    ], details)

    # Графік — окрема діагностика; оракули методів уже завершили роботу.
    plot_oracle = Oracle(objective)
    sigmas = np.linspace(MIN_FILTER_SIGMA, MAX_FILTER_SIGMA, 200)
    values = [plot_oracle.peek(float(sigma)) for sigma in sigmas]
    if plot_oracle.calls != 0:
        raise RuntimeError("Plotting changed the search call counter.")
    write_csv("outputs/lab1_experiment_curve.csv", ["sigma", "MSE"],
              zip(sigmas, values))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (name, result) in zip(axes.flat, results):
        ax.plot(sigmas, values, color="royalblue", label="MSE(sigma)")
        ax.scatter([r.sigma for r in result.rows],
                   [r.function_value for r in result.rows],
                   color="crimson", zorder=3, label="Svenn points")
        ax.axvspan(result.a, result.b, color="green", alpha=0.12)
        for row in result.rows:
            ax.annotate(str(row.call), (row.sigma, row.function_value),
                        xytext=(3, 4), textcoords="offset points")
        ax.set(title=f"{name}: Nf={result.evaluations}, p={result.direction}",
               xlabel="sigma", ylabel="MSE",
               xlim=(MIN_FILTER_SIGMA, MAX_FILTER_SIGMA))
        ax.grid(alpha=0.3)
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig("outputs/lab1_experiment_svenn.png", dpi=150)
    plt.close(fig)
    print("p=0: right, or the starting point already brackets a minimum; p=1: left.")
    print("LAB 1 EXPERIMENTS COMPLETE")
    return {"summary": summary, "results": results}


def main():
    run_experiments(prepare_project())


if __name__ == "__main__":
    main()
