"""ЛР2: точність 1e-6, delta=epsilon та ручний вибір sigma."""

import math
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiment_utils import report, write_csv
from src.lab1 import create_objective, svenn
from src.lab2 import TAU, dichotomy, golden_section
from src.oracle import Oracle
from src.preparation import prepare_project
from src.variant import EPSILON, SVENN_DELTA0, SVENN_X0


def run_experiments(project):
    Path("outputs").mkdir(exist_ok=True)
    print("\nLAB 2 - REQUIRED EXPERIMENTS")
    objective = create_objective(project.noisy, project.clean)
    bracket = svenn(Oracle(objective), SVENN_X0, SVENN_DELTA0)
    a, b = bracket.a, bracket.b
    print(f"Initial interval = [{a:.9f}; {b:.9f}]")
    print(f"Svenn preparation calls = {bracket.evaluations} (separate)")

    print("\nEXPERIMENT 1 - TOLERANCE")
    results, rows = {}, []
    for epsilon in (EPSILON, 1e-6):
        for name, method in (("Dichotomy", dichotomy),
                             ("Golden section", golden_section)):
            oracle = Oracle(objective)
            result = method(oracle, a, b, epsilon, max_iter=500)
            if result.evaluations != oracle.calls:
                raise RuntimeError("Search call counter disagrees with Oracle.")
            key = (name, epsilon)
            results[key] = result
            length = result.b - result.a
            contraction = (length / (b - a)) ** (1.0 / result.evaluations)
            base = results[(name, EPSILON)]
            rows.append([
                name, epsilon, result.x, result.fx, result.iterations,
                result.evaluations, length, contraction,
                result.iterations - base.iterations,
                result.evaluations - base.evaluations, result.converged,
            ])
            trace = [list(asdict(row).values()) for row in result.rows]
            filename = name.lower().replace(" ", "_")
            write_csv(f"outputs/lab2_experiment_{filename}_{epsilon:.0e}.csv", [
                "k", "a_before", "b_before", "x1", "x2", "f1", "f2",
                "length_before", "decision",
            ], trace)
    report("outputs/lab2_experiment_accuracy.csv", [
        "Method", "epsilon", "sigma", "MSE", "Iterations", "Nf", "Final_length",
        "Contraction_per_call", "Extra_iterations", "Extra_Nf", "Converged",
    ], rows)
    factor = EPSILON / 1e-6
    print(f"Accuracy factor for this variant = {factor:g}")
    print(f"Theory: extra dichotomy iterations about {math.log(factor, 2):.3f}; "
          f"extra golden iterations about {math.log(factor) / math.log(1.0 / TAU):.3f}.")

    fig, ax = plt.subplots(figsize=(9, 5))
    for (name, epsilon), result in results.items():
        lengths = [row.length for row in result.rows] + [result.b - result.a]
        ax.semilogy(range(len(lengths)), lengths,
                    label=f"{name}, eps={epsilon:.0e}")
    ax.set(xlabel="iteration k", ylabel="interval length",
           title="Lab 2: influence of tolerance")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("outputs/lab2_experiment_accuracy.png", dpi=150)
    plt.close(fig)

    print("\nEXPERIMENT 2 - DELTA EQUAL TO EPSILON")
    oracle = Oracle(objective)
    stalled = dichotomy(oracle, a, b, EPSILON, delta=EPSILON, max_iter=500)
    baseline = results[("Dichotomy", EPSILON)]
    delta_rows = [[label, delta, r.iterations, r.evaluations,
                   r.b - r.a, (r.b - r.a) - EPSILON, r.converged]
                  for label, delta, r in [
                      ("delta = epsilon / 3", EPSILON / 3.0, baseline),
                      ("delta = epsilon", EPSILON, stalled),
                  ]]
    report("outputs/lab2_experiment_delta.csv", [
        "Case", "delta", "Iterations", "Nf", "Final_length",
        "Length_minus_epsilon", "Converged",
    ], delta_rows)
    write_csv("outputs/lab2_experiment_delta_trace.csv", [
        "k", "a_before", "b_before", "x1", "x2", "f1", "f2",
        "length_before", "decision",
    ], [list(asdict(row).values()) for row in stalled.rows])
    print("For dichotomy L_next=(L+delta)/2, so lengths approach delta.")
    print("With delta=epsilon exact arithmetic approaches epsilon from above; "
          "floating-point rounding and the iteration limit determine the exit.")

    print("\nEXPERIMENT 3 - MANUAL SIGMA")
    optimum = results[("Golden section", EPSILON)]
    optimal_psnr = -10.0 * math.log10(optimum.fx)
    manual_rows = []
    for sigma in (0.3, 0.5, 0.8, 1.5):
        manual_oracle = Oracle(objective)
        value = manual_oracle.value(sigma)
        quality = -10.0 * math.log10(value)
        manual_rows.append(["manual", sigma, value, quality,
                            optimal_psnr - quality, manual_oracle.calls])
    manual_rows.append(["golden optimum", optimum.x, optimum.fx,
                        optimal_psnr, 0.0, optimum.evaluations])
    report("outputs/lab2_experiment_manual.csv", [
        "Case", "sigma", "MSE", "PSNR_dB", "PSNR_loss_dB", "Nf",
    ], manual_rows)
    # Ця діагностична сітка не змінює жодного лічильника методів.
    grid = np.linspace(0.3, 1.5, 200)
    qualities = [-10.0 * math.log10(objective(float(s))) for s in grid]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grid, qualities, label="PSNR(sigma)")
    ax.scatter([r[1] for r in manual_rows], [r[3] for r in manual_rows],
               color="crimson", zorder=3)
    ax.axvline(optimum.x, color="green", linestyle="--", label="golden optimum")
    ax.set(xlabel="sigma", ylabel="PSNR, dB", title="Lab 2: manual sigma selection")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("outputs/lab2_experiment_manual.png", dpi=150)
    plt.close(fig)
    print("LAB 2 EXPERIMENTS COMPLETE")
    return {"results": results, "delta": stalled, "manual": manual_rows}


def main():
    run_experiments(prepare_project())


if __name__ == "__main__":
    main()
