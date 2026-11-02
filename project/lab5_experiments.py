"""Чотири обов'язкові досліди ЛР5; сире спостереження з частини 0."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiment_utils import report, write_csv
from lab4_experiments import find_optimal_lambda
from src.image_io import save_image
from src.lab4 import energy, energy_grad
from src.lab5 import penalty_method, project, violation
from src.preparation import prepare_project


def projected_with_criterion(raw, lam, lipschitz, criterion,
                             max_iterations=200, tolerance=1e-6):
    """Той самий проєкційний спуск; змінюється лише критерій зупинки."""
    if criterion not in ("mapping", "gradient"):
        raise ValueError("Criterion must be 'mapping' or 'gradient'.")
    image = project(raw.copy())
    step = 1.0 / lipschitz
    rows = []
    n_f = n_grad = 0
    converged = False
    for k in range(max_iterations + 1):
        gradient = energy_grad(image, raw, lam)
        n_grad += 1
        candidate = project(image - step * gradient)
        gradient_norm = float(np.linalg.norm(gradient))
        mapping_norm = float(np.linalg.norm(candidate - image) / step)
        current_energy = energy(image, raw, lam)
        n_f += 1
        rows.append([k, current_energy, gradient_norm, mapping_norm, n_f, n_grad])
        stopping_norm = mapping_norm if criterion == "mapping" else gradient_norm
        if stopping_norm <= tolerance:
            converged = True
            break
        if k == max_iterations:
            break
        image = candidate
    return {"image": image, "iterations": rows[-1][0], "energy": rows[-1][1],
            "gradient_norm": rows[-1][2], "gradient_mapping": rows[-1][3],
            "n_f": n_f, "n_grad": n_grad, "converged": converged, "rows": rows}


def experiment_stopping(raw, lam, lipschitz):
    print("\nEXPERIMENT 1 - WRONG STOPPING CRITERION")
    results, rows = {}, []
    for criterion in ("mapping", "gradient"):
        result = projected_with_criterion(raw, lam, lipschitz, criterion)
        results[criterion] = result
        rows.append([criterion, result["iterations"], result["n_f"], result["n_grad"],
                     result["n_f"] + result["n_grad"], result["gradient_norm"],
                     result["gradient_mapping"], result["energy"], result["converged"]])
        write_csv(f"outputs/lab5_experiment_stop_{criterion}_trace.csv", [
            "k", "E", "Gradient_norm", "Mapping_norm", "Nf", "Ngrad",
        ], result["rows"])
    report("outputs/lab5_experiment_stopping.csv", [
        "Criterion", "Iterations", "Nf", "Ngrad", "Calls", "Gradient_norm",
        "Mapping_norm", "Energy", "Converged",
    ], rows)
    wrong = results["gradient"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy([r[0] for r in wrong["rows"]],
                [max(r[2], 1e-16) for r in wrong["rows"]], label="gradient norm")
    ax.semilogy([r[0] for r in wrong["rows"]],
                [max(r[3], 1e-16) for r in wrong["rows"]], label="gradient mapping norm")
    ax.axhline(1e-6, color="gray", linestyle="--", label="tolerance")
    ax.axvline(results["mapping"]["iterations"], color="green", linestyle=":",
               label="correct stopping iteration")
    ax.set(xlabel="iteration k", ylabel="norm", title="Lab 5: stopping criterion")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("outputs/lab5_experiment_stopping.png", dpi=150)
    plt.close(fig)
    return results


def experiment_growth(raw, lam, lipschitz, reference_energy):
    print("\nEXPERIMENT 2 - PENALTY GROWTH")
    results, rows = {}, []
    for growth in (2.0, 10.0, 100.0):
        result = penalty_method(raw, lam, lipschitz, growth=growth)
        results[growth] = result
        v = violation(result["image"])
        # Один додатковий виклик E для діагностики допустимого зображення.
        # Він записаний окремо й не входить до вартості штрафного розв'язувача.
        feasible_energy_gap = energy(project(result["image"]), raw, lam) - reference_energy
        rows.append([growth, result["outer_iterations"], result["n_f"], result["n_grad"],
                     result["n_f"] + result["n_grad"], v["max"], v["count"],
                     result["rows"][-1]["rP"], feasible_energy_gap, 1])
        trace = result["rows"]
        write_csv(f"outputs/lab5_experiment_growth_{growth:g}_trace.csv",
                  list(trace[0]), [[row[key] for key in trace[0]] for row in trace])
    report("outputs/lab5_experiment_growth.csv", [
        "Growth", "Outer_iterations", "Nf", "Ngrad", "Solver_calls", "Max_violation",
        "Outside", "Final_rP", "E_projected_minus_E_reference", "Diagnostic_Nf",
    ], rows)
    print("Small violation alone does not prove energy minimization.")
    print("Projected energy gap uses one separate diagnostic E call per growth.")
    fig, ax = plt.subplots(figsize=(8, 5))
    for growth, result in results.items():
        ax.semilogy([r["s"] for r in result["rows"]],
                    [max(r["violation"], 1e-16) for r in result["rows"]],
                    "o-", label=f"growth={growth:g}")
    ax.set(xlabel="outer iteration s", ylabel="maximum violation",
           title="Lab 5: penalty growth")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("outputs/lab5_experiment_growth.png", dpi=150)
    plt.close(fig)
    return results


def experiment_conditioning(penalty_result):
    print("\nEXPERIMENT 3 - CONDITIONING COST")
    rows = [[r["s"], r["r"], r["L_r"], r["step"], r["inner"],
             r["rP"], r["violation"]] for r in penalty_result["rows"]]
    report("outputs/lab5_experiment_conditioning.csv", [
        "s", "r", "L_r", "Step", "Inner_iterations", "rP", "Max_violation",
    ], rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog([r[1] for r in rows], [r[3] for r in rows], "o-")
    ax.set(xlabel="penalty r", ylabel="step = 1/(L+2r)",
           title="Lab 5: shrinking step")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("outputs/lab5_experiment_conditioning.png", dpi=150)
    plt.close(fig)
    return rows


def experiment_active_constraints(clean, image):
    print("\nEXPERIMENT 4 - ACTIVE CONSTRAINTS")
    lower = image <= 1e-12
    upper = image >= 1.0 - 1e-12
    active_map = np.full_like(image, 0.5)
    active_map[lower] = 0.0
    active_map[upper] = 1.0
    rows = [["lower (black)", int(np.count_nonzero(lower))],
            ["upper (white)", int(np.count_nonzero(upper))],
            ["free (gray)", int(np.count_nonzero(~(lower | upper)))]]
    report("outputs/lab5_experiment_active.csv", ["Constraint", "Pixels"], rows)
    save_image(active_map, "outputs/lab5_experiment_active_map.png")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, data, title in zip(axes, [clean, image, active_map],
                               ["Clean", "Constrained solution", "Active constraints"]):
        ax.imshow(data, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig("outputs/lab5_experiment_active_comparison.png", dpi=150)
    plt.close(fig)
    return rows


def run_experiments(project):
    Path("outputs").mkdir(exist_ok=True)
    print("\nLAB 5 - REQUIRED EXPERIMENTS")
    # Lambda налаштовуємо на обрізаному noisy, як у ЛР4; задачу ЛР5 — на raw.
    lam, calls = find_optimal_lambda(project.clean, project.noisy)
    lipschitz = 1.0 + 8.0 * lam
    print(f"lambda={lam:.9f}; L={lipschitz:.9f}; lambda search calls={calls} (separate)")
    stopping = experiment_stopping(project.raw_noisy, lam, lipschitz)
    correct = stopping["mapping"]
    growth = experiment_growth(project.raw_noisy, lam, lipschitz, correct["energy"])
    conditioning = experiment_conditioning(growth[10.0])
    active = experiment_active_constraints(project.clean, correct["image"])
    print("LAB 5 EXPERIMENTS COMPLETE")
    return {"lambda": lam, "stopping": stopping, "growth": growth,
            "conditioning": conditioning, "active": active}


def main():
    run_experiments(prepare_project())


if __name__ == "__main__":
    main()
