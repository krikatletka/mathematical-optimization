"""ЛР4: обумовленість, нестійкий крок та вплив lambda на PSNR."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiment_utils import report, write_csv
from src.lab4 import (
    _golden_lambda, _mse, _psnr, energy, energy_grad,
    gradient_method, tikhonov_exact,
)
from src.oracle import Oracle
from src.preparation import prepare_project
from src.variant import LAMBDA0


def find_optimal_lambda(clean, noisy):
    """Та сама точність пошуку, що в виправленій основній ЛР4."""
    oracle = Oracle(lambda lam: _mse(tikhonov_exact(noisy, lam), clean))
    value = _golden_lambda(oracle.value, start=LAMBDA0, eps=1e-3)
    return value, oracle.calls


def experiment_large_condition_number(clean, noisy, optimal_lambda=None):
    print("\nEXPERIMENT 1 - LARGE CONDITION NUMBER")
    cases = [("lambda = 6", 6.0)]
    if optimal_lambda is not None:
        cases.insert(0, ("lambda*", optimal_lambda))
    rows, histories = [], []
    for case, lam in cases:
        lipschitz = 1.0 + 8.0 * lam
        root_l = math.sqrt(lipschitz)
        beta_hb = ((root_l - 1.0) / (root_l + 1.0)) ** 2
        beta_nag = (root_l - 1.0) / (root_l + 1.0)
        reference = tikhonov_exact(noisy, lam)
        print(f"{case}: L={lipschitz:.9f}, mu=1, kappa={lipschitz:.9f}; "
              f"beta_hb={beta_hb:.9f}, beta_nag={beta_nag:.9f}")
        case_history = []
        for name, kind, beta in [
            ("Gradient descent", "gd", None),
            ("Heavy ball optimal beta", "hb", beta_hb),
            ("Nesterov strongly convex", "nag", beta_nag),
        ]:
            result = gradient_method(
                noisy, lam, lipschitz, kind=kind, beta=beta,
                max_iter=3000, tol=1e-6, reference=reference,
            )
            rows.append([
                case, lam, lipschitz, name, result["iterations"],
                result["n_f"], result["n_grad"], result["n_f"] + result["n_grad"],
                result["energy"], _psnr(result["image"], clean), result["converged"],
            ])
            case_history.append((name, result))
        histories.append((case, case_history))
    report("outputs/lab4_experiment_conditioning.csv", [
        "Case", "lambda", "kappa", "Method", "Iterations", "Nf", "Ngrad",
        "Calls", "Energy", "PSNR_dB", "Converged",
    ], rows)
    fig, axes = plt.subplots(1, len(histories), figsize=(7 * len(histories), 5),
                             squeeze=False)
    for ax, (case, methods) in zip(axes.flat, histories):
        for name, result in methods:
            ax.semilogy([r[0] for r in result["rows"]],
                        [max(r[2], 1e-16) for r in result["rows"]], label=name)
        ax.axhline(1e-6, color="gray", linestyle="--")
        ax.set(title=case, xlabel="iteration k", ylabel="gradient norm at query point")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig("outputs/lab4_experiment_conditioning.png", dpi=150)
    plt.close(fig)
    return rows


def unstable_gradient_descent(noisy, lam, iterations=50):
    """Рівно iterations переходів; початковий і кінцевий стани теж вимірюються."""
    lipschitz = 1.0 + 8.0 * lam
    step = 2.1 / lipschitz
    image = noisy.copy()
    rows = []
    n_f = n_grad = 0
    for k in range(iterations + 1):
        current_energy = energy(image, noisy, lam)
        n_f += 1
        gradient = energy_grad(image, noisy, lam)
        n_grad += 1
        rows.append([k, current_energy, float(np.linalg.norm(gradient)),
                     n_f, n_grad, n_f + n_grad])
        if k < iterations:
            image = image - step * gradient
    return rows, step, 2.0 / lipschitz


def experiment_unstable_step(noisy, optimal_lambda):
    print("\nEXPERIMENT 2 - STEP ABOVE THE STABILITY LIMIT")
    rows, step, stability_limit = unstable_gradient_descent(noisy, optimal_lambda)
    headers = ["Iteration", "Energy", "Gradient_norm", "Nf", "Ngrad", "Calls"]
    print(f"lambda*={optimal_lambda:.9f}, h=2.1/L={step:.9f}, "
          f"2/L={stability_limit:.9f}")
    report("outputs/lab4_experiment_unstable_step.csv", headers,
           [r for r in rows if r[0] % 10 == 0 or r is rows[-1]])
    write_csv("outputs/lab4_experiment_unstable_trace.csv", headers, rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy([r[0] for r in rows], [r[1] for r in rows], color="crimson")
    ax.set(xlabel="iteration k", ylabel="E(I_k)",
           title="Lab 4: unstable step h = 2.1/L")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("outputs/lab4_experiment_unstable_step.png", dpi=150)
    plt.close(fig)
    return rows


def experiment_lambda_influence(clean, noisy, optimal_lambda):
    print("\nEXPERIMENT 3 - INFLUENCE OF LAMBDA")
    rows = []
    for lam in (0.05, 0.24, optimal_lambda, 1.0, 5.0):
        image = tikhonov_exact(noisy, lam)
        rows.append([lam, _mse(image, clean), _psnr(image, clean),
                     "optimal" if lam == optimal_lambda else "test"])
    report("outputs/lab4_experiment_lambda.csv", ["lambda", "MSE", "PSNR_dB", "Type"], rows)
    grid = np.sort(np.append(np.geomspace(0.01, 10.0, 100), optimal_lambda))
    quality = [_psnr(tikhonov_exact(noisy, float(lam)), clean) for lam in grid]
    write_csv("outputs/lab4_experiment_lambda_curve.csv", ["lambda", "PSNR_dB"],
              zip(grid, quality))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(grid, quality, label="PSNR(lambda)")
    ax.scatter([r[0] for r in rows], [r[2] for r in rows], color="black", zorder=3)
    ax.axvline(optimal_lambda, color="red", linestyle="--",
               label=f"lambda*={optimal_lambda:.6f}")
    ax.set(xlabel="lambda", ylabel="PSNR, dB", title="Lab 4: influence of lambda")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("outputs/lab4_experiment_lambda.png", dpi=150)
    plt.close(fig)
    return rows


def run_experiments(project):
    Path("outputs").mkdir(exist_ok=True)
    print("\nLAB 4 - REQUIRED EXPERIMENTS")
    lam, calls = find_optimal_lambda(project.clean, project.noisy)
    print(f"Optimal lambda = {lam:.9f}; search calls = {calls}")
    conditioning = experiment_large_condition_number(project.clean, project.noisy, lam)
    unstable = experiment_unstable_step(project.noisy, lam)
    influence = experiment_lambda_influence(project.clean, project.noisy, lam)
    print("Reference construction and plotting are separate from solver counts.")
    print("LAB 4 EXPERIMENTS COMPLETE")
    return {"lambda": lam, "lambda_calls": calls, "conditioning": conditioning,
            "unstable": unstable, "influence": influence}


def main():
    run_experiments(prepare_project())


if __name__ == "__main__":
    main()
