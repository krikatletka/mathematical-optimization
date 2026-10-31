"""Лабораторна робота №4: градієнтні методи відновлення зображення."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn


def grad_xy(image):
    gx = np.zeros_like(image)
    gy = np.zeros_like(image)
    gx[:, :-1] = image[:, 1:] - image[:, :-1]
    gy[:-1, :] = image[1:, :] - image[:-1, :]
    return gx, gy


def divergence(px, py):
    result = np.zeros_like(px)
    result[:, 0] += px[:, 0]
    result[:, 1:-1] += px[:, 1:-1] - px[:, :-2]
    result[:, -1] -= px[:, -2]
    result[0, :] += py[0, :]
    result[1:-1, :] += py[1:-1, :] - py[:-2, :]
    result[-1, :] -= py[-2, :]
    return result


def laplacian(image):
    return divergence(*grad_xy(image))


def energy(image, noisy, lam):
    gx, gy = grad_xy(image)
    return 0.5 * float(np.sum((image - noisy) ** 2)) + \
        0.5 * lam * float(np.sum(gx * gx + gy * gy))


def energy_grad(image, noisy, lam):
    return (image - noisy) - lam * laplacian(image)


def tikhonov_exact(noisy, lam):
    height, width = noisy.shape
    iy = np.arange(height)[:, None]
    ix = np.arange(width)[None, :]
    eigenvalues = (2.0 - 2.0 * np.cos(np.pi * iy / height)) + \
                  (2.0 - 2.0 * np.cos(np.pi * ix / width))
    spectrum = dctn(noisy, type=2, norm="ortho")
    return idctn(spectrum / (1.0 + lam * eigenvalues), type=2, norm="ortho")


def _mse(a, b):
    return float(np.mean((a - b) ** 2))


def _psnr(a, b):
    value = _mse(a, b)
    return float("inf") if value == 0 else 10.0 * math.log10(1.0 / value)


def _save_image(image, path):
    data = np.uint8(np.rint(np.clip(image, 0.0, 1.0) * 255.0))
    Image.fromarray(data, mode="L").save(path)


def check_operators(shape=(128, 128), seed=168081):
    rng = np.random.default_rng(seed)
    u = rng.normal(size=shape)
    px = rng.normal(size=shape)
    py = rng.normal(size=shape)
    gx, gy = grad_xy(u)
    left = float(np.sum(gx * px + gy * py))
    right = -float(np.sum(u * divergence(px, py)))
    return abs(left - right) / max(1.0, abs(left), abs(right))


def check_gradient(noisy, lam, seed=168081):
    rng = np.random.default_rng(seed)
    image = rng.random(noisy.shape)
    direction = rng.normal(size=noisy.shape)
    direction /= np.linalg.norm(direction)
    analytic = float(np.sum(energy_grad(image, noisy, lam) * direction))
    step = 1e-6
    numeric = (energy(image + step * direction, noisy, lam) -
               energy(image - step * direction, noisy, lam)) / (2.0 * step)
    return abs(analytic - numeric) / max(1.0, abs(analytic), abs(numeric))


def _golden_lambda(function, start=1.4817, eps=1e-4):
    # Спочатку локалізуємо мінімум методом Свенна.
    x0 = max(float(start), 1e-6)
    delta = 0.2
    f0 = function(x0)
    right = x0 + delta
    fr = function(right)
    left = max(1e-6, x0 - delta)
    fl = function(left)
    if fr >= f0 and fl >= f0:
        a, b = left, right
    else:
        if fl < fr:
            delta = -delta
            previous, current, fc = x0, left, fl
        else:
            previous, current, fc = x0, right, fr
        for power in range(1, 30):
            candidate = max(1e-6, current + delta * (2**power))
            fn = function(candidate)
            if fn >= fc or candidate <= 1e-6:
                a, b = sorted((previous, candidate))
                break
            previous, current, fc = current, candidate, fn
        else:
            a, b = sorted((previous, current))

    tau = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = a + (1.0 - tau) * (b - a)
    x2 = a + tau * (b - a)
    f1, f2 = function(x1), function(x2)
    while b - a > eps:
        if f1 <= f2:
            b, x2, f2 = x2, x1, f1
            x1 = a + (1.0 - tau) * (b - a)
            f1 = function(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + tau * (b - a)
            f2 = function(x2)
    return (a + b) / 2.0


def gradient_method(noisy, lam, lipschitz, kind="gd", beta=None, step=None,
                    max_iter=200, tol=1e-6, reference=None):
    h = float(step if step is not None else 1.0 / lipschitz)
    image = noisy.copy()
    previous = image.copy()
    trial = h
    n_f = n_grad = 0
    rows = []
    converged = False

    for k in range(1, max_iter + 1):
        if kind == "nag":
            momentum = (k - 1) / (k + 2) if beta is None else beta
            point = image + momentum * (image - previous)
            gradient = energy_grad(point, noisy, lam)
            n_grad += 1
            grad_norm = float(np.linalg.norm(gradient))
            new_image = point - h * gradient
        else:
            gradient = energy_grad(image, noisy, lam)
            n_grad += 1
            grad_norm = float(np.linalg.norm(gradient))
            if kind == "armijo":
                local_step = trial
                current_energy = energy(image, noisy, lam)
                n_f += 1
                gradient2 = float(np.sum(gradient * gradient))
                for _ in range(40):
                    candidate = image - local_step * gradient
                    candidate_energy = energy(candidate, noisy, lam)
                    n_f += 1
                    if candidate_energy <= current_energy - 1e-4 * local_step * gradient2:
                        break
                    local_step *= 0.5
                new_image = candidate
                trial = local_step * 1.5
                h = local_step
            elif kind == "hb":
                new_image = image - h * gradient + beta * (image - previous)
            else:
                new_image = image - h * gradient

        current_energy = energy(image, noisy, lam)
        n_f += 1
        distance = (float(np.linalg.norm(image - reference))
                    if reference is not None else float("nan"))
        rows.append((k - 1, current_energy, grad_norm, h, n_f, n_grad, distance))

        if grad_norm <= tol:
            converged = True
            break
        previous, image = image, new_image

    return {
        "image": image,
        "iterations": len(rows) - 1,
        "n_f": n_f,
        "n_grad": n_grad,
        "energy": energy(image, noisy, lam),
        "grad": rows[-1][2],
        "converged": converged,
        "rows": rows,
    }


def _write_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["k", "E", "grad_norm", "step", "Nf", "Ngrad", "distance"])
        writer.writerows(rows)


def run_lab4(clean, noisy, lambda0=1.4817):
    Path("outputs").mkdir(exist_ok=True)
    lambda_calls = 0

    def lambda_objective(lam):
        nonlocal lambda_calls
        lambda_calls += 1
        return _mse(tikhonov_exact(noisy, lam), clean)

    lam = _golden_lambda(lambda_objective, lambda0)
    reference = tikhonov_exact(noisy, lam)
    e_star = energy(reference, noisy, lam)
    mu = 1.0
    lipschitz = 1.0 + 8.0 * lam
    root_l = math.sqrt(lipschitz)
    beta_hb = ((root_l - 1.0) / (root_l + 1.0)) ** 2
    beta_nag = (root_l - 1.0) / (root_l + 1.0)

    adjoint_error = check_operators(noisy.shape)
    gradient_error = check_gradient(noisy, lam)
    results = {
        "Gradient": gradient_method(noisy, lam, lipschitz, "gd", reference=reference),
        "Armijo": gradient_method(noisy, lam, lipschitz, "armijo", reference=reference),
        "Heavy ball optimal": gradient_method(
            noisy, lam, lipschitz, "hb", beta=beta_hb, reference=reference),
        "Heavy ball beta=0.9": gradient_method(
            noisy, lam, lipschitz, "hb", beta=0.9, reference=reference),
        "Nesterov convex": gradient_method(
            noisy, lam, lipschitz, "nag", beta=None, reference=reference),
        "Nesterov strongly convex": gradient_method(
            noisy, lam, lipschitz, "nag", beta=beta_nag, reference=reference),
    }

    for name, result in results.items():
        filename = name.lower().replace(" ", "_").replace("=", "_")
        _write_rows(f"outputs/lab4_{filename}.csv", result["rows"])

    plt.figure(figsize=(9, 5))
    for name, result in results.items():
        values = [max(row[1] - e_star, 1e-16) for row in result["rows"]]
        plt.semilogy(values, label=name)
    plt.xlabel("iteration k")
    plt.ylabel("E(I_k) - E*")
    plt.title("Lab 4: convergence by energy")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("outputs/lab4_convergence_f.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    for name, result in results.items():
        values = [max(row[6], 1e-16) for row in result["rows"]]
        plt.semilogy(values, label=name)
    plt.xlabel("iteration k")
    plt.ylabel("||I_k - I*||")
    plt.title("Lab 4: convergence by distance")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("outputs/lab4_convergence_dist.png", dpi=150)
    plt.close()

    best = results["Nesterov strongly convex"]["image"]
    _save_image(best, "outputs/04_gradient.png")

    # Обов'язковий дослід: вплив lambda на PSNR.
    lambda_grid = np.geomspace(0.01, 10.0, 60)
    psnr_grid = [_psnr(tikhonov_exact(noisy, value), clean) for value in lambda_grid]
    plt.figure(figsize=(8, 5))
    plt.semilogx(lambda_grid, psnr_grid)
    plt.axvline(lam, color="red", ls="--", label=f"lambda*={lam:.6f}")
    plt.xlabel("lambda")
    plt.ylabel("PSNR, dB")
    plt.title("Lab 4: influence of lambda")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/lab4_lambda_psnr.png", dpi=150)
    plt.close()

    print("\nLAB 4 - GRADIENT METHODS")
    print(f"lambda* = {lam:.9f}, search calls = {lambda_calls}")
    print(f"L = {lipschitz:.9f}, mu = {mu:.1f}, kappa = {lipschitz / mu:.6f}")
    print(f"Adjoint check error = {adjoint_error:.3e}")
    print(f"Gradient check error = {gradient_error:.3e}")
    print(f"Exact energy E* = {e_star:.9f}")
    print("+-----------------------------+------------+------+-------+-----------+")
    print("| Method                      | Iterations | Nf   | Ngrad | Converged |")
    print("+-----------------------------+------------+------+-------+-----------+")
    for name, result in results.items():
        print(f"| {name:<27} | {result['iterations']:<10} | "
              f"{result['n_f']:<4} | {result['n_grad']:<5} | "
              f"{str(result['converged']):<9} |")
    print("+-----------------------------+------------+------+-------+-----------+")
    print(f"Restored PSNR = {_psnr(best, clean):.6f} dB")
    print("Saved image = outputs/04_gradient.png")
    print("Saved plots = outputs/lab4_convergence_f.png, "
          "outputs/lab4_convergence_dist.png, outputs/lab4_lambda_psnr.png")
    print("LAB 4 COMPLETE")
    return {"lambda": lam, "L": lipschitz, "E_star": e_star,
            "results": results, "image": best}
