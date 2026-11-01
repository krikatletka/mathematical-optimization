"""Лабораторна робота №3: білатеральний фільтр, Нелдер—Мід і Пауелл."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


LO = np.array([0.3, 0.01], dtype=float)
HI = np.array([5.0, 1.0], dtype=float)


def _mse(a, b):
    return float(np.mean((a - b) ** 2))


def _psnr(a, b):
    value = _mse(a, b)
    return float("inf") if value == 0 else 10.0 * math.log10(1.0 / value)


def _save_image(image, path):
    data = np.uint8(np.rint(np.clip(image, 0.0, 1.0) * 255.0))
    Image.fromarray(data, mode="L").save(path)


def bilateral_filter(img, sigma_spatial, sigma_range):
    sigma_spatial = max(float(sigma_spatial), 1e-3)
    sigma_range = max(float(sigma_range), 1e-3)
    radius = int(min(max(round(2.0 * sigma_spatial), 1), 5))
    height, width = img.shape
    padded = np.pad(img, radius, mode="reflect")
    numerator = np.zeros_like(img, dtype=float)
    denominator = np.zeros_like(img, dtype=float)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = padded[
                radius + dy:radius + dy + height,
                radius + dx:radius + dx + width,
            ]
            spatial = math.exp(
                -(dx * dx + dy * dy) / (2.0 * sigma_spatial**2)
            )
            photometric = np.exp(
                -((shifted - img) ** 2) / (2.0 * sigma_range**2)
            )
            weight = spatial * photometric
            numerator += weight * shifted
            denominator += weight

    return numerator / denominator


class Objective2:
    def __init__(self, noisy, clean):
        self.noisy = noisy
        self.clean = clean
        self.calls = 0

    def __call__(self, theta):
        self.calls += 1
        point = np.asarray(theta, dtype=float)
        clamped = np.clip(point, LO, HI)
        restored = bilateral_filter(self.noisy, clamped[0], clamped[1])
        gap = point - clamped
        return _mse(restored, self.clean) + 1000.0 * float(gap @ gap)


def nelder_mead(function, x0, edge, tol=1e-7, max_iter=200):
    x0 = np.asarray(x0, dtype=float)
    simplex = np.array([
        x0,
        x0 + np.array([edge, 0.0]),
        x0 + np.array([0.0, 0.4 * edge]),
    ])
    values = np.array([function(x) for x in simplex])
    rows = []
    path = []

    for iteration in range(max_iter):
        order = np.argsort(values)
        simplex, values = simplex[order], values[order]
        best, middle, worst = simplex
        fb, fm, fw = values
        path.append(best.copy())
        size = max(np.linalg.norm(simplex[i] - best) for i in (1, 2))
        rows.append((iteration, *best, fb, size, "sort"))

        if size <= tol and fw - fb <= tol:
            break

        center = (best + middle) / 2.0
        reflected = center + (center - worst)
        fr = function(reflected)

        if fr < fb:
            expanded = center + 2.0 * (reflected - center)
            fe = function(expanded)
            if fe < fr:
                simplex[2], values[2], action = expanded, fe, "expansion"
            else:
                simplex[2], values[2], action = reflected, fr, "reflection"
        elif fr < fm:
            simplex[2], values[2], action = reflected, fr, "reflection"
        else:
            if fr < fw:
                contracted = center + 0.5 * (reflected - center)
                limit = fr
            else:
                contracted = center + 0.5 * (worst - center)
                limit = fw
            fc = function(contracted)
            if fc < limit:
                simplex[2], values[2], action = contracted, fc, "contraction"
            else:
                simplex[1] = best + 0.5 * (simplex[1] - best)
                simplex[2] = best + 0.5 * (simplex[2] - best)
                values[1] = function(simplex[1])
                values[2] = function(simplex[2])
                action = "reduction"
        rows[-1] = (*rows[-1][:-1], action)

    order = np.argsort(values)
    return {
        "x": simplex[order[0]].copy(),
        "f": float(values[order[0]]),
        "iterations": len(rows) - 1,
        "rows": rows,
        "path": np.asarray(path),
    }

def _alpha_bounds(x, direction):
    """Допустимый интервал alpha: LO <= x + alpha*d <= HI."""
    left = -float("inf")
    right = float("inf")

    for i in range(len(x)):
        if abs(direction[i]) < 1e-15:
            continue

        a1 = (LO[i] - x[i]) / direction[i]
        a2 = (HI[i] - x[i]) / direction[i]

        left = max(left, min(a1, a2))
        right = min(right, max(a1, a2))

    return left, right


def _golden_line(function, x, direction, eps=1e-4):
    """Метод Свенна + золотий переріз уздовж заданого напрямку."""
    x = np.asarray(x, dtype=float)
    direction = np.asarray(direction, dtype=float)

    if np.linalg.norm(direction) < 1e-15:
        return 0.0

    alpha_min, alpha_max = _alpha_bounds(x, direction)

    def phi(alpha):
        alpha = float(np.clip(alpha, alpha_min, alpha_max))
        point = x + alpha * direction
        return function(point)

    # Метод Свенна
    alpha0 = float(np.clip(0.0, alpha_min, alpha_max))
    step = 0.1

    f0 = phi(alpha0)

    alpha_right = float(np.clip(alpha0 + step, alpha_min, alpha_max))
    alpha_left = float(np.clip(alpha0 - step, alpha_min, alpha_max))

    f_right = phi(alpha_right)
    f_left = phi(alpha_left)

    if f_right >= f0 and f_left >= f0:
        left, right = sorted((alpha_left, alpha_right))

    else:
        if f_left < f_right:
            step = -step
            previous = alpha0
            current = alpha_left
            f_current = f_left
        else:
            previous = alpha0
            current = alpha_right
            f_current = f_right

        left = min(previous, current)
        right = max(previous, current)

        for power in range(1, 30):
            candidate = current + step * (2 ** power)
            candidate = float(np.clip(candidate, alpha_min, alpha_max))

            # Досягнуто межі допустимої області.
            if abs(candidate - current) < 1e-15:
                left, right = sorted((previous, current))
                break

            f_candidate = phi(candidate)

            if f_candidate >= f_current:
                left, right = sorted((previous, candidate))
                break

            previous = current
            current = candidate
            f_current = f_candidate
        else:
            left, right = sorted((previous, current))

    if right - left <= eps:
        return (left + right) / 2.0

    # Метод золотого перерізу
    tau = (math.sqrt(5.0) - 1.0) / 2.0

    a = left
    b = right

    x1 = a + (1.0 - tau) * (b - a)
    x2 = a + tau * (b - a)

    f1 = phi(x1)
    f2 = phi(x2)

    while b - a > eps:
        if f1 <= f2:
            b = x2
            x2 = x1
            f2 = f1

            x1 = a + (1.0 - tau) * (b - a)
            f1 = phi(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2

            x2 = a + tau * (b - a)
            f2 = phi(x2)

    return (a + b) / 2.0

def powell(function, x0, tol=1e-4, max_cycles=20):
    x = np.clip(np.asarray(x0, dtype=float), LO, HI)

    # Початкові координатні напрямки.
    directions = np.eye(2, dtype=float)

    rows = []
    path = [x.copy()]

    for cycle in range(1, max_cycles + 1):
        cycle_start = x.copy()
        f_start = function(x)

        gains = []

        # Пошук уздовж усіх поточних напрямків.
        for direction in directions:
            f_before = function(x)

            alpha = _golden_line(
                function,
                x,
                direction,
                eps=tol,
            )

            x = np.clip(x + alpha * direction, LO, HI)
            f_after = function(x)

            gains.append(f_before - f_after)
            path.append(x.copy())

        # Новий напрямок Пауелла.
        shift = x - cycle_start
        shift_norm = np.linalg.norm(shift)

        if shift_norm > 1e-15:
            new_direction = shift.copy()

            # ВАЖЛИВО: додатковий лінійний пошук
            # уздовж нового напрямку.
            alpha = _golden_line(
                function,
                x,
                new_direction,
                eps=tol,
            )

            x = np.clip(
                x + alpha * new_direction,
                LO,
                HI,
            )

            path.append(x.copy())

            # Замінюємо напрямок, який дав найбільше зменшення.
            worst_direction = int(np.argmax(gains))
            directions[worst_direction] = new_direction

        f_end = function(x)

        rows.append(
            (
                cycle,
                x[0],
                x[1],
                f_end,
                np.linalg.norm(x - cycle_start),
            )
        )

        # Обидві умови мають виконуватися одночасно.
        point_change = np.linalg.norm(x - cycle_start)
        function_change = abs(f_start - f_end)

        if point_change <= tol and function_change <= tol:
            break

        # Перезапуск напрямків кожні три цикли.
        if cycle % 3 == 0:
            directions = np.eye(2, dtype=float)

    return {
        "x": x.copy(),
        "f": float(function(x)),
        "cycles": len(rows),
        "rows": rows,
        "path": np.asarray(path),
    }

def _write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def run_lab3(clean, noisy, gaussian_sigma, simplex_edge=0.6976):
    Path("outputs").mkdir(exist_ok=True)
    sigma_s0 = float(np.clip(2.0 * gaussian_sigma, 0.5, 4.0))
    sigma_r0 = float(np.clip(0.15 + 0.1 * 0.1449, 0.02, 0.9))
    x0 = np.array([sigma_s0, sigma_r0])

    objective_nm = Objective2(noisy, clean)
    nm = nelder_mead(objective_nm, x0, simplex_edge)
    nm["nf"] = objective_nm.calls

    objective_powell = Objective2(noisy, clean)
    pw = powell(objective_powell, x0)
    pw["nf"] = objective_powell.calls

    best = nm if nm["f"] <= pw["f"] else pw
    theta = np.clip(best["x"], LO, HI)
    restored = bilateral_filter(noisy, theta[0], theta[1])
    restored_psnr = _psnr(restored, clean)
    _save_image(restored, "outputs/03_bilateral.png")

    _write_csv("outputs/lab3_nelder_mead.csv",
               ["k", "sigma_s", "sigma_r", "f", "simplex_size", "action"], nm["rows"])
    _write_csv("outputs/lab3_powell.csv",
               ["cycle", "sigma_s", "sigma_r", "f", "shift_norm"], pw["rows"])

    # Контурний графік обчислюємо на помірній сітці, щоб запуск не був надто довгим.
    ss = np.linspace(0.5, 3.2, 28)
    sr = np.linspace(0.05, 0.55, 28)
    zz = np.empty((len(sr), len(ss)))
    probe = Objective2(noisy, clean)
    for i, range_sigma in enumerate(sr):
        for j, spatial_sigma in enumerate(ss):
            zz[i, j] = probe([spatial_sigma, range_sigma])
    plt.figure(figsize=(8, 6))
    plt.contour(ss, sr, zz, levels=18)
    plt.plot(nm["path"][:, 0], nm["path"][:, 1], "o-", ms=3, label="Nelder-Mead")
    plt.plot(pw["path"][:, 0], pw["path"][:, 1], "s-", ms=3, label="Powell")
    plt.xlabel("sigma_spatial")
    plt.ylabel("sigma_range")
    plt.title("Lab 3: optimization paths")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/lab3_contours.png", dpi=150)
    plt.close()

    print("\nLAB 3 - NELDER-MEAD AND POWELL")
    print(f"Initial point = ({x0[0]:.6f}; {x0[1]:.6f})")
    print(f"Nelder-Mead: ({nm['x'][0]:.6f}; {nm['x'][1]:.6f}), "
          f"iterations = {nm['iterations']}, Nf = {nm['nf']}")
    print(f"Powell:      ({pw['x'][0]:.6f}; {pw['x'][1]:.6f}), "
          f"cycles = {pw['cycles']}, Nf = {pw['nf']}")
    print(f"Best parameters = ({theta[0]:.6f}; {theta[1]:.6f})")
    print(f"Restored PSNR = {restored_psnr:.6f} dB")
    print("Saved image = outputs/03_bilateral.png")
    print("Saved plot = outputs/lab3_contours.png")
    print("LAB 3 COMPLETE")
    return {"nelder_mead": nm, "powell": pw, "theta": theta,
            "image": restored, "psnr": restored_psnr}
