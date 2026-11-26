"""Spread pricing, dynamic Jacobi correlation, and two-futures hedging."""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri
from scipy.stats import qmc


@dataclass(frozen=True)
class SpreadModel:
    sigma1: float = 0.32
    sigma2: float = 0.27
    kappa: float = 3.0
    mean: float = 0.45
    xi: float = 0.58
    rate: float = 0.02
    linkage: float = 0.0

    def __post_init__(self):
        if abs(self.mean) >= 1 or min(self.sigma1, self.sigma2, self.kappa, self.xi) <= 0:
            raise ValueError("Invalid spread parameters.")
        if self.kappa * (1 - abs(self.mean)) < self.xi**2:
            raise ValueError("Jacobi boundary condition is violated.")
        if abs(self.linkage) >= 1:
            raise ValueError("Correlation-driver loading must be between -1 and 1.")

    def average(self, rho, t):
        return self.mean + (rho - self.mean) * (-np.expm1(-self.kappa * t)) / (self.kappa * t)


def _black_forward(f, g, t, sd, r):
    total = np.maximum(sd * np.sqrt(t), 1e-15)
    d = np.log(f / g) / total + total / 2
    return np.exp(-r * t) * (f * ndtr(d) - g * ndtr(d - total))


def margrabe(f, g, t, rho, p):
    vol = np.sqrt(p.sigma1**2 + p.sigma2**2 - 2 * rho * p.sigma1 * p.sigma2)
    return _black_forward(f, g, t, vol, p.rate)


def kirk(f, g, k, t, rho, p):
    b = g / (g + k)
    vol = np.sqrt(p.sigma1**2 + (b * p.sigma2) ** 2 - 2 * rho * b * p.sigma1 * p.sigma2)
    return _black_forward(f, g + k, t, vol, p.rate)


def bjerksund_stensland(f, g, k, t, rho, p):
    """Price the payoff on the log-linear approximate exercise region."""
    b = g / (g + k)
    a, c = p.sigma1, p.sigma2
    w = np.maximum(np.sqrt(t * (a * a + b * b * c * c - 2 * b * rho * a * c)), 1e-15)
    base = (np.log(f / (g + k)) + (-a * a + b * c * c) * t / 2) / w
    return np.exp(-p.rate * t) * (
        f * ndtr(base + (a * a - b * rho * a * c) * t / w)
        - g * ndtr(base + (rho * a * c - b * c * c) * t / w)
        - k * ndtr(base)
    )


def constant_price(f, g, k, t, rho, p, order=36):
    """Condition on the second futures leg and integrate the lognormal first leg."""
    if k == 0:
        return margrabe(f, g, t, rho, p)
    nodes, weights = hermgauss(order)
    rho = np.asarray(rho)[..., None]
    f, g = np.asarray(f)[..., None], np.asarray(g)[..., None]
    z = np.sqrt(2) * nodes
    s1, s2 = p.sigma1 * np.sqrt(t), p.sigma2 * np.sqrt(t)
    second = g * np.exp(-s2 * s2 / 2 + s2 * z)
    first = f * np.exp(-((s1 * rho) ** 2) / 2 + s1 * rho * z)
    values = _black_forward(first, second + k, t, p.sigma1 * np.sqrt(np.maximum(1 - rho * rho, 0)), p.rate)
    return values @ (weights / np.sqrt(np.pi))


def _project_rho(x):
    return 1 - np.abs((x + 1) % 4 - 2)


def _evolve(f, g, rho, z, dt, p):
    z1, z2, z3 = z
    z2_linked = rho * z1 + np.sqrt(np.maximum(1 - rho**2, 0)) * z2
    # W_rho shares W_1. The loading representation gives a PSD matrix.
    zr = p.linkage * z1 + np.sqrt(1 - p.linkage**2) * z3
    factor = np.sqrt(dt)
    f = f * np.exp(-(p.sigma1**2) * dt / 2 + p.sigma1 * factor * z1)
    g = g * np.exp(-(p.sigma2**2) * dt / 2 + p.sigma2 * factor * z2_linked)
    rho = _project_rho(
        rho + p.kappa * (p.mean - rho) * dt + p.xi * np.sqrt(np.maximum(1 - rho * rho, 0)) * factor * zr
    )
    return f, g, rho


def driver_matrix(rho, link):
    return np.array([[1, rho, link], [rho, 1, rho * link], [link, rho * link, 1]])


def normals(steps, budget, seed, dimension=3):
    exponent = int(np.ceil(np.log2(budget)))
    u = qmc.Sobol(steps * dimension, scramble=True, seed=seed).random_base2(exponent)
    return ndtri(np.clip(u, 1e-12, 1 - 1e-12)).T.reshape(steps, dimension, -1)


def dynamic_price(f, g, rho, k, t, p, z):
    """Conditional price by randomized QMC, Rao–Blackwellized in the orthogonal case."""
    dt = t / len(z)
    corr = np.full(z.shape[-1], rho)
    a, b = np.full_like(corr, f), np.full_like(corr, g)
    if p.linkage == 0:
        integral = np.zeros_like(corr)
        for _, _, zr in z:
            integral += corr / len(z)
            corr = _project_rho(
                corr
                + p.kappa * (p.mean - corr) * dt
                + p.xi * np.sqrt(np.maximum(1 - corr * corr, 0) * dt) * zr
            )
        return float(np.mean(constant_price(f, g, k, t, integral, p, 20)))
    for zz in z:
        a, b, corr = _evolve(a, b, corr, zz, dt, p)
    return float(np.exp(-p.rate * t) * np.mean(np.maximum(a - b - k, 0)))


def implied_correlation(value, f, g, k, t, p):
    low, high = kirk(f, g, k, t, 1.0, p), kirk(f, g, k, t, -1.0, p)
    if value < low or value > high:
        return (1.0 if value < low else -1.0), True
    return brentq(lambda rho: kirk(f, g, k, t, rho, p) - value, -1, 1), False


def sensitivities(fn, f, g, rho, h=0.1, hr=0.002):
    up, down = min(rho + hr, 1.0), max(rho - hr, -1.0)
    return np.array(
        [
            (fn(f + h, g, rho) - fn(f - h, g, rho)) / (2 * h),
            (fn(f, g + h, rho) - fn(f, g - h, rho)) / (2 * h),
            (fn(f, g, up) - fn(f, g, down)) / (up - down),
            (fn(f + h, g + h, rho) - fn(f - h, g + h, rho) - fn(f + h, g - h, rho) + fn(f - h, g - h, rho))
            / (4 * h * h),
        ]
    )


def local_risk_hedge(gradient, f, g, rho, p):
    loading = np.array([p.sigma1 * f, p.sigma2 * g])
    cov = driver_matrix(rho, p.linkage)
    if np.linalg.eigvalsh(cov)[0] < -1e-10:
        raise ValueError("Instantaneous driver matrix is not positive semidefinite.")
    corr_vol = p.xi * np.sqrt(np.maximum(1 - rho * rho, 0))
    matrix = loading[:, None] * cov[:2, :2] * loading
    cross = loading * cov[:2, 2] * corr_vol
    projection = np.linalg.pinv(matrix, rcond=1e-10) @ cross
    residual = gradient[2] ** 2 * max(corr_vol**2 - cross @ projection, 0)
    return gradient[:2] + projection * gradient[2], residual


def _spread_path(job):
    """One hedged sample path; its outer shocks are drawn in advance by the caller."""
    p, shocks, steps, inner_paths, seed = job
    initial_f, initial_g, k, t = 104.0, 100.0, 8.0, 0.25
    dt = t / steps
    f, g, rho = initial_f, initial_g, p.mean
    margin = np.zeros(len(SPREAD_METHODS))
    clipped = residual = 0.0
    for day in range(steps):
        remaining = t - day * dt
        draws = normals(max(6, min(18, steps - day)), inner_paths, seed + day + 37)

        def dynamic(a, b, c):
            return dynamic_price(a, b, c, k, remaining, p, draws)

        true_mark = dynamic(f, g, rho)
        if day == 0:
            margin[:] = true_mark
        # Fixed reference futures contract held constant across re-marking.
        reference_t = 0.5 - day * dt
        reference = dynamic_price(f, g, rho, 0.0, reference_t, p, draws)
        iv_corr, flag = implied_correlation(reference, f, g, 0.0, reference_t, p)
        clipped += flag
        average = p.average(rho, remaining)

        def exact(a, b, c):
            return constant_price(a, b, k, remaining, average, p)

        def avg_kirk(a, b, c):
            return kirk(a, b, k, remaining, average, p)

        def implied(a, b, c):
            return kirk(a, b, k, remaining, iv_corr, p)

        true_grad = sensitivities(dynamic, f, g, rho)
        projection, unhedgeable = local_risk_hedge(true_grad, f, g, rho, p)
        residual += unhedgeable * dt * np.exp(2 * p.rate * (remaining - dt))
        deltas = np.array(
            [
                true_grad[:2],
                sensitivities(exact, f, g, rho)[:2],
                sensitivities(avg_kirk, f, g, rho)[:2],
                sensitivities(implied, f, g, rho)[:2],
                projection,
            ]
        )
        next_f, next_g, rho = _evolve(f, g, rho, shocks[day], dt, p)
        margin = margin * np.exp(p.rate * dt) + deltas @ np.array([next_f - f, next_g - g])
        f, g = next_f, next_g
    return margin - max(f - g - k, 0), clipped, residual


SPREAD_METHODS = ("dynamic_delta", "constant_exact", "kirk_average", "kirk_implied", "dynamic_local_risk")


def spread_hedge(p, paths, steps, inner_paths, seed, mapper=None):
    """Draw the outer shocks in the serial order, then hedge the paths independently."""
    shocks = np.random.default_rng(seed).standard_normal((paths, steps, 3))
    jobs = [(p, shocks[sample], steps, inner_paths, seed) for sample in range(paths)]
    results = (mapper or (lambda fn, items: [fn(x) for x in items]))(_spread_path, jobs)
    pnl = np.array([row[0] for row in results]).reshape(paths, len(SPREAD_METHODS))
    clipped = np.array([row[1] for row in results], dtype=float)
    residual = np.array([row[2] for row in results], dtype=float)
    return SPREAD_METHODS, pnl, clipped / steps, residual
