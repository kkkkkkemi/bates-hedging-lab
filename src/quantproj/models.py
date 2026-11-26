"""Risk-neutral dynamics and log-return characteristic functions."""

from dataclasses import dataclass, replace

import numpy as np
from scipy.special import ndtr


@dataclass(frozen=True)
class Bates:
    v0: float = 0.0576
    kappa: float = 2.03
    theta: float = 0.04
    xi: float = 0.38
    rho: float = -0.7
    intensity: float = 0.59
    jump_mean: float = -0.05
    jump_vol: float = 0.07
    rate: float = 0.02

    def __post_init__(self) -> None:
        values = [
            self.v0,
            self.kappa,
            self.theta,
            self.xi,
            self.rho,
            self.intensity,
            self.jump_mean,
            self.jump_vol,
            self.rate,
        ]
        if not np.all(np.isfinite(values)):
            raise ValueError("Model parameters must be finite.")
        if min(self.v0, self.theta, self.intensity, self.jump_vol) < 0:
            raise ValueError("Variance, jump intensity and jump volatility must be nonnegative.")
        if min(self.kappa, self.xi) <= 0 or abs(self.rho) >= 1:
            raise ValueError("Require positive kappa and xi, and absolute rho below one.")

    @property
    def jump_compensator(self) -> float:
        return float(self.intensity * np.expm1(self.jump_mean + self.jump_vol**2 / 2))

    @property
    def feller_margin(self) -> float:
        return 2 * self.kappa * self.theta - self.xi**2

    def no_jumps(self) -> "Bates":
        return replace(self, intensity=0)

    def affine(self, u, t):
        """Return A, D such that E[exp(iu log(S_T/S_0))] = exp(A + D*v0)."""
        iu = 1j * np.asarray(u)
        b = self.kappa - self.rho * self.xi * iu
        d = np.sqrt(b**2 - self.xi**2 * (iu**2 - iu) + 0j)
        g = (b - d) / (b + d)
        decay_minus_one = np.expm1(-d * t)
        log_ratio = np.log1p(-g * decay_minus_one / (1 - g))
        d_coef = -(b - d) * decay_minus_one / (self.xi**2 * (1 - g * (1 + decay_minus_one)))
        a_coef = self.kappa * self.theta / self.xi**2 * ((b - d) * t - 2 * log_ratio)
        jump_cf_minus_one = np.expm1(iu * self.jump_mean + iu**2 * self.jump_vol**2 / 2)
        a_coef += t * (iu * (self.rate - self.jump_compensator) + self.intensity * jump_cf_minus_one)
        return a_coef, d_coef

    def cf(self, u, t, variance=None):
        a, d = self.affine(u, t)
        return np.exp(a + d * (self.v0 if variance is None else variance))


@dataclass
class Paths:
    spot: np.ndarray
    variance: np.ndarray
    jump_count: np.ndarray


def diffusion_step(log_s, v, dt, p: Bates, zv, zs, scheme: str):
    """Full-truncation Euler or Andersen QE with conditional martingale correction."""
    v_plus = np.maximum(v, 0)
    drift = p.rate - p.jump_compensator
    if scheme == "euler":
        dw = np.sqrt(v_plus * dt)
        x_next = log_s + (drift - v_plus / 2) * dt + dw * (p.rho * zv + np.sqrt(1 - p.rho**2) * zs)
        v_next = v + p.kappa * (p.theta - v_plus) * dt + p.xi * dw * zv
        return x_next, v_next
    if scheme != "qe":
        raise ValueError(f"Unknown scheme {scheme!r}")
    decay = np.exp(-p.kappa * dt)
    m = p.theta * (1 - decay) + v_plus * decay
    s2 = p.xi**2 / p.kappa * (v_plus * decay * (1 - decay) + p.theta * (1 - decay) ** 2 / 2)
    psi = np.maximum(s2 / np.maximum(m, 1e-150) ** 2, 1e-12)
    quadratic = psi <= 1.5
    inv = 2 / np.minimum(psi, 1.5)
    b2 = inv - 1 + np.sqrt(inv * (inv - 1))
    a = m / (1 + b2)
    atom = np.maximum((psi - 1) / (psi + 1), 0)
    beta = (1 - atom) / np.maximum(m, 1e-150)
    u = np.minimum(ndtr(zv), 1 - np.finfo(float).eps)
    v_exp = np.maximum(np.log((1 - atom) / (1 - u)) / beta, 0)
    v_next = np.where(quadratic, a * (np.sqrt(b2) + zv) ** 2, v_exp)
    kr = p.kappa * p.rho / p.xi - 0.5
    k2 = dt * kr / 2 + p.rho / p.xi
    k3 = dt / 2 * (1 - p.rho**2)
    exponent = k2 + k3 / 2
    quad_denom = 1 - 2 * exponent * a
    if np.any(np.where(quadratic, quad_denom <= 0, exponent >= beta)):
        raise ValueError("QE martingale correction moment unavailable. Refine the time grid.")
    quad_denom = np.where(quadratic, quad_denom, 1)
    quad_mgf_log = -0.5 * np.log(quad_denom) + exponent * a * b2 / quad_denom
    exp_mgf = atom + (1 - atom) * beta / np.where(quadratic, beta, beta - exponent)
    log_mgf = np.where(quadratic, quad_mgf_log, np.log(exp_mgf))
    dx = drift * dt - k3 * v_plus / 2 - log_mgf + k2 * v_next + np.sqrt(k3 * (v_plus + v_next)) * zs
    return log_s + dx, v_next


def simulate(
    spot: float,
    t: float,
    p: Bates,
    steps: int,
    paths: int,
    seed: int,
    scheme: str = "qe",
    bs_vol: float | None = None,
) -> Paths:
    """Exact conditional compound-Poisson increments per step."""
    if min(spot, t, steps, paths) <= 0:
        raise ValueError("Require positive spot, time, steps and paths.")
    rng = np.random.default_rng(seed)
    dt = t / steps
    log_s = np.full(paths, np.log(spot))
    v = np.full(paths, p.v0 if bs_vol is None else bs_vol**2)
    s_out = np.empty((steps + 1, paths))
    v_out = np.empty_like(s_out)
    counts = np.zeros((steps, paths), dtype=np.int16)
    s_out[0], v_out[0] = spot, v
    for i in range(steps):
        zv, zs = rng.standard_normal((2, paths))
        if bs_vol is None:
            log_s, v = diffusion_step(log_s, v, dt, p, zv, zs, scheme)
            if p.intensity:
                n = rng.poisson(p.intensity * dt, paths)
                log_s += n * p.jump_mean + np.sqrt(n) * p.jump_vol * rng.standard_normal(paths)
                counts[i] = n
        else:
            log_s += (p.rate - bs_vol**2 / 2) * dt + bs_vol * np.sqrt(dt) * zs
        s_out[i + 1], v_out[i + 1] = np.exp(log_s), np.maximum(v, 0)
    return Paths(s_out, v_out, counts)
