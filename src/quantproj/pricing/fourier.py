"""Fang–Oosterlee COS and Carr–Madan FFT pricing."""

import numpy as np

from quantproj.models import Bates


def _cos_rows(p: Bates, s, k, t: float, variance, n: int, width: float):
    ivar = p.theta * t + (variance - p.theta) * (-np.expm1(-p.kappa * t)) / p.kappa
    jump_var = p.intensity * t * (p.jump_vol**2 + p.jump_mean**2)
    # Wider than a Gaussian interval, with a jump-tail allowance for short expiries.
    half = width * np.maximum(
        np.sqrt(np.maximum(ivar + jump_var, 1e-12)), (p.xi**2 * p.theta * t**3 / 3) ** 0.25
    )
    half += 3 * p.jump_vol * min(1.0, np.sqrt(p.intensity * t) * 4)
    center = (p.rate - p.jump_compensator + p.intensity * p.jump_mean) * t - ivar / 2
    a, b = center - half, center + half
    u = np.pi * np.arange(n) / (2 * half[:, None])
    phi_a, phi_d = p.affine(u, t)
    modes = np.exp(phi_a + phi_d * variance[:, None] - 1j * u * a[:, None])
    modes[:, 0] *= 0.5
    log_k = np.log(k / s)
    cutoff = np.clip(log_k, a, b)

    # Price the OTM side directly (and use parity for the ITM side). On the
    # call side use F-K + put if the direct call loses cancellation at wide L.
    def integral(left, right, put):
        ul, ur = u * (left - a)[:, None], u * (right - a)[:, None]
        chi = (
            np.exp(right)[:, None] * (np.cos(ur) + u * np.sin(ur))
            - np.exp(left)[:, None] * (np.cos(ul) + u * np.sin(ul))
        ) / (1 + u * u)
        psi = np.empty_like(chi)
        psi[:, 0] = right - left
        psi[:, 1:] = (np.sin(ur[:, 1:]) - np.sin(ul[:, 1:])) / u[:, 1:]
        coeff = k[:, None] * psi - s[:, None] * chi if put else s[:, None] * chi - k[:, None] * psi
        weight = np.exp(-p.rate * t) / half
        value = weight * np.sum(modes.real * coeff, axis=1)
        return value

    put = integral(a, cutoff, True)
    parity = s - k * np.exp(-p.rate * t)
    return put, parity


def cos_price(p: Bates, s, k, t: float, *, variance=None, put=False, n=256, width=11.0):
    s, k, v = np.broadcast_arrays(
        np.asarray(s, float), np.asarray(k, float), p.v0 if variance is None else variance
    )
    if t < 0 or n < 2 or np.any(s <= 0) or np.any(k <= 0) or np.any(v < 0):
        raise ValueError("Invalid COS inputs.")
    if t == 0:
        return np.maximum((k - s) if put else (s - k), 0)
    flat_s, flat_k, flat_v = s.ravel(), k.ravel(), v.ravel()
    out = np.empty(flat_s.size)
    for i in range(0, out.size, 1536):
        batch = slice(i, i + 1536)
        pv, parity = _cos_rows(p, flat_s[batch], flat_k[batch], t, flat_v[batch], n, width)
        lower = np.maximum(-parity if put else parity, 0)
        upper = flat_k[batch] * np.exp(-p.rate * t) if put else flat_s[batch]
        out[batch] = np.clip(pv if put else pv + parity, lower, upper)
    return out.reshape(s.shape)


def cos_greeks(p: Bates, spot: float, strike: float, t: float, n=256, width=11.0):
    """Analytic differentiation on the fixed log-return COS grid."""
    v = p.v0
    ivar = p.theta * t + (v - p.theta) * (-np.expm1(-p.kappa * t)) / p.kappa
    jump_var = p.intensity * t * (p.jump_vol**2 + p.jump_mean**2)
    half = width * max(np.sqrt(ivar + jump_var), (p.xi**2 * p.theta * t**3 / 3) ** 0.25)
    half += 3 * p.jump_vol * min(1.0, np.sqrt(p.intensity * t) * 4)
    center = (p.rate - p.jump_compensator + p.intensity * p.jump_mean) * t - ivar / 2
    a, b = center - half, center + half
    x = float(np.clip(np.log(strike / spot), a, b))
    u = np.arange(n) * np.pi / (b - a)
    aa, dd = p.affine(u, t)
    modes = np.exp(aa + dd * v - 1j * u * a)
    modes[0] *= 0.5
    z = u * (x - a)
    chi = (np.exp(x) * (np.cos(z) + u * np.sin(z)) - np.exp(a)) / (1 + u * u)
    psi = np.r_[x - a, np.sin(z[1:]) / u[1:]]
    factor = np.exp(-p.rate * t) / half
    return {
        "delta": 1 - factor * np.dot(modes.real, chi),
        "gamma": factor * strike / spot**2 * np.dot(modes.real, np.cos(z)) if a < x < b else 0.0,
        "variance": factor * np.dot((modes * dd).real, strike * psi - spot * chi),
    }


def fft_call(p: Bates, s: float, strikes, t: float, *, alpha=1.25, n=4096, eta=0.13):
    if alpha <= 0 or eta <= 0 or n < 16 or n % 2 or t <= 0:
        raise ValueError("FFT needs positive damping, spacing and maturity, and even n >= 16.")
    # Guard the positive moment before evaluating the shifted characteristic function.
    moment = p.cf(-1j * (1 + alpha), t)
    if not np.isfinite(moment) or moment.real <= 0 or abs(moment.imag) > 1e-8 * moment.real:
        raise ValueError("Damped moment is unavailable for these inputs.")
    j = np.arange(n)
    u = eta * j
    spacing = 2 * np.pi / (n * eta)
    half = np.pi / eta
    k_log = -half + spacing * j
    denom = (alpha + 1j * u) * (alpha + 1 + 1j * u)
    integrand = np.exp(-p.rate * t) * p.cf(u - 1j * (alpha + 1), t) / denom
    weights = (3 + (-1) ** (j + 1) - (j == 0)) / 3  # Carr–Madan Simpson weights
    fft = np.fft.fft(np.exp(1j * half * u) * integrand * eta * weights)
    values = s * np.exp(-alpha * k_log) * fft.real / np.pi
    from scipy.interpolate import CubicSpline

    targets = np.log(np.asarray(strikes) / s)
    if np.any(np.abs(targets) > min(half / 2, 5)):
        raise ValueError("Strikes outside the supported FFT interpolation region.")
    local = np.abs(k_log) <= min(half, 6)
    return CubicSpline(k_log[local], values[local])(targets)
