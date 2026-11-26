"""Black–Scholes under a continuous rate and zero dividend yield."""

import numpy as np
from scipy.special import ndtr


def density(x):
    return np.exp(-0.5 * np.asarray(x) ** 2) / np.sqrt(2 * np.pi)


def greeks(s, k, t, sigma, r=0.02, put=False):
    s, k, t, sigma = np.broadcast_arrays(s, k, t, sigma)
    root = np.sqrt(np.maximum(t, 0))
    w = np.maximum(sigma * root, 1e-15)
    d1 = (np.log(s / k) + r * t) / w + w / 2
    d2 = d1 - w
    disc = np.exp(-r * t)
    sign = -1 if put else 1
    delta = sign * ndtr(sign * d1)
    value = sign * (s * ndtr(sign * d1) - k * disc * ndtr(sign * d2))
    vega = s * density(d1) * root
    theta = -s * density(d1) * sigma / (2 * np.maximum(root, 1e-15)) - sign * r * k * disc * ndtr(sign * d2)
    return {
        "value": value,
        "delta": delta,
        "gamma": density(d1) / (s * w),
        "vega": vega,
        "theta": np.where(t > 0, theta, 0),
        "vanna": -density(d1) * d2 / np.maximum(sigma, 1e-15),
        "volga": vega * d1 * d2 / np.maximum(sigma, 1e-15),
    }


def price(s, k, t, sigma, r=0.02, put=False):
    s, k, t, sigma = np.broadcast_arrays(s, k, t, sigma)
    w = np.maximum(sigma * np.sqrt(np.maximum(t, 0)), 1e-15)
    d1 = (np.log(s / k) + r * t) / w + w / 2
    d2 = d1 - w
    sign = -1 if put else 1
    return sign * (s * ndtr(sign * d1) - k * np.exp(-r * t) * ndtr(sign * d2))


def implied_vol(value, s, k, t, r=0.02, put=False):
    """Vectorized bisection on the OTM premium. NaN denotes an invalid quote."""
    value, s, k, t = np.broadcast_arrays(value, s, k, t)
    strike_pv = k * np.exp(-r * t)
    otm_put = s >= strike_pv
    intrinsic = np.maximum((strike_pv - s) if put else (s - strike_pv), 0)
    premium = value - intrinsic
    cap = np.minimum(s, strike_pv)
    tolerance = 1e-10 * np.maximum(s, k)
    valid = (t > 0) & (premium >= -tolerance) & (premium < cap)
    lo, hi = np.zeros_like(premium, dtype=float), np.full_like(premium, 20.0, dtype=float)
    for _ in range(41):
        mid = (lo + hi) * 0.5
        call = price(s, k, t, mid, r)
        pv = np.where(otm_put, price(s, k, t, mid, r, put=True), call)
        lo = np.where(pv < premium, mid, lo)
        hi = np.where(pv >= premium, mid, hi)
    return np.where(valid, np.where(premium <= 0, 0, (lo + hi) * 0.5), np.nan)
