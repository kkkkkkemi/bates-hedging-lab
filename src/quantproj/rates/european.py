"""Normal quotes and Gaussian European payer/receiver swaption pricing."""

from functools import lru_cache

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import brentq
from scipy.special import ndtr

from quantproj.rates.models import root_cov


@lru_cache
def normals(order: int):
    x, w = hermgauss(order)
    return x * np.sqrt(2), w / np.sqrt(np.pi)


def normal_call(forward, strike, expiry, vol):
    diff = np.asarray(forward) - strike
    std = np.asarray(vol) * np.sqrt(max(expiry, 0))
    h = diff / np.maximum(std, 1e-16)
    return np.where(
        std > 0, diff * ndtr(h) + std * np.exp(-h * h / 2) / np.sqrt(2 * np.pi), np.maximum(diff, 0)
    )


def normal_iv(value, forward, strike, expiry, annuity):
    normalized = value / annuity
    if expiry <= 0 or normalized < max(forward - strike, 0) - 1e-12:
        raise ValueError("Invalid normal option premium.")
    if normalized <= max(forward - strike, 0) + 1e-14:
        return 0.0
    upper = max(0.02, 4 * normalized / np.sqrt(expiry))
    while normal_call(forward, strike, expiry, upper) < normalized:
        upper *= 2
    return brentq(lambda v: normal_call(forward, strike, expiry, v) - normalized, 0, upper, xtol=1e-13)


def observables(contract, discount):
    df = discount(contract.dates)
    annuity = float(df[1:] @ contract.accrual)
    return (df[0] - df[-1]) / annuity, annuity, df


def coupon_weights(contract):
    weights = contract.fixed * contract.accrual
    weights[-1] += 1.0
    return weights


def _boundary(base, beta, weights):
    """Exercise threshold along the integration factor, including negative coupons."""
    lower = np.full(base.shape[:-1], -12.0)
    upper = np.full(base.shape[:-1], 12.0)

    def residual(y):
        log_bonds = np.clip(base - y[..., None] * beta, -650, 650)
        if np.any(weights < 0):
            # Dividing by the final bond leaves a strictly increasing function for K < 0.
            scaled = np.exp(np.clip(log_bonds - log_bonds[..., -1:], -650, 650))
            return np.exp(np.clip(-log_bonds[..., -1], -650, 650)) - np.sum(weights * scaled, axis=-1)
        return 1 - np.sum(weights * np.exp(log_bonds), axis=-1)

    for _ in range(10):
        left, right = residual(lower) > 0, residual(upper) < 0
        if not (np.any(left) or np.any(right)):
            break
        lower = np.where(left, lower * 2, lower)
        upper = np.where(right, upper * 2, upper)
    if base.ndim == 1:
        return brentq(lambda y: float(residual(np.array(y))), float(lower), float(upper), xtol=1e-11)
    for _ in range(36):
        mid = (lower + upper) / 2
        exercise = residual(mid) > 0
        upper = np.where(exercise, mid, upper)
        lower = np.where(exercise, lower, mid)
    return (lower + upper) / 2


def jamshidian_from_bonds(discounts, loads, variance, weights):
    """Jamshidian decomposition into expiry-forward zero-coupon puts."""
    forward = discounts[1:] / discounts[0]
    if variance < 1e-16:
        return max(discounts[0] - weights @ discounts[1:], 0.0)
    if weights[-1] <= 0:
        return discounts[0] - weights @ discounts[1:]
    std = np.sqrt(variance)
    d = loads * std
    base = np.log(forward) - d * d / 2
    y = float(_boundary(base, d, weights))
    if np.any(weights < 0):
        # Mixed coupons use the same unique boundary and signed truncated bond moments.
        return float(discounts[0] * (ndtr(-y) - weights @ (forward * ndtr(-y - d))))
    bond_strikes = np.exp(np.clip(base - y * d, -700, 700))
    puts = bond_strikes * ndtr(-y) - forward * ndtr(-y - d)
    return max(float(discounts[0] * (weights @ puts)), 0.0)


def jamshidian(model, t, state, contract, receiver=False):
    maturity = contract.dates[0]
    df = model.bond(t, contract.dates, state)
    weights = coupon_weights(contract)
    b = model.loads(contract.dates[1:] - maturity)[:, 0]
    variance = model.moments(t, maturity)[0, 0]
    value = jamshidian_from_bonds(df, b, variance, weights)
    return value - (df[0] - weights @ df[1:]) if receiver else value


def _g2_payer_and_factor_delta(model, t, state, contract, order):
    """Conditional exercise probabilities and truncated bond moments."""
    expiry = contract.dates[0]
    df = model.bond(t, contract.dates, state)
    weights = coupon_weights(contract)
    forward = df[1:] / df[0]
    if expiry <= t + 1e-12 or weights[-1] <= 0:
        expected_exercise = float(df[0] > weights @ df[1:])
        bond_terms = df[1:] * expected_exercise
    else:
        cov = model.moments(t, expiry)[:2, :2]
        b = model.loads(contract.dates[1:] - expiry)
        sd0 = np.sqrt(cov[0, 0])
        regression = cov[0, 1] / sd0
        conditional = np.sqrt(max(cov[1, 1] - regression**2, 0))
        z, probability = normals(order)
        exposure = b[:, 0] * sd0 + b[:, 1] * regression
        base = np.log(forward) - np.sum((b @ cov) * b, axis=1) / 2 - z[:, None] * exposure
        beta = b[:, 1] * conditional
        if conditional < 1e-10:
            active = 1 - np.exp(base) @ weights > 0
            expected_exercise = probability @ active
            bond_terms = df[0] * ((probability * active) @ np.exp(base))
        else:
            cutoff = _boundary(base, beta, weights)
            expected_exercise = probability @ ndtr(-cutoff)
            bond_terms = df[0] * (probability @ (np.exp(base + beta**2 / 2) * ndtr(-cutoff[:, None] - beta)))
    exercise_bond = df[0] * expected_exercise
    cashflows = np.r_[exercise_bond, -weights * bond_terms]
    delta = -cashflows @ model.loads(contract.dates - t)
    return max(float(cashflows.sum()), 0.0), delta


def g2_european(model, t, state, contract, order=28, receiver=False):
    """Brigo–Mercurio conditional integral with analytical truncated bond moments."""
    price, _ = _g2_payer_and_factor_delta(model, t, state, contract, order)
    if not receiver:
        return price
    df = model.bond(t, contract.dates, state)
    return price - (df[0] - coupon_weights(contract) @ df[1:])


def european_factor_delta(model, t, state, contract):
    if model.dimension == 2:
        return _g2_payer_and_factor_delta(model, t, state, contract, 28)
    value = jamshidian(model, t, state, contract)
    h = 1e-5
    delta = (jamshidian(model, t, state + h, contract) - jamshidian(model, t, state - h, contract)) / (2 * h)
    return value, np.array([delta])


def european(model, t, state, contract):
    return (
        jamshidian(model, t, state, contract)
        if model.dimension == 1
        else g2_european(model, t, state, contract)
    )


def monte_carlo(model, t, state, contract, n, seed):
    expiry = contract.dates[0]
    mean, cov = model.conditional_forward(t, expiry, state, expiry)
    x = mean + np.random.default_rng(seed).standard_normal((n, model.dimension)) @ root_cov(cov).T
    df = model.bond(expiry, contract.dates[1:], x)
    samples = model.bond(t, expiry, state) * np.maximum(1 - df @ coupon_weights(contract), 0.0)
    return float(samples.mean()), float(samples.std(ddof=1) / np.sqrt(n))
