"""IFT quote sensitivities and reverse-mode differentiation of the HW CN price."""

from functools import lru_cache

import numpy as np
from scipy.special import ndtr

from quantproj.rates.european import _boundary, coupon_weights, observables


def interpolation_weights(nodes, times):
    times = np.asarray(times)
    nodes = np.asarray(nodes)
    i = np.clip(np.searchsorted(nodes, times) - 1, 0, len(nodes) - 2)
    w = (times - nodes[i]) / (nodes[i + 1] - nodes[i])
    eye = np.eye(len(nodes))
    return (1 - w[..., None]) * eye[i] + w[..., None] * eye[i + 1]


def ois_ift(market):
    curve = market.curve
    return np.linalg.inv(np.stack([_ois_row(curve, s) for s in market.swaps]))


def _ois_row(curve, contract):
    f, ann, df = observables(contract, curve.discount)
    weights = interpolation_weights(curve.nodes, contract.dates)[:, 1:]
    dw = df[:, None] * weights
    return (dw[0] - dw[-1] - f * contract.accrual @ dw[1:]) / ann


def normal_density(z):
    return np.exp(-(np.asarray(z) ** 2) / 2) / np.sqrt(2 * np.pi)


def european_partials(curve, contract, variance, a):
    """Derivative with respect to log discount factors and exercise-date variance."""
    _, _, df = observables(contract, curve.discount)
    dt = contract.dates[1:] - contract.dates[0]
    b = -np.expm1(-a * dt) / a
    w = coupon_weights(contract)
    if variance <= 1e-14:
        exercise = df[0] - w @ df[1:] > 0.0
        return np.r_[df[0], -w * df[1:]] * exercise, 0.0
    vol = b * np.sqrt(variance)
    y = float(_boundary(np.log(df[1:] / df[0]) - vol**2 / 2, vol, w))
    dp = np.r_[df[0] * ndtr(-y), -df[1:] * w * ndtr(-y - vol)]
    dv = (w * df[1:] * normal_density(y + vol)) @ b / (2 * np.sqrt(variance))
    return dp, float(dv)


def calibration_ift(market, fit, a):
    """Solve the per-expiry equation differential without differentiating its root iterations."""
    if fit.fallback or np.any(fit.sigma2 <= 1e-12):
        raise ValueError("The unconstrained calibration differential is unavailable at an active constraint.")
    kq = ois_ift(market)
    dk, dv = [], []
    for s, v, w in zip(market.strip, market.vols, fit.w):
        f, ann, df = observables(s, market.curve.discount)
        h = (f - s.fixed) / (v * np.sqrt(s.dates[0]))
        z = ndtr(h)
        da = -(s.fixed * z) + v * np.sqrt(s.dates[0]) * normal_density(h)
        quoted = np.r_[z, s.accrual * da]
        quoted[-1] -= z
        dp, dpdw = european_partials(market.curve, s, w, a)
        weights = interpolation_weights(market.curve.nodes, s.dates)[:, 1:]
        dk.append(((quoted * df - dp) @ weights) @ kq / dpdw)
        dv.append(ann * np.sqrt(s.dates[0]) * normal_density(h) / dpdw)
    return kq, np.asarray(dk), np.diag(dv)


@lru_cache
def _aad_kernel(n, steps):
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    def _solve(lower, diagonal, upper, rhs):
        return jax.lax.linalg.tridiagonal_solve(lower, diagonal, upper, rhs[:, None])[:, 0]

    solve = jax.custom_vjp(_solve)

    def forward(lower, diagonal, upper, rhs):
        x = _solve(lower, diagonal, upper, rhs)
        return x, (lower, diagonal, upper, x)

    def backward(state, cotangent):
        lower, diagonal, upper, x = state
        db = _solve(jnp.r_[0.0, upper[:-1]], diagonal, jnp.r_[lower[1:], 0.0], cotangent)
        return (-db * jnp.r_[0.0, x[:-1]], -db * x, -db * jnp.r_[x[1:], 0.0], db)

    solve.defvjp(forward, backward)

    def value(logs, w, dates, accrual, k, a):
        intervals = jnp.diff(jnp.r_[0.0, dates[:-1]])
        decay = jnp.exp(-2 * a * intervals)
        sigma2 = jnp.maximum(
            (w - decay * jnp.r_[0.0, w[:-1]]) * 2 * a / (-jnp.expm1(-2 * a * intervals)), 0.0
        )
        width = jnp.maximum(7 * jnp.sqrt(jnp.maximum(w[-1], 1e-16)), 0.015)
        x = jnp.linspace(-width, width, n)
        h = x[1] - x[0]
        stencil_idx = jnp.arange(len(dates))

        def back(v, items):
            i, dt, vol, variance = items
            maturity = dates[i]
            b = -jnp.expm1(-a * jnp.maximum(dates - maturity, 0.0)) / a
            b = b - b[-1]
            ratio = jnp.exp(logs - logs[-1] - x[:, None] * b[None, :] - 0.5 * b * b * variance)
            pay = ratio[:, i] - ratio[:, -1] - k * (ratio * (accrual * (stencil_idx > i))).sum(axis=1)
            active = (pay > v) & (pay > 0.0)
            v = jnp.maximum(v, jnp.maximum(pay, 0.0))
            drift = -a * x
            diff = vol / (2 * h * h)
            lo, hi = diff - drift / (2 * h), diff + drift / (2 * h)
            bad = (lo < 0) | (hi < 0)
            lo = jnp.where(bad, diff + jnp.maximum(-drift, 0) / h, lo)
            hi = jnp.where(bad, diff + jnp.maximum(drift, 0) / h, hi)
            lo = lo.at[0].set(0.0).at[-1].set(2 * diff + jnp.maximum(-drift[-1], 0) / h)
            hi = hi.at[-1].set(0.0).at[0].set(2 * diff + jnp.maximum(drift[0], 0) / h)
            diag = -lo - hi

            def step(u, time, theta):
                lower = -theta * time * lo
                m = 1 - theta * time * diag
                r = -theta * time * hi
                rhs = u + (1 - theta) * time * (
                    diag * u + jnp.r_[0.0, lo[1:] * u[:-1]] + jnp.r_[hi[:-1] * u[1:], 0.0]
                )
                return solve(lower, m, r, rhs)

            v = step(step(v, dt / (2 * steps), 1.0), dt / (2 * steps), 1.0)
            v = jax.lax.fori_loop(0, steps - 1, lambda _, y: step(y, dt / steps, 0.5), v)
            return v, active

        v, regions = jax.lax.scan(
            back, jnp.zeros(n), (jnp.arange(len(w))[::-1], intervals[::-1], sigma2[::-1], w[::-1])
        )
        return jnp.exp(logs[-1]) * v[n // 2], regions

    return jax.jit(jax.value_and_grad(value, argnums=(0, 1), has_aux=True))


def aad_price(curve, fit, target, a, n=39, steps=4):
    """Return price, log-bond and state-variance adjoints, and exercise mask."""
    if len(target.dates) != len(fit.w) + 1 or not np.allclose(target.dates[:-1], fit.times, atol=1e-10):
        raise ValueError("AAD target and the co-terminal calibration expiries must align.")
    logs = curve.log_discount(target.dates)
    (value, mask), (dlog, dw) = _aad_kernel(n, steps)(
        logs, fit.w, target.dates, np.r_[0.0, target.accrual], target.fixed, a
    )
    return float(value), np.asarray(dlog), np.asarray(dw), np.asarray(mask)


def quote_greeks(market, fit, target, a, n=39, steps=4):
    """Reverse-mode PV gradients chained through the exact implicit market equations."""
    value, dlog, dw, mask = aad_price(market.curve, fit, target, a, n, steps)
    kq, kappa, vega = calibration_ift(market, fit, a)
    bond_weights = interpolation_weights(market.curve.nodes, target.dates)[:, 1:]
    return value, dlog @ bond_weights @ kq + dw @ kappa, dw @ vega, mask
