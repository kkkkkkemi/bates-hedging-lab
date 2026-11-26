"""Observable quote generation, no-static-arbitrage filtering and calibration."""

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import differential_evolution, least_squares, minimize_scalar

from quantproj.models import Bates
from quantproj.pricing import black
from quantproj.pricing.fourier import cos_price

NAMES = ("v0", "kappa", "theta", "xi", "rho")
BOUNDS = np.array([[0.001, 0.45], [0.12, 16], [0.003, 0.35], [0.025, 1.8], [-0.97, 0.97]])
JUMP_BOUNDS = np.array([[0.02, 3.0], [-0.25, 0.15], [0.01, 0.35]])
START = Bates(v0=0.05, kappa=1.6, theta=0.045, xi=0.32, rho=-0.55, intensity=0)


@dataclass
class Surface:
    spot: float
    rates: float
    times: np.ndarray
    log_moneyness: np.ndarray
    strikes: np.ndarray
    clean: np.ndarray
    observed: np.ndarray
    clean_iv: np.ndarray
    iv: np.ndarray
    mask: np.ndarray
    half_spread_iv: np.ndarray
    half_spread_price: np.ndarray
    noise: np.ndarray

    def quote_or_smile(self, maturity, strike):
        row, col = np.where(
            (abs(self.times[:, None] - maturity) < 1e-12) & (abs(self.strikes - strike) < 1e-8) & self.mask
        )
        return float(self.iv[row[0], col[0]]) if len(row) else self.smile(maturity, strike)

    def smile(self, maturity, strike, mode="fitted", exclude=None):
        """Interpolate smile nodes in forward log moneyness and total variance."""
        kk = np.log(strike / (self.spot * np.exp(self.rates * maturity)))
        values, dates = [], []
        for i, t in enumerate(self.times):
            keep = np.ones(len(self.log_moneyness), bool) if mode == "clean" else self.mask[i].copy()
            if exclude is not None:
                target_k, target_t = exclude
                keep &= ~((abs(t - target_t) < 1e-10) & (np.abs(self.strikes[i] - target_k) < 1e-7))
            if not keep.any():
                continue
            x = self.log_moneyness[keep]
            selected = self.clean_iv if mode == "clean" else self.iv
            y = selected[i, keep]
            if mode == "raw":
                estimate = y[np.argmin(abs(x - kk))]
            elif mode == "fitted" and len(x) >= 3:
                # Local quadratic fit with distance weights.
                off = x - kk
                basis = np.column_stack((np.ones(len(x)), off, off**2))
                w = 1 / (1 + (off / 0.07) ** 2)
                estimate = np.linalg.lstsq(basis * w[:, None], y * w, rcond=None)[0][0]
                if estimate <= 0 or kk < x.min() or kk > x.max():
                    estimate = np.interp(kk, x, y)
            else:
                estimate = np.interp(kk, x, y)
            values.append(estimate**2 * t)
            dates.append(t)
        if not dates:
            return np.nan
        return np.sqrt(
            np.interp(
                maturity,
                dates,
                values,
                left=values[0] * maturity / dates[0],
                right=values[-1] * maturity / dates[-1],
            )
            / maturity
        )


def noise_scale(k, t, atm):
    t = np.maximum(t, 7 / 365)
    return 0.0015 / np.sqrt(t) * np.minimum(1 + np.abs(k) / (2 * atm * np.sqrt(t)), 3)


def static_mask(iv, calls, spots, strikes, dates, rate, log_k):
    """Deterministic ATM-outward acceptance with strike and calendar constraints.

    Strike checks apply to each observed slice. Calendar checks compare
    forward-normalized call prices on the shared log-moneyness grid,
    equivalent to monotone total implied variance at common nodes.
    """
    mask = np.zeros_like(iv, bool)
    order = np.argsort(abs(log_k), kind="stable")
    for i, t in enumerate(dates):
        disc = np.exp(-rate * t)
        for j in order:
            eps = 1e-9 * spots
            call = calls[i, j]
            if not (
                np.isfinite(iv[i, j])
                and iv[i, j] > 0
                and max(spots - strikes[i, j] * disc, 0) - eps <= call <= spots + eps
            ):
                continue
            prior = np.flatnonzero(mask[:i, j])
            if len(prior) and iv[i, j] ** 2 * t + 1e-12 < iv[prior[-1], j] ** 2 * dates[prior[-1]]:
                continue
            selected = np.sort(np.r_[np.flatnonzero(mask[i]), j])
            slopes = np.diff(calls[i, selected]) / np.diff(strikes[i, selected])
            if np.any(slopes > 1e-10) or np.any(slopes < -disc - 1e-10) or np.any(np.diff(slopes) < -1e-10):
                continue
            mask[i, j] = True
    return mask


def generate_surface(p, spot, variance, times, log_k, rng, noise=1.0, modes=128):
    times = np.asarray(times, dtype=float)
    log_k = np.asarray(log_k, dtype=float)
    strikes = spot * np.exp(p.rate * times[:, None] + log_k)
    clean = np.array([cos_price(p, spot, k, t, variance=variance, n=modes) for k, t in zip(strikes, times)])
    clean_iv = black.implied_vol(clean, spot, strikes, times[:, None], p.rate)
    atm = np.array([np.interp(0, log_k, x) for x in clean_iv])
    sd = noise_scale(log_k, times[:, None], np.maximum(atm[:, None], 1e-8))
    shock = (
        noise
        * sd
        * (
            np.sqrt(0.8) * rng.standard_normal(clean.shape)
            + np.sqrt(0.2) * rng.standard_normal((len(times), 1))
        )
    )
    proposed = clean_iv + shock
    observed = black.price(spot, strikes, times[:, None], np.where(proposed > 0, proposed, np.nan), p.rate)
    iv = black.implied_vol(observed, spot, strikes, times[:, None], p.rate)
    mask = static_mask(iv, observed, spot, strikes, times, p.rate, log_k)
    spread = 2.5758293035489004 * sd
    price_spread = black.greeks(spot, strikes, times[:, None], clean_iv, p.rate)["vega"] * spread
    return Surface(
        spot,
        p.rate,
        times,
        log_k,
        strikes,
        clean,
        observed,
        clean_iv,
        np.where(mask, iv, np.nan),
        mask,
        spread,
        price_spread,
        shock,
    )


def physical(p, bates=False):
    x = [getattr(p, name) for name in NAMES]
    return np.asarray(x + [p.intensity, p.jump_mean, p.jump_vol] if bates else x)


def encode(x):
    y = x.copy()
    y[[0, 1, 2, 3]] = np.log(y[[0, 1, 2, 3]])
    y[4] = np.arctanh(y[4])
    if len(y) > 5:
        y[[5, 7]] = np.log(y[[5, 7]])
    return y


def decode(x):
    y = x.copy()
    y[[0, 1, 2, 3]] = np.exp(np.clip(y[[0, 1, 2, 3]], -14, 5))
    y[4] = np.clip(np.tanh(y[4]), -0.99999, 0.99999)
    if len(y) > 5:
        y[[5, 7]] = np.exp(np.clip(y[[5, 7]], -14, 3))
    return y


def _model(x, rate):
    return Bates(*x, rate=rate) if len(x) == 8 else Bates(*x, intensity=0, rate=rate)


@dataclass
class Fit:
    params: Bates
    rmse_bp: float
    success: bool
    boundary: bool
    residual: np.ndarray
    jacobian: np.ndarray
    nfev: int


def calibrate(
    surface,
    start=None,
    *,
    clean=False,
    matched=False,
    global_iterations=0,
    max_evaluations=70,
    state_only=False,
    seed=0,
    modes=96,
    fixed=None,
    bates=False,
    tolerance_bp=250,
):
    """Vega-weighted price residuals approximate implied-volatility residuals."""
    p0 = start or replace(START, rate=surface.rates, intensity=0.45 if bates else 0)
    initial = physical(p0, bates)
    bounds = np.vstack((BOUNDS, JUMP_BOUNDS)) if bates else BOUNDS
    mask = surface.mask if not clean or matched else np.isfinite(surface.clean_iv)
    prices = surface.clean if clean else surface.observed
    iv = surface.clean_iv if clean else surface.iv
    vega = black.greeks(surface.spot, surface.strikes, surface.times[:, None], iv, surface.rates)["vega"]
    keep = mask & (vega > 1e-4)
    n = int(keep.sum())
    free = [0] if state_only else list(range(len(initial)))
    if fixed is not None:
        fixed_index, fixed_value = fixed
        initial[fixed_index] = fixed_value
        free.remove(fixed_index)
    if n < len(free):
        empty = np.full(prices.shape, np.nan)
        return Fit(p0, np.inf, False, False, empty, np.empty((0, len(initial))), 0)
    start_encoded = encode(initial)

    def assemble(y):
        result = start_encoded.copy()
        result[free] = y
        return decode(result)

    rows = [
        (i, t, surface.strikes[i, keep[i]], prices[i, keep[i]], vega[i, keep[i]])
        for i, t in enumerate(surface.times)
        if keep[i].any()
    ]

    def residual(x):
        pp = _model(x, surface.rates)
        return np.concatenate(
            [(cos_price(pp, surface.spot, k, t, n=modes) - price) / vg for _, t, k, price, vg in rows]
        )

    y0 = start_encoded[free]
    if global_iterations:
        limits = list(zip(encode(bounds[:, 0])[free], encode(bounds[:, 1])[free]))
        opt = differential_evolution(
            lambda y: np.mean(residual(assemble(y)) ** 2),
            limits,
            maxiter=global_iterations,
            popsize=5,
            seed=seed,
            polish=False,
            x0=np.clip(y0, *np.asarray(limits).T),
        )
        y0 = opt.x
    if state_only:

        def scalar_loss(value):
            vec = initial.copy()
            vec[0] = value
            return np.mean(residual(vec) ** 2)

        state = minimize_scalar(
            scalar_loss, bounds=tuple(bounds[0]), method="bounded", options={"xatol": 1e-8}
        )
        x = initial.copy()
        x[0] = state.x
        success, nfev = state.success, state.nfev
        jacobian = np.empty((n, 0))
    else:
        opt = least_squares(
            lambda y: residual(assemble(y)),
            y0,
            method="lm",
            max_nfev=max_evaluations,
            ftol=2e-7,
            xtol=2e-7,
            gtol=2e-7,
        )
        x = assemble(opt.x)
        scales = np.r_[x[:4], 1 - x[4] ** 2, np.array([x[5], 1, x[7]]) if bates else []]
        jacobian = opt.jac / np.maximum(scales[free], 1e-12)
        success, nfev = opt.success, opt.nfev
    margin = (bounds[free, 1] - bounds[free, 0]) * 1e-4
    boundary = bool(np.any((x[free] <= bounds[free, 0] + margin) | (x[free] >= bounds[free, 1] - margin)))
    fitted = _model(x, surface.rates)
    residual_iv = np.full(prices.shape, np.nan)
    for i, t, k, _, _ in rows:
        value = cos_price(fitted, surface.spot, k, t, n=modes)
        residual_iv[i, keep[i]] = (
            black.implied_vol(value, surface.spot, k, t, surface.rates) - iv[i, keep[i]]
        ) * 1e4
    rmse = float(np.sqrt(np.nanmean(residual_iv**2)))
    return Fit(
        fitted,
        rmse,
        bool(success and np.isfinite(rmse) and rmse <= tolerance_bp),
        boundary,
        residual_iv,
        jacobian,
        nfev,
    )


def hessian_diagnostics(fit, surface=None, modes=96):
    """Hessian of half the vega-weighted sum of squared residuals in physical coordinates."""
    x = physical(fit.params)
    scales = np.r_[x[:4], 1 - x[4] ** 2]
    gn = fit.jacobian.T @ fit.jacobian
    h = gn.copy()
    if surface is not None:
        spots, strikes, dates = surface.spot, surface.strikes, surface.times
        vega = black.greeks(spots, strikes, dates[:, None], surface.clean_iv, surface.rates)["vega"]
        selected = vega > 1e-4

        def objective(point):
            model = _model(point, surface.rates)
            prices = np.array([cos_price(model, spots, k, t, n=modes) for k, t in zip(strikes, dates)])
            diff = (prices[selected] - surface.clean[selected]) / vega[selected]
            return 0.5 * np.dot(diff, diff)

        base = objective(x)
        for i, ei in enumerate(np.eye(len(x)) * scales * 2e-4):
            h[i, i] = (objective(x + ei) + objective(x - ei) - 2 * base) / ei[i] ** 2
            for j, ej in enumerate(np.eye(len(x)) * scales * 2e-4):
                if j < i:
                    h[i, j] = h[j, i] = (
                        objective(x + ei + ej)
                        - objective(x - ei + ej)
                        - objective(x + ei - ej)
                        + objective(x - ei - ej)
                    ) / (4 * ei[i] * ej[j])
    standardized = scales[:, None] * h * scales
    values, vectors = np.linalg.eigh(standardized)
    return {
        "hessian": h,
        "gauss_newton_hessian": gn,
        "scaled_hessian": standardized,
        "eigenvalues": values,
        "eigenvectors": vectors,
        "condition": np.linalg.cond(standardized),
    }


def daily_update(surface, previous, day, *, weekly=False, bates=False, modes=96, tolerance=250):
    fit = calibrate(
        surface,
        previous,
        state_only=weekly and day % 5 != 0,
        bates=bates,
        tolerance_bp=tolerance,
        modes=modes,
    )
    failed = not fit.success or fit.boundary
    return (previous if failed else fit.params), failed, fit.rmse_bp


def filter_variance(surface, previous, dt, modes=96):
    """One-dimensional state update with a CIR moment prediction and a quote likelihood."""
    decay = np.exp(-previous.kappa * dt)
    predicted = previous.theta + (previous.v0 - previous.theta) * decay
    transition_var = (
        previous.xi**2
        / previous.kappa
        * (previous.v0 * decay * (1 - decay) + previous.theta * (1 - decay) ** 2 / 2)
    )
    valid = surface.mask & (surface.half_spread_price > 1e-5)
    if not valid.any():
        return replace(previous, v0=float(predicted))
    sd = surface.half_spread_price / 2.5758293035489004

    def loss(v):
        terms = np.array(
            [
                cos_price(previous, surface.spot, k, t, variance=v, n=modes)
                for k, t in zip(surface.strikes, surface.times)
            ]
        )
        diff = (terms[valid] - surface.observed[valid]) / sd[valid]
        return np.dot(diff, diff) + (v - predicted) ** 2 / max(transition_var, 1e-7)

    fit = minimize_scalar(
        loss,
        bounds=(
            max(BOUNDS[0, 0], predicted - 8 * np.sqrt(transition_var)),
            min(BOUNDS[0, 1], predicted + 8 * np.sqrt(transition_var)),
        ),
        method="bounded",
    )
    return replace(previous, v0=float(fit.x if fit.success else predicted))
