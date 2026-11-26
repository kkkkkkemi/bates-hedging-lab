"""Co-terminal observations and piecewise HW calibration in state-variance coordinates."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, least_squares

from quantproj.rates.curves import DiscountCurve, Swap, bootstrap, ois_quotes, quote_swaps
from quantproj.rates.european import (
    coupon_weights,
    european,
    jamshidian_from_bonds,
    normal_call,
    normal_iv,
    observables,
)
from quantproj.rates.models import hw


@dataclass
class Market:
    swaps: tuple
    curve: DiscountCurve
    ois: np.ndarray
    strip: tuple
    vols: np.ndarray
    clean_ois: np.ndarray
    clean_vols: np.ndarray
    clean_curve: DiscountCurve
    quotes: list
    deleted: int
    curve_fallback: bool = False


def remaining(swap, clock):
    return tuple(swap.tail(i).shift(clock) for i, t in enumerate(swap.dates[:-1]) if t > clock + 1e-10)


def observe(model, clock, state, asof, swap, rng, curve_noise, vol_noise, *, smile_offsets=(), order=18):
    instruments = quote_swaps(asof)

    def true_df(x):
        return model.bond(clock, np.asarray(x) + clock, state)

    clean_ois = ois_quotes(instruments, true_df)
    # A documented synthetic quote process, independent of true market marks.
    ois = clean_ois + rng.normal(0.0, curve_noise * 1e-4, len(clean_ois))
    clean_curve = bootstrap(instruments, clean_ois)
    curve_fallback = False
    try:
        curve = bootstrap(instruments, ois)
    except ValueError:
        curve_fallback = True
        # Repair failed quote nodes using adjacent observable tenors only.
        repaired = ois.copy()
        for i in range(len(ois)):
            peers = np.arange(len(ois)) != i
            repaired[i] = np.interp(
                instruments[i].dates[-1],
                [s.dates[-1] for j, s in enumerate(instruments) if j != i],
                ois[peers],
            )
            try:
                curve = bootstrap(instruments, repaired)
                ois = repaired
                break
            except ValueError:
                repaired = ois.copy()
        else:
            raise ValueError("The observed OIS snapshot has no admissible single-node repair.") from None
    strip, vols, clean_vols, rows = [], [], [], []
    deleted = 0
    for r in remaining(swap, clock):
        strike = r.fair_rate(curve.discount)
        offsets = sorted(set((*smile_offsets, 0.0)))
        for offset in offsets:
            k = strike + offset * 1e-4
            local = Swap(r.dates, r.accrual, k)
            true = Swap(local.dates + clock, local.accrual, k)
            forward, ann, _ = observables(local, true_df)
            premium = (
                european(model, clock, state, true)
                if model.dimension == 1
                else _g2(model, clock, state, true, order)
            )
            clean = normal_iv(premium, forward, k, local.dates[0], ann)
            v = clean + rng.normal(0.0, vol_noise * 1e-4)
            observed_forward, observed_ann, _ = observables(local, curve.discount)
            quote = float(observed_ann * normal_call(observed_forward, k, local.dates[0], max(v, 0.0)))
            rows.append([local.dates[0], k, clean, v, quote, True])
            if offset == 0.0:
                strip.append(local)
                vols.append(v)
                clean_vols.append(clean)
        smile = rows[-len(offsets) :]
        # Keep ATM first, then admit wings only if price decreases convexly with strike.
        admitted = []
        scale = observables(local, curve.discount)[1]
        for j in sorted(range(len(offsets)), key=lambda x: abs(offsets[x])):
            item = smile[j]
            tentative = sorted([*admitted, j])
            slopes = np.diff([smile[m][4] for m in tentative]) / np.diff([smile[m][1] for m in tentative])
            passes = item[3] > 0 and np.all(slopes <= 1e-10) and np.all(slopes >= -scale - 1e-8)
            passes = passes and np.all(np.diff(slopes) >= -1e-10)
            item[-1] = bool(passes)
            if passes:
                admitted.append(j)
            else:
                deleted += 1
                if offsets[j] == 0:
                    vols[-1] = np.nan
    return Market(
        instruments,
        curve,
        ois,
        tuple(strip),
        np.asarray(vols),
        clean_ois,
        np.asarray(clean_vols),
        clean_curve,
        rows,
        deleted,
        curve_fallback,
    )


def _g2(model, t, state, contract, order):
    from quantproj.rates.european import g2_european

    return g2_european(model, t, state, contract, order)


@dataclass
class Calibration:
    model: object
    w: np.ndarray
    sigma2: np.ndarray
    times: np.ndarray
    residual_bp: np.ndarray
    fallback: bool
    missing: int
    converged: bool


def covariance_map(a, times):
    dt = np.diff(np.r_[0.0, times])
    causal = times[:, None] - times[None, :] >= -1e-12
    decay = np.exp(-2 * a * np.maximum(times[:, None] - times[None, :], 0.0))
    return causal * decay * (-np.expm1(-2 * a * dt)) / (2 * a)


def calibrate(curve, strip, vols, a, *, prior=0.01):
    times = np.asarray([s.dates[0] for s in strip])
    if not len(times):
        raise ValueError("Cannot fit a strip without future expiries.")
    data = []
    for contract, vol in zip(strip, vols):
        f, ann, df = observables(contract, curve.discount)
        t = contract.dates[0]
        load = -np.expm1(-a * (contract.dates[1:] - t)) / a
        vegas = (
            ann
            * np.sqrt(t)
            * np.exp(-0.5 * ((f - contract.fixed) / max(vol * np.sqrt(t), 1e-12)) ** 2)
            / np.sqrt(2 * np.pi)
        )
        target = ann * normal_call(f, contract.fixed, t, vol)
        data.append((f, ann, df, load, coupon_weights(contract), float(target), max(vegas, 1e-8)))
    matrix = covariance_map(a, times)
    valid = np.isfinite(vols) & (vols > 0)
    desired = np.full(len(times), np.nan)
    for i, (_, _, df, b, c, target, _) in enumerate(data):
        if not valid[i]:
            continue

        def diff(w):
            return jamshidian_from_bonds(df, b, w, c) - target

        # A quoted premium with no measurable time value does not identify variance.
        if diff(0.0) >= -1e-13:
            valid[i] = False
            continue
        upper = 0.05
        while diff(upper) < 0 and upper < 20:
            upper *= 2
        if diff(upper) >= 0:
            desired[i] = brentq(diff, 0.0, upper, xtol=7e-13)
        else:
            valid[i] = False
    missing = int((~valid).sum())
    unconstrained = np.linalg.solve(matrix, desired) if not missing else np.full(len(times), np.nan)
    fallback = bool(missing or np.any(unconstrained < -1e-12))
    converged = True
    if fallback and valid.any():

        def error(x):
            curve_w = matrix @ x
            output = []
            for i, (_, _, df, b, c, target, weight) in enumerate(data):
                if valid[i]:
                    output.append((jamshidian_from_bonds(df, b, curve_w[i], c) - target) / weight * 1e4)
            if missing:
                output.extend(0.05 * (np.sqrt(np.maximum(x[~valid], 0)) - prior) * 1e4)
            return np.array(output)

        fitted = least_squares(
            error,
            np.maximum(np.nan_to_num(unconstrained, nan=prior**2), 1e-8),
            bounds=(0, np.inf),
            xtol=1e-9,
            ftol=1e-8,
            max_nfev=60,
        )
        sigma2 = fitted.x
        converged = fitted.success
    elif fallback:
        sigma2 = np.full(len(times), prior**2)
        converged = False
    else:
        sigma2 = np.maximum(unconstrained, 0.0)
    w = matrix @ sigma2
    residuals = np.full(len(times), np.nan)
    for j, (f, ann, df, b, c, _, _) in enumerate(data):
        if valid[j]:
            vv = normal_iv(jamshidian_from_bonds(df, b, w[j], c), f, strip[j].fixed, times[j], ann)
            residuals[j] = (vv - vols[j]) * 1e4
    model = hw(curve, a, np.sqrt(sigma2), times[:-1])
    return Calibration(model, w, sigma2, times, residuals, fallback, missing, converged)


def fit_market(market, a, clean=False):
    if clean:
        return calibrate(market.clean_curve, market.strip, market.clean_vols, a)
    return calibrate(market.curve, market.strip, market.vols, a)
