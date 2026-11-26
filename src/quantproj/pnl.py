"""Frozen-position explain and 250-day MAR32 attribution metrics."""

import numpy as np
from scipy.stats import ks_2samp, spearmanr

from quantproj.calibration import generate_surface
from quantproj.models import simulate
from quantproj.pricing import black
from quantproj.pricing.fourier import cos_price

MODES = ("clean", "raw", "fitted")


def zone(spearman, ks):
    if not np.isfinite(spearman) or not np.isfinite(ks):
        return "undefined"
    if spearman > 0.80 and ks < 0.09:
        return "green"
    if spearman < 0.70 or ks > 0.12:
        return "red"
    return "amber"


def pla(hpl, rtpl):
    if len(hpl) != 250 or len(rtpl) != 250 or not np.all(np.isfinite(np.r_[hpl, rtpl])):
        raise ValueError("PLA requires 250 consecutive finite observations.")
    rho = float(spearmanr(hpl, rtpl).statistic)
    # The two 250-observation empirical CDFs have steps of exactly 1/250.
    ks = round(250 * float(ks_2samp(hpl, rtpl, method="asymp").statistic)) / 250
    return rho, ks, zone(rho, ks)


def atm_iv(surface, t, excluded):
    return {
        method: surface.smile(t, surface.spot * np.exp(surface.rates * t), method, exclude=excluded)
        for method in MODES
    }


def explain(s0, s1, k, t0, t1, iv0, iv1, r, stock):
    ds, dv = s1 - s0, iv1 - iv0
    g = black.greeks(s0, k, t0, iv0, r)
    contributions = np.array(
        [
            (stock - g["delta"]) * ds,
            -g["vega"] * dv,
            -g["theta"] * (t0 - t1),
            -0.5 * g["gamma"] * ds**2,
            -g["vanna"] * ds * dv,
            -0.5 * g["volga"] * dv**2,
        ]
    )
    v0 = g["value"]
    a = black.price(s1, k, t0, iv0, r)
    b = black.price(s1, k, t0, iv1, r)
    c = black.price(s1, k, t1, iv1, r)
    sequential = np.array([stock * ds - (a - v0), -(b - a), -(c - b)])
    return contributions, sequential


def run_attribution(p, paths, seed, times, log_k, *, noise=1, modes=96, days=250, path_ids=None):
    """Each path uses its own quote seed, so any subset reproduces the full loop."""
    dt = 1 / 252
    tape = simulate(100, days * dt, p, days, paths, seed, scheme="qe")
    rows, metrics = [], []
    for a in range(paths) if path_ids is None else path_ids:
        rng = np.random.default_rng(seed + 5879 + a)
        k, t_expiry, holding = 100 * np.exp(p.rate * 63 * dt), 63, 0.0
        previous = {}
        for day in range(days + 1):
            spot, v = tape.spot[day, a], tape.variance[day, a]
            t = (t_expiry - day) * dt
            date_nodes = np.unique(np.r_[times, t, 63 * dt])
            target_node = np.log(k / (spot * np.exp(p.rate * t)))
            nodes = np.unique(np.round(np.r_[log_k, target_node], 12))
            surface = generate_surface(p, spot, v, date_nodes, nodes, rng, noise, modes)
            target = float(cos_price(p, spot, k, t, variance=v, n=modes))
            today = atm_iv(surface, t, (k, t))
            if day:
                ds = spot - previous["s"]
                hpl = holding * ds - (target - previous["target"])
                risk_values = {
                    method: holding * ds
                    - (
                        black.price(spot, k, t, today[method], p.rate)
                        - black.price(previous["s"], k, previous["t"], previous[method], p.rate)
                    )
                    for method in MODES
                }
                # Explain with the trader's own fitted-smile BS marks; PLA uses ATM-only inputs.
                iv = surface.quote_or_smile(t, k)
                parts, ordered = explain(
                    previous["s"], spot, k, previous["t"], t, previous["smile"], iv, p.rate, holding
                )
                full = ordered.sum()
                rows.append(
                    [
                        a,
                        day,
                        int(tape.jump_count[day - 1, a] > 0),
                        hpl,
                        *risk_values.values(),
                        full,
                        parts[:3].sum(),
                        parts.sum(),
                        full - parts[:3].sum(),
                        full - parts.sum(),
                        hpl - full,
                        *parts,
                        *ordered,
                    ]
                )
            if day and day % 21 == 0:
                t_expiry = day + 63
                t = 63 * dt
                k = spot * np.exp(p.rate * t)
                target = float(cos_price(p, spot, k, t, variance=v, n=modes))
                today = atm_iv(surface, t, (k, t))
            smile = surface.quote_or_smile(t, k)
            holding = float(black.greeks(spot, k, t, smile, p.rate)["delta"])
            previous = {"s": spot, "target": target, "t": t, "smile": smile, **today}
        series = np.asarray(rows[-days:])
        if days >= 250:
            for j, method in enumerate(MODES):
                metrics.append([a, method, *pla(series[-250:, 3], series[-250:, 4 + j])])
    names = [
        "path",
        "day",
        "jump",
        "hpl",
        "rtpl_clean",
        "rtpl_raw",
        "rtpl_fitted",
        "bs_full",
        "taylor_first",
        "taylor_second",
        "first_residual",
        "second_residual",
        "market_model_gap",
        "delta",
        "vega",
        "theta",
        "gamma",
        "vanna",
        "volga",
        "spot_step",
        "iv_step",
        "time_step",
    ]
    return names, rows, metrics
