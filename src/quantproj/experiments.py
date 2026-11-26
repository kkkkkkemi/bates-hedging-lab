"""Configuration-driven experiments and machine-readable artifacts."""

import csv
import json
from dataclasses import replace
from time import perf_counter

import numpy as np
from scipy.optimize import brentq

from quantproj import calibration, commodity, greeks, parallel, pnl
from quantproj.hedging.engine import HedgeSettings, decomposition, hedge_bs_control, hedge_market
from quantproj.hedging.events import BrownianJumpTape
from quantproj.models import simulate
from quantproj.pricing import black, monte_carlo
from quantproj.pricing.fourier import cos_greeks, cos_price, fft_call
from quantproj.pricing.pde import adi_grid, bs_discrete_reference, bs_grid, continuous_upout


def write_csv(root, name, headers, rows):
    with (root / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def write_json(root, name, data):
    def clean(node):
        if isinstance(node, np.ndarray):
            return clean(node.tolist())
        if isinstance(node, np.generic):
            return clean(node.item())
        if isinstance(node, float) and not np.isfinite(node):
            return None
        if isinstance(node, dict):
            return {k: clean(v) for k, v in node.items()}
        if isinstance(node, (list, tuple)):
            return [clean(x) for x in node]
        return node

    (root / f"{name}.json").write_text(
        json.dumps(clean(data), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def scenarios(p):
    gap = p.intensity * (0.10**2 - p.jump_vol**2)
    return {
        "base": p,
        "heavy_a": replace(p, jump_vol=0.10),
        "heavy_b": replace(p, jump_vol=0.10, v0=p.v0 - gap, theta=p.theta - gap),
        "no_jump": p.no_jumps(),
    }


def _american_iv(value, s, k, t, r):
    def tree(sigma):
        n = 240
        dt = t / n
        base = np.exp(sigma * np.sqrt(dt))
        prob = (np.exp(r * dt) - 1 / base) / (base - 1 / base)
        if not 0 <= prob <= 1:
            return max(k - s, 0)
        price = s * base ** (2 * np.arange(n + 1) - n)
        cash = np.maximum(k - price, 0)
        for i in range(n, 0, -1):
            price = price[:-1] * base
            cash = np.maximum(k - price, np.exp(-r * dt) * ((1 - prob) * cash[:-1] + prob * cash[1:]))
        return cash[0]

    if value <= max(k - s, 0) or value >= k:
        return np.nan
    return brentq(lambda v: tree(v) - value, 0.005, 4.0)


def pricing_experiment(root, p, cfg, seed):
    n, days = cfg["mc_paths"], cfg["days"]
    k, s, t, b = 100.0, 100.0, days / 252, 125.0
    r, vol = p.rate, 0.24
    h = p.no_jumps()
    rows = []

    def bench(model, product, method, callback, ref, put=False, mc=False):
        start = perf_counter()
        output = callback()
        elapsed = perf_counter() - start
        result = float(np.mean(output))
        se = float(np.std(output, ddof=1) / np.sqrt(len(output))) if mc else np.nan
        if product == "barrier":
            units, error = "notional_bp", 1e4 * (result - ref) / s
        elif product == "american_put":
            units, error = (
                "american_bs_iv_bp",
                1e4 * (_american_iv(result, s, k, t, r) - _american_iv(ref, s, k, t, r)),
            )
        else:
            units, error = (
                "european_bs_iv_bp",
                1e4 * (black.implied_vol(result, s, k, t, r, put) - black.implied_vol(ref, s, k, t, r, put)),
            )
            error = float(error)
        rows.append(
            [model, product, method, result, ref, result - ref, abs(result - ref), error, units, se, elapsed]
        )

    for label, model in [("Heston", h), ("Bates", p)]:
        ref = float(cos_price(model, s, k, t, n=1024))
        for n_modes in (64, 128, 256):
            for width in (8, 11, 14):
                bench(
                    label,
                    "call",
                    f"COS_{n_modes}_L{width}",
                    lambda: cos_price(model, s, k, t, n=n_modes, width=width),
                    ref,
                )
        short_t = 5 / 252
        short_strike = 104.0
        short_ref = float(cos_price(model, s, short_strike, short_t, n=1024, width=14))
        for order in (64, 128, 256):
            start = perf_counter()
            value = float(cos_price(model, s, short_strike, short_t, n=order, width=11))
            elapsed = perf_counter() - start
            difference = 1e4 * (
                black.implied_vol(value, s, short_strike, short_t, r)
                - black.implied_vol(short_ref, s, short_strike, short_t, r)
            )
            rows.append(
                [
                    label,
                    "short_call",
                    f"COS_{order}",
                    value,
                    short_ref,
                    value - short_ref,
                    abs(value - short_ref),
                    difference,
                    "european_bs_iv_bp",
                    np.nan,
                    elapsed,
                ]
            )
        for alpha in (0.75, 1.25, 1.75):
            bench(label, "call", f"FFT_alpha_{alpha:g}", lambda: fft_call(model, s, k, t, alpha=alpha), ref)
        for scheme in ("qe", "euler"):
            bench(
                label,
                "call",
                f"MC_{scheme}",
                lambda: monte_carlo.payoff(simulate(s, t, model, days * 2, n, seed, scheme), k, t, r),
                ref,
                mc=True,
            )
        if label == "Heston":
            for method in ("douglas", "cs", "hv"):
                bench(
                    label,
                    "call",
                    f"ADI_{method}",
                    lambda: (
                        adi_grid(
                            model, k, t, nx=cfg["pde_nx"], nv=cfg["pde_nv"], steps=days * 3, method=method
                        )
                        .interpolator()
                        .ev(s, model.v0)
                    ),
                    ref,
                )
        barrier_ref = float(
            adi_grid(model, k, t, nx=cfg["pde_nx"] * 2, nv=cfg["pde_nv"] * 2, steps=days * 8, barrier=b)
            .interpolator()
            .ev(s, model.v0)
        )
        bench(
            label,
            "barrier",
            "IMEX_ADI" if model.intensity else "HV_ADI",
            lambda: (
                adi_grid(model, k, t, nx=cfg["pde_nx"], nv=cfg["pde_nv"], steps=days * 4, barrier=b)
                .interpolator()
                .ev(s, model.v0)
            ),
            barrier_ref,
        )
        bench(
            label,
            "barrier",
            "discrete_QE_MC",
            lambda: monte_carlo.payoff(
                simulate(s, t, model, days * 2, n, seed),
                k,
                t,
                r,
                barrier=b,
                monitor_steps=np.arange(2, 2 * days + 1, 2),
            ),
            barrier_ref,
            mc=True,
        )
        if model.intensity == 0:
            am_ref = float(
                adi_grid(
                    model,
                    k,
                    t,
                    nx=cfg["pde_nx"] * 2,
                    nv=cfg["pde_nv"] * 2,
                    steps=days * 6,
                    put=True,
                    american=True,
                )
                .interpolator()
                .ev(s, model.v0)
            )
            bench(
                label,
                "american_put",
                "ADI_IT",
                lambda: (
                    adi_grid(
                        model,
                        k,
                        t,
                        nx=cfg["pde_nx"],
                        nv=cfg["pde_nv"],
                        steps=days * 3,
                        put=True,
                        american=True,
                    )
                    .interpolator()
                    .ev(s, model.v0)
                ),
                am_ref,
                put=True,
            )
            bench(
                label,
                "american_put",
                "cross_fitted_LSMC",
                lambda: monte_carlo.lsmc(simulate(s, t, model, days * 2, n, seed), k, t, r),
                am_ref,
                put=True,
                mc=True,
            )
    ref = float(black.price(s, k, t, vol, r))
    for nx in (100, 200, 400):
        bench("BS", "call", f"CN_Rannacher_{nx}", lambda: bs_grid(k, t, vol, r, nx=nx, steps=nx)(s), ref)
    bs_ref = float(bs_discrete_reference(s, k, t, vol, r, b, 1 / 252, 1 / 252, nx=720, steps=days * 12)[0])
    bench(
        "BS",
        "barrier",
        "CN_daily",
        lambda: bs_grid(k, t, vol, r, nx=500, steps=days * 8, barrier=b)(s),
        bs_ref,
    )
    bench(
        "BS",
        "barrier",
        "BGK_continuity_correction",
        lambda: continuous_upout(s, k, t, b * np.exp(0.5826 * vol / np.sqrt(252)), vol, r),
        bs_ref,
    )
    bench(
        "BS",
        "barrier",
        "discrete_BS_MC",
        lambda: monte_carlo.payoff(
            simulate(s, t, p, days, n, seed, bs_vol=vol),
            k,
            t,
            r,
            barrier=b,
            monitor_steps=np.arange(1, days + 1),
        ),
        bs_ref,
        mc=True,
    )
    am_ref = float(bs_grid(k, t, vol, r, nx=400, steps=days * 4, put=True, american=True)(s))
    bench(
        "BS",
        "american_put",
        "CN_PSOR",
        lambda: bs_grid(k, t, vol, r, nx=200, steps=days * 2, put=True, american=True)(s),
        am_ref,
        put=True,
    )
    bench(
        "BS",
        "american_put",
        "cross_fitted_LSMC",
        lambda: monte_carlo.lsmc(
            simulate(s, t, p, days * 2, n, seed, bs_vol=vol), k, t, r, stochastic_variance=False
        ),
        am_ref,
        put=True,
        mc=True,
    )
    write_csv(
        root,
        "pricing",
        [
            "model",
            "product",
            "method",
            "price",
            "reference",
            "price_error",
            "absolute_price_error",
            "error_bp",
            "metric",
            "mc_se",
            "seconds",
        ],
        rows,
    )
    return rows


def _calibration_task(job):
    surface, start, kwargs = job
    return calibration.calibrate(surface, start, **kwargs)


def calibration_experiment(root, p, cfg, seed, times, log_k, modes):
    rng = np.random.default_rng(seed)
    rows, masks, residuals, profiles = [], [], [], []
    diagnostics = {}
    for label, model in [("Heston", p.no_jumps()), ("Bates", p)]:
        clean = calibration.generate_surface(model, 100.0, model.v0, times, log_k, rng, 0, modes)
        optimal = calibration.calibrate(
            clean,
            clean=True,
            global_iterations=cfg["global_iterations"],
            seed=seed,
            modes=modes,
            max_evaluations=160,
        )
        diagnostics[label] = calibration.hessian_diagnostics(optimal, clean, modes)
        short = calibration.generate_surface(
            model, 100.0, model.v0, times[: max(2, len(times) // 2)], log_k, rng, 0, modes
        )
        shortfit = calibration.calibrate(short, optimal.params, clean=True, modes=modes, max_evaluations=160)
        diagnostics[label + "_short"] = calibration.hessian_diagnostics(shortfit, short, modes)

        def entry(name, noise, repl, fit):
            rows.append(
                [
                    label,
                    name,
                    noise,
                    repl,
                    fit.success,
                    fit.boundary,
                    fit.rmse_bp,
                    fit.params.feller_margin,
                    *calibration.physical(fit.params),
                ]
            )
            for i, t in enumerate(times):
                for j, k in enumerate(log_k):
                    residuals.append([label, name, noise, repl, t, k, fit.residual[i, j]])

        entry("clean", 0, 0, optimal)
        # Quote draws keep the serial RNG order; the fits themselves use no randomness.
        draws = []
        for s in (0.5, 1.0, 2.0):
            for i in range(cfg["bootstrap"]):
                surface = calibration.generate_surface(model, 100.0, model.v0, times, log_k, rng, s, modes)
                masks.append(
                    [
                        label,
                        s,
                        i,
                        1 - surface.mask.mean(),
                        np.mean(abs(surface.noise) <= surface.half_spread_iv),
                        np.mean(surface.noise[surface.mask]),
                        np.mean(surface.noise),
                    ]
                )
                draws.append((s, i, surface))
        jobs = []
        for _, _, surface in draws:
            jobs.append((surface, optimal.params, dict(modes=modes, max_evaluations=100)))
            jobs.append(
                (surface, optimal.params, dict(clean=True, matched=True, modes=modes, max_evaluations=100))
            )
        x = calibration.physical(optimal.params)
        levels = []
        for j, name in enumerate(calibration.NAMES):
            scaled = np.r_[x[:4], max(1 - x[4] ** 2, 0.05)]
            for offset in np.linspace(-0.3, 0.3, cfg["profile_points"]):
                level = np.clip(x[j] + scaled[j] * offset, *calibration.BOUNDS[j])
                if j < 4:
                    level = max(level, 1e-4)
                levels.append((name, level))
                jobs.append(
                    (
                        clean,
                        optimal.params,
                        dict(clean=True, fixed=(j, level), modes=modes, max_evaluations=140),
                    )
                )
        fits = parallel.pmap(_calibration_task, jobs)
        for n, (s, i, _) in enumerate(draws):
            entry("noisy", s, i, fits[2 * n])
            entry("matched_clean", s, i, fits[2 * n + 1])
        for (name, level), fit in zip(levels, fits[2 * len(draws) :]):
            profiles.append([label, name, level, fit.rmse_bp, fit.success, *calibration.physical(fit.params)])
    names = list(calibration.NAMES)
    write_csv(
        root,
        "calibration",
        [
            "world",
            "experiment",
            "noise",
            "replicate",
            "converged",
            "boundary",
            "rmse_iv_bp",
            "feller_margin",
            *names,
        ],
        rows,
    )
    write_csv(
        root,
        "quotes",
        [
            "world",
            "noise",
            "replicate",
            "deletion_rate",
            "spread_coverage",
            "mean_retained_noise",
            "mean_proposed_noise",
        ],
        masks,
    )
    write_csv(
        root,
        "calibration_residuals",
        ["world", "experiment", "noise", "replicate", "tenor", "log_moneyness", "residual_iv_bp"],
        residuals,
    )
    write_csv(
        root, "profile", ["world", "fixed_parameter", "level", "rmse_iv_bp", "converged", *names], profiles
    )
    correlations = {}
    for label in ("Heston", "Bates"):
        for noise in (0.5, 1.0, 2.0):
            select = np.array(
                [row[-5:] for row in rows if row[0] == label and row[1] == "noisy" and row[2] == noise]
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                correlations[f"{label}_{noise:g}"] = np.corrcoef(select.T)
    write_json(root, "calibration_diagnostics", {"hessian": diagnostics, "correlations": correlations})
    return rows, diagnostics


def greek_experiment(root, p, cfg, seed):
    t, b, k, vol, r = cfg["days"] / 252, 125.0, 100.0, 0.24, p.rate
    n = cfg["paths"]
    heston = p.no_jumps()
    rng = np.random.default_rng(seed)
    rows, curves, timing = [], [], []

    def add(target, model, instrument, section, s, methods, reference):
        for method, (series, seconds) in methods.items():
            for greek, samples in series.items():
                values = np.atleast_1d(samples)
                estimate = np.mean(values)
                spread = np.std(values, ddof=1) if len(values) > 1 else 0.0
                se = spread / np.sqrt(len(values))
                true = reference[greek]
                target.append(
                    [
                        model,
                        instrument,
                        section,
                        s,
                        method,
                        greek,
                        true,
                        estimate,
                        estimate - true,
                        spread,
                        se,
                        estimate - 1.96 * se,
                        estimate + 1.96 * se,
                        seconds,
                    ]
                )

    def add_jax(out, model, instrument, section, s, dates, barrier, volatility, params=None):
        if not cfg["aad"]:
            return
        data, elapsed = greeks.jax_batch(s, k, volatility, r, dates, barrier, n, seed, p=params)
        if instrument == "vanilla":
            data.pop("gamma")
        out["AAD_JAX"] = data, elapsed["run_first_s"] + elapsed["run_second_s"]
        timing.append([model, instrument, section, s, *elapsed.values()])

    dates = greeks.time_calendar(t, 1 / 252)
    z = rng.standard_normal((len(dates), n))
    true = {j: black.greeks(100, k, t, vol, r)[j] for j in ("delta", "gamma", "vega")}
    methods = greeks.bs_estimators(100, k, t, vol, r, z, dates)
    methods["analytic"] = ({j: np.array([value]) for j, value in true.items()}, 0.0)
    add_jax(methods, "BS", "vanilla", "vanilla", 100, dates, None, vol)
    add(rows, "BS", "vanilla", "vanilla", 100, methods, true)
    sg = cos_greeks(heston, 100, k, t, n=512)
    href = {"delta": sg["delta"], "gamma": sg["gamma"], "vega": 2 * np.sqrt(heston.v0) * sg["variance"]}
    z_h = rng.standard_normal((len(dates), 2, n))
    methods = greeks.heston_estimators(heston, 100, k, t, z_h, dates)
    methods["analytic_CF"] = ({j: np.array([value]) for j, value in href.items()}, 0.0)
    add_jax(methods, "Heston", "vanilla", "vanilla", 100, dates, None, np.sqrt(heston.v0), heston)
    add(rows, "Heston", "vanilla", "vanilla", 100, methods, href)

    for first_label, first in [("one_day", 1 / 252), ("near_monitor", 0.08 / 252)]:
        dates = greeks.time_calendar(t, 1 / 252, first)
        zs = rng.standard_normal((len(dates), n))
        zh = rng.standard_normal((len(dates), 2, n))
        hv = adi_grid(
            heston,
            k,
            t,
            nx=cfg["heston_nx"],
            nv=cfg["heston_nv"],
            steps=cfg["days"] * 12,
            barrier=b,
            first=first,
        ).interpolator()
        for s in np.linspace(116, 129, cfg["scan_points"]):
            _, delta, gamma = bs_discrete_reference(
                s, k, t, vol, r, b, 1 / 252, first, nx=660, steps=cfg["days"] * 12
            )
            bump = 0.001
            vega = (
                bs_discrete_reference(
                    s, k, t, vol + bump, r, b, 1 / 252, first, nx=660, steps=cfg["days"] * 12
                )[0]
                - bs_discrete_reference(
                    s, k, t, vol - bump, r, b, 1 / 252, first, nx=660, steps=cfg["days"] * 12
                )[0]
            ) / (2 * bump)
            out = greeks.bs_estimators(s, k, t, vol, r, zs, dates, b)
            add(curves, "BS", "barrier", first_label, s, out, dict(delta=delta, gamma=gamma, vega=vega))
            if abs(s - b) == min(abs(np.linspace(116, 129, cfg["scan_points"]) - b)):
                add_jax(out, "BS", "barrier", first_label, s, dates, b, vol)
                add(rows, "BS", "barrier", first_label, s, out, dict(delta=delta, gamma=gamma, vega=vega))
                hh = greeks.heston_estimators(heston, s, k, t, zh, dates, b)
                add_jax(hh, "Heston", "barrier", first_label, s, dates, b, np.sqrt(heston.v0), heston)
                add(
                    rows,
                    "Heston",
                    "barrier",
                    first_label,
                    s,
                    hh,
                    {
                        "delta": hv.ev(s, heston.v0, dx=1),
                        "gamma": hv.ev(s, heston.v0, dx=2),
                        "vega": 2 * np.sqrt(heston.v0) * hv.ev(s, heston.v0, dy=1),
                    },
                )
    columns = [
        "model",
        "product",
        "monitor_slice",
        "spot",
        "algorithm",
        "greek",
        "reference",
        "estimate",
        "error",
        "sample_sd",
        "standard_error",
        "ci_low",
        "ci_high",
        "seconds",
    ]
    write_csv(root, "greeks", columns, rows)
    write_csv(root, "barrier_greeks", columns, curves)
    write_csv(
        root,
        "aad_timing",
        [
            "model",
            "product",
            "slice",
            "spot",
            "compile_first_s",
            "compile_second_s",
            "run_first_s",
            "run_second_s",
        ],
        timing,
    )
    return rows, curves


def _risk(x):
    boundary = np.quantile(x, 0.05)
    return np.mean(x), np.std(x, ddof=1), -np.mean(x[x <= boundary])


def _paired(first, second, seed, n=600):
    rng = np.random.default_rng(seed)
    index = rng.integers(len(first), size=(n, len(first)))
    diff = first - second
    var_diff = np.var(first[index], axis=1, ddof=1) - np.var(second[index], axis=1, ddof=1)
    return (
        np.mean(diff),
        *np.quantile(diff[index].mean(axis=1), [0.025, 0.975]),
        *np.quantile(var_diff, [0.025, 0.975]),
    )


def _hedge_task(job):
    args, kwargs, ids = job
    return hedge_market(*args, **kwargs, path_ids=ids)


def _hedge_merge(parts, npaths):
    """Combine single-path ledgers in path order, as the serial loop records them."""
    data = {"pnl": {}, "cost": {}, "calibration": [], "corrections": [], "quotes": []}
    for ids, part in parts:
        for book in ("pnl", "cost"):
            for case, values in part[book].items():
                data[book].setdefault(case, np.zeros(npaths))[ids] = values[ids]
        for name in ("calibration", "corrections", "quotes"):
            data[name].extend(part[name])
        data["dt"], data["premium"] = part["dt"], part["premium"]
    return data


def hedge_experiment(root, p, cfg, seed, times, log_k, experiment_noise, modes):
    days, substeps = cfg["days"], cfg["substeps"]
    t = days / 252
    fine_steps = days * substeps
    frequencies = cfg["frequencies"]
    results, paired, drift, corrections, quotes, reconciled, raw = [], [], [], [], [], [], []
    regressions = []
    settings = HedgeSettings(
        paths_daily=cfg["paths"],
        calibration_modes=modes,
        noise=experiment_noise,
        stock_cost=cfg["stock_cost"],
        option_cap=cfg["option_cap"],
        tool_tenor_days=cfg.get("tool_tenor_days", HedgeSettings.tool_tenor_days),
    )
    tape = BrownianJumpTape(t, fine_steps, cfg["control_paths"], p, seed)
    bs_tape = tape.market(settings.spot, p, fine_steps, bs_vol=0.24)
    control = hedge_bs_control(bs_tape, t, p.rate, 0.24, frequencies, settings)
    for every, x in control.items():
        results.append(
            ["vanilla", "BS", "true_delta", every, t / fine_steps * every, 0.0, len(x), *_risk(x), 0.0]
        )
    # Every (scenario, product, path) ledger is independent; submit them to one pool.
    contexts, jobs = [], []
    for scenario, world in scenarios(p).items():
        for product in ["vanilla", "barrier"] if scenario == "base" else ["vanilla"]:
            name = "heston" if scenario == "no_jump" else "bates"
            n = cfg["barrier_paths"] if product == "barrier" else cfg["paths"]
            trade_tape = BrownianJumpTape(t, fine_steps, n, world, seed + 1)
            market = trade_tape.market(settings.spot, world, fine_steps)
            args = (market, t, world, frequencies, cfg["costs"], settings, seed + 2)
            kwargs = dict(
                substeps=substeps,
                surface_times=times,
                log_k=log_k,
                product=product,
                world=name,
                extension=cfg.get("extension", False) and scenario == "base" and product == "vanilla",
            )
            contexts.append((scenario, product, name, n))
            jobs.extend((len(contexts) - 1, (args, kwargs, [a])) for a in range(n))
    outputs = parallel.pmap(_hedge_task, [job for _, job in jobs])
    for index, (scenario, product, name, n) in enumerate(contexts):
        parts = [(job[2], out) for (owner, job), out in zip(jobs, outputs) if owner == index]
        data = _hedge_merge(parts, n)
        drift.extend([[product, scenario, *row] for row in data["calibration"]])
        corrections.extend([[product, scenario, *row] for row in data["corrections"]])
        quotes.extend([[product, scenario, *row] for row in data["quotes"]])
        for (strategy, every, cost), x in data["pnl"].items():
            cost_sample = data["cost"][(strategy, every, cost)]
            results.append(
                [
                    product,
                    scenario,
                    strategy,
                    every,
                    every * data["dt"],
                    cost,
                    len(x),
                    *_risk(x),
                    np.mean(cost_sample),
                ]
            )
            raw.extend(
                [
                    [product, scenario, strategy, every, cost, a, value, fee]
                    for a, (value, fee) in enumerate(zip(x, cost_sample))
                ]
            )
        if name == "bates":
            for every in frequencies:
                pairs = [
                    ("sticky_strike", "calibrated_mv"),
                    ("delta_vega", "calibrated_two"),
                    ("sticky_strike", "full_stock"),
                    ("calibrated_mv", "full_stock"),
                    ("delta_vega", "full_two"),
                    ("calibrated_two", "full_two"),
                ]
                if cfg.get("extension", False) and scenario == "base" and product == "vanilla":
                    for suffix, last in [
                        ("stock", ("calibrated_mv", "sticky_strike")),
                        ("two", ("calibrated_two", "delta_vega")),
                    ]:
                        chain = [
                            "full_" + suffix,
                            "bates_filtered_true_" + suffix,
                            "bates_filtered_calibrated_" + suffix,
                            *last,
                        ]
                        pairs.extend(zip(chain[1:], chain[:-1]))
                for x, y in pairs:
                    a, b = data["pnl"][(x, every, 0.0)], data["pnl"][(y, every, 0.0)]
                    paired.append(
                        [
                            product,
                            scenario,
                            every,
                            x,
                            y,
                            np.var(a, ddof=1) - np.var(b, ddof=1),
                            *_paired(a, b, seed + 56),
                        ]
                    )
    deep = BrownianJumpTape(t, fine_steps * cfg["reconcile_factor"], cfg["control_paths"], p, seed + 9)
    rng = np.random.default_rng(seed + 2)
    start_surface = calibration.generate_surface(
        p, settings.spot, p.v0, np.unique(np.r_[times, t]), log_k, rng, experiment_noise, modes
    )
    iv = start_surface.smile(t, settings.strike)
    for level in (1, cfg["reconcile_factor"]):
        market = deep.market(settings.spot, p, fine_steps * level)
        target, sim, diff, jump = decomposition(
            market, p, t, settings.strike, iv, [f * level for f in frequencies]
        )
        for every, values in sim.items():
            for a, value in enumerate(values):
                reconciled.append(
                    [
                        level,
                        every // level,
                        t * every / (fine_steps * level),
                        a,
                        value,
                        target[a],
                        diff[a],
                        jump[a],
                        value - target[a],
                    ]
                )
    for product, world, strategy in sorted({tuple(row[:3]) for row in results}):
        items = [row for row in results if tuple(row[:3]) == (product, world, strategy) and row[5] == 0.0]
        if len(items) < 2:
            continue
        dt = np.array([row[4] for row in items])
        sd = np.array([row[8] for row in items])
        # a + b*dt is a small-step expansion, so the intercept uses intervals of at most one
        # trading day. The log-log slope keeps every interval.
        local = dt <= 1 / 252 + 1e-12
        if local.sum() < 2:
            local = np.argsort(dt) < 2
        intercept, slope = np.polynomial.polynomial.polyfit(dt[local], sd[local] ** 2, 1)
        log_slope = np.polynomial.polynomial.polyfit(np.log(dt), np.log(np.maximum(sd, 1e-12)), 1)[1]
        regressions.append([product, world, strategy, intercept, slope, log_slope, float(dt[local].max())])
    write_csv(
        root,
        "hedge",
        [
            "product",
            "world",
            "strategy",
            "step",
            "dt",
            "stock_cost",
            "paths",
            "mean",
            "sd",
            "cvar_loss_95",
            "mean_cost",
        ],
        results,
    )
    write_csv(
        root, "hedge_path", ["product", "world", "strategy", "step", "stock_cost", "path", "pnl", "cost"], raw
    )
    write_csv(
        root,
        "hedge_variance_fit",
        [
            "product",
            "world",
            "strategy",
            "residual_variance_intercept",
            "dt_slope",
            "log_sd_slope",
            "intercept_fit_max_dt",
        ],
        regressions,
    )
    write_csv(
        root,
        "paired",
        [
            "product",
            "world",
            "step",
            "strategy",
            "reference",
            "variance_difference",
            "mean_pnl_difference",
            "mean_ci_low",
            "mean_ci_high",
            "variance_ci_low",
            "variance_ci_high",
        ],
        paired,
    )
    write_csv(
        root,
        "reconciliation",
        [
            "grid_refinement",
            "step",
            "trade_dt",
            "path",
            "actual",
            "predicted",
            "diffusion",
            "jump",
            "residual",
        ],
        reconciled,
    )
    write_csv(
        root,
        "daily_calibration",
        [
            "product",
            "world",
            "path",
            "day",
            "full_fallback",
            "weekly_fallback",
            "clean_fallback",
            "full_rmse_bp",
            "weekly_rmse_bp",
            *[f"{prefix}_{j}" for prefix in ("full", "weekly", "clean") for j in calibration.NAMES],
        ],
        drift,
    )
    write_csv(
        root,
        "hedge_corrections",
        ["product", "world", "path", "step", "sticky_delta", "exact_mv", "approx_mv"],
        corrections,
    )
    write_csv(root, "hedge_quotes", ["product", "world", "deletion", "coverage"], quotes)
    frontier = []
    for product, world, strategy in sorted({tuple(row[:3]) for row in results}):
        for c in sorted({row[5] for row in results if row[5] > 0}):
            candidates = [
                row for row in results if tuple(row[:3]) == (product, world, strategy) and row[5] == c
            ]
            for a in candidates:
                dominated = any(
                    b[8] <= a[8] and b[10] <= a[10] and (b[8] < a[8] or b[10] < a[10]) for b in candidates
                )
                frontier.append([*a, not dominated])
    write_csv(
        root,
        "cost_frontier",
        [
            "product",
            "world",
            "strategy",
            "step",
            "dt",
            "stock_cost",
            "paths",
            "mean",
            "sd",
            "cvar_95",
            "mean_cost",
            "efficient_sd",
        ],
        frontier,
    )
    return results, regressions, reconciled, drift


def _attribution_task(job):
    args, kwargs, ids = job
    return pnl.run_attribution(*args, **kwargs, path_ids=ids)


def pnl_experiment(root, p, cfg, seed, times, log_k, noise, modes):
    args, kwargs = (p, cfg["paths"], seed, times, log_k), dict(noise=noise, modes=modes)
    parts = parallel.pmap(_attribution_task, [(args, kwargs, [a]) for a in range(cfg["paths"])])
    headers = parts[0][0]
    rows = [row for part in parts for row in part[1]]
    metrics = [row for part in parts for row in part[2]]
    write_csv(root, "pnl_explain", headers, rows)
    write_csv(root, "pla", ["path", "method", "spearman", "ks", "zone"], metrics)
    probabilities = {
        method: {
            zone: sum(row[4] == zone for row in metrics if row[1] == method) / cfg["paths"]
            for zone in ("green", "amber", "red", "undefined")
        }
        for method in pnl.MODES
    }
    risk = []
    x = np.array(rows)
    for name, keep in [("all", np.ones(len(rows), bool)), ("jump", x[:, 2] > 0), ("no_jump", x[:, 2] == 0)]:
        if not keep.any():
            continue
        arr = x[keep]

        def ratio(residual, reference):
            return np.sqrt(np.mean(residual**2)) / max(np.sqrt(np.mean(reference**2)), 1e-16)

        risk.append(
            [
                name,
                len(arr),
                ratio(arr[:, 10], arr[:, 7]),
                ratio(arr[:, 11], arr[:, 7]),
                ratio(arr[:, 12], arr[:, 3]),
            ]
        )
    write_csv(
        root,
        "pnl_residual_summary",
        ["sample", "count", "first_rmse_over_bs", "second_rmse_over_bs", "model_gap_rmse_over_market"],
        risk,
    )
    write_json(root, "pla_zone_probability", probabilities)
    return rows, metrics, risk


def commodity_experiment(root, cfg, seed):
    p = commodity.SpreadModel()
    rows = []
    for rho in (-0.6, -0.2, 0.2, 0.5, 0.8, 0.95):
        for k in (0.0, 5.0, 12.0, 24.0, 40.0):
            f, g, t = 104.0, 100.0, 0.25
            reference = commodity.constant_price(f, g, k, t, rho, p, 64)

            def rr(a, b, c):
                return commodity.constant_price(a, b, k, t, c, p)

            gradient = commodity.sensitivities(rr, f, g, rho)
            kirk = commodity.kirk(f, g, k, t, rho, p)
            bs = commodity.bjerksund_stensland(f, g, k, t, rho, p)
            draws = np.random.default_rng(seed).standard_normal((2, cfg["mc_paths"]))
            a = f * np.exp(-(p.sigma1**2) * t / 2 + p.sigma1 * np.sqrt(t) * draws[0])
            b = g * np.exp(
                -(p.sigma2**2) * t / 2
                + p.sigma2 * np.sqrt(t) * (rho * draws[0] + np.sqrt(1 - rho * rho) * draws[1])
            )
            mc = np.exp(-p.rate * t) * np.maximum(a - b - k, 0)
            exchange = commodity.margrabe(f, g, t, rho, p) if k == 0 else np.nan
            rows.append(
                [
                    rho,
                    k,
                    reference,
                    exchange,
                    kirk,
                    bs,
                    np.mean(mc),
                    np.std(mc, ddof=1) / np.sqrt(len(mc)),
                    kirk - reference,
                    bs - reference,
                    *gradient,
                ]
            )
    hedge = []
    dynamic_rows = []
    for label, link in [("orthogonal", 0.0), ("linked", 0.35)]:
        model = replace(p, linkage=link)
        methods, x, clip, residual = commodity.spread_hedge(
            model, cfg["paths"], cfg["steps"], cfg["inner_paths"], seed, parallel.pmap
        )
        for j, method in enumerate(methods):
            hedge.append([label, method, len(x), *_risk(x[:, j]), np.mean(clip), np.mean(residual)])
        for a in range(len(x)):
            dynamic_rows.append([label, a, clip[a], residual[a], *x[a]])
    write_csv(
        root,
        "spread_prices",
        [
            "rho",
            "strike",
            "quadrature_price",
            "margrabe",
            "kirk",
            "bjerksund_stensland",
            "mc",
            "mc_se",
            "kirk_error",
            "bjerksund_error",
            "delta_f1",
            "delta_f2",
            "rho_greek",
            "cross_gamma",
        ],
        rows,
    )
    write_csv(
        root,
        "spread_hedge",
        [
            "driver",
            "strategy",
            "paths",
            "mean",
            "sd",
            "cvar_95",
            "boundary_rate",
            "continuous_residual_variance_estimate",
        ],
        hedge,
    )
    write_csv(
        root,
        "spread_hedge_paths",
        ["driver", "path", "boundary_rate", "integrated_residual", *methods],
        dynamic_rows,
    )
    return rows, hedge
