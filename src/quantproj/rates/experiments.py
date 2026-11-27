"""Reproducible rates experiments R1 through R3."""

import csv
import json
from dataclasses import replace
from datetime import date

import numpy as np

from quantproj.core.parallel import configure, pmap
from quantproj.rates.attribution import run_pla
from quantproj.rates.bermudan import g2_adi, hw_pde, hw_tree, primal_dual
from quantproj.rates.calibration import calibrate, fit_market, observe
from quantproj.rates.curves import (
    TENORS,
    NelsonSiegel,
    act360,
    bootstrap,
    business_dates,
    maturity,
    nc1,
    ois_quotes,
    quote_swaps,
)
from quantproj.rates.european import european, monte_carlo, normal_iv, observables
from quantproj.rates.greeks import aad_price, quote_greeks
from quantproj.rates.hedging import evaluate_exercise, hedge_path
from quantproj.rates.models import g2, hw, simulate
from quantproj.rates.reconciliation import reconcile


def write_csv(path, name, header, rows):
    with (path / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def summarize(values):
    x = np.asarray(values)
    se = float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan
    return [float(np.mean(x)), se, float(np.mean(x) - 1.96 * se), float(np.mean(x) + 1.96 * se)]


def _trader_market(model, product, origin, settings, seed, noisy):
    return observe(
        model,
        0.0,
        np.zeros(model.dimension),
        origin,
        product,
        np.random.default_rng(seed),
        settings["ois_noise_bp"] * noisy,
        settings["normal_noise_bp"] * noisy,
        smile_offsets=settings["smile_offsets_bp"],
        order=settings["integration_order"],
    )


def r1(output, config, curve, worlds, product, origin):
    seed = config["seed"]
    settings = config["calibration"]
    a = settings["mean_reversion"]
    instruments = quote_swaps(origin)
    q = ois_quotes(instruments, curve.discount)
    result = []
    curves = {}
    tenor_check = []
    for method in ("log_linear", "monotone_convex"):
        constructed = bootstrap(instruments, q, method)
        curves[method] = constructed
        fit = ois_quotes(instruments, constructed.discount)
        xx = np.linspace(0.002, 30, 3000)
        f = constructed.forward(xx)
        total_var = np.sum(np.diff(f) ** 2) / np.mean(np.diff(xx))
        result.append([method, np.max(abs(fit - q)) * 1e4, total_var, f.min(), f.max()])
        tenor_check.extend(
            [method, tenor, s.dates[-1], reference, model_q, (model_q - reference) * 1e4]
            for tenor, s, reference, model_q in zip(TENORS, instruments, q, fit)
        )
        write_csv(output, f"forward_{method}", ["time", "forward"], zip(xx, f))
    write_csv(
        output,
        "curve_repricing",
        ["interpolation", "max_par_error_bp", "forward_roughness", "min", "max"],
        result,
    )
    write_csv(
        output,
        "curve_instruments",
        ["interpolation", "tenor", "maturity", "input", "repriced", "error_bp"],
        tenor_check,
    )
    show = []
    for label, model in worlds.items():
        for i in (0, 3, 7, 8):
            swap = product.tail(i)
            state = np.zeros(model.dimension)
            pv = european(model, 0, state, swap)
            mc, se = monte_carlo(model, 0, state, swap, config["pricing"]["european_paths"], seed + i)
            f, ann, _ = observables(swap, curve.discount)
            vol = normal_iv(pv, f, swap.fixed, swap.dates[0], ann)
            mc_vol = normal_iv(mc, f, swap.fixed, swap.dates[0], ann)
            show.append(
                [
                    label,
                    swap.dates[0],
                    "Jamshidian" if model.dimension == 1 else "G2 conditional integral",
                    pv,
                    mc,
                    se,
                    vol * 1e4,
                    mc_vol * 1e4,
                    (mc_vol - vol) * 1e4,
                    (mc - pv) / se,
                ]
            )
    write_csv(
        output,
        "european",
        [
            "world",
            "expiry",
            "method",
            "analytic",
            "mc",
            "mc_se",
            "analytic_normal_bp",
            "mc_normal_bp",
            "error_normal_bp",
            "z_score",
        ],
        show,
    )
    market = _trader_market(worlds["g2_base"], product, origin, settings, seed, False)
    interp = []
    bump = 1e-5
    for name, c in curves.items():
        fit = calibrate(c, market.strip, market.clean_vols, a)
        pv = hw_pde(fit.model, product, n=config["pricing"]["hw_grid"], steps_per_year=10).price(0.0, [0])
        for i, tenor in enumerate(TENORS):
            delta = np.eye(len(q))[i] * bump
            marks = []
            for sign in (1, -1):
                bumped = bootstrap(instruments, q + sign * delta, name)
                case = calibrate(bumped, market.strip, market.clean_vols, a)
                marks.append(
                    hw_pde(case.model, product, n=config["pricing"]["hw_grid"], steps_per_year=10).price(
                        0.0, [0]
                    )
                )
            interp.append([name, tenor, pv, pv * 1e4, (marks[0] - marks[1]) / (2 * bump) * 1e-4])
    write_csv(
        output,
        "interpolation_risk",
        ["interpolation", "tenor", "bermudan", "pv_notional_bp", "key_rate_pv_per_bp"],
        interp,
    )
    for name, model in worlds.items():
        rows = []
        tape = np.array([0.0, 0.25, 1.0, 2.0, 5.0])
        x, b = simulate(model, tape, config["pricing"]["european_paths"], seed + 31)
        bond = model.curve.discount(10.0)
        for t, state, bank in zip(tape, x, b):
            sample = model.bond(t, 10.0, state) / bank
            rows.append([name, t, bond, *summarize(sample)])
        with (output / "discounted_bond_martingale.csv").open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)


def r2(output, config, worlds, product, origin):
    pricing = config["pricing"]
    cal = config["calibration"]
    seed = config["seed"]
    a = cal["mean_reversion"]
    rows = []
    models = []
    quote_rows = []
    bounds_rows = []
    fine = {}
    for label, model in worlds.items():
        n = pricing["hw_grid"] if model.dimension == 1 else pricing["adi_grid"]
        solver = hw_pde if model.dimension == 1 else g2_adi
        x = np.zeros(model.dimension)
        for i, grid in enumerate((n, 2 * n - 1)):
            sol = solver(model, product, n=grid, steps_per_year=(i + 1) * pricing["adi_steps"])
            pv = sol.price(0.0, x)
            rows.append([label, "HW CN Rannacher" if model.dimension == 1 else "G2 HV-ADI", grid, pv * 1e4])
        fine[label] = sol
        if model.dimension == 1:
            for steps in (pricing["adi_steps"], 2 * pricing["adi_steps"]):
                rows.append(
                    [label, "HW trinomial", steps, hw_tree(model, product, steps_per_year=steps) * 1e4]
                )
        lower, upper, _ = primal_dual(
            model, product, pricing["lsmc_training"], pricing["dual_outer"], pricing["dual_inner"], seed + 170
        )
        for method, values in [("OOS LSMC lower", lower), ("Andersen-Broadie nested upper", upper)]:
            bounds_rows.append([label, method, len(values), *summarize(values * 1e4)])
        for level in (False, True):
            for rep in range(cal["replicates"] if level else 1):
                market = _trader_market(model, product, origin, cal, seed + rep, level)
                for expiry, k, clean_v, noisy_v, quote, keep in market.quotes:
                    quote_rows.append(
                        [label, int(level), rep, expiry, k, clean_v * 1e4, noisy_v * 1e4, quote, keep]
                    )
                for aa in cal["mean_reversion_grid"]:
                    fit = fit_market(market, aa)
                    model_pv = hw_pde(
                        fit.model, product, n=pricing["hw_grid"], steps_per_year=pricing["adi_steps"]
                    ).price(0, [0])
                    for i, t in enumerate(fit.times):
                        models.append(
                            [
                                label,
                                int(level),
                                rep,
                                aa,
                                t,
                                np.sqrt(fit.sigma2[i]) * 1e4,
                                fit.residual_bp[i],
                                int(fit.fallback),
                                fit.missing,
                                int(fit.converged),
                                model_pv * 1e4,
                                pv * 1e4,
                            ]
                        )
    adjusted = [
        [world, method, resolution, value, fine[world].price(0.0, np.zeros(worlds[world].dimension)) * 1e4]
        for world, method, resolution, value in rows
    ]
    write_csv(
        output,
        "bermudan",
        ["world", "method", "resolution", "notional_bp", "reference_bp", "error_notional_bp"],
        [row + [row[3] - row[4]] for row in adjusted],
    )
    write_csv(
        output,
        "bermudan_bounds",
        ["world", "method", "paths", "mean_bp", "se_bp", "ci_low_bp", "ci_high_bp"],
        bounds_rows,
    )
    write_csv(
        output,
        "hw_calibration",
        [
            "world",
            "noisy",
            "replicate",
            "a",
            "expiry",
            "sigma_bp",
            "iv_residual_bp",
            "fallback",
            "missing",
            "converged",
            "hw_bermudan_bp",
            "market_adi_bp",
        ],
        models,
    )
    write_csv(
        output,
        "co_terminal_quotes",
        ["world", "noisy", "replicate", "expiry", "strike", "clean_bp", "observed_bp", "premium", "retained"],
        quote_rows,
    )
    aad_check = []
    for label, model in worlds.items():
        market = _trader_market(model, product, origin, cal, seed, False)
        fit = fit_market(market, a)
        if fit.fallback or np.any(fit.sigma2 <= 1e-12):
            aad_check.append([label, "all", 0, 0.0, np.nan, np.nan, np.nan, True, False])
            continue
        n = cal["desk_grid"]
        steps = cal["desk_steps"]
        v, dq, dv, mask = quote_greeks(market, fit, product, a, n, steps)
        for i, key in enumerate(
            [("ois", t) for t in TENORS] + [("normal_vol", i + 1) for i in range(len(fit.w))]
        ):
            key_type, node = key
            test_bump = 1e-6
            values = []
            switched = False
            constrained = False
            for sign in (-1, 1):
                if key_type == "ois":
                    q = market.ois.copy()
                    q[i] += sign * test_bump
                    changed = replace(market, ois=q, curve=bootstrap(market.swaps, q))
                else:
                    vols = market.vols.copy()
                    vols[i - len(dq)] += sign * test_bump
                    changed = replace(market, vols=vols)
                new = fit_market(changed, a)
                val, _, _, region = aad_price(changed.curve, new, product, a, n, steps)
                values.append(val)
                constrained |= new.fallback
                switched |= bool(np.any(region != mask))
            reference = (values[1] - values[0]) / (2 * test_bump)
            greek = np.r_[dq, dv][i]
            aad_check.append(
                [
                    label,
                    key_type,
                    node,
                    v,
                    greek * 1e-4,
                    reference * 1e-4,
                    (greek - reference) * 1e-4,
                    constrained,
                    switched,
                ]
            )
    write_csv(
        output,
        "quote_greeks",
        [
            "world",
            "risk",
            "node",
            "pv",
            "aad_pv_per_bp",
            "bump_pv_per_bp",
            "difference",
            "constrained",
            "exercise_switch",
        ],
        aad_check,
    )
    return fine


def _paired(lhs, rhs, rng):
    lhs, rhs = np.asarray(lhs), np.asarray(rhs)
    difference = lhs - rhs
    boot = rng.integers(0, len(lhs), (1000, len(lhs)))
    spread = np.var(lhs, ddof=1) - np.var(rhs, ddof=1)
    interval = np.quantile(
        np.var(lhs[boot], axis=1, ddof=1) - np.var(rhs[boot], axis=1, ddof=1), [0.025, 0.975]
    )
    mean_interval = np.quantile(difference[boot].mean(axis=1), [0.025, 0.975])
    return [difference.mean(), *mean_interval, spread, *interval]


def _hedge_job(job):
    *args, holder = job
    return hedge_path(*args, holder=holder)


def _exercise_job(job):
    *args, path = job
    return evaluate_exercise(*args, path_ids=[path])


def _pla_job(job):
    *args, path = job
    return run_pla(*args, path_ids=[path])


def r3(output, config, worlds, product, origin, solutions):
    settings = {**config["calibration"], **config["hedging"]}
    a = settings["mean_reversion"]
    seed = config["seed"]
    end = maturity(origin, "10Y")
    asofs = business_dates(origin, end)
    times = np.array([act360(origin, d) for d in asofs])
    exercise = []
    paths = []
    fits = []
    summaries = []
    pairs = []
    residual = []
    for label, model in worlds.items():
        grid = solutions[label]
        if model.dimension == 2:
            r = np.r_[0.0, product.dates]
            ids = [int(np.argmin(abs(times - t))) for t in r]
            xs, bank = simulate(model, r, config["exercise"]["paths"], seed + 700)
            # Each path draws its quotes from a path-keyed seed, so path jobs match the serial loop.
            exercise_jobs = [
                (
                    model,
                    product,
                    grid,
                    r,
                    xs,
                    bank,
                    [asofs[i] for i in ids],
                    settings["mean_reversion_grid"],
                    settings,
                    seed + 800,
                    p,
                )
                for p in range(xs.shape[1])
            ]
            for rows in pmap(_exercise_job, exercise_jobs):
                exercise.extend([label, *row] for row in rows)
        x, b = simulate(model, times, settings["paths"], seed + 900)
        choices = ("true", "hw") if model.dimension == 2 else ("true",)
        a_choices = settings.get("hedge_mean_reversions", [a]) if model.dimension == 2 else [a]
        groups = [(aa, holder) for aa in a_choices for holder in choices]
        # One pool per world keeps every worker busy across mean reversions and holders.
        hedge_jobs = [
            (
                model,
                product,
                grid,
                times,
                x[:, p],
                b[:, p],
                asofs,
                settings,
                aa,
                seed + 1000 + p,
                holder,
            )
            for aa, holder in groups
            for p in range(x.shape[1])
        ]
        results = pmap(_hedge_job, hedge_jobs)
        for g, (aa, holder) in enumerate(groups):
            collected = {}
            group_results = results[g * x.shape[1] : (g + 1) * x.shape[1]]
            for p, (data, cal) in enumerate(group_results):
                if holder == "true":
                    fits.extend([label, aa, p, *row] for row in cal)
                for row in data:
                    strategy, freq, cost, *metrics = row
                    paths.append([label, aa, holder, p, *row])
                    collected.setdefault((strategy, freq, cost), []).append(metrics)
            for (strategy, freq, cost), values in collected.items():
                values = np.asarray(values, dtype=float)
                pnl = values[:, 0]
                tail = np.sort(pnl)[: max(1, int(np.ceil(0.05 * len(pnl))))]
                summaries.append(
                    [
                        label,
                        aa,
                        holder,
                        strategy,
                        freq,
                        cost,
                        len(pnl),
                        float(pnl.mean()),
                        float(pnl.std(ddof=1)),
                        float(-tail.mean()),
                        float(values[:, 2].mean()),
                        float(np.mean(values[:, 4])),
                        float(values[:, 5].sum() / max(values[:, 6].sum(), 1)),
                    ]
                )
            if model.dimension == 2:
                compare = [
                    ("key_rate", "full_info_swaps"),
                    ("key_rate_vega", "full_info_all"),
                    ("single_delta", "key_rate"),
                    ("key_rate", "key_rate_vega"),
                ]
                rng = np.random.default_rng(seed + 1100)
                for freq in settings["frequencies"]:
                    for cost in settings["cost_multipliers"]:
                        for lhs, rhs in compare:
                            lhs_pnl = np.asarray(collected[lhs, freq, cost])[:, 0]
                            rhs_pnl = np.asarray(collected[rhs, freq, cost])[:, 0]
                            pairs.append(
                                [label, aa, holder, lhs, rhs, freq, cost, *_paired(lhs_pnl, rhs_pnl, rng)]
                            )
            for strategy in sorted({k[0] for k in collected}):
                steps = [f for f in settings["frequencies"] if (strategy, f, 0.0) in collected]
                if len(steps) > 1:
                    step_var = [np.var(np.asarray(collected[strategy, f, 0.0])[:, 0], ddof=1) for f in steps]
                    slope, intercept = np.polyfit(np.asarray(steps) / 252, np.asarray(step_var), 1)
                    residual.append([label, aa, holder, strategy, float(intercept), float(slope)])
            print(f"  {label}, a={aa:g}, holder={holder}: {x.shape[1]} hedge paths", flush=True)
    write_csv(
        output,
        "exercise_paths",
        [
            "world",
            "path",
            "a",
            "adi_pv",
            "hw_pv",
            "loss_pv",
            "adi_date_index",
            "hw_date_index",
            "decisions_different",
        ],
        exercise,
    )
    groups = sorted({(r[0], r[2]) for r in exercise})
    exercise_stats = []
    for label, aa in groups:
        matched = np.array([r[3:6] for r in exercise if r[0] == label and r[2] == aa])
        exercise_stats.append(
            [
                label,
                aa,
                len(matched),
                float(matched[:, 0].mean() * 1e4),
                float(matched[:, 1].mean() * 1e4),
                *summarize(matched[:, 2] * 1e4),
            ]
        )
    write_csv(
        output,
        "exercise_loss",
        [
            "world",
            "a",
            "paths",
            "adi_mean_bp",
            "hw_mean_bp",
            "loss_mean_bp",
            "loss_se_bp",
            "loss_ci_low_bp",
            "loss_ci_high_bp",
        ],
        exercise_stats,
    )
    write_csv(
        output,
        "hedge_paths",
        [
            "world",
            "a",
            "holder",
            "path",
            "strategy",
            "days",
            "cost_multiple",
            "discounted_pnl",
            "year10_pnl",
            "discounted_cost",
            "stop_day",
            "exercised",
            "mismatches",
            "decisions",
        ],
        paths,
    )
    write_csv(
        output,
        "hedge_summary",
        [
            "world",
            "a",
            "holder",
            "strategy",
            "days",
            "cost_multiple",
            "paths",
            "mean",
            "sd",
            "cvar_95",
            "cost",
            "exercise_rate",
            "mismatch_rate",
        ],
        summaries,
    )
    write_csv(
        output,
        "hedge_paired",
        [
            "world",
            "a",
            "holder",
            "lhs",
            "rhs",
            "days",
            "cost_multiple",
            "mean_delta",
            "mean_ci_low",
            "mean_ci_high",
            "variance_delta",
            "variance_ci_low",
            "variance_ci_high",
        ],
        pairs,
    )
    write_csv(
        output,
        "hedge_variance_fit",
        ["world", "a", "holder", "strategy", "variance_intercept", "variance_slope"],
        residual,
    )
    write_csv(
        output,
        "daily_calibration",
        [
            "world",
            "a",
            "path",
            "day",
            "expiry",
            "sigma",
            "clean_sigma",
            "iv_residual_bp",
            "fallback",
            "missing",
            "quotes_deleted",
            "ois_fallback",
        ],
        fits,
    )
    base = worlds["g2_base"]
    ti = times[times <= product.dates[0] + 1e-8]
    data = reconcile(
        base,
        product,
        config["reconciliation"]["paths"],
        seed + 1200,
        ti,
        tuple(config["reconciliation"]["grid_refinements"]),
        tuple(config["reconciliation"]["frequencies"]),
    )
    write_csv(
        output,
        "annuity_reconciliation",
        ["grid_multiple", "rehedge_days", "paths", "rmse", "pnl_mean", "theoretical_mean", "residual_mean"],
        data,
    )
    begin = date.fromisoformat(config["pnl"]["start"])
    warmup = int(np.searchsorted(times, act360(origin, begin)))
    n = warmup + 250
    settings = {**settings, **config["pnl"]}
    xs, bs = simulate(base, times[:n], config["pnl"]["paths"], seed + 1350)
    pnl, metrics = [], []
    extras: dict[str, list] = {"sensitivity_daily": [], "sensitivity": [], "calibration": []}
    pla_jobs = [
        (base, times[:n], asofs[:n], xs, bs, product, solutions["g2_base"], warmup, settings, seed + 1450, p)
        for p in range(xs.shape[1])
    ]
    for rows, stats, extra in pmap(_pla_job, pla_jobs):
        pnl.extend(rows)
        metrics.extend(stats)
        for key in extras:
            extras[key].extend(extra[key])
    write_csv(
        output,
        "pnl_explain",
        [
            "path",
            "day",
            "date",
            "event",
            "hpl",
            "rtpl",
            "taylor_first",
            "taylor_second",
            "residual_first",
            "residual_second",
            "mark_gap",
            "hpl_front_office",
            "rtpl_g2",
        ],
        pnl,
    )
    write_csv(
        output,
        "pla",
        [
            "path",
            "first_day",
            "last_day",
            "spearman",
            "ks",
            "zone",
            "exercises",
            "expiries",
            "aligned_hw_spearman",
            "aligned_hw_ks",
            "aligned_hw_zone",
            "aligned_g2_spearman",
            "aligned_g2_ks",
            "aligned_g2_zone",
        ],
        metrics,
    )
    write_csv(
        output,
        "pla_sensitivity_daily",
        ["path", "day", "date", "variant", "hpl_front_office", "rtpl_g2"],
        extras["sensitivity_daily"],
    )
    write_csv(output, "pla_sensitivity", ["path", "variant", "spearman", "ks", "zone"], extras["sensitivity"])
    write_csv(
        output,
        "pla_desk_calibration",
        [
            "path",
            "day",
            "date",
            "phase",
            "desk",
            "sigma",
            "eta",
            "rho",
            "rmse_bp",
            "success",
            "fallback",
        ],
        extras["calibration"],
    )


def run(config, output):
    count = configure(config.get("workers", 1))
    print(f"Using up to {count} worker processes for independent paths", flush=True)
    origin = date.fromisoformat(config["origin"])
    curve = NelsonSiegel(**config["initial_curve"])
    m = config["market"]
    worlds = {
        "hw_control": hw(curve, m["hw_a"], [m["hw_sigma"]]),
        "g2_base": g2(curve, "base"),
        "g2_second": g2(curve, "second"),
    }
    product = nc1(origin, origin, curve.discount)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(
        output,
        "discounted_bond_martingale",
        ["world", "clock", "initial_bond", "mean", "se", "ci_low", "ci_high"],
        [],
    )
    print("R1 | Curve bootstrap and European pricing", flush=True)
    r1(output, config, curve, worlds, product, origin)
    print("R2 | Bermudan solvers, bounds, calibration and Greeks", flush=True)
    grids = r2(output, config, worlds, product, origin)
    print("R3 | Exercise, hedge ledgers, annuity reconciliation and PLA", flush=True)
    r3(output, config, worlds, product, origin, grids)
    (output / "manifest.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
