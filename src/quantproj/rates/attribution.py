"""Frozen Bermudan HPL/RTPL and 250-day rolling PLA."""

from dataclasses import replace

import numpy as np

from quantproj.core.hedge import positions
from quantproj.core.pnl import pla
from quantproj.rates import desk_g2
from quantproj.rates.bermudan import exercise_value, g2_adi
from quantproj.rates.calibration import fit_market, observe
from quantproj.rates.curves import bootstrap, nc1, swap
from quantproj.rates.hedging import (
    _bank_view,
    _swap_risk,
    coupon_cash,
    key_rate_buckets,
    swap_mark,
    trader_value,
)


def _frozen_mark(market, t, contract, idx, event, hedges, positions_, bank_at, a, settings):
    def df(ends):
        return market.curve.discount(ends - t)

    assets = sum(q * swap_mark(s, t, bank_at, df) for q, s in zip(positions_, hedges))
    if event == "exercise":
        liability = swap_mark(contract.tail(idx), t, bank_at, df)
    elif event == "expiry":
        liability = 0.0
    else:
        start = int(np.searchsorted(contract.dates[:-1], t + 1e-10))
        fit = fit_market(market, a)
        liability = trader_value(market, fit, contract.tail(start).shift(t), a, settings)
    return assets - liability


def _aligned_liability(market, t, contract, idx, event, bank_at, value_tail):
    """Front-office or two-factor desk liability on the shared quote snapshot."""
    if event == "exercise":
        return swap_mark(contract.tail(idx), t, bank_at, lambda ends: market.curve.discount(ends - t))
    if event == "expiry":
        return 0.0
    start = int(np.searchsorted(contract.dates[:-1], t + 1e-10))
    return value_tail(market, contract.tail(start).shift(t))


def _aligned_metrics(daily, hpl_column, rtpl_column):
    if not all(np.isfinite(r[hpl_column]) and np.isfinite(r[rtpl_column]) for r in daily):
        return [np.nan, np.nan, "undefined"]
    return list(pla([r[hpl_column] for r in daily], [r[rtpl_column] for r in daily]))


def desk_plan(settings):
    """Distinct G2++ desk configurations and the HPL/RTPL pair of each PLA comparison.

    The main comparison pairs ``front_office`` with ``risk_g2``. Each entry of
    ``pla_sensitivity`` overrides either side; identical configurations share one
    daily calibration and valuation.
    """
    if "front_office" not in settings or "risk_g2" not in settings:
        return {}, {}
    desks: dict[str, dict] = {}

    def register(label, cfg):
        for key, known in desks.items():
            if known == cfg:
                return key
        desks[label] = cfg
        return label

    pairs = {
        "main": (
            register("front_office", dict(settings["front_office"])),
            register("risk_g2", dict(settings["risk_g2"])),
        )
    }
    for variant in settings.get("pla_sensitivity", ()):
        name = variant["name"]
        fo = {**settings["front_office"], **variant.get("front_office", {})}
        risk = {**settings["risk_g2"], **variant.get("risk_g2", {})}
        pairs[name] = (register(f"{name}/front_office", fo), register(f"{name}/risk_g2", risk))
    return desks, pairs


def run_pla(model, dates, asofs, states, banks, product, price, warmup, settings, seed, path_ids=None):
    """Frozen-book HPL and RTPL over a fixed 250-day window.

    The diagnostic HPL uses the true G2++ marks. When the settings name a front-office
    and a two-factor risk desk, every desk also values the book on the same noisy quote
    snapshot. The aligned HPL then uses the front-office G2++ desk, as MAR32 data
    alignment allows, and the one- and two-factor risk models supply competing RTPLs.
    """
    a = settings["mean_reversion"]
    desks, pairs = desk_plan(settings)
    aligned = bool(desks)
    smile = settings.get("daily_smile_offsets_bp", ()) if aligned else ()
    pnl, metrics = [], []
    sensitivity_daily, sensitivity, calibration_log = [], [], []
    for path in range(states.shape[1]) if path_ids is None else path_ids:
        bank_at = _bank_view(dates, banks[:, path])
        contract, solution = product, price
        x0 = states[0, path]
        warm: dict[str, tuple | None] = {name: None for name in desks}
        stamp = {"day": None, "phase": "close"}

        def desk(name):
            cfg = desks[name]

            def value_tail(snapshot, tail):
                fit = desk_g2.calibrate(
                    snapshot,
                    cfg["reversions"],
                    wings=cfg.get("wings", True),
                    start=warm[name],
                    rho=cfg.get("rho"),
                )
                fallback = not fit.success and warm[name] is not None
                if fallback:
                    # A failed fit keeps the previous day's parameters and is not a warm start.
                    used = desk_g2.G2Fit(*warm[name], fit.rmse_bp, False)
                else:
                    used = fit
                    warm[name] = (fit.sigma, fit.eta, fit.rho)
                calibration_log.append(
                    [
                        path,
                        stamp["day"],
                        asofs[stamp["day"]].isoformat(),
                        stamp["phase"],
                        name,
                        used.sigma,
                        used.eta,
                        used.rho,
                        fit.rmse_bp,
                        int(fit.success),
                        int(fallback),
                    ]
                )
                return desk_g2.bermudan(snapshot, cfg["reversions"], used, tail, cfg["grid"], cfg["steps"])

            return value_tail

        def first_df(dates):
            return model.bond(0.0, dates, x0)

        hedges = []
        for tenor in settings["swap_tenors"]:
            s = swap(asofs[0], tenor, asofs[0])
            hedges.append(replace(s, fixed=s.fair_rate(first_df)))
        h = np.zeros(len(hedges))
        daily = []
        previous = None
        for day in range(warmup - 1, warmup + 250):
            t, x = dates[day], states[day, path]

            def disc(ends):
                return model.bond(t, ends, x)

            stamp.update(day=day, phase="close")
            snapshot_seed = np.random.SeedSequence([seed, path, day])
            market = observe(
                model,
                t,
                x,
                asofs[day],
                contract,
                np.random.default_rng(snapshot_seed),
                settings["ois_noise_bp"],
                settings["normal_noise_bp"],
                smile_offsets=smile,
            )
            idx = int(np.searchsorted(contract.dates[:-1], t - 1e-10))
            event = "none"
            if idx < len(contract.accrual) and abs(t - contract.dates[idx]) < 1e-9:
                h_ex = exercise_value(model, t, x, contract.tail(idx))
                if h_ex > max(solution.price(t, x, continue_only=True), 0.0):
                    event = "exercise"
                elif idx == len(contract.accrual) - 1:
                    event = "expiry"
            true_assets = sum(q * swap_mark(s, t, bank_at, disc) for q, s in zip(h, hedges))
            actual_liability = (
                h_ex
                if event == "exercise"
                else (
                    0.0
                    if event == "expiry"
                    else solution.price(
                        t,
                        x,
                        continue_only=idx < len(contract.accrual) and abs(t - contract.dates[idx]) < 1e-9,
                    )
                )
            )
            true_value = true_assets - actual_liability
            model_value = _frozen_mark(market, t, contract, idx, event, hedges, h, bank_at, a, settings)
            observed_assets = sum(
                q * swap_mark(s, t, bank_at, lambda ends: market.curve.discount(ends - t))
                for q, s in zip(h, hedges)
            )
            liabilities = {}
            for name in desks:
                liabilities[name] = _aligned_liability(market, t, contract, idx, event, bank_at, desk(name))
            if previous is not None:
                old_true, old_risk, old_quotes, old_vols, old_t, old_market, old_aligned = previous
                coupon = sum(q * coupon_cash(s, t, bank_at) for q, s in zip(h, hedges))
                hpl = true_value - old_true + coupon
                rtpl = model_value - old_risk + coupon

                def desk_pnl(name):
                    return observed_assets - liabilities[name] - old_aligned[name] + coupon

                if aligned:
                    hpl_fo, rtpl_g2 = (desk_pnl(name) for name in pairs["main"])
                    for variant, (fo_name, risk_name) in pairs.items():
                        if variant != "main":
                            sensitivity_daily.append(
                                [
                                    path,
                                    day,
                                    asofs[day].isoformat(),
                                    variant,
                                    desk_pnl(fo_name),
                                    desk_pnl(risk_name),
                                ]
                            )
                else:
                    hpl_fo = rtpl_g2 = np.nan
                previous_vols = dict(zip(np.round(old_market + old_t, 9), old_vols))
                base_vol = np.array(
                    [
                        previous_vols.get(round(t + i.dates[0], 9), v)
                        for i, v in zip(market.strip, market.vols)
                    ]
                )
                # Quotes that were deleted retain their availability mask for numerical Taylor only.
                base_vol = np.where(
                    np.isfinite(market.vols), np.where(np.isfinite(base_vol), base_vol, market.vols), np.nan
                )

                def evaluate(factor):
                    q = old_quotes + factor * (market.ois - old_quotes)
                    v = base_vol + factor * (market.vols - base_vol)
                    curve = bootstrap(market.swaps, q)
                    strip = tuple(
                        replace(contract, fixed=contract.fair_rate(curve.discount))
                        for contract in market.strip
                    )
                    bumped = replace(market, ois=q, vols=v, curve=curve, strip=strip)
                    return _frozen_mark(bumped, t, contract, idx, event, hedges, h, bank_at, a, settings)

                base = evaluate(0.0)
                step = 0.10
                plus, minus = evaluate(step), evaluate(-step)
                first = base - old_risk + coupon + (plus - minus) / (2 * step)
                second = first + (plus - 2 * base + minus) / (2 * step**2)
                daily.append(
                    [
                        path,
                        day,
                        asofs[day].isoformat(),
                        event,
                        hpl,
                        rtpl,
                        first,
                        second,
                        rtpl - first,
                        rtpl - second,
                        hpl - rtpl,
                        hpl_fo,
                        rtpl_g2,
                    ]
                )
            if event != "none":
                contract = nc1(asofs[day], asofs[0], disc)
                solution = g2_adi(
                    model, contract, t, n=settings["adi_grid"], steps_per_year=settings["adi_steps"]
                )
                # One close has one quote snapshot. The same seed redraws the identical OIS
                # quotes first, then the same noise sequence for the new trade's strip by expiry.
                market = observe(
                    model,
                    t,
                    x,
                    asofs[day],
                    contract,
                    np.random.default_rng(snapshot_seed),
                    settings["ois_noise_bp"],
                    settings["normal_noise_bp"],
                    smile_offsets=smile,
                )
                stamp.update(phase="roll")
                # The new trade is valued on the same close snapshot by every desk.
                for name in liabilities:
                    liabilities[name] = _aligned_liability(
                        market, t, contract, -1, "none", bank_at, desk(name)
                    )
            for j, s in enumerate(hedges):
                if s.dates[-1] <= t + 1e-9:
                    fresh = swap(asofs[day], settings["swap_tenors"][j], asofs[0])
                    hedges[j] = replace(fresh, fixed=fresh.fair_rate(disc))
                    h[j] = 0.0
            live_index = int(np.searchsorted(contract.dates[:-1], t + 1e-10))
            fit = fit_market(market, a)
            # Quote-risk fit uses the same hedging matrix and tie-break as experiment B.
            value, greeks, maps, _ = trader_value(
                market, fit, contract.tail(live_index).shift(t), a, settings, True
            )

            def obs(ends):
                return market.curve.discount(ends - t)

            rows = [_swap_risk(s, t, bank_at, market.curve, maps[0], obs) for s in hedges]
            bucket = key_rate_buckets(market, [s.dates[-1] - t for s in hedges])
            h = positions(
                bucket @ np.asarray(rows).T, bucket @ greeks[: len(market.ois)], h, settings["position_cap"]
            )
            continuation = (
                event == "none" and live_index > 0 and abs(contract.dates[live_index - 1] - t) < 1e-9
            )
            old_true = sum(q * swap_mark(s, t, bank_at, disc) for q, s in zip(h, hedges)) - solution.price(
                t, x, continue_only=continuation
            )
            new_assets = sum(q * swap_mark(s, t, bank_at, obs) for q, s in zip(h, hedges))
            old_risk = new_assets - value
            old_aligned = {name: new_assets - liability for name, liability in liabilities.items()}
            expiry_array = np.array([s.dates[0] for s in market.strip])
            previous = (old_true, old_risk, market.ois, market.vols, t, expiry_array, old_aligned)
        pnl.extend(daily)
        rho, ks, zone = pla([r[4] for r in daily], [r[5] for r in daily])
        metrics.append(
            [
                path,
                daily[0][2],
                daily[-1][2],
                rho,
                ks,
                zone,
                sum(r[3] == "exercise" for r in daily),
                sum(r[3] == "expiry" for r in daily),
                # Aligned snapshot: front-office G2++ HPL against each risk model.
                *_aligned_metrics(daily, 11, 5),
                *_aligned_metrics(daily, 11, 12),
            ]
        )
        for variant in pairs:
            if variant != "main":
                rows = [r for r in sensitivity_daily if r[0] == path and r[3] == variant]
                sensitivity.append([path, variant, *_aligned_metrics(rows, 4, 5)])
    return (
        pnl,
        metrics,
        {
            "sensitivity_daily": sensitivity_daily,
            "sensitivity": sensitivity,
            "calibration": calibration_log,
        },
    )
