"""Bermudan exercise policies, OIS cash flows and matched hedge books."""

from dataclasses import replace

import numpy as np

from quantproj.core.hedge import positions
from quantproj.core.ledger import Ledger
from quantproj.rates.bermudan import exercise_value
from quantproj.rates.calibration import calibrate, fit_market, observe
from quantproj.rates.curves import bootstrap, swap
from quantproj.rates.european import coupon_weights, european, european_factor_delta, normal_iv, observables
from quantproj.rates.greeks import (
    aad_price,
    calibration_ift,
    european_partials,
    interpolation_weights,
    ois_ift,
)
from quantproj.rates.models import root_cov

STRATEGIES = {
    "single_delta": "single",
    "key_rate": "swaps",
    "full_info_swaps": "swaps",
    "full_info_two": "two",
    "key_rate_vega": "all",
    "full_info_all": "all",
}


def _at(a, b):
    return bool(abs(a - b) < 1e-9)


def swap_components(contract, t, bank_at, *, before_payment=False):
    """Return discount dates and signed payer cash-flow weights for the surviving swap."""
    dates, accrual = contract.dates, contract.accrual
    if t < dates[0] - 1e-10:
        return dates, np.r_[1.0, -coupon_weights(contract)]
    side = "left" if before_payment else "right"
    idx = int(np.searchsorted(dates, t + (0 if before_payment else 1e-10), side=side)) - 1
    idx = max(idx, 0)
    if idx >= len(accrual):
        return np.empty(0), np.empty(0)
    ratio = bank_at(t) / bank_at(dates[idx])
    tail = contract.tail(idx)
    return np.r_[t, tail.dates[1:]], np.r_[ratio, -coupon_weights(tail)]


def coupon_cash(contract, t, bank_at):
    indices = np.where(abs(contract.dates[1:] - t) < 1e-9)[0]
    if not len(indices):
        return 0.0
    j = indices[0]
    return float(bank_at(t) / bank_at(contract.dates[j]) - 1 - contract.fixed * contract.accrual[j])


def swap_mark(contract, t, bank_at, discount, *, before_payment=False):
    dates, weights = swap_components(contract, t, bank_at, before_payment=before_payment)
    return float(weights @ discount(dates))


def _swap_risk(contract, t, bank_at, curve, quote_map, discount):
    dates, weights = swap_components(contract, t, bank_at)
    row = (weights * discount(dates)) @ interpolation_weights(curve.nodes, dates - t)[:, 1:] @ quote_map
    return row


def _risky_fit(market, fit, a, bump=1e-5):
    """Calibration and quote Jacobians. Refit finite differences on constrained samples."""
    if not fit.fallback and np.all(fit.sigma2 > 1e-12):
        return calibration_ift(market, fit, a)
    curve = market.curve
    # Keep the set of missing quotes fixed while bumping active calibration inputs.
    q, w = [], []
    for i in range(len(market.ois)):
        step = np.eye(len(market.ois))[i] * bump
        curves = [bootstrap(market.swaps, market.ois + sign * step) for sign in (1, -1)]
        states = [calibrate(x, market.strip, market.vols, a).w for x in curves]
        q.append((states[0] - states[1]) / (2 * bump))
    for i in range(len(market.vols)):
        if np.isfinite(market.vols[i]) and market.vols[i] > bump:
            step = np.eye(len(market.vols))[i] * bump
            states = [calibrate(curve, market.strip, market.vols + sign * step, a).w for sign in (1, -1)]
            w.append((states[0] - states[1]) / (2 * bump))
        else:
            w.append(np.zeros(len(market.vols)))
    return ois_ift(market), np.asarray(q).T, np.asarray(w).T


def trader_value(market, fit, tail, a, settings, greeks=False):
    value, dlog, dw, mask = aad_price(
        market.curve, fit, tail, a, settings["desk_grid"], settings["desk_steps"]
    )
    if not greeks:
        return value
    log_q, w_q, w_vol = _risky_fit(market, fit, a)
    bond_q = interpolation_weights(market.curve.nodes, tail.dates)[:, 1:] @ log_q
    return value, np.r_[dlog @ bond_q + dw @ w_q, dw @ w_vol], (log_q, w_q, w_vol), mask


def _swap_factor(contract, t, state, model, bank_at):
    dates, weights = swap_components(contract, t, bank_at)
    df = model.bond(t, dates, state)
    return -(weights * df) @ model.loads(dates - t)


def _state_gradient(fn, state):
    result = []
    for j in range(len(state)):
        step = np.eye(len(state))[j] * 2e-5
        result.append((fn(state + step) - fn(state - step)) / 4e-5)
    return np.array(result)


def _option_risk(contract, clock, market, fit, a, maps):
    shifted = contract.shift(clock)
    index = int(np.argmin(abs(fit.times - shifted.dates[0])))
    if not _at(fit.times[index], shifted.dates[0]):
        raise ValueError("Option hedge expiry is outside the remaining co-terminal strip.")
    log_q, w_q, w_vol = maps
    dlog, dw = european_partials(market.curve, shifted, fit.w[index], a)
    return np.r_[
        dlog @ interpolation_weights(market.curve.nodes, shifted.dates)[:, 1:] @ log_q + dw * w_q[index],
        dw * w_vol[index],
    ]


def _desk_decision(market, fit, contract, idx, clock, a, settings):
    tail = contract.tail(idx).shift(clock)
    intrinsic = 1 - coupon_weights(tail) @ market.curve.discount(tail.dates[1:])
    continuation = (
        0.0 if fit is None else trader_value(market, fit, contract.tail(idx + 1).shift(clock), a, settings)
    )
    return intrinsic > max(continuation, 0.0)


def _fitted_day(model, clock, state, asof, contract, rng, a, settings):
    market = observe(
        model,
        clock,
        state,
        asof,
        contract,
        rng,
        settings["ois_noise_bp"],
        settings["normal_noise_bp"],
        smile_offsets=settings.get("daily_smile_offsets_bp", ()),
    )
    clean = fit_market(market, a, clean=True) if len(market.strip) else None
    fit = fit_market(market, a) if len(market.strip) else None
    return market, fit, clean


def evaluate_exercise(
    model, contract, solution, dates, states, banks, asof_dates, a_values, settings, seed, path_ids=None
):
    """Independent holder evaluation with common paths and early-stop recording."""
    days = [int(np.argmin(abs(dates - t))) for t in contract.dates[:-1]]
    results = []
    for path in range(states.shape[1]) if path_ids is None else path_ids:
        paid = {str(a): 0.0 for a in a_values}
        stopped = {str(a): -1 for a in a_values}
        paid["adi"] = 0.0
        stopped["adi"] = -1
        disagreement = {str(a): 0 for a in a_values}
        for idx, i in enumerate(days):
            t, x, B = dates[i], states[i, path], banks[i, path]
            if all(event >= 0 for event in stopped.values()):
                break
            h = exercise_value(model, t, x, contract.tail(idx))
            opt = h > max(solution.price(t, x, continue_only=True), 0.0)
            if stopped["adi"] < 0 and opt:
                paid["adi"], stopped["adi"] = h / B, idx
            rng = np.random.default_rng(np.random.SeedSequence([seed, path, i]))
            market = observe(
                model,
                t,
                x,
                asof_dates[i],
                contract,
                rng,
                settings["ois_noise_bp"],
                settings["normal_noise_bp"],
            )
            for a in a_values:
                key = str(a)
                if stopped[key] >= 0:
                    continue
                fit = fit_market(market, a) if len(market.strip) else None
                choice = _desk_decision(market, fit, contract, idx, t, a, settings)
                disagreement[key] += int(opt != choice)
                if choice:
                    paid[key], stopped[key] = h / B, idx
        for a in a_values:
            key = str(a)
            results.append(
                [
                    path,
                    a,
                    paid["adi"],
                    paid[key],
                    paid["adi"] - paid[key],
                    stopped["adi"],
                    stopped[key],
                    disagreement[key],
                ]
            )
    return results


def _bank_view(times, banks):
    def lookup(time):
        idx = int(np.searchsorted(times, time - 1e-10))
        if idx >= len(times) or abs(times[idx] - time) > 1e-8:
            raise ValueError("A contractual accrual date is missing from the bank-account tape.")
        return banks[idx]

    return lookup


def hedge_path(model, contract, solution, dates, states, banks, asofs, settings, a, seed, *, holder="true"):
    """Matched, cash-settled terminal P&L for all hedge strategies and trade grids."""
    exercise_days = {int(np.argmin(abs(dates - t))): j for j, t in enumerate(contract.dates[:-1])}
    stop = max(exercise_days)
    bank_at = _bank_view(dates, banks)

    def first_discount(t):
        return model.bond(0, np.asarray(t), states[0])

    payer_swap = contract
    par_swaps = []
    for tenor in settings["swap_tenors"]:
        s = swap(asofs[0], tenor, asofs[0])
        par_swaps.append(replace(s, fixed=s.fair_rate(first_discount)))
    swaps = [payer_swap, *par_swaps]
    options = [contract.tail(j) for j in range(len(contract.accrual))]
    nsw, n = len(swaps), len(swaps) + len(options)
    cases = []
    for f in settings["frequencies"]:
        for costs in settings["cost_multipliers"]:
            for name in STRATEGIES:
                if model.dimension == 1 and name != "single_delta":
                    continue
                cases.append((f, costs, name))
    premium = solution.price(0, states[0])
    ledgers = {case: Ledger(premium, 1.0, np.zeros(n)) for case in cases}
    fits = []
    gap = ""
    for day in range(stop + 1):
        t, x, B = dates[day], states[day], banks[day]
        observation_date = day in exercise_days or any(day % f == 0 for f in settings["frequencies"])
        settlement = any(np.any(abs(s.dates[1:] - t) < 1e-9) for s in swaps)
        if not observation_date and not settlement:
            continue
        market, fit, clean = None, None, None

        def discount(ends):
            return model.bond(t, ends, x)

        cashflows = np.zeros(n)
        for j, s in enumerate(swaps):
            cashflows[j] = coupon_cash(s, t, bank_at)
        active = [j for j, s in enumerate(swaps) if s.dates[-1] > t + 1e-9]
        active_opt = []
        for j, o in enumerate(options, nsw):
            if _at(o.dates[0], t):
                cashflows[j] = max(exercise_value(model, t, x, o), 0.0)
            elif o.dates[0] > t:
                active_opt.append(j)
        for ledger in ledgers.values():
            ledger.accrue(B)
            ledger.settle(cashflows)
            for j in set(range(n)) - set(active + active_opt):
                ledger.holdings[j] = 0.0
        mark = np.zeros(n)
        for j in active:
            mark[j] = swap_mark(swaps[j], t, bank_at, discount)
        # Quotes are generated with a date-keyed seed on every decision or hedge date.
        if observation_date and model.dimension == 2:
            market, fit, clean = _fitted_day(
                model,
                t,
                x,
                asofs[day],
                contract,
                np.random.default_rng(np.random.SeedSequence([seed, day])),
                a,
                settings,
            )
            if fit is not None:
                for k, t_exp in enumerate(fit.times):
                    fits.append(
                        [
                            day,
                            t_exp,
                            np.sqrt(fit.sigma2[k]),
                            np.sqrt(clean.sigma2[k]),
                            fit.residual_bp[k],
                            int(fit.fallback),
                            fit.missing,
                            market.deleted,
                            int(market.curve_fallback),
                        ]
                    )
        if day in exercise_days:
            idx = exercise_days[day]
            h = exercise_value(model, t, x, contract.tail(idx))
            optimal = h > max(solution.price(t, x, continue_only=True), 0.0)
            desk = optimal if market is None else _desk_decision(market, fit, contract, idx, t, a, settings)
            choose = optimal if holder == "true" else desk
            gap += str(int(desk != optimal))
            if choose or day == stop:
                for j in active_opt:
                    mark[j] = european(model, t, x, options[j - nsw])
                for key, ledger in ledgers.items():
                    c = key[1]
                    spreads = _spreads(swaps, options, active, active_opt, t, x, model, settings, mark) * c
                    ledger.trade(np.zeros(n), mark, spreads)
                    ledger.cash -= h if choose else 0.0
                break
        if not observation_date:
            continue
        option_deltas = {}
        for j in active_opt:
            mark[j], option_deltas[j] = european_factor_delta(model, t, x, options[j - nsw])
        for j in range(1, nsw):
            if j not in active:
                rolled = swap(asofs[day], settings["swap_tenors"][j - 1], asofs[0])
                swaps[j] = replace(rolled, fixed=rolled.fair_rate(discount))
                active.append(j)
        active.sort()
        spreads = _spreads(swaps, options, active, active_opt, t, x, model, settings, mark)
        subset = active + active_opt
        # Orthogonal standard-normal coordinates give the local covariance risk objective.
        sd = model.sigma_at(t)
        shock = root_cov(sd[:, None] * model.corr * sd)
        skip = day in exercise_days
        frac = _state_gradient(lambda state: solution.price(t, state, continue_only=skip), x) @ shock
        model_factors = []
        for j in subset:
            if j < nsw:
                f = _swap_factor(swaps[j], t, x, model, bank_at)
            else:
                f = option_deltas[j]
            model_factors.append(f @ shock)
        true_load = np.asarray(model_factors).T
        locations = {j: i for i, j in enumerate(subset)}
        obs_load = None
        if market is not None and fit is not None:
            next_index = int(np.searchsorted(contract.dates[:-1], t + 1e-10))
            local = contract.tail(next_index).shift(t)
            _, obs_target, maps, _ = trader_value(market, fit, local, a, settings, True)

            def desk_df(ends):
                return market.curve.discount(ends - t)

            matrices = []
            for j in subset:
                if j < nsw:
                    row = _swap_risk(swaps[j], t, bank_at, market.curve, maps[0], desk_df)
                    matrices.append(np.r_[row, np.zeros(len(fit.w))])
                else:
                    matrices.append(_option_risk(options[j - nsw], t, market, fit, a, maps))
            obs_load = np.asarray(matrices).T
            # Key-rate risk is bucketed onto the hedge-swap maturities before the fit.
            bucket = key_rate_buckets(market, [swaps[j].dates[-1] - t for j in active[1:]])
            nq = len(market.ois)
            obs_load_b = np.vstack((bucket @ obs_load[:nq], obs_load[nq:]))
            obs_target_b = np.r_[bucket @ obs_target[:nq], obs_target[nq:]]
            # Model spot delta spans one direction, computed by shifting today's HW state.
            local_delta = obs_target[: len(market.ois)] @ _hw_shift_quotes(market, a)
            hedge_delta = matrices[0][: len(market.ois)] @ _hw_shift_quotes(market, a)
        buckets = {
            "single": [0],
            "swaps": active[1:],
            "two": [active[1], active[-1]],
            "all": active[1:] + active_opt,
        }
        for key, ledger in ledgers.items():
            every, c, name = key
            if day % every:
                continue
            ids = buckets[STRATEGIES[name]]
            vol_map = None
            if name == "key_rate_vega" and market is not None and fit is not None:
                # Options expiring by the next rebalance are not hedge tools. Their vega
                # node moves to the first expiry that survives the rebalance interval.
                horizon = dates[min(day + every, len(dates) - 1)]
                keep = [j for j in active_opt if options[j - nsw].dates[0] > horizon + 1e-9]
                ids = active[1:] + keep
                vol_map = vega_buckets(fit.times + t, [options[j - nsw].dates[0] for j in keep])
            m = [locations[j] for j in ids]
            prev = ledger.holdings[ids]
            if name == "full_info_two":
                singular = np.linalg.svd(true_load[:, m], compute_uv=False)
                if singular[-1] <= singular[0] * 1e-8:
                    raise ValueError("The two-swaps control must span both Gaussian factors.")
            if name.startswith("full_info") or model.dimension == 1:
                caps = np.inf if name == "full_info_two" else settings["position_cap"]
                target = positions(true_load[:, m], frac, prev, caps)
            elif name == "single_delta":
                target = positions(np.array([[hedge_delta]]), [local_delta], prev, settings["position_cap"])
            elif name == "key_rate":
                nb = len(bucket)
                target = positions(obs_load_b[:nb, m], obs_target_b[:nb], prev, settings["position_cap"])
            else:
                nb = len(bucket)
                load = np.vstack((obs_load_b[:nb, m], vol_map @ obs_load_b[nb:, m]))
                goal = np.r_[obs_target_b[:nb], vol_map @ obs_target_b[nb:]]
                target = positions(load, goal, prev, settings["position_cap"])
            holdings = np.zeros(n)
            holdings[ids] = target
            ledger.trade(holdings, mark, c * spreads)
    output = []
    for (d, c, key), ledger in ledgers.items():
        pv = ledger.discounted()
        output.append(
            [key, d, c, pv, pv * banks[-1], ledger.cost_pv, day, int(choose), gap.count("1"), len(gap)]
        )
    return output, fits


def key_rate_buckets(market, pillars):
    """Map OIS key-rate risk onto hedge maturities, preserving the total of each risk row.

    Each OIS node is split linearly between the two nearest hedge maturities. Nodes
    outside the hedge range go to the nearest hedge. A fit against the bucketed rows
    therefore keeps the parallel DV01 that an unweighted node fit gives up.
    """
    nodes = np.array([s.dates[-1] for s in market.swaps])
    pillars = np.asarray(pillars, dtype=float)
    order = np.argsort(pillars, kind="stable")
    sorted_p = pillars[order]
    weights = np.zeros((len(pillars), len(nodes)))
    for i, node in enumerate(nodes):
        k = int(np.searchsorted(sorted_p, node))
        if k == 0 or len(sorted_p) == 1:
            weights[order[0 if k == 0 else -1], i] = 1.0
        elif k == len(sorted_p):
            weights[order[-1], i] = 1.0
        else:
            left, right = sorted_p[k - 1], sorted_p[k]
            share = (node - left) / max(right - left, 1e-12)
            weights[order[k - 1], i] += 1 - share
            weights[order[k], i] += share
    return weights


def vega_buckets(nodes, pillars):
    """Move each vega node to the earliest hedge-option expiry on or after it.

    Nodes after the last pillar stay on the last pillar. Without pillars the vega
    rows drop out and the hedge reduces to bucketed key-rate delta.
    """
    nodes = np.asarray(nodes, dtype=float)
    pillars = np.asarray(pillars, dtype=float)
    weights = np.zeros((len(pillars), len(nodes)))
    if not len(pillars):
        return weights
    for i, node in enumerate(nodes):
        k = min(int(np.searchsorted(pillars, node - 1e-9)), len(pillars) - 1)
        weights[k, i] = 1.0
    return weights


def _hw_shift_quotes(market, a):
    rows = []
    for s in market.swaps:
        f, ann, df = observables(s, market.curve.discount)
        b = -np.expm1(-a * s.dates) / a
        d = -b * df
        rows.append((d[0] - d[-1] - f * s.accrual @ d[1:]) / ann)
    return np.asarray(rows)


def _spreads(swaps, options, active, active_opt, t, state, model, settings, marks):
    spreads = np.zeros(len(swaps) + len(options))
    for j in active:
        dates = swaps[j].dates
        pay = dates[1:] > t + 1e-10
        spreads[j] = (
            settings["swap_cost_bp"] * 1e-4 * (swaps[j].accrual[pay] @ model.bond(t, dates[1:][pay], state))
        )
    for j in active_opt:
        s = options[j - len(swaps)]
        df = model.bond(t, s.dates, state)
        ann = s.accrual @ df[1:]
        f = (df[0] - df[-1]) / ann
        expiry = s.dates[0] - t
        vega = ann * np.sqrt(expiry) / np.sqrt(2 * np.pi)
        # A fixed half-spread in quoted normal-volatility units, converted by fair vega.
        mid_vol = normal_iv(marks[j], f, s.fixed, expiry, ann)
        vega *= np.exp(-0.5 * ((f - s.fixed) / max(mid_vol * np.sqrt(expiry), 1e-14)) ** 2)
        spreads[j] = settings["option_half_spread_bp"] * 1e-4 * vega
    return spreads
