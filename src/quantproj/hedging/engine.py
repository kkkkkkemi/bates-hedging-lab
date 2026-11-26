"""Self-financing, common-path hedging and local covariance projection."""

from dataclasses import dataclass, replace

import numpy as np
from numpy.polynomial.hermite import hermgauss

from quantproj.calibration import START, calibrate, daily_update, filter_variance, generate_surface, physical
from quantproj.pricing import black
from quantproj.pricing.fourier import cos_greeks, cos_price
from quantproj.pricing.pde import adi_grid, bs_grid


@dataclass(frozen=True)
class HedgeSettings:
    strike: float = 100.0
    barrier: float = 125.0
    spot: float = 100.0
    stock_cost: float = 0.0005
    option_cap: float = 3.0
    # Longer than the 63-day target, so the tool is not a near copy of the hedged claim.
    tool_tenor_days: int = 126
    tool_roll_days: int = 10
    risk_aversion: float = 0.15
    paths_daily: int = 12
    calibration_modes: int = 96
    volatility_window: int = 20
    noise: float = 1.0


@dataclass(frozen=True)
class Leg:
    value: float
    delta: float
    variance: float


def local_covariance(s, v, p, legs, jump_changes):
    """Instantaneous quadratic-covariation matrix with compound-Poisson shocks."""
    sensitivities = np.array([[1, 0]] + [[leg.delta, leg.variance] for leg in legs])
    diffusion = v * np.array([[s * s, p.rho * p.xi * s], [p.rho * p.xi * s, p.xi**2]])
    matrix = sensitivities @ diffusion @ sensitivities.T
    if p.intensity:
        weights = hermgauss(jump_changes.shape[0])[1] / np.sqrt(np.pi)
        matrix += p.intensity * (jump_changes.T * weights) @ jump_changes
    return matrix


def projected_positions(cov, cap=None, two=False):
    if cov[0, 0] <= 1e-18:
        return 0.0, 0.0
    if not two:
        return cov[0, -1] / cov[0, 0], 0.0
    sigma = cov[:2, :2]
    b = cov[:2, -1]
    position = np.linalg.pinv(sigma, rcond=1e-12) @ b
    option = position[1] if cap is None else np.clip(position[1], -cap, cap)
    stock = (b[0] - sigma[0, 1] * option) / sigma[0, 0]
    return float(stock), float(option)


def _leg(p, s, k, t, modes):
    g = cos_greeks(p, s, k, t, n=modes)
    return Leg(float(cos_price(p, s, k, t, n=modes)), g["delta"], g["variance"])


def _smile_slope(surface, k, t):
    h = 0.004 * k
    return (surface.smile(t, k + h) - surface.smile(t, k - h)) / (2 * h)


def _interp_surface_halfspread(surface, t, k, clean_iv):
    from quantproj.calibration import noise_scale

    moneyness = np.log(k / (surface.spot * np.exp(surface.rates * t)))
    atm = surface.smile(t, surface.spot * np.exp(surface.rates * t), mode="clean")
    return (
        2.5758293035489004
        * noise_scale(moneyness, t, atm)
        * black.greeks(surface.spot, k, t, clean_iv, surface.rates)["vega"]
    )


def _initialize_book(names, npaths):
    return {name: np.zeros(npaths) for name in names}


def hedge_bs_control(market, maturity, rate, vol, frequencies, settings):
    s = market.paths.spot
    payoff = np.maximum(s[-1] - settings.strike, 0)
    results = {}
    for every in frequencies:
        cash = np.full(s.shape[1], black.price(settings.spot, settings.strike, maturity, vol, rate))
        holding = np.zeros_like(cash)
        steps = len(s) - 1
        previous = 0
        for i in range(0, steps, every):
            cash *= np.exp(rate * (i - previous) * maturity / steps)
            delta = black.greeks(s[i], settings.strike, maturity * (1 - i / steps), vol, rate)["delta"]
            cash -= (delta - holding) * s[i]
            holding = delta
            previous = i
        results[every] = (
            cash * np.exp(rate * (steps - previous) * maturity / steps) + holding * s[-1] - payoff
        )
    return results


def hedge_market(
    market,
    maturity,
    p,
    frequencies,
    cost_levels,
    settings,
    seed,
    *,
    substeps,
    surface_times,
    log_k,
    product="vanilla",
    world="bates",
    extension=False,
    path_ids=None,
):
    """Audit-series ledger for several strategies and frequency/cost settings.

    All sampled strategies at each trading decision share the observations.
    Surface refreshes and tool rolls occur on the same calendar at every frequency.
    Each path draws quotes from its own seed, so ``path_ids`` runs any subset of
    paths with the same results as the full loop. Unlisted paths stay zero.
    """
    s_path, v_path = market.paths.spot, market.paths.variance
    steps, npaths = s_path.shape[0] - 1, s_path.shape[1]
    dt, r, k = maturity / steps, p.rate, settings.strike
    modes = settings.calibration_modes
    is_barrier = product == "barrier"
    barrier = settings.barrier
    day_interval = dt * substeps
    if is_barrier:
        market_grid = adi_grid(
            p,
            k,
            maturity,
            nx=100,
            nv=24,
            steps=steps * max(2, int(np.ceil(40 / steps))),
            barrier=barrier,
            monitor=day_interval,
            store=True,
        )
        barrier_indices = {round(time, 11): i for i, time in enumerate(market_grid.days)}
        barrier_marks = {
            i: market_grid.interpolator(barrier_indices[round(i * dt, 11)]) for i in range(steps)
        }
    else:
        market_grid = None
        barrier_marks = {}
    normals = np.sqrt(2) * hermgauss(14)[0]
    if world == "heston":
        names = ["true_mv", "true_two", "true_two_uncapped"]
    else:
        names = [
            "fixed_iv",
            "sticky_strike",
            "sticky_delta",
            "realized_vol",
            "oracle_vol",
            "calibrated_mv",
            "calibrated_mv_approx",
            "calibrated_weekly_mv",
            "delta_vega",
            "calibrated_two",
            "full_stock",
            "full_two",
            "no_trade_band",
        ]
        if extension:
            names += [
                "bates_filtered_true_stock",
                "bates_filtered_true_two",
                "bates_filtered_calibrated_stock",
                "bates_filtered_calibrated_two",
            ]
    cases = [
        (name, q, c)
        for name in names
        for q in frequencies
        for c in cost_levels
        if name != "no_trade_band" or c > 0
    ]
    pnl = _initialize_book(cases, npaths)
    cost = _initialize_book(cases, npaths)
    drift_rows = []
    corrections = []
    diagnostics = []
    premium = np.nan
    for a in range(npaths) if path_ids is None else path_ids:
        rng = np.random.default_rng(seed + a * 11009)
        start = replace(START, rate=r)
        calibrated = weekly = pseudo = start
        bates_cal = replace(start, intensity=0.4)
        state_params = replace(p, v0=start.v0)
        tool_expiry = settings.tool_tenor_days * substeps
        tool_k = s_path[0, a] * np.exp(r * tool_expiry * dt)
        history = []
        observed = None
        initial_iv = None
        last_iv = last_tool_iv = np.sqrt(START.v0)
        smile_slope = 0.0
        start_s = s_path[0, a]
        if is_barrier:
            premium = float(barrier_marks[0].ev(start_s, p.v0))
        else:
            premium = float(cos_price(p, start_s, k, maturity, n=modes))
        wallet = np.full(len(cases), premium)
        stock, option, total_cost = (np.zeros(len(cases)) for _ in range(3))
        knocked_out = False
        for i in range(steps + 1):
            s, v = float(s_path[i, a]), float(v_path[i, a])
            t = max(maturity - i * dt, 0)
            refresh = i % substeps == 0
            if i:
                wallet *= np.exp(r * dt)
                total_cost *= np.exp(r * dt)
            remaining_tool = max((tool_expiry - i) * dt, 0)
            true_p = replace(p, v0=v)
            tool_mark = float(cos_price(true_p, s, tool_k, remaining_tool, n=modes))

            def close(stock_price, option_price):
                if observed is None:
                    option_spread = 0.0
                else:
                    maturity_tool = max(remaining_tool, dt)
                    iv = observed.smile(maturity_tool, tool_k, mode="clean")
                    option_spread = _interp_surface_halfspread(observed, maturity_tool, tool_k, iv)
                charges = np.array([c for _, _, c in cases])
                fees = charges * abs(stock) * stock_price + (charges > 0) * abs(option) * option_spread
                return stock * stock_price + option * option_price - fees, fees

            if is_barrier and i > 0 and refresh and s >= barrier:
                knocked_out = True
            if knocked_out or i == steps:
                cashflow, fees = close(s, tool_mark)
                wallet += cashflow
                total_cost += fees
                payoff = 0.0 if knocked_out else max(s - k, 0)
                values = wallet - payoff
                if i < steps:
                    values *= np.exp(r * (steps - i) * dt)
                    total_cost *= np.exp(r * (steps - i) * dt)
                for j, case in enumerate(cases):
                    pnl[case][a], cost[case][a] = values[j], total_cost[j]
                break
            if refresh:
                history.append(s)
                if world != "heston":
                    dates = np.unique(
                        np.r_[surface_times, t, remaining_tool, settings.tool_tenor_days * day_interval]
                    )
                    dates = dates[dates > 1e-10]
                    target_node = np.log(k / (s * np.exp(r * t)))
                    tool_node = np.log(tool_k / (s * np.exp(r * remaining_tool)))
                    nodes = np.unique(np.round(np.r_[log_k, target_node, tool_node], 12))
                    observed = generate_surface(p, s, v, dates, nodes, rng, settings.noise, modes)
                    day = i // substeps
                    calibrated, fail, rmse = daily_update(observed, calibrated, day, modes=modes)
                    weekly, wfail, wrmse = daily_update(observed, weekly, day, weekly=True, modes=modes)
                    clean_fit = calibrate(observed, pseudo, clean=True, modes=modes)
                    clean_fail = not clean_fit.success or clean_fit.boundary
                    if not clean_fail:
                        pseudo = clean_fit.params
                    drift_rows.append(
                        [
                            a,
                            day,
                            int(fail),
                            int(wfail),
                            int(clean_fail),
                            rmse,
                            wrmse,
                            *physical(calibrated),
                            *physical(weekly),
                            *physical(pseudo),
                        ]
                    )
                    diagnostics.append(
                        (1 - observed.mask.mean(), np.mean(abs(observed.noise) <= observed.half_spread_iv))
                    )
                    if extension:
                        state_params = filter_variance(observed, state_params, day_interval, modes)
                        if day % 5 == 0:
                            bates_cal, _, _ = daily_update(observed, bates_cal, day, bates=True, modes=modes)
                        else:
                            bates_cal = filter_variance(observed, bates_cal, day_interval, modes)
            if not any(i % every == 0 for every in frequencies) and not refresh:
                continue
            roll = refresh and remaining_tool <= settings.tool_roll_days * day_interval
            if roll:
                charges = np.array([c for _, _, c in cases])
                if observed is None:
                    spread = 0.0
                else:
                    old_iv = observed.smile(max(remaining_tool, dt), tool_k, mode="clean")
                    spread = _interp_surface_halfspread(observed, max(remaining_tool, dt), tool_k, old_iv)
                fee = (charges > 0) * abs(option) * spread
                wallet += option * tool_mark - fee
                total_cost += fee
                option.fill(0)
                tool_k, tool_expiry = (
                    s * np.exp(r * settings.tool_tenor_days * day_interval),
                    i + settings.tool_tenor_days * substeps,
                )
                remaining_tool = (tool_expiry - i) * dt
                tool_mark = float(cos_price(true_p, s, tool_k, remaining_tool, n=modes))
            elif not any(i % every == 0 for every in frequencies):
                continue

            next_monitor = (substeps - i % substeps) * dt
            interpolators = {}
            bs_interpolators = {}

            def barrier_interpolator(pp):
                if pp == true_p:
                    return barrier_marks[i]
                if pp not in interpolators:
                    interpolators[pp] = adi_grid(
                        pp,
                        k,
                        t,
                        nx=56,
                        nv=14,
                        steps=max(int(t / day_interval) * 2, 12),
                        barrier=barrier,
                        monitor=day_interval,
                        first=next_monitor,
                    ).interpolator()
                return interpolators[pp]

            def market_leg(pp):
                if not is_barrier:
                    return _leg(pp, s, k, t, modes)
                f = barrier_interpolator(pp)
                return Leg(float(f.ev(s, pp.v0)), float(f.ev(s, pp.v0, dx=1)), float(f.ev(s, pp.v0, dy=1)))

            def black_leg(vol):
                g = black.greeks(s, k, t, vol, r)
                if not is_barrier:
                    return g

                def fn(volatility):
                    if volatility not in bs_interpolators:
                        bs_interpolators[volatility] = bs_grid(
                            k,
                            t,
                            volatility,
                            r,
                            barrier=barrier,
                            monitor=day_interval,
                            first=next_monitor,
                            nx=220,
                            steps=max(12, int(t / day_interval) * 3),
                        )
                    return bs_interpolators[volatility]

                f = fn(vol)
                return {
                    "value": float(f(s)),
                    "delta": float(f(s, 1)),
                    "gamma": float(f(s, 2)),
                    "vega": float(
                        (fn(vol + 0.002)(s) - fn(max(vol - 0.002, 0.001))(s))
                        / (vol + 0.002 - max(vol - 0.002, 0.001))
                    ),
                }

            tool = _leg(true_p, s, tool_k, remaining_tool, modes)

            def local(pp):
                tr = market_leg(pp)
                unit = _leg(pp, s, tool_k, remaining_tool, modes)
                s_jump = s * np.exp(pp.jump_mean + pp.jump_vol * normals)
                if pp.intensity and is_barrier:
                    special = barrier_interpolator(pp)
                    jump_v = special.ev(s_jump, np.full(len(s_jump), pp.v0))
                elif pp.intensity:
                    jump_v = cos_price(pp, s_jump, k, t, n=modes)
                else:
                    jump_v = np.zeros_like(s_jump)
                jump_u = (
                    cos_price(pp, s_jump, tool_k, remaining_tool, n=modes)
                    if pp.intensity
                    else np.zeros_like(s_jump)
                )
                increments = np.column_stack((s_jump - s, jump_u - unit.value, jump_v - tr.value))
                return local_covariance(s, pp.v0, pp, [unit, tr], increments)

            positions = {}
            if world == "heston":
                target = market_leg(true_p)
                cov = local(true_p)
                h = target.variance / tool.variance if abs(tool.variance) > 1e-9 else 0.0
                limited = np.clip(h, -settings.option_cap, settings.option_cap)
                positions.update(
                    true_mv=projected_positions(cov),
                    true_two=(target.delta - limited * tool.delta, limited),
                    true_two_uncapped=(target.delta - h * tool.delta, h),
                )
                tool_spread = 0.0
            else:
                assert observed is not None
                if refresh:
                    quote_iv = observed.smile(t, k) if is_barrier else observed.quote_or_smile(t, k)
                    if np.isfinite(quote_iv):
                        last_iv = quote_iv
                    if initial_iv is None:
                        initial_iv = last_iv
                    fresh_tool_iv = observed.quote_or_smile(remaining_tool, tool_k)
                    if np.isfinite(fresh_tool_iv):
                        last_tool_iv = fresh_tool_iv
                    smile_slope = _smile_slope(observed, k, t)
                iv = float(np.clip(last_iv, 0.008, 2.0))
                realized = np.diff(np.log(history[-settings.volatility_window - 1 :]))
                historical = (
                    float(np.sqrt(np.mean(realized**2) / day_interval)) if len(realized) >= 5 else initial_iv
                )
                g = black_leg(iv)
                correction = float(-k / s * g["vega"] * smile_slope)
                positions.update(
                    fixed_iv=(black_leg(initial_iv)["delta"], 0.0),
                    sticky_strike=(g["delta"], 0.0),
                    sticky_delta=(g["delta"] + correction, 0.0),
                    realized_vol=(black_leg(max(historical, 0.008))["delta"], 0.0),
                    oracle_vol=(black_leg(max(np.sqrt(v), 0.008))["delta"], 0.0),
                    no_trade_band=(g["delta"], 0.0),
                )
                cg = market_leg(calibrated)
                wg = market_leg(weekly)
                exact_correction = calibrated.rho * calibrated.xi / s * cg.variance
                approx_correction = (
                    calibrated.rho * calibrated.xi / s * g["vega"] / (2 * np.sqrt(calibrated.v0))
                )
                positions.update(
                    calibrated_mv=(cg.delta + exact_correction, 0.0),
                    calibrated_mv_approx=(g["delta"] + approx_correction, 0.0),
                    calibrated_weekly_mv=(wg.delta + weekly.rho * weekly.xi / s * wg.variance, 0.0),
                )
                corrections.append((a, i, correction, exact_correction, approx_correction))
                bg = black.greeks(s, tool_k, remaining_tool, last_tool_iv, r)
                h = np.clip(g["vega"] / max(bg["vega"], 1e-8), -settings.option_cap, settings.option_cap)
                positions["delta_vega"] = (g["delta"] - h * bg["delta"], h)
                hc = _leg(calibrated, s, tool_k, remaining_tool, modes)
                opt = np.clip(cg.variance / max(hc.variance, 1e-8), -settings.option_cap, settings.option_cap)
                net = cg.variance - opt * hc.variance
                positions["calibrated_two"] = (
                    cg.delta - opt * hc.delta + calibrated.rho * calibrated.xi / s * net,
                    opt,
                )
                cov = local(true_p)
                positions["full_stock"] = projected_positions(cov)
                positions["full_two"] = projected_positions(cov, settings.option_cap, True)
                if extension:
                    for prefix, estimated in [
                        ("bates_filtered_true", replace(p, v0=state_params.v0)),
                        ("bates_filtered_calibrated", bates_cal),
                    ]:
                        matrix = local(estimated)
                        positions[prefix + "_stock"] = projected_positions(matrix)
                        positions[prefix + "_two"] = projected_positions(matrix, settings.option_cap, True)
                live_iv = black.implied_vol(tool_mark, s, tool_k, remaining_tool, r)
                tool_spread = _interp_surface_halfspread(observed, remaining_tool, tool_k, live_iv)
            for j, (name, every, c) in enumerate(cases):
                if i % every and not roll:
                    continue
                target_s, target_u = positions[name]
                # A roll off the trading grid re-hedges both legs, so the stock
                # position matches the new tool instead of the expired one.
                if i % every and abs(target_u) < 1e-12:
                    continue
                if name == "no_trade_band":
                    width = (1.5 * c * s * g["gamma"] ** 2 / settings.risk_aversion) ** (1 / 3)
                    target_s = np.clip(stock[j], target_s - width, target_s + width)
                ds, du = target_s - stock[j], target_u - option[j]
                fee = c * abs(ds) * s + (c > 0) * abs(du) * tool_spread
                wallet[j] -= ds * s + du * tool_mark + fee
                total_cost[j] += fee
                stock[j], option[j] = target_s, target_u
    return {
        "pnl": pnl,
        "cost": cost,
        "calibration": drift_rows,
        "corrections": corrections,
        "quotes": diagnostics,
        "dt": dt,
        "premium": premium,
    }


def decomposition(market, p, maturity, strike, vol, frequencies):
    s, v = market.paths.spot, market.paths.variance
    steps = len(s) - 1
    t = np.linspace(0, maturity, steps + 1)
    dt = t[1]
    r = p.rate
    premium = float(cos_price(p, s[0, 0], strike, maturity))
    start_difference = np.exp(r * maturity) * (premium - black.price(s[0, 0], strike, maturity, vol, r))
    gamma = black.greeks(s[:-1], strike, (maturity - t[:-1])[:, None], vol, r)["gamma"]
    diffusion = -0.5 * np.sum(
        np.exp(r * (maturity - t[:-1, None])) * gamma * s[:-1] ** 2 * (v[:-1] - vol**2) * dt, axis=0
    )
    jump = np.zeros(s.shape[1])
    for event in market.events:
        rem = maturity - event.time
        now = black.greeks(event.before, strike, rem, vol, r)
        jump[event.path] -= np.exp(r * rem) * (
            black.price(event.after, strike, rem, vol, r)
            - now["value"]
            - now["delta"] * (event.after - event.before)
        )
    predicted = start_difference + diffusion + jump
    pnl = hedge_bs_control(market, maturity, r, vol, frequencies, HedgeSettings(strike=strike, spot=s[0, 0]))
    for every in frequencies:
        pnl[every] += start_difference
    return predicted, pnl, diffusion, jump
