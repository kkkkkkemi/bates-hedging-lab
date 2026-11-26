"""Monte Carlo Greeks for vanilla calls and discretely monitored up-and-outs."""

from time import perf_counter

import numpy as np
from scipy.special import ndtr, ndtri


def time_calendar(maturity, interval, first=None):
    first = interval if first is None else first
    return np.r_[np.arange(first, maturity - 1e-12, interval), maturity]


def _bs_samples(s, k, t, vol, r, z, dates, barrier=None):
    dts = np.diff(np.r_[0, dates])
    w = np.cumsum(np.sqrt(dts)[:, None] * z, axis=0)
    path = s * np.exp((r - vol * vol / 2) * dates[:, None] + vol * w)
    survive = np.ones(path.shape[1], bool) if barrier is None else (path < barrier).all(axis=0)
    terminal = path[-1]
    itm = (terminal > k) & survive
    disc = np.exp(-r * t)
    value = disc * np.maximum(terminal - k, 0) * survive
    return value, disc * terminal / s * itm, disc * terminal * (w[-1] - vol * t) * itm


def _bumped(value, s, vol, h, vol_bump=0.001):
    base = value(s, vol)
    up, down = value(s + h, vol), value(s - h, vol)
    return {
        "delta": (up - down) / (2 * h),
        "gamma": (up - 2 * base + down) / h**2,
        "vega": (value(s, vol + vol_bump) - value(s, vol - vol_bump)) / (2 * vol_bump),
    }


def _survival_value(s, k, t, vol, r, uniform, dates, barrier, stop_first=False):
    dts = np.diff(np.r_[0, dates])
    x = np.full(uniform.shape[1], np.log(s))
    weight = np.ones_like(x)
    for i, dt in enumerate(dts):
        mu = x + (r - vol**2 / 2) * dt
        sd = vol * np.sqrt(dt)
        cap = ndtr((np.log(barrier) - mu) / sd)
        if i == 0 and stop_first:
            x = mu + sd * ndtri(np.clip(cap * uniform[i], 1e-300, 1 - 1e-15))
            first_s, first_z = np.exp(x), (x - mu) / sd
        if i == len(dts) - 1 and not (i == 0 and stop_first):
            floor = ndtr((np.log(k) - mu) / sd)
            asset_mass = ndtr((np.log(barrier) - mu) / sd - sd) - ndtr((np.log(k) - mu) / sd - sd)
            value = np.exp(mu + sd**2 / 2) * asset_mass - k * np.maximum(cap - floor, 0)
            cash = np.exp(-r * t) * weight * np.maximum(value, 0)
            if stop_first:
                return cash, first_s, first_z
            return cash
        weight *= cap
        if not (i == 0 and stop_first):
            x = mu + sd * ndtri(np.clip(cap * uniform[i], 1e-300, 1 - 1e-15))
    cash = np.exp(-r * t) * weight * np.maximum(np.exp(x) - k, 0)
    return (cash, first_s, first_z) if stop_first else cash


def bs_estimators(s, k, t, vol, r, z, dates, barrier=None, bumps=(0.08, 0.35, 1.0)):
    measurements = {}
    before = perf_counter()
    value, delta, vega = _bs_samples(s, k, t, vol, r, z, dates, barrier)
    measurements["pathwise"] = ({"delta": delta, "vega": vega}, perf_counter() - before)
    before = perf_counter()
    dt = dates[0]
    sd = vol * np.sqrt(dt)
    if barrier is None:
        dt = t
        sd = vol * np.sqrt(t)
        score_z = (np.sqrt(np.diff(np.r_[0, dates]))[:, None] * z).sum(axis=0) / np.sqrt(t)
        score_vol = (score_z**2 - 1) / vol - np.sqrt(t) * score_z
    else:
        score_z = z[0]
        score_vol = ((z**2 - 1) / vol - np.sqrt(np.diff(np.r_[0, dates]))[:, None] * z).sum(axis=0)
    score = score_z / (s * sd)
    measurements["likelihood_ratio"] = (
        {
            "delta": value * score,
            "gamma": value * ((score_z**2 - 1) / (s * sd) ** 2 - score / s),
            "vega": value * score_vol,
        },
        perf_counter() - before,
    )
    if barrier is None:
        before = perf_counter()
        measurements["hybrid"] = ({"gamma": delta * (score - 1 / s)}, perf_counter() - before)
    for bump in bumps:
        before = perf_counter()

        def sample(ss, vv):
            return _bs_samples(ss, k, t, vv, r, z, dates, barrier)[0]

        measurements[f"crn_fd_{bump:g}"] = (_bumped(sample, s, vol, bump), perf_counter() - before)
    if barrier is not None:
        before = perf_counter()
        uniform = ndtr(z)

        def sample(ss, vv):
            return _survival_value(ss, k, t, vv, r, uniform, dates, barrier)

        measurements["one_step_survival"] = (_bumped(sample, s, vol, bumps[0]), perf_counter() - before)
        before = perf_counter()
        cash, _, first_z = _survival_value(s, k, t, vol, r, uniform, dates, barrier, True)
        # Integrate the discontinuous first monitor with survival sampling;
        # differentiate its Gaussian transition density on the surviving domain.
        measurements["survival_lr"] = (
            {
                "delta": cash * first_z / (s * sd),
                "gamma": cash * ((first_z**2 - 1) / (s * sd) ** 2 - first_z / (s * s * sd)),
            },
            perf_counter() - before,
        )
    return measurements


def heston_estimators(p, s, k, t, z, dates, barrier=None, bump=0.25):
    """Euler scores for the joint first-step (log spot, variance) transition."""
    dts = np.diff(np.r_[0, dates])
    rho = p.rho
    root_rho = np.sqrt(1 - rho**2)

    def sample(spot, vol, derivative=False):
        var = np.full(z.shape[-1], vol**2)
        x = np.full_like(var, np.log(spot))
        dx, dv = np.zeros_like(var), np.full_like(var, 2 * vol)
        alive = np.ones_like(var, bool)
        for dt, (zv, zs) in zip(dts, z):
            vv = np.maximum(var, 0)
            dw = np.sqrt(vv * dt)
            correlated = rho * zv + root_rho * zs
            if derivative:
                dp = dv * (var > 0)
                ddw = np.divide(dt * dp, 2 * dw, out=np.zeros_like(dp), where=dw > 0)
                dx += -dt * dp / 2 + ddw * correlated
                dv += -p.kappa * dt * dp + p.xi * ddw * zv
            x += (p.rate - p.jump_compensator - vv / 2) * dt + dw * correlated
            var += p.kappa * (p.theta - vv) * dt + p.xi * dw * zv
            if barrier is not None:
                alive &= x < np.log(barrier)
        st = np.exp(x)
        itm = alive & (st > k)
        disc = np.exp(-p.rate * t)
        val = disc * np.maximum(st - k, 0) * alive
        return val, disc * st / spot * itm, disc * st * dx * itm

    vol = np.sqrt(p.v0)
    before = perf_counter()
    val, delta, vega = sample(s, vol, True)
    out = {"pathwise": ({"delta": delta, "vega": vega}, perf_counter() - before)}
    before = perf_counter()
    dt = dates[0]
    zv, zs = z[0]
    sd = vol * np.sqrt(dt)
    ss = zs / (sd * root_rho)
    # Score with respect to initial vol via mean and covariance of Euler joint increment.
    mean_x = -vol * dt
    mean_v = 2 * vol * (1 - p.kappa * dt)
    mean_v_standard = mean_v / (p.xi * sd)
    mean_orth = (mean_x / sd - rho * mean_v_standard) / root_rho
    sv = (zs**2 + zv**2 - 2) / vol + mean_orth * zs + mean_v_standard * zv
    out["likelihood_ratio"] = (
        {
            "delta": val * ss / s,
            "gamma": val * ((zs**2 - 1) / (s * sd * root_rho) ** 2 - ss / s**2),
            "vega": val * sv,
        },
        perf_counter() - before,
    )
    if barrier is None:
        out["hybrid"] = ({"gamma": delta * (ss - 1) / s}, 0.0)
    before = perf_counter()
    out["crn_fd"] = (_bumped(lambda a, b: sample(a, b)[0], s, vol, bump), perf_counter() - before)
    return out


def jax_batch(s, k, vol, r, dates, barrier, paths, seed, p=None, replicates=8):
    """Reverse first order, forward-over-reverse gamma, and separate compile timing."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    steps = jnp.asarray(np.diff(np.r_[0, dates]))
    key = jax.random.key(seed)
    keys = jax.random.split(key, replicates)
    each = max(paths // replicates, 1)
    barrier_log = np.inf if barrier is None else np.log(barrier)

    def value(params, rng):
        spot, sigma = params
        random = jax.random.normal(rng, (len(dates), 2, each), dtype=jnp.float64)
        start = (jnp.full(each, jnp.log(spot)), jnp.full(each, sigma**2), jnp.ones(each, dtype=bool))

        def advance(state, args):
            x, v, alive = state
            dt, z = args
            vp = jnp.maximum(v, 0)
            sd = jnp.sqrt(jnp.maximum(vp * dt, 1e-30))
            if p is None:
                x = x + (r - sigma**2 / 2) * dt + sigma * jnp.sqrt(dt) * z[1]
            else:
                x = x + (r - vp / 2) * dt + sd * (p.rho * z[0] + jnp.sqrt(1 - p.rho**2) * z[1])
                v = v + p.kappa * (p.theta - vp) * dt + p.xi * sd * z[0]
            return (x, v, alive & (x < barrier_log)), None

        (terminal, _, alive), _ = jax.lax.scan(jax.checkpoint(advance), start, (steps, random))
        return jnp.exp(-r * steps.sum()) * jnp.mean(jnp.maximum(jnp.exp(terminal) - k, 0) * alive)

    gradient = jax.grad(value)

    def gamma(params, rng):
        return jax.jvp(lambda x: gradient(x, rng)[0], (params,), (jnp.array([1.0, 0.0]),))[1]

    fn = jax.jit(jax.vmap(gradient, in_axes=(None, 0)))
    second = jax.jit(jax.vmap(gamma, in_axes=(None, 0)))
    inp = jnp.array([s, vol])
    before = perf_counter()
    executable = fn.lower(inp, keys).compile()
    compile_gradient = perf_counter() - before
    before = perf_counter()
    second_exec = second.lower(inp, keys).compile()
    compile_gamma = perf_counter() - before
    before = perf_counter()
    grad = np.asarray(executable(inp, keys))
    seconds_gradient = perf_counter() - before
    before = perf_counter()
    gammas = np.asarray(second_exec(inp, keys))
    seconds_gamma = perf_counter() - before
    return {"delta": grad[:, 0], "vega": grad[:, 1], "gamma": gammas}, {
        "compile_first_s": compile_gradient,
        "compile_second_s": compile_gamma,
        "run_first_s": seconds_gradient,
        "run_second_s": seconds_gamma,
    }
