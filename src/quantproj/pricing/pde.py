"""Log-spot finite differences with date-aligned discrete monitoring."""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy import sparse
from scipy.interpolate import CubicSpline, RectBivariateSpline
from scipy.sparse.linalg import splu

from quantproj.core.numerics.adi import THETA, adi_step
from quantproj.models import Bates
from quantproj.pricing.black import price as black_price


def _spot_grid(strike, nx, barrier):
    h = 3.6 / nx
    if barrier is None:
        return np.log(strike) + h * (np.arange(nx + 1) - nx // 2)
    # The discontinuity lies between adjacent cell centers at every resolution.
    n_left = round(nx * 0.58)
    return np.log(barrier) + h * (np.arange(nx + 1) - n_left + 0.5)


def _monitor_nodes(t, dt, first=None):
    first = dt if first is None else first
    if dt <= 0 or first <= 0:
        raise ValueError("Monitoring intervals must be positive.")
    dates = np.arange(first, t + 1e-12, dt)
    return np.unique(np.round(np.r_[dates[dates < t - 1e-12], t], 13))


def bs_grid(
    strike,
    t,
    vol,
    r=0.02,
    *,
    put=False,
    barrier=None,
    monitor=1 / 252,
    first=None,
    nx=400,
    steps=200,
    american=False,
    continuous=False,
):
    """CN with implicit-Euler half steps at payoff/monitoring discontinuities.

    The input state is immediately after the previous fixing or between fixings.
    For a continuous barrier use strictly zero value on and above the barrier.
    """
    x = _spot_grid(strike, nx, barrier)
    if continuous:
        if barrier is None:
            raise ValueError("A continuous barrier needs a barrier level.")
        x = np.r_[x[x < np.log(barrier)], np.log(barrier)]
    s = np.exp(x)
    payoff = np.maximum((strike - s) if put else (s - strike), 0)
    alive = np.ones_like(s, bool) if barrier is None else s < barrier
    value = payoff * alive
    rows = range(1, len(s) - 1)
    mu = r - vol**2 / 2
    op = sparse.lil_matrix((len(s), len(s)))
    for i in rows:
        left = x[i] - x[i - 1]
        right = x[i + 1] - x[i]
        op[i, i - 1] = (vol**2 - mu * right) / (left * (left + right))
        op[i, i] = (-(vol**2) + mu * (right - left)) / (left * right) - r
        op[i, i + 1] = (vol**2 + mu * left) / (right * (left + right))
    op = op.tocsc()
    identity = sparse.eye(len(s), format="csc")
    dates = [] if barrier is None or continuous else _monitor_nodes(t, monitor, first)[:-1]
    segments = np.r_[t, dates[::-1], 0.0]
    elapsed = 0.0
    factors = {}
    for seg, (old, new) in enumerate(zip(segments[:-1], segments[1:])):
        n = max(1, int(np.ceil((old - new) * steps / t - 1e-10)))
        # Split the first step into two fully implicit half steps.
        sizes = [(old - new) / (2 * n)] * 2 + [(old - new) / n] * (n - 1)
        for i, htime in enumerate(sizes):
            theta = 1.0 if i < 2 else 0.5
            elapsed += htime
            key = (round(htime, 14), theta)
            if key not in factors:
                a = identity - theta * htime * op
                factors[key] = a if american else splu(a)
            rhs = value + (1 - theta) * htime * (op @ value)
            rhs[0] = max(strike * np.exp(-r * elapsed) - s[0], 0) if put else 0.0
            rhs[-1] = max(s[-1] - strike * np.exp(-r * elapsed), 0) if not put and barrier is None else 0.0
            if american:
                rhs[0] = max(rhs[0], payoff[0])
                rhs[-1] = max(rhs[-1], payoff[-1])
                value = _psor(factors[key], rhs, payoff, value)
            else:
                value = factors[key].solve(rhs)
        if seg < len(segments) - 2:
            value *= alive
    return CubicSpline(s, value, extrapolate=True)


def _psor(matrix, rhs, payoff, initial, tol=1e-9):
    d = matrix.diagonal()
    low = matrix.diagonal(-1)
    up = matrix.diagonal(1)
    value = initial.copy()
    for _ in range(16000):
        change = 0.0
        for i, diagonal in enumerate(d):
            adjusted = rhs[i]
            if i:
                adjusted -= low[i - 1] * value[i - 1]
            if i + 1 < len(d):
                adjusted -= up[i] * value[i + 1]
            new = max(payoff[i], value[i] + 1.2 * (adjusted / diagonal - value[i]))
            change = max(change, abs(value[i] - new))
            value[i] = new
        if change < tol:
            return value
    raise RuntimeError("PSOR did not converge.")


@dataclass
class ADIGrid:
    days: np.ndarray
    spot: np.ndarray
    variance: np.ndarray
    values: np.ndarray

    def interpolator(self, index=-1):
        return RectBivariateSpline(self.spot, self.variance, self.values[index], s=0)


def adi_grid(
    p: Bates,
    strike,
    t,
    *,
    nx=106,
    nv=32,
    steps=126,
    vmax=0.6,
    method="hv",
    put=False,
    american=False,
    barrier=None,
    monitor=1 / 252,
    first=None,
    store=False,
    jump_order=16,
):
    """Douglas, Craig–Sneyd and Hundsdorfer–Verwer ADI with explicit jumps.

    Project at contractual monitor dates only. American exercise uses the
    Ikonen–Toivanen multiplier update on the same time grid.
    """
    theta = THETA[method]
    x = _spot_grid(strike, nx, barrier)
    s = np.exp(x)
    v = np.linspace(0, vmax, nv + 1)
    dx, dv = x[1] - x[0], v[1] - v[0]
    shape = (len(s), len(v))
    size = np.prod(shape)
    aa = [sparse.lil_matrix((size, size)) for _ in range(3)]
    a0, ax, av = aa
    jump = sparse.lil_matrix((size, size))
    nodes, weights = hermgauss(jump_order)
    if p.intensity:
        ex = np.clip((x[:, None] + p.jump_mean + np.sqrt(2) * p.jump_vol * nodes - x[0]) / dx, 0, nx)
        jl = np.minimum(ex.astype(int), nx - 1)
        jf = ex - jl
    for i in range(1, nx):
        for j, vv in enumerate(v):
            m = i * (nv + 1) + j
            mu = p.rate - p.jump_compensator - vv / 2
            drift = p.kappa * (p.theta - vv)
            a = vv / (2 * dx**2)
            if a >= abs(mu) / (2 * dx):
                ax[m, m - (nv + 1)] = a - mu / (2 * dx)
                ax[m, m + (nv + 1)] = a + mu / (2 * dx)
            else:
                ax[m, m - (nv + 1)] = a + max(-mu, 0) / dx
                ax[m, m + (nv + 1)] = a + max(mu, 0) / dx
            ax[m, m] = -ax[m, m - (nv + 1)] - ax[m, m + (nv + 1)] - p.rate / 2
            if 0 < j < nv:
                diff = p.xi**2 * vv / (2 * dv**2)
                if diff >= abs(drift) / (2 * dv):
                    av[m, m - 1], av[m, m + 1] = diff - drift / (2 * dv), diff + drift / (2 * dv)
                else:
                    av[m, m - 1], av[m, m + 1] = diff + max(-drift, 0) / dv, diff + max(drift, 0) / dv
                mixed = p.rho * p.xi * vv / (4 * dx * dv)
                for ix, jv in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                    a0[m, m + ix * (nv + 1) + jv] = mixed * ix * jv
            elif j == 0:
                av[m, m + 1] = max(drift, 0) / dv
            else:
                av[m, m - 1] = max(-drift, 0) / dv
            av[m, m] = -sum(av.data[m]) - p.rate / 2
            if p.intensity:
                for left, f, w in zip(jl[i], jf[i], weights / np.sqrt(np.pi)):
                    jump[m, left * (nv + 1) + j] += p.intensity * w * (1 - f)
                    jump[m, (left + 1) * (nv + 1) + j] += p.intensity * w * f
                jump[m, m] -= p.intensity
    ax, av = ax.tocsc(), av.tocsc()
    mixed = a0.tocsc()
    jump = jump.tocsc()
    a0 = mixed + jump
    total = a0 + ax + av
    identity = sparse.eye(size, format="csc")
    payoff = np.broadcast_to(np.maximum((strike - s) if put else (s - strike), 0)[:, None], shape).ravel()
    alive = np.repeat(np.ones_like(s, bool) if barrier is None else s < barrier, len(v))
    value = payoff * alive
    multiplier = np.zeros_like(value)
    dates = [] if barrier is None else _monitor_nodes(t, monitor, first)[:-1]
    calendar = np.r_[t, dates[::-1], 0.0]
    times, snapshots = [t], [value.reshape(shape).copy()]
    factors = {}
    for seg, (old, new) in enumerate(zip(calendar[:-1], calendar[1:])):
        n = max(1, int(np.ceil((old - new) * steps / t - 1e-10)))
        step = (old - new) / n
        key = round(step, 13)
        if key not in factors:
            factors[key] = splu(identity - theta * step * ax), splu(identity - theta * step * av)
        sx, sv = factors[key]
        for sub in range(n):
            current = old - (sub + 1) * step
            remaining = t - current
            low = max(strike * np.exp(-p.rate * remaining) - s[0], 0) if put else 0.0
            high = (
                max(s[-1] - strike * np.exp(-p.rate * remaining), 0) if not put and barrier is None else 0.0
            )
            if american:
                low, high = max(low, payoff[0]), max(high, payoff[-1])

            def boundary(arr):
                arr[: len(v)] = low
                arr[-len(v) :] = high
                return arr

            y = adi_step(
                value,
                step,
                method,
                a0,
                ax,
                av,
                total,
                (sx.solve, sv.solve),
                boundary=boundary,
                source=multiplier,
            )
            if american:
                value = np.maximum(y - step * multiplier, payoff)
                multiplier = np.maximum(0, multiplier + (payoff - y) / step)
            else:
                value = y
            if store:
                times.append(current)
                snapshots.append(value.reshape(shape).copy())
        if seg < len(calendar) - 2:
            value *= alive
        if not store and seg == len(calendar) - 2:
            times.append(new)
            snapshots.append(value.reshape(shape).copy())
    return ADIGrid(np.array(times), s, v, np.array(snapshots))


def continuous_upout(s, k, t, b, sigma, r=0.02):
    """Reiner–Rubinstein zero-rebate up-and-out call, K < B."""
    s = np.asarray(s)
    if k >= b:
        return np.zeros_like(s)

    def truncated(x):
        return (
            black_price(x, k, t, sigma, r)
            - black_price(x, b, t, sigma, r)
            - (b - k) * np.exp(-r * t) * _digital(x, b, t, sigma, r)
        )

    beta = 2 * r / sigma**2 - 1
    return np.where(s < b, np.maximum(truncated(s) - (b / s) ** beta * truncated(b * b / s), 0), 0)


def _digital(s, k, t, vol, r):
    from scipy.special import ndtr

    return ndtr((np.log(s / k) + (r - vol**2 / 2) * t) / (vol * np.sqrt(t)))


def bs_discrete_reference(s, k, t, sigma, r, b, interval, first, nx=700, steps=700):
    """Spatial/time Richardson extrapolation on grids with aligned barrier cells."""
    coarse = bs_grid(k, t, sigma, r, barrier=b, monitor=interval, first=first, nx=nx, steps=steps)
    fine = bs_grid(k, t, sigma, r, barrier=b, monitor=interval, first=first, nx=2 * nx, steps=2 * steps)
    return np.array([(4 * fine(s, d) - coarse(s, d)) / 3 for d in range(3)])
