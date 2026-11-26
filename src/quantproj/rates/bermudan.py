"""Gaussian Bermudan PDE, trinomial lattice and out-of-sample primal-dual bounds."""

from dataclasses import dataclass
from math import ceil

import numpy as np
from scipy import sparse
from scipy.interpolate import CubicSpline, RectBivariateSpline
from scipy.linalg import solve_banded
from scipy.sparse.linalg import splu

from quantproj.core.numerics.adi import THETA, adi_step
from quantproj.core.numerics.regression import fit_continuation
from quantproj.rates.european import coupon_weights
from quantproj.rates.models import terminal_paths


def exercise_value(model, t, state, forward_swap):
    return 1.0 - model.bond(t, forward_swap.dates[1:], state) @ coupon_weights(forward_swap)


def normalized_payoff(model, swap, index, state):
    t = swap.dates[index]
    return np.maximum(exercise_value(model, t, state, swap.tail(index)), 0.0) / model.bond(
        t, swap.dates[-1], state
    )


def _segments(start, swap, model, steps_per_year):
    events = np.unique(np.r_[start, swap.dates[:-1], model.knots])
    events = events[(events >= start) & (events <= swap.dates[-2])]
    pieces = [
        np.linspace(left, right, max(1, ceil((right - left) * steps_per_year)) + 1)[1:]
        for left, right in zip(events[:-1], events[1:])
    ]
    return np.r_[start, *pieces]


def _immediate(model, swap, index, z):
    state = z + model.centered_mean(swap.dates[index], swap.dates[-1])
    return normalized_payoff(model, swap, index, state)


def _slices(times, swap):
    return {
        int(np.argmin(abs(times - t))): i
        for i, t in enumerate(swap.dates[:-1])
        if times[0] <= t + 1e-10 and times[-1] >= t - 1e-10
    }


def _range(model, swap, start, n):
    var = np.diag(model.moments(0, swap.dates[-2]))[: model.dimension]
    width = np.maximum(7 * np.sqrt(var), 0.015)
    return tuple(np.linspace(-w, w, n) for w in width)


@dataclass
class Solution:
    model: object
    swap: object
    times: np.ndarray
    axes: tuple
    values: dict
    continuation: dict

    def _interp(self, grid, z):
        coords = np.asarray(z)
        if len(self.axes) == 1:
            return float(CubicSpline(self.axes[0], grid)(coords[0]))
        return float(RectBivariateSpline(*self.axes, grid).ev(*coords))

    def normalized(self, t, state, continue_only=False):
        z = np.asarray(state) - self.model.centered_mean(t, self.swap.dates[-1])
        if continue_only:
            idx = int(np.argmin(abs(self.times - t)))
            if abs(self.times[idx] - t) > 1e-9 or idx not in self.continuation:
                raise ValueError("Continuation requested outside an exercise date.")
            return max(self._interp(self.continuation[idx], z), 0.0)
        m = int(np.clip(np.searchsorted(self.times, t), 0, len(self.times) - 1))
        if m == 0 or abs(self.times[m] - t) < 1e-10:
            return max(self._interp(self.values[m], z), 0.0)
        q = (t - self.times[m - 1]) / (self.times[m] - self.times[m - 1])
        # The limit from the left includes that date's exercise opportunity.
        return max(self._interp(self.values[m - 1] * (1 - q) + self.values[m] * q, z), 0.0)

    def price(self, t, state, continue_only=False):
        return self.normalized(t, state, continue_only) * self.model.bond(t, self.swap.dates[-1], state)


def _direction(nodes, a, vol):
    h = nodes[1] - nodes[0]
    drift = -a * nodes
    diffusion = vol**2 / (2 * h * h)
    lo = diffusion - drift / (2 * h)
    hi = diffusion + drift / (2 * h)
    # Reflect at far-field boundaries and use upwind where central coefficients turn negative.
    invalid = (lo < 0) | (hi < 0)
    lo[invalid] = diffusion + np.maximum(-drift[invalid], 0) / h
    hi[invalid] = diffusion + np.maximum(drift[invalid], 0) / h
    lo[0], hi[0] = 0, 2 * diffusion + max(drift[0], 0) / h
    hi[-1], lo[-1] = 0, 2 * diffusion + max(-drift[-1], 0) / h
    return lo, -lo - hi, hi


def _hw_step(v, coeffs, dt, theta):
    lo, diag, hi = coeffs
    rhs = v + (1 - theta) * dt * (diag * v + np.r_[0.0, lo[1:] * v[:-1]] + np.r_[hi[:-1] * v[1:], 0.0])
    left = np.array(
        [np.r_[0.0, -theta * dt * hi[:-1]], 1 - theta * dt * diag, np.r_[-theta * dt * lo[1:], 0.0]]
    )
    return solve_banded((1, 1), left, rhs, check_finite=False)


def hw_pde(model, swap, start=0.0, n=121, steps_per_year=20):
    times = _segments(start, swap, model, steps_per_year)
    exercises = _slices(times, swap)
    axes = _range(model, swap, start, n)
    x = axes[0]
    values, continuation = {}, {}
    v = np.zeros(len(x))
    for j in range(len(times) - 1, -1, -1):
        restart = j in exercises
        if restart:
            continuation[j] = v.copy()
            v = np.maximum(v, _immediate(model, swap, exercises[j], x[:, None]))
        values[j] = v
        if j:
            dt = times[j] - times[j - 1]
            c = _direction(x, model.reversion[0], model.sigma_at((times[j] + times[j - 1]) / 2)[0])
            if restart:
                v = _hw_step(_hw_step(v, c, dt / 2, 1.0), c, dt / 2, 1.0)
            else:
                v = _hw_step(v, c, dt, 0.5)
    return Solution(model, swap, times, axes, values, continuation)


def hw_tree(model, swap, start=0.0, state=0.0, steps_per_year=24):
    times = _segments(start, swap, model, steps_per_year)
    exercises = _slices(times, swap)
    z = np.atleast_1d(state) - model.centered_mean(start, swap.dates[-1])
    if len(times) == 1:
        h = _immediate(model, swap, exercises[0], z[:, None])
        return float(h[0] * model.bond(start, swap.dates[-1], [state]))
    nodes = [z]
    transitions = []
    for left, right in zip(times[:-1], times[1:]):
        decay, cov = model.state_steps(left, right)
        variance = max(float(cov[0, 0]), 1e-14)
        dx = np.sqrt(3 * variance)
        means = nodes[-1] * decay[0]
        centers = np.rint(means / dx).astype(int)
        first, last = min(centers) - 1, max(centers) + 1
        delta = means / dx - centers
        q = 1 / 3 + delta**2
        probs = np.stack(((q - delta) / 2, 1 - q, (q + delta) / 2), axis=1)
        transitions.append((centers[:, None] - first + np.array([-1, 0, 1]), probs))
        nodes.append(dx * np.arange(first, last + 1))
    v = np.zeros(len(nodes[-1]))
    for i in range(len(times) - 1, -1, -1):
        if i in exercises:
            v = np.maximum(v, _immediate(model, swap, exercises[i], nodes[i][:, None]))
        if i:
            ids, probs = transitions[i - 1]
            v = np.sum(v[ids] * probs, axis=1)
    return float(v[0] * model.bond(start, swap.dates[-1], [state]))


def g2_adi(model, swap, start=0.0, n=55, steps_per_year=24, method="hv"):
    times = _segments(start, swap, model, steps_per_year)
    exercises = _slices(times, swap)
    x, y = _range(model, swap, start, n)
    state = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1)
    sig = model.sigma_at(start)
    theta = THETA[method]
    eye = sparse.eye(n, format="csc")
    op = []
    for axis, a, s in zip((x, y), model.reversion, sig):
        lo, diag, hi = _direction(axis, a, s)
        op.append(sparse.diags([lo[1:], diag, hi[:-1]], [-1, 0, 1], format="csc"))
    first = sparse.kron(op[0], eye, format="csc")
    second = sparse.kron(eye, op[1], format="csc")
    d = []
    for axis in (x, y):
        off = np.r_[np.full(n - 2, 1 / (2 * (axis[1] - axis[0]))), 0.0]
        d.append(sparse.diags([-off, off[::-1]], [-1, 1], format="csc"))
    mixed = model.rho * np.prod(sig) * sparse.kron(*d, format="csc")
    total = first + second + mixed
    identity = sparse.eye(n * n, format="csc")
    factors = {}
    v = np.zeros(n * n)
    values, continuation = {}, {}
    for j in range(len(times) - 1, -1, -1):
        if j in exercises:
            continuation[j] = v.reshape(n, n).copy()
            v = np.maximum(v, _immediate(model, swap, exercises[j], state).ravel())
        values[j] = v.reshape(n, n)
        if j:
            dt = times[j] - times[j - 1]
            k = round(dt, 12)
            if k not in factors:
                factors[k] = (
                    splu(identity - theta * dt * first).solve,
                    splu(identity - theta * dt * second).solve,
                )
            v = adi_step(v, dt, method, mixed, first, second, total, factors[k])
    return Solution(model, swap, times, (x, y), values, continuation)


def _basis(state, scale):
    z = np.asarray(state) / scale
    terms = [np.ones(z.shape[:-1])]
    for j in range(z.shape[-1]):
        terms.extend((z[..., j], z[..., j] ** 2, z[..., j] ** 3))
    if z.shape[-1] == 2:
        terms.extend((z[..., 0] * z[..., 1], z[..., 0] ** 2 * z[..., 1], z[..., 0] * z[..., 1] ** 2))
    return np.stack(terms, axis=-1)


@dataclass
class Policy:
    coefs: list
    scale: np.ndarray

    def exercise(self, idx, states, payoff):
        return (payoff > 0) & (payoff >= np.maximum(_basis(states, self.scale) @ self.coefs[idx], 0))


def _policy_cash(model, swap, policy, states, start=0):
    samples = np.zeros(states.shape[1:-1])
    for i in range(len(states) - 1, -1, -1):
        idx = start + i
        pay = normalized_payoff(model, swap, idx, states[i])
        samples = np.where(policy.exercise(idx, states[i], pay), pay, samples)
    return samples


def primal_dual(model, swap, training, outer, inner, seed):
    """Fit once, evaluate on independent paths and nest an Andersen–Broadie dual."""
    dates = swap.dates[:-1]
    final = swap.dates[-1]
    scale = np.maximum(np.sqrt(np.diag(model.moments(0, dates[-1]))[: model.dimension]), 1e-5)
    rng = np.random.default_rng(seed)
    x = terminal_paths(model, dates, 0.0, np.zeros((training, model.dimension)), final, rng)
    coeffs = [np.zeros(_basis(x[0], scale).shape[-1]) for _ in dates]
    train_cash = np.zeros(training)
    for j in range(len(dates) - 1, -1, -1):
        h = normalized_payoff(model, swap, j, x[j])
        design = _basis(x[j], scale)
        itm = h > 0
        if j < len(dates) - 1 and itm.sum() >= 2 * design.shape[-1]:
            coeffs[j] = fit_continuation(design[itm], train_cash[itm])
        exercise = (h > 0) & (h >= np.maximum(design @ coeffs[j], 0))
        train_cash = np.where(exercise, h, train_cash)
    policy = Policy(coeffs, scale)
    xn = terminal_paths(model, dates, 0.0, np.zeros((outer, model.dimension)), final, rng)
    lower = _policy_cash(model, swap, policy, xn)
    previous_state = np.zeros((outer, model.dimension))
    previous_time = 0.0
    m = np.zeros(outer)
    upper = np.full(outer, -np.inf)
    for j, t in enumerate(dates):
        h = normalized_payoff(model, swap, j, xn[j])
        exercised = policy.exercise(j, xn[j], h)
        start_states = np.repeat(xn[j][:, None, :], inner, axis=1)
        if j + 1 < len(dates):
            nested = terminal_paths(model, dates[j + 1 :], t, start_states, final, rng)
            decision_value = np.where(
                exercised, h, _policy_cash(model, swap, policy, nested, j + 1).mean(axis=1)
            )
        else:
            decision_value = h
        before = np.repeat(previous_state[:, None, :], inner, axis=1)
        nested = terminal_paths(model, dates[j:], previous_time, before, final, rng)
        conditional = _policy_cash(model, swap, policy, nested, j).mean(axis=1)
        m += decision_value - conditional
        upper = np.maximum(upper, h - m)
        previous_time, previous_state = t, xn[j]
    df = model.bond(0.0, final, np.zeros(model.dimension))
    return df * lower, df * upper, policy
