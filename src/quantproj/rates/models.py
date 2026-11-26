"""Exact Gaussian factor and integrated short-rate transitions."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss

from quantproj.rates.curves import NelsonSiegel


@lru_cache
def quadrature(order: int):
    return leggauss(order)


@dataclass(frozen=True, eq=False)
class Gaussian:
    curve: object
    reversion: tuple[float, ...]
    sigmas: tuple[tuple[float, ...], ...]
    knots: tuple[float, ...] = ()
    rho: float = 0.0

    def __post_init__(self):
        if len(self.reversion) not in (1, 2) or any(a <= 0 for a in self.reversion):
            raise ValueError("One or two strictly positive mean reversions are required.")
        if len(self.sigmas) != len(self.knots) + 1 or any(len(v) != self.dimension for v in self.sigmas):
            raise ValueError("Provide one vector of volatilities per segment.")
        if (
            np.any(np.asarray(self.sigmas) < 0)
            or np.any(np.diff((0.0, *self.knots)) <= 0)
            or abs(self.rho) >= 1
        ):
            raise ValueError("Invalid volatility, segment dates or correlation.")

    @property
    def dimension(self) -> int:
        return len(self.reversion)

    @property
    def corr(self):
        return np.eye(1) if self.dimension == 1 else np.array([[1.0, self.rho], [self.rho, 1.0]])

    def loads(self, delta):
        a = np.asarray(self.reversion)
        return -np.expm1(-np.asarray(delta)[..., None] * a) / a

    def sigma_at(self, t):
        return np.asarray(self.sigmas)[np.searchsorted(self.knots, t, side="right")]

    def moments(self, start, end):
        return self._moments(round(float(start), 12), round(float(end), 12))

    @lru_cache(maxsize=24000)
    def _moments(self, start, end):
        """Return covariance of factor innovations and their integrated innovations."""
        if end < start:
            raise ValueError("Transition end precedes start.")
        d = self.dimension
        cov = np.zeros((2 * d, 2 * d))
        if end == start:
            return cov
        boundaries = [start, *[k for k in self.knots if start < k < end], end]
        a = np.array(self.reversion)
        x, w = quadrature(12)
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            delta = end - (left + right) / 2 - (right - left) * x / 2
            kernel = np.concatenate((np.exp(-delta[:, None] * a), self.loads(delta)), axis=1)
            sd = self.sigma_at((left + right) / 2)
            inst = self.corr * sd[:, None] * sd
            cov += ((kernel.T * w) @ kernel) * np.tile(inst, (2, 2)) * (right - left) / 2
        return cov

    def integral_variance(self, start, end):
        return self.moments(start, end)[self.dimension :, self.dimension :].sum()

    def phi_integral(self, start, end):
        return float(
            self.curve.log_discount(start)
            - self.curve.log_discount(end)
            + (self.integral_variance(0, end) - self.integral_variance(0, start)) / 2
        )

    def log_bond(self, t, ends, state):
        state = np.asarray(state)
        ends = np.asarray(ends, dtype=float)
        if np.any(ends < t - 1e-10):
            raise ValueError("Cannot value a matured zero-coupon bond.")
        deterministic = np.array(
            [-self.phi_integral(t, u) + self.integral_variance(t, u) / 2 for u in ends.ravel()]
        ).reshape(ends.shape)
        return deterministic - np.tensordot(state, self.loads(ends - t), axes=([-1], [-1]))

    def bond(self, t, ends, state):
        return np.exp(self.log_bond(t, ends, state))

    def centered_mean(self, t, numeraire_end):
        cov = self.moments(0, t)
        d = self.dimension
        return -cov[:d, d:].sum(axis=1) - cov[:d, :d] @ self.loads(numeraire_end - t)

    def state_steps(self, start, end):
        d = self.dimension
        return np.exp(-np.asarray(self.reversion) * (end - start)), self.moments(start, end)[:d, :d]

    def conditional_forward(self, t, expiry, state, bond_end):
        d = self.dimension
        decay, q = self.state_steps(t, expiry)
        cross = self.moments(t, expiry)[:d, d:].sum(axis=1)
        return np.asarray(state) * decay - cross - q @ self.loads(bond_end - expiry), q


def hw(curve, a, vols, knots=()):
    return Gaussian(curve, (float(a),), tuple((float(v),) for v in vols), tuple(knots))


def g2(curve=None, case="base"):
    curve = NelsonSiegel() if curve is None else curve
    params = {
        "base": (0.7735, 0.082, 0.0223, 0.0104, -0.702),
        "second": (0.0693, 0.0531, 0.0116, 0.0057, 0.121),
    }
    a, b, sigma, eta, rho = params[case]
    return Gaussian(curve, (a, b), ((sigma, eta),), rho=rho)


def root_cov(cov):
    eigen, vec = np.linalg.eigh(cov)
    return vec * np.sqrt(np.maximum(eigen, 0))


def simulate(model: Gaussian, times, n, seed, normals=None):
    """Exact OU factors and stochastic bank accounts, with reproducible tape input."""
    times = np.asarray(times)
    d = model.dimension
    if times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Simulation times must start at zero and be increasing.")
    if normals is None:
        normals = np.random.default_rng(seed).standard_normal((len(times) - 1, n, 2 * d))
    states = np.zeros((len(times), n, d))
    log_bank = np.zeros((len(times), n))
    for i, (a, b) in enumerate(zip(times[:-1], times[1:])):
        shock = normals[i] @ root_cov(model.moments(a, b)).T
        states[i + 1] = states[i] * np.exp(-np.asarray(model.reversion) * (b - a)) + shock[:, :d]
        log_bank[i + 1] = (
            log_bank[i] + model.phi_integral(a, b) + states[i] @ model.loads(b - a) + shock[:, d:].sum(axis=1)
        )
    return states, np.exp(log_bank)


def terminal_paths(model, times, start, states, numeraire, rng):
    """Exact forward-measure transitions starting from risk-neutral factor coordinates."""
    out = []
    for end in times:
        mean, cov = model.conditional_forward(start, end, states, numeraire)
        states = mean + rng.standard_normal(mean.shape) @ root_cov(cov).T
        out.append(states)
        start = end
    return np.asarray(out)
