"""Weekend-only OIS contracts, ACT/360 and discount curve bootstrapping."""

from calendar import monthrange
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq, root

DAY = timedelta(days=1)
TENORS = ("ON", "1W", "1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y")


def act360(start: date, end: date) -> float:
    return (end - start).days / 360


def following(d: date) -> date:
    return d + timedelta(days=(7 - d.weekday()) % 7) if d.weekday() >= 5 else d


def modified_following(d: date) -> date:
    adjusted = following(d)
    return d - timedelta(days=d.weekday() - 4) if adjusted.month != d.month else adjusted


def months(d: date, n: int) -> date:
    year, month = divmod(d.year * 12 + d.month - 1 + n, 12)
    month += 1
    return date(year, month, min(d.day, monthrange(year, month)[1]))


def maturity(d: date, tenor: str) -> date:
    if tenor == "ON":
        return following(d + DAY)
    n = int(tenor[:-1])
    unit = tenor[-1]
    raw = d + timedelta(weeks=n) if unit == "W" else months(d, n * (12 if unit == "Y" else 1))
    return modified_following(raw)


@dataclass(frozen=True)
class Swap:
    """Fixed payer swap. Dates are ACT/360 from a common epoch."""

    dates: np.ndarray
    accrual: np.ndarray
    fixed: float

    def shift(self, origin: float) -> "Swap":
        return Swap(self.dates - origin, self.accrual, self.fixed)

    def tail(self, index: int) -> "Swap":
        return Swap(self.dates[index:], self.accrual[index:], self.fixed)

    def fair_rate(self, discount) -> float:
        df = discount(self.dates)
        return (df[0] - df[-1]) / np.dot(self.accrual, df[1:])


def swap(start: date, tenor: str, epoch: date, fixed: float = 0.0) -> Swap:
    end = maturity(start, tenor)
    payments = [start]
    j = 1
    while modified_following(months(start, 12 * j)) < end:
        payments.append(modified_following(months(start, 12 * j)))
        j += 1
    payments.append(end)
    times = np.array([act360(epoch, d) for d in payments])
    return Swap(times, np.diff(times), fixed)


def nc1(issue: date, epoch: date, discount=None, strike: float | None = None) -> Swap:
    """10-year final date with 9 co-terminal exercise opportunities."""
    dates = [modified_following(months(issue, 12 * k)) for k in range(1, 11)]
    times = np.array([act360(epoch, d) for d in dates])
    result = Swap(times, np.diff(times), 0.0 if strike is None else strike)
    return result if strike is not None else Swap(times, result.accrual, result.fair_rate(discount))


def business_dates(origin: date, end: date) -> list[date]:
    return [
        origin + timedelta(days=i) for i in range((end - origin).days + 1) if (origin.weekday() + i) % 7 < 5
    ]


@dataclass(frozen=True)
class NelsonSiegel:
    level: float = 0.037
    slope: float = -0.018
    curvature: float = 0.006
    tau: float = 3.0

    def log_discount(self, t):
        u = np.asarray(t) / self.tau
        return (
            -self.level * np.asarray(t)
            + (self.slope + self.curvature) * self.tau * np.expm1(-u)
            + (self.curvature * np.asarray(t) * np.exp(-u))
        )

    def discount(self, t):
        return np.exp(self.log_discount(t))

    def forward(self, t):
        u = np.asarray(t) / self.tau
        return self.level + (self.slope + self.curvature * u) * np.exp(-u)


def _hagan_integral(x, g0, g1):
    """Integral of the monotone-convex forward correction on a unit interval."""
    if max(abs(g0), abs(g1)) < 1e-13:
        return 0.0
    if min(abs(g0), abs(g1)) < 1e-13:
        return g0 * (x - 2 * x * x + x**3) + g1 * (-x * x + x**3)
    if g0 * g1 < 0:
        if abs(g1) > 2 * abs(g0):
            eta = (g1 + 2 * g0) / (g1 - g0)
            return g0 * x + (g1 - g0) * max(x - eta, 0) ** 3 / (3 * (1 - eta) ** 2)
        if abs(g0) > 2 * abs(g1):
            return -_hagan_integral(1 - x, g1, g0)
        return g0 * (x - 2 * x * x + x**3) + g1 * (-x * x + x**3)
    eta = g1 / (g0 + g1)
    floor = -g0 * g1 / (g0 + g1)
    if x <= eta:
        return floor * x + (g0 - floor) * (x - x * x / eta + x**3 / (3 * eta**2))
    return floor * x + (g0 - floor) * eta / 3 + (g1 - floor) * (x - eta) ** 3 / (3 * (1 - eta) ** 2)


@dataclass(frozen=True)
class DiscountCurve:
    nodes: np.ndarray
    logs: np.ndarray
    method: str = "log_linear"

    def _forwards(self):
        dt = np.diff(self.nodes)
        margins = -np.diff(self.logs) / dt
        node_f = np.empty(len(self.nodes))
        node_f[1:-1] = (dt[1:] * margins[:-1] + dt[:-1] * margins[1:]) / (dt[:-1] + dt[1:])
        node_f[0] = (3 * margins[0] - node_f[1]) / 2
        node_f[-1] = (3 * margins[-1] - node_f[-2]) / 2
        # The positivity limiter is used where both adjacent discrete forwards are positive.
        for i, f in enumerate(node_f):
            adjacent = margins[max(0, i - 1) : min(i + 1, len(margins))]
            if min(adjacent) > 0:
                node_f[i] = np.clip(f, 0, 2 * min(adjacent))
        return margins, node_f

    def log_discount(self, t):
        t = np.asarray(t, dtype=float)
        if self.method == "log_linear":
            i = np.clip(np.searchsorted(self.nodes, t) - 1, 0, len(self.nodes) - 2)
            x = (t - self.nodes[i]) / (self.nodes[i + 1] - self.nodes[i])
            return self.logs[i] + x * (self.logs[i + 1] - self.logs[i])
        margins, f = self._forwards()
        out = np.empty(t.size)
        for k, point in enumerate(t.ravel()):
            j = int(np.searchsorted(self.nodes, point) - 1)
            if j < 0:
                out[k] = -f[0] * point
            elif j >= len(margins):
                out[k] = self.logs[-1] - f[-1] * (point - self.nodes[-1])
            else:
                w = self.nodes[j + 1] - self.nodes[j]
                x = (point - self.nodes[j]) / w
                g = _hagan_integral(x, f[j] - margins[j], f[j + 1] - margins[j])
                out[k] = self.logs[j] - w * (margins[j] * x + g)
        return out.reshape(t.shape)

    def discount(self, t):
        return np.exp(self.log_discount(t))

    def forward(self, t):
        return -(self.log_discount(np.asarray(t) + 1e-5) - self.log_discount(np.asarray(t) - 1e-5)) / 2e-5


@lru_cache(maxsize=16384)
def quote_swaps(asof: date, tenors: tuple[str, ...] = TENORS) -> tuple[Swap, ...]:
    return tuple(swap(asof, t, asof) for t in tenors)


def ois_quotes(instruments, discount):
    return np.array([inst.fair_rate(discount) for inst in instruments])


def bootstrap(instruments, quotes, method="log_linear") -> DiscountCurve:
    quotes = np.asarray(quotes)
    nodes = np.array([0.0, *[o.dates[-1] for o in instruments]])
    if len(quotes) != len(instruments) or np.any(np.diff(nodes) <= 0):
        raise ValueError("OIS instruments and quotes must be supplied in strictly increasing maturity order.")
    logs = np.zeros_like(nodes)
    for i, (inst, q) in enumerate(zip(instruments, quotes), 1):
        # Short dated single-payment instruments have a closed-form discount factor.
        if len(inst.accrual) == 1:
            if 1 + q * inst.accrual[0] <= 0:
                raise ValueError("OIS quote implies a nonpositive discount factor.")
            logs[i] = -np.log1p(q * inst.accrual[0])
            continue

        def value(last):
            ds = np.exp(np.interp(inst.dates[1:], nodes[: i + 1], np.r_[logs[:i], last]))
            return q * np.dot(inst.accrual, ds) + ds[-1] - 1

        logs[i] = brentq(value, -60, 30, xtol=1e-13)
    if method == "monotone_convex":
        solution = root(
            lambda z: ois_quotes(instruments, DiscountCurve(nodes, np.r_[0.0, z], method).discount) - quotes,
            logs[1:],
            tol=1e-10,
        )
        if not np.isfinite(solution.x).all() or np.max(abs(solution.fun)) > 1e-9:
            raise ValueError("Monotone-convex OIS bootstrap did not reprice input quotes.")
        logs[1:] = solution.x
    elif method != "log_linear":
        raise ValueError(f"Unknown curve interpolation method {method!r}.")
    return DiscountCurve(nodes, logs, method)
