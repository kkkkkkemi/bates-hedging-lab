"""Coupled market grids with individual compound-Poisson event states."""

from dataclasses import dataclass

import numpy as np

from quantproj.models import Bates, Paths, diffusion_step


@dataclass
class JumpEvent:
    path: int
    time: float
    before: float
    after: float


@dataclass
class MarketPath:
    paths: Paths
    events: list[JumpEvent]


class BrownianJumpTape:
    """Share Brownian values and jump draws when refining a simulation grid."""

    def __init__(self, maturity, steps, npaths, p: Bates, seed):
        self.t, self.steps, self.npaths = maturity, steps, npaths
        self.dt = maturity / steps
        rng = np.random.default_rng(seed)
        self.w = np.zeros((steps + 1, 2, npaths))
        self.w[1:] = np.cumsum(rng.normal(0, np.sqrt(self.dt), (steps, 2, npaths)), axis=0)
        self.events = []
        for path, count in enumerate(rng.poisson(p.intensity * maturity, npaths)):
            ts = np.sort(rng.uniform(0, maturity, count))
            ys = rng.normal(p.jump_mean, p.jump_vol, count)
            last_idx, last_t, last_w = -1, 0.0, np.zeros(2)
            for time, jump in zip(ts, ys):
                i = min(int(time / self.dt), steps - 1)
                if last_idx != i:
                    last_t, last_w = i * self.dt, self.w[i, :, path]
                right = (i + 1) * self.dt
                a = (time - last_t) / (right - last_t)
                w = (
                    (1 - a) * last_w
                    + a * self.w[i + 1, :, path]
                    + np.sqrt((time - last_t) * (1 - a)) * rng.standard_normal(2)
                )
                self.events.append((time, path, jump, w))
                last_idx, last_t, last_w = i, time, w
        self.events.sort(key=lambda row: row[0])

    def market(self, spot, p, steps, *, scheme="qe", bs_vol=None):
        if self.steps % steps:
            raise ValueError("The fine Brownian grid must be divisible by every requested grid.")
        stride = self.steps // steps
        dt = self.t / steps
        s = np.empty((steps + 1, self.npaths))
        v = np.empty_like(s)
        counts = np.zeros((steps, self.npaths), dtype=np.int16)
        events = []
        x = np.full(self.npaths, np.log(spot))
        var = np.full(self.npaths, p.v0 if bs_vol is None else bs_vol**2)
        s[0], v[0] = spot, var
        cursor = 0
        for i in range(steps):
            begin, end = i * dt, (i + 1) * dt
            w0, w1 = self.w[i * stride], self.w[(i + 1) * stride]
            affected = {}
            while cursor < len(self.events) and self.events[cursor][0] <= end and bs_vol is None:
                time, path, jump, w = self.events[cursor]
                old_t, old_w = affected.get(path, (begin, w0[:, path]))
                z = (w - old_w) / np.sqrt(time - old_t)
                x[path], var[path] = diffusion_step(x[path], var[path], time - old_t, p, *z, scheme)
                before = float(np.exp(x[path]))
                x[path] += jump
                events.append(JumpEvent(path, time, before, float(np.exp(x[path]))))
                counts[i, path] += 1
                affected[path] = time, w
                cursor += 1
            if bs_vol is None:
                before = np.ones(self.npaths, bool)
                before[list(affected)] = False
                z = (w1[:, before] - w0[:, before]) / np.sqrt(dt)
                if before.any():
                    x[before], var[before] = diffusion_step(x[before], var[before], dt, p, *z, scheme)
                for path, (last, w) in affected.items():
                    remaining = end - last
                    if remaining > 1e-14:
                        z = (w1[:, path] - w) / np.sqrt(remaining)
                        x[path], var[path] = diffusion_step(x[path], var[path], remaining, p, *z, scheme)
            else:
                x += (p.rate - bs_vol**2 / 2) * dt + bs_vol * (w1[1] - w0[1])
            s[i + 1], v[i + 1] = np.exp(x), np.maximum(var, 0)
        return MarketPath(Paths(s, v, counts), events)
