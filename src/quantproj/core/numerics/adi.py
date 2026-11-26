"""Directional ADI sweeps shared by Heston/Bates and Gaussian rate PDEs."""

import numpy as np

THETA = {"douglas": 0.5, "cs": 0.5, "hv": 0.5 + np.sqrt(3) / 6}


def adi_step(value, step, method, mixed, first, second, total, solves, *, boundary=None, source=0.0):
    """One Douglas, Craig–Sneyd or Hundsdorfer–Verwer step."""
    theta = THETA[method]
    explicit = value + step * (total @ value + source)
    sx, sy = solves

    def sweep(rhs, previous):
        rhs = rhs - theta * step * (first @ previous)
        if boundary is not None:
            rhs = boundary(rhs)
        rhs = sx(rhs) - theta * step * (second @ previous)
        if boundary is not None:
            rhs = boundary(rhs)
        return sy(rhs)

    y = sweep(explicit, value)
    if method == "cs":
        return sweep(explicit + 0.5 * step * (mixed @ (y - value)), value)
    if method == "hv":
        return sweep(explicit + 0.5 * step * (total @ (y - value)), y)
    return y
