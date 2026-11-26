"""Independent Monte Carlo and cross-fitted least-squares early exercise."""

import numpy as np

from quantproj.models import Paths


def payoff(
    paths: Paths, strike, maturity, rate, *, put=False, barrier=None, monitor_steps=None, knock_in=False
):
    spot = paths.spot
    cash = np.maximum(strike - spot[-1] if put else spot[-1] - strike, 0)
    if barrier is not None:
        if monitor_steps is None:
            raise ValueError("Provide explicit monitoring indices.")
        hit = np.any(spot[monitor_steps] >= barrier, axis=0)
        cash *= hit if knock_in else ~hit
    return np.exp(-rate * maturity) * cash


def lsmc(paths: Paths, strike, maturity, rate, *, put=True, stochastic_variance=True):
    """Two-fold cross-fitting. Each policy is evaluated on paths it never trained on."""
    s = paths.spot / strike
    v = paths.variance
    dt = maturity / (len(s) - 1)
    intrinsic = strike * np.maximum(1 - s if put else s - 1, 0)
    ids = np.arange(s.shape[1])
    out = np.zeros(s.shape[1])
    discount = np.exp(-rate * dt)
    for fold in (0, 1):
        training = ids % 2 == fold
        evaluation = ~training
        cash = intrinsic[-1].copy()
        for i in range(len(s) - 2, 0, -1):
            cash *= discount
            itm = intrinsic[i] > 0
            train = itm & training
            candidates = itm & evaluation
            x = s[i] - 1
            y = v[i] - 0.04
            columns = [np.ones_like(x), x, x * x, x * x * x]
            if stochastic_variance:
                columns.extend([y, y * y, x * y])
            basis = np.column_stack(columns)
            if train.sum() < 3 * basis.shape[1]:
                continue
            coef = np.linalg.lstsq(basis[train], cash[train], rcond=1e-8)[0]
            continuation = basis @ coef
            exercise = (train | candidates) & (intrinsic[i] > np.maximum(continuation, 0))
            cash[exercise] = intrinsic[i, exercise]
        out[evaluation] = cash[evaluation] * discount
    return out
