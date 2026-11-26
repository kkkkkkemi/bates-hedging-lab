"""European Bachelier annuity-hedge verification on coupled exact Gaussian paths."""

import numpy as np
from scipy.special import ndtr

from quantproj.rates.european import european, normal_call, normal_iv
from quantproj.rates.greeks import normal_density
from quantproj.rates.models import simulate


def _market(model, t, state, swap):
    df = model.bond(t, swap.dates, state)
    ann = df[:, 1:] @ swap.accrual
    swap_pv = df[:, 0] - df[:, -1] - swap.fixed * ann
    f = (df[:, 0] - df[:, -1]) / ann
    loads = model.loads(swap.dates - t)
    derivative = -df[..., None] * loads
    d_ann = np.einsum("nkd,k->nd", derivative[:, 1:], swap.accrual)
    d_f = (derivative[:, 0] - derivative[:, -1] - f[:, None] * d_ann) / ann[:, None]
    scale = model.sigma_at(t)
    cov = scale[:, None] * model.corr * scale
    nu = np.einsum("ni,ij,nj->n", d_f, cov, d_f)
    return ann, f, swap_pv, nu


def reconcile(model, contract, npaths, seed, base_times, multipliers=(1, 4), frequencies=(1, 5, 10)):
    """Integrate and trade on nested grids with independently chosen mesh coarseness."""
    n = max(multipliers)
    fine = np.r_[
        base_times[0],
        np.concatenate([np.linspace(a, b, n + 1)[1:] for a, b in zip(base_times[:-1], base_times[1:])]),
    ]
    expiry = contract.dates[0]
    states, bank = simulate(model, fine, npaths, seed)
    spot = np.zeros(model.dimension)
    premium = european(model, 0.0, spot, contract)
    a, f, _, _ = _market(model, 0.0, np.array([spot]), contract)
    vol = normal_iv(premium, float(f[0]), contract.fixed, expiry, float(a[0]))
    values = [_market(model, t, state, contract) for t, state in zip(fine, states)]
    ann, fw, sw, nu = (np.asarray(x) for x in zip(*values))
    unit = np.array([normal_call(f, contract.fixed, expiry - t, vol) for t, f in zip(fine, fw)])
    residual_rows = []
    for resolution in multipliers:
        sample = np.arange(0, len(fine), n // resolution)
        eval_times = fine[sample]
        unit_sd = vol * np.sqrt(expiry - eval_times[:-1])
        z = (fw[sample[:-1]] - contract.fixed) / unit_sd[:, None]
        gamma = normal_density(z) / unit_sd[:, None]
        drift_term = (
            np.sum(
                ann[sample[:-1]]
                / bank[sample[:-1]]
                * gamma
                * (nu[sample[:-1]] - vol**2)
                * np.diff(eval_times)[:, None],
                axis=0,
            )
            / 2
        )
        expected = premium - float(a[0]) * unit[0, 0] - drift_term
        for frequency in frequencies:
            rehedges = np.r_[np.arange(0, len(fine) - 1, n * frequency), len(fine) - 1]
            pv = np.full(npaths, premium) - ann[-1] * unit[-1] / bank[-1]
            for i, j in zip(rehedges[:-1], rehedges[1:]):
                h = ndtr((fw[i] - contract.fixed) / (vol * np.sqrt(expiry - fine[i])))
                ann_pos = unit[i] - (fw[i] - contract.fixed) * h
                pv += h * (sw[j] / bank[j] - sw[i] / bank[i]) + ann_pos * (
                    ann[j] / bank[j] - ann[i] / bank[i]
                )
            diff = pv - expected
            residual_rows.append(
                [
                    resolution,
                    frequency,
                    npaths,
                    float(np.sqrt(np.mean(diff**2))),
                    float(np.mean(pv)),
                    float(np.mean(expected)),
                    float(np.mean(diff)),
                ]
            )
    return residual_rows
