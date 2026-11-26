"""Two-factor G2++ desk valuation on the daily observed quote snapshot.

The mean reversions are fixed desk inputs, as for the one-factor desk. The two
factor volatilities and their correlation are refitted each day to the observed
swaption quotes on the observed OIS curve, so the model sees exactly the market
data used by the other desk models.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from quantproj.rates.bermudan import g2_adi
from quantproj.rates.curves import Swap
from quantproj.rates.european import g2_european, observables
from quantproj.rates.models import Gaussian

RHO_LIMIT = 0.995


@dataclass(frozen=True)
class G2Fit:
    sigma: float
    eta: float
    rho: float
    rmse_bp: float
    success: bool


def _params(x):
    return float(np.exp(x[0])), float(np.exp(x[1])), float(RHO_LIMIT * np.tanh(x[2]))


def _unparams(sigma, eta, rho):
    return np.array([np.log(sigma), np.log(eta), np.arctanh(np.clip(rho / RHO_LIMIT, -0.999, 0.999))])


def model(curve, reversions, fit):
    return Gaussian(curve, tuple(float(a) for a in reversions), ((fit.sigma, fit.eta),), rho=fit.rho)


def _quote_set(market, wings):
    """Retained quotes with their contracts, normalized by ATM-equivalent vega."""
    strips = {round(float(s.dates[0]), 9): s for s in market.strip}
    rows = []
    for expiry, strike, _, vol, premium, keep in market.quotes:
        contract = strips.get(round(float(expiry), 9))
        if contract is None or not keep or not np.isfinite(vol) or vol <= 0:
            continue
        if not wings and abs(strike - contract.fixed) > 1e-12:
            continue
        local = Swap(contract.dates, contract.accrual, strike)
        forward, annuity, _ = observables(local, market.curve.discount)
        sd = vol * np.sqrt(expiry)
        vega = annuity * np.sqrt(expiry) * np.exp(-0.5 * ((forward - strike) / sd) ** 2) / np.sqrt(2 * np.pi)
        rows.append((local, premium, max(vega, 1e-8)))
    return rows


def calibrate(market, reversions, *, wings=True, start=None, rho=None, order=24, max_nfev=60):
    """Fit sigma, eta and, unless it is fixed, rho to observed quotes, in vol bp."""
    rows = _quote_set(market, wings)
    if start is not None and rho is not None:
        start = (start[0], start[1], rho)
    if len(rows) < 3:
        # Too few quotes identify the parameters. Keep the previous day's fit.
        if start is None:
            raise ValueError("A G2++ desk fit needs at least three retained quotes.")
        return G2Fit(*start, np.nan, False)
    x0 = _unparams(*(start if start is not None else (0.01, 0.01, 0.0 if rho is None else rho)))
    fixed = None if rho is None else x0[2]
    if fixed is not None:
        x0 = x0[:2]

    def full(x):
        return x if fixed is None else np.r_[x, fixed]

    def residual(x):
        trial = G2Fit(*_params(full(x)), np.nan, False)
        m = model(market.curve, reversions, trial)
        out = [
            (g2_european(m, 0.0, np.zeros(2), contract, order) - premium) / vega * 1e4
            for contract, premium, vega in rows
        ]
        return np.asarray(out)

    solution = least_squares(
        residual, x0, x_scale=1.0, diff_step=1e-4, xtol=1e-8, ftol=1e-8, max_nfev=max_nfev
    )
    sigma, eta, fitted_rho = _params(full(solution.x))
    return G2Fit(sigma, eta, fitted_rho, float(np.sqrt(np.mean(solution.fun**2))), bool(solution.success))


def bermudan(market, reversions, fit, tail, grid, steps):
    """Bermudan PV for a contract shifted to today's clock, on the observed curve."""
    m = model(market.curve, reversions, fit)
    return float(g2_adi(m, tail, 0.0, n=grid, steps_per_year=steps).price(0.0, np.zeros(2)))
