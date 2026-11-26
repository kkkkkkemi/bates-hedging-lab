"""Controls for information sets, calibration, jump ledger and P&L attribution."""

import numpy as np
import pytest

from quantproj.calibration import calibrate, generate_surface, hessian_diagnostics, physical
from quantproj.commodity import (
    SpreadModel,
    bjerksund_stensland,
    constant_price,
    driver_matrix,
    kirk,
    local_risk_hedge,
    margrabe,
)
from quantproj.greeks import bs_estimators, time_calendar
from quantproj.hedging.engine import Leg, decomposition, local_covariance, projected_positions
from quantproj.hedging.events import BrownianJumpTape
from quantproj.models import Bates
from quantproj.pnl import explain, pla, zone
from quantproj.pricing import black


def test_clean_heston_recovery_and_mask_reproducibility():
    p = Bates().no_jumps()
    times = [0.04, 0.08, 0.16, 0.25, 0.5, 1.0]
    log_k = np.linspace(-0.14, 0.14, 7)

    def surface(noise):
        return generate_surface(p, 100, p.v0, times, log_k, np.random.default_rng(7), noise)

    clean, low, high = surface(0), surface(0.5), surface(2.0)
    np.testing.assert_allclose(low.noise * 4, high.noise, atol=1e-12)
    np.testing.assert_array_equal(low.half_spread_iv, high.half_spread_iv)
    np.testing.assert_array_equal(low.mask, surface(0.5).mask)
    fit = calibrate(clean, clean=True, max_evaluations=120)
    assert fit.success and not fit.boundary
    np.testing.assert_allclose(physical(fit.params), physical(p), rtol=0.002, atol=0.00002)
    diag = hessian_diagnostics(fit, clean)
    np.testing.assert_allclose(diag["hessian"], diag["hessian"].T)
    assert diag["condition"] > 1
    for prices, strikes, keep in zip(high.observed, high.strikes, high.mask):
        slopes = np.diff(prices[keep]) / np.diff(strikes[keep])
        assert np.all(slopes <= 1e-10) and np.all(np.diff(slopes) >= -1e-10)
    for iv, keep in zip(high.iv.T, high.mask.T):
        assert np.all(np.diff(np.square(iv[keep]) * np.array(times)[keep]) >= -1e-11)


def test_full_information_projection_reoptimizes_stock_after_cap():
    p = Bates().no_jumps()
    tool, target = Leg(2.0, 0.3, 0.45), Leg(10.0, 0.6, 2.0)
    cov = local_covariance(100, 0.04, p, [tool, target], np.zeros((14, 3)))
    stock, opt = projected_positions(cov, two=True)
    assert opt == pytest.approx(target.variance / tool.variance)
    assert stock == pytest.approx(target.delta - tool.delta * opt)
    stock, opt = projected_positions(cov, cap=1.0, two=True)
    assert opt == 1
    assert stock == (cov[0, -1] - cov[0, 1] * opt) / cov[0, 0]


def test_coupled_jumps_and_pathwise_reconciliation():
    p = Bates()
    tape = BrownianJumpTape(0.25, 256, 700, p, 22)
    coarse = tape.market(100, p, 128)
    fine = tape.market(100, p, 256)
    assert len(fine.events) == fine.paths.jump_count.sum() > 0
    assert [(a.path, a.time) for a in coarse.events] == [(a.path, a.time) for a in fine.events]
    np.testing.assert_allclose(
        [a.after / a.before for a in coarse.events], [a.after / a.before for a in fine.events]
    )
    rhs, lhs, diff, jump = decomposition(fine, p, 0.25, 100.0, 0.245, [1, 16])
    assert np.isfinite(rhs).all()
    assert np.any(jump < 0) and np.isfinite(diff).all()
    assert np.std(lhs[1] - rhs) < np.std(lhs[16] - rhs)


def test_pla_boundary_and_frozen_repricing():
    assert zone(0.80, 0.02) == "amber"
    assert zone(0.81, 0.09) == "amber"
    assert zone(0.70, 0.12) == "amber"
    assert zone(0.699, 0.09) == "red" and zone(0.9, 0.121) == "red"
    rho, ks, classification = pla(np.arange(250.0), np.arange(250.0))
    assert rho == pytest.approx(1)
    assert (ks, classification) == (0.0, "green")
    with pytest.raises(ValueError, match="250"):
        pla(np.arange(10.0), np.arange(10.0))
    parts, step = explain(100, 101, 100, 0.25, 0.246, 0.23, 0.232, 0.02, 0.55)
    exact = 0.55 - (black.price(101, 100, 0.246, 0.232) - black.price(100, 100, 0.25, 0.23))
    assert step.sum() == pytest.approx(exact)
    assert abs(parts.sum() - exact) < abs(parts[:3].sum() - exact)


def test_spread_prices_and_projection():
    p = SpreadModel()
    for rho in [-0.8, -0.1, 0.4, 0.9]:
        assert constant_price(104, 100, 0, 0.25, rho, p) == margrabe(104, 100, 0.25, rho, p)
        assert kirk(104, 100, 0, 0.25, rho, p) == pytest.approx(margrabe(104, 100, 0.25, rho, p))
        assert bjerksund_stensland(104, 100, 0, 0.25, rho, p) == pytest.approx(
            margrabe(104, 100, 0.25, rho, p)
        )
        assert np.linalg.eigvalsh(driver_matrix(rho, 0.35))[0] >= 0
    gradient = np.array([0.53, -0.48, -0.6, 0.0])
    deltas, var = local_risk_hedge(gradient, 104, 100, 0.4, p)
    np.testing.assert_array_equal(deltas, gradient[:2])
    assert var > 0


def test_vanilla_pathwise_and_likelihood_ratio():
    args = (100, 100, 0.25, 0.24, 0.02)
    dates = time_calendar(0.25, 1 / 252)
    z = np.random.default_rng(7).standard_normal((len(dates), 26000))
    estimates = bs_estimators(*args, z, dates, bumps=(0.1,))
    reference = black.greeks(*args)
    for name, keys in [
        ("pathwise", ["delta", "vega"]),
        ("likelihood_ratio", ["delta", "gamma"]),
        ("hybrid", ["gamma"]),
    ]:
        for key in keys:
            x = estimates[name][0][key]
            assert np.mean(x) == pytest.approx(reference[key], abs=5 * x.std(ddof=1) / np.sqrt(len(x)))


@pytest.mark.slow
def test_jax_float64_aad():
    pytest.importorskip("jax")
    from quantproj.greeks import jax_batch

    estimates, seconds = jax_batch(100, 100, 0.24, 0.02, np.linspace(0.025, 0.25, 10), None, 10000, 3)
    expected = black.greeks(100, 100, 0.25, 0.24)
    for key in ("delta", "vega"):
        x = estimates[key]
        assert x.dtype == np.float64
        assert np.mean(x) == pytest.approx(expected[key], abs=5 * x.std(ddof=1) / np.sqrt(len(x)))
    assert all(value >= 0 for value in seconds.values())


def test_optional_bates_filter_recovers_state_from_clean_quotes():
    from dataclasses import replace

    from quantproj.calibration import filter_variance

    p = Bates()
    prior = replace(p, v0=0.052)
    market = generate_surface(
        p,
        100.0,
        p.v0,
        [0.04, 0.08, 0.16, 0.25, 0.5, 1.0],
        np.linspace(-0.14, 0.14, 7),
        np.random.default_rng(122),
        0.0,
    )
    result = filter_variance(market, prior, 1 / 252)
    assert abs(result.v0 - p.v0) < abs(prior.v0 - p.v0)
    assert result.intensity == p.intensity
