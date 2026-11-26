"""Independent solver, calibration, risk and study-level numerical controls."""

from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from quantproj.core.hedge import positions
from quantproj.rates.bermudan import g2_adi, hw_pde, hw_tree, primal_dual
from quantproj.rates.calibration import fit_market, observe
from quantproj.rates.curves import NelsonSiegel, bootstrap, nc1
from quantproj.rates.european import european, g2_european, jamshidian, monte_carlo
from quantproj.rates.greeks import aad_price, quote_greeks
from quantproj.rates.models import g2, hw
from quantproj.rates.reconciliation import reconcile

ASOF = date(2025, 1, 2)


@pytest.mark.parametrize("kind", ("hw", "base", "second"))
def test_european_parity_and_independent_monte_carlo(kind):
    curve = NelsonSiegel()
    model = hw(curve, 0.06, [0.01, 0.012], (1.0,)) if kind == "hw" else g2(curve, kind)
    contract = nc1(ASOF, ASOF, curve.discount).tail(3)
    value = european(model, 0.0, np.zeros(model.dimension), contract)
    estimate, error = monte_carlo(model, 0.0, np.zeros(model.dimension), contract, 48000, 219)
    assert abs(estimate - value) < 4 * error
    fn = jamshidian if model.dimension == 1 else g2_european
    payer = fn(model, 0.0, np.zeros(model.dimension), contract)
    receiver = fn(model, 0.0, np.zeros(model.dimension), contract, receiver=True)
    df = curve.discount(contract.dates)
    assert payer - receiver == pytest.approx(df[0] - df[-1] - contract.fixed * df[1:] @ contract.accrual)


def test_hw_tree_cn_and_european_limit():
    curve = NelsonSiegel()
    h = hw(curve, 0.06, [0.009, 0.014, 0.010], (1.2, 4.1))
    swap = nc1(ASOF, ASOF, curve.discount)
    cn = hw_pde(h, swap, n=111, steps_per_year=26).price(0.0, [0])
    tree = hw_tree(h, swap, steps_per_year=36)
    assert cn == pytest.approx(tree, abs=1.5e-4)
    single = swap.tail(8)
    x = hw_pde(h, single, n=121, steps_per_year=36).price(0, [0])
    assert x == pytest.approx(jamshidian(h, 0.0, [0], single), abs=4e-5)


@pytest.mark.slow
def test_adi_grid_stability_and_primal_dual_independent_paths():
    g = g2()
    s = nc1(ASOF, ASOF, g.curve.discount)
    coarse = g2_adi(g, s, n=39, steps_per_year=16).price(0.0, [0.0, 0.0])
    fine = g2_adi(g, s, n=71, steps_per_year=32).price(0.0, [0.0, 0.0])
    assert abs(coarse - fine) < 2.5e-4
    lower, upper, _ = primal_dual(g, s, 1800, 512, 36, 126)
    assert np.mean(lower) < np.mean(upper)
    assert np.mean(lower) - 4 * np.std(lower, ddof=1) / np.sqrt(len(lower)) < fine
    assert fine < np.mean(upper) + 4 * np.std(upper, ddof=1) / np.sqrt(len(upper))


def test_variance_bootstrap_and_constrained_fallback():
    curve = NelsonSiegel()
    h = hw(curve, 0.06, [0.011])
    s = nc1(ASOF, ASOF, curve.discount)
    m = observe(h, 0.0, np.zeros(1), ASOF, s, np.random.default_rng(1), 0, 0)
    fit = fit_market(m, 0.06)
    assert not fit.fallback
    assert np.max(abs(fit.residual_bp)) < 1e-5
    np.testing.assert_allclose(np.sqrt(fit.sigma2), 0.011, rtol=0.015)
    vols = m.vols.copy()
    vols[1] *= 0.25
    fitted = fit_market(replace(m, vols=vols), 0.06)
    assert fitted.fallback
    assert np.all(fitted.sigma2 >= 0)
    assert np.max(abs(fitted.residual_bp)) > 1


@pytest.mark.slow
def test_ift_quote_greeks_against_full_refit():
    pytest.importorskip("jax")
    curve = NelsonSiegel()
    h = hw(curve, 0.06, [0.011])
    s = nc1(ASOF, ASOF, curve.discount)
    m = observe(h, 0.0, np.zeros(1), ASOF, s, np.random.default_rng(1), 0, 0)
    fit = fit_market(m, 0.06)
    _, dk, dv, mask = quote_greeks(m, fit, s, 0.06, n=33, steps=3)
    for kind, j in (("ois", 10), ("vol", 2)):
        levels = []
        eps = 1e-6
        for sign in (1, -1):
            if kind == "ois":
                q = m.ois.copy()
                q[j] += sign * eps
                changed = replace(m, ois=q, curve=bootstrap(m.swaps, q))
            else:
                v = m.vols.copy()
                v[j] += sign * eps
                changed = replace(m, vols=v)
            other = fit_market(changed, 0.06)
            val, _, _, bump_mask = aad_price(changed.curve, other, s, 0.06, n=33, steps=3)
            assert not other.fallback
            np.testing.assert_array_equal(mask, bump_mask)
            levels.append(val)
        assert (levels[0] - levels[1]) / (2 * eps) == pytest.approx(
            (dk if kind == "ois" else dv)[j], rel=1e-3, abs=2e-5
        )


def test_rank_deficient_hedge_tie_break_under_caps():
    a = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
    h = positions(a, [1.0, 0.0], [0.8, 0.2, 0.0], 1.0)
    np.testing.assert_allclose(h, [0.8, 0.2, 0.0], atol=1e-9)
    h = positions(a, [1.5, 0.0], [1.0, 0.0, 0.0], 0.8)
    np.testing.assert_allclose(h, [0.8, 0.7, 0.0], atol=1e-8)


def test_annuity_reconciliation_reduces_on_finer_rehedge_grid():
    model = g2()
    s = nc1(ASOF, ASOF, model.curve.discount)
    stats = reconcile(model, s, 800, 287, np.linspace(0, s.dates[0], 82), (1, 2), (1, 8))
    assert stats[0][3] < stats[1][3]
    assert stats[2][3] < stats[3][3]


def test_two_factor_desk_recovers_noise_free_g2_parameters():
    from quantproj.rates import desk_g2

    world = g2()
    s = nc1(ASOF, ASOF, world.curve.discount)
    m = observe(world, 0.0, np.zeros(2), ASOF, s, np.random.default_rng(3), 0, 0, smile_offsets=(-25, 25))
    fit = desk_g2.calibrate(m, world.reversion, start=(0.02, 0.01, -0.5))
    np.testing.assert_allclose([fit.sigma, fit.eta, fit.rho], [0.0223, 0.0104, -0.702], rtol=0.01, atol=0.003)
    assert fit.rmse_bp < 0.05
    adi = g2_adi(world, s, n=41, steps_per_year=16).price(0.0, [0.0, 0.0])
    # The desk discounts on the bootstrapped log-linear OIS curve, not the Nelson–Siegel curve.
    assert desk_g2.bermudan(m, world.reversion, fit, s, 41, 16) == pytest.approx(adi, abs=2e-4)
