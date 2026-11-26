"""Cross-solver checks and fixed numerical regression points."""

from dataclasses import replace

import numpy as np
import pytest

from quantproj.experiments import scenarios
from quantproj.models import Bates, simulate
from quantproj.pricing import black, monte_carlo
from quantproj.pricing.fourier import cos_greeks, cos_price, fft_call
from quantproj.pricing.pde import adi_grid, bs_grid, continuous_upout


def test_black_parity_and_iv():
    s, k, t, r, vol = 100.0, np.array([82.0, 100.0, 118.0]), 0.25, 0.02, 0.24
    call = black.price(s, k, t, vol, r)
    put = black.price(s, k, t, vol, r, put=True)
    np.testing.assert_allclose(call - put, s - k * np.exp(-r * t), atol=1e-12)
    np.testing.assert_allclose(black.implied_vol(call, s, k, t, r), vol, atol=1e-9)
    np.testing.assert_allclose(black.implied_vol(put, s, k, t, r, True), vol, atol=1e-9)


def test_scenarios_characteristic_function_and_regression():
    p = Bates()
    models = scenarios(p)
    assert all(x.feller_margin > 0 for x in models.values())

    def qv(pp, t):
        return (
            pp.theta
            + (pp.v0 - pp.theta) * (-np.expm1(-pp.kappa * t)) / (pp.kappa * t)
            + pp.intensity * (pp.jump_mean**2 + pp.jump_vol**2)
        )

    for t in (0.02, 0.25, 2.0):
        assert qv(p, t) == pytest.approx(qv(models["heavy_b"], t), abs=1e-13)
        assert p.cf(0, t) == pytest.approx(1.0)
        assert p.cf(-1j, t) == pytest.approx(np.exp(p.rate * t))
    assert cos_price(p, 100, 100, 0.25) == pytest.approx(4.9455929636, abs=2e-8)
    assert cos_price(p.no_jumps(), 100, 100, 0.25) == pytest.approx(4.7702573602, abs=2e-8)


@pytest.mark.parametrize("jumps", [False, True])
def test_cos_fft_parity_and_greeks(jumps):
    p = Bates() if jumps else Bates().no_jumps()
    s, k, t = 100.0, np.array([84.0, 100.0, 114.0]), 0.25
    call = cos_price(p, s, k, t)
    put = cos_price(p, s, k, t, put=True)
    np.testing.assert_allclose(call - put, s - k * np.exp(-p.rate * t), atol=1e-10)
    np.testing.assert_allclose(fft_call(p, s, k, t), call, atol=0.00015)
    h = 0.02
    d = 0.00003
    v0 = cos_price(p, s, 100.0, t)
    vp = cos_price(p, s + h, 100.0, t)
    vm = cos_price(p, s - h, 100.0, t)
    g = cos_greeks(p, s, 100.0, t)
    assert g["delta"] == pytest.approx((vp - vm) / (2 * h), abs=2e-6)
    assert g["gamma"] == pytest.approx((vp - 2 * v0 + vm) / h**2, abs=2e-6)
    assert g["variance"] == pytest.approx(
        (cos_price(replace(p, v0=p.v0 + d), s, 100.0, t) - cos_price(replace(p, v0=p.v0 - d), s, 100.0, t))
        / (2 * d),
        abs=2e-3,
    )


def test_monte_carlo_barrier_parity_and_forward():
    p = Bates()
    n, steps = 20000, 42
    mc = simulate(100, 0.125, p, steps, n, 21)
    call = monte_carlo.payoff(mc, 100, 0.125, p.rate)
    put = monte_carlo.payoff(mc, 100, 0.125, p.rate, put=True)
    out = monte_carlo.payoff(mc, 100, 0.125, p.rate, barrier=125, monitor_steps=np.arange(2, steps + 1, 2))
    in_ = monte_carlo.payoff(
        mc, 100, 0.125, p.rate, barrier=125, monitor_steps=np.arange(2, steps + 1, 2), knock_in=True
    )
    np.testing.assert_allclose(in_ + out, call, atol=1e-13)
    np.testing.assert_allclose(call - put, np.exp(-p.rate * 0.125) * (mc.spot[-1] - 100), atol=1e-13)
    exact = cos_price(p, 100, 100, 0.125)
    assert abs(call.mean() - exact) < 4 * call.std(ddof=1) / np.sqrt(n)
    aver = mc.spot[-1] * np.exp(-p.rate * 0.125)
    assert abs(aver.mean() - 100) < 4 * aver.std(ddof=1) / np.sqrt(n)


def test_cn_convergence_psor_and_continuous_limit():
    t = 0.25
    exact = black.price(100, 100, t, 0.24)
    err = []
    for n in (100, 200, 400):
        err.append(abs(bs_grid(100, t, 0.24, nx=n, steps=n)(100) - exact))
    assert err[0] / err[1] > 3.7 and err[1] / err[2] > 3.7
    american = bs_grid(100, t, 0.24, put=True, american=True, nx=160, steps=70)(100)
    assert american > black.price(100, 100, t, 0.24, put=True)
    cont = continuous_upout(100, 100, t, 125, 0.24)
    assert bs_grid(100, t, 0.24, nx=800, steps=440, barrier=125, continuous=True)(100) == pytest.approx(
        cont, abs=0.006
    )
    discrete = bs_grid(100, t, 0.24, nx=500, steps=252, barrier=125)(100)
    assert discrete > cont
    assert bs_grid(100, t, 0.24, nx=500, steps=252, barrier=125)(126) > 0.0


@pytest.mark.parametrize("method", ["douglas", "cs", "hv"])
def test_adi_against_cos_and_european_parity(method):
    p = Bates().no_jumps()
    opts = {"nx": 116, "nv": 28, "steps": 90, "method": method}
    call = adi_grid(p, 100, 0.16, **opts).interpolator().ev(100, p.v0)
    put = adi_grid(p, 100, 0.16, put=True, **opts).interpolator().ev(100, p.v0)
    reference = cos_price(p, 100, 100, 0.16)
    assert call == pytest.approx(reference, abs=0.13)
    assert call - put == pytest.approx(100 * (1 - np.exp(-p.rate * 0.16)), abs=0.006)


@pytest.mark.slow
def test_heston_and_bates_barrier_pide_against_mc():
    for p in (Bates(), Bates().no_jumps()):
        t = 21 / 252
        n = 45000
        sim = simulate(100, t, p, 84, n, 567)
        pay = monte_carlo.payoff(sim, 100, t, p.rate, barrier=125, monitor_steps=np.arange(4, 85, 4))
        ad = adi_grid(p, 100, t, nx=192, nv=46, steps=160, barrier=125).interpolator().ev(100, p.v0)
        assert ad == pytest.approx(pay.mean(), abs=0.055 + 4 * pay.std(ddof=1) / np.sqrt(n))


@pytest.mark.slow
def test_mc_standard_error_rate():
    sizes = np.array([2048, 8192, 32768])
    errors = []
    for n in sizes:
        p = Bates()
        x = monte_carlo.payoff(simulate(100, 0.25, p, 28, n, 55), 100, 0.25, p.rate)
        errors.append(x.std(ddof=1) / np.sqrt(n))
    assert np.polyfit(np.log(sizes), np.log(errors), 1)[0] == pytest.approx(-0.5, abs=0.06)


@pytest.mark.slow
@pytest.mark.parametrize("method,ratio", [("douglas", (1.75, 2.25)), ("cs", (3.75, 4.3)), ("hv", (3.2, 4.3))])
def test_adi_time_convergence_on_common_spatial_grid(method, ratio):
    p = Bates().no_jumps()
    values = [
        float(
            adi_grid(p, 100.0, 0.25, method=method, nx=60, nv=18, steps=steps).interpolator().ev(103.0, p.v0)
        )
        for steps in (64, 128, 256, 1800)
    ]
    errors = np.abs(np.asarray(values[:-1]) - values[-1])
    assert np.all((errors[:-1] / errors[1:] >= ratio[0]) & (errors[:-1] / errors[1:] <= ratio[1]))


def test_bates_barrier_grid_stores_intraday_interpolation_nodes():
    p = Bates()
    maturity = 3 / 252
    grid = adi_grid(p, 100.0, maturity, nx=50, nv=12, steps=24, barrier=125.0, store=True)
    for i in range(6):
        index = np.argmin(np.abs(grid.days - i * maturity / 6))
        assert grid.days[index] == pytest.approx(i * maturity / 6, abs=1e-12)
        assert grid.interpolator(index).ev(100, p.v0) > 0
