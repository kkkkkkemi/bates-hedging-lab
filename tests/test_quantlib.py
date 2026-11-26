"""External validation only. QuantLib is never imported by feature code."""

import pytest

from quantproj.models import Bates
from quantproj.pricing.fourier import cos_price
from quantproj.pricing.pde import bs_grid, continuous_upout

ql = pytest.importorskip("QuantLib")


def _shared():
    today = ql.Date(20, 1, 2025)
    ql.Settings.instance().evaluationDate = today
    expiry = today + 91
    dc = ql.Actual365Fixed()
    date = ql.Date(today.serialNumber())
    spot = ql.QuoteHandle(ql.SimpleQuote(100.0))
    r = ql.YieldTermStructureHandle(ql.FlatForward(date, 0.02, dc))
    q = ql.YieldTermStructureHandle(ql.FlatForward(date, 0.0, dc))
    v = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(date, ql.NullCalendar(), 0.24, dc))
    bs = ql.BlackScholesMertonProcess(spot, q, r, v)
    return today, expiry, spot, r, q, bs, dc.yearFraction(today, expiry)


def test_quantlib_analytic_heston():
    _, expiry, s, r, q, _, t = _shared()
    p = Bates().no_jumps()
    process = ql.HestonProcess(r, q, s, p.v0, p.kappa, p.theta, p.xi, p.rho)
    option = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, 100.0), ql.EuropeanExercise(expiry))
    option.setPricingEngine(ql.AnalyticHestonEngine(ql.HestonModel(process), 144))
    assert cos_price(p, 100.0, 100.0, t) == pytest.approx(option.NPV(), abs=2e-5)


def test_quantlib_american_and_continuous_barrier():
    today, expiry, _, _, _, bs, t = _shared()
    put = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Put, 100.0), ql.AmericanExercise(today, expiry))
    put.setPricingEngine(ql.FdBlackScholesVanillaEngine(bs, 480, 480))
    assert bs_grid(100, t, 0.24, nx=360, steps=200, put=True, american=True)(100) == pytest.approx(
        put.NPV(), abs=0.015
    )
    barrier = ql.BarrierOption(
        ql.Barrier.UpOut,
        125.0,
        0.0,
        ql.PlainVanillaPayoff(ql.Option.Call, 100.0),
        ql.EuropeanExercise(expiry),
    )
    barrier.setPricingEngine(ql.AnalyticBarrierEngine(bs))
    assert continuous_upout(100, 100, t, 125, 0.24) == pytest.approx(barrier.NPV(), abs=2e-8)
