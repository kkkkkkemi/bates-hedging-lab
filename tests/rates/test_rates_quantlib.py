"""QuantLib is an external test benchmark for constant-volatility models."""

from datetime import date, timedelta

import numpy as np
import pytest

from quantproj.rates.curves import NelsonSiegel, nc1
from quantproj.rates.european import g2_european, jamshidian
from quantproj.rates.models import g2, hw

ql = pytest.importorskip("QuantLib")


def test_constant_hull_white_and_g2_against_quantlib():
    asof = date(2025, 1, 2)
    curve = NelsonSiegel(0.03, 0.0, 0.0)
    contract = nc1(asof, asof, curve.discount)

    def qld(p):
        return ql.Date(p.day, p.month, p.year)

    t0 = qld(asof)
    ql.Settings.instance().evaluationDate = t0
    dc = ql.Actual360()
    ts = ql.YieldTermStructureHandle(ql.FlatForward(t0, 0.03, dc))
    cal = ql.WeekendsOnly()
    dates = [qld(asof + timedelta(days=round(t * 360))) for t in contract.dates]
    leg = ql.Schedule(dates, cal, ql.ModifiedFollowing)
    # At zero lag a forecast simple annual floating coupon telescopes to the
    # same curve discount difference as the compounded OIS floating leg.
    index = ql.IborIndex(
        "Annual one-curve benchmark",
        ql.Period(1, ql.Years),
        0,
        ql.EURCurrency(),
        cal,
        ql.ModifiedFollowing,
        False,
        dc,
        ts,
    )
    swap = ql.VanillaSwap(ql.VanillaSwap.Payer, 1.0, leg, contract.fixed, dc, leg, index, 0.0, dc)
    option = ql.Swaption(swap, ql.EuropeanExercise(dates[0]))
    option.setPricingEngine(ql.JamshidianSwaptionEngine(ql.HullWhite(ts, 0.06, 0.011)))
    np.testing.assert_allclose(
        jamshidian(hw(curve, 0.06, [0.011]), 0.0, [0.0], contract), option.NPV(), atol=2e-9
    )
    option.setPricingEngine(ql.G2SwaptionEngine(ql.G2(ts, 0.7735, 0.0223, 0.082, 0.0104, -0.702), 8.0, 180))
    np.testing.assert_allclose(
        g2_european(g2(curve), 0.0, [0.0, 0.0], contract, order=64), option.NPV(), atol=6e-7
    )
