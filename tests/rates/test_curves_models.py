"""Calendar, quote repricing and exact funding controls."""

from datetime import date

import numpy as np
import pytest

from quantproj.core.ledger import Ledger
from quantproj.rates.curves import (
    NelsonSiegel,
    act360,
    bootstrap,
    maturity,
    modified_following,
    nc1,
    ois_quotes,
    quote_swaps,
    swap,
)
from quantproj.rates.hedging import coupon_cash, swap_mark
from quantproj.rates.models import g2, hw, simulate

ORIGIN = date(2025, 1, 2)


def test_contract_schedule_and_act360():
    assert modified_following(date(2026, 1, 31)) == date(2026, 1, 30)
    assert maturity(date(2026, 1, 30), "ON") == date(2026, 2, 2)
    assert act360(date(2025, 1, 3), date(2025, 1, 6)) == pytest.approx(3 / 360)
    p = nc1(ORIGIN, ORIGIN, NelsonSiegel().discount)
    assert len(p.dates[:-1]) == 9
    assert p.dates[-1] == act360(ORIGIN, maturity(ORIGIN, "10Y"))
    assert p.fair_rate(NelsonSiegel().discount) == pytest.approx(p.fixed)


def test_bootstrap_reprices_and_monotone_convex_has_continuous_forwards():
    inst = quote_swaps(ORIGIN)
    q = ois_quotes(inst, NelsonSiegel().discount)
    for kind in ("log_linear", "monotone_convex"):
        fitted = bootstrap(inst, q, kind)
        np.testing.assert_allclose(ois_quotes(inst, fitted.discount), q, atol=1e-11)
        if kind == "monotone_convex":
            at = fitted.nodes[1:-1]
            np.testing.assert_allclose(fitted.forward(at - 2e-5), fitted.forward(at + 2e-5), atol=2e-5)


@pytest.mark.parametrize("world", ("hw", "base", "second"))
def test_curve_fit_and_discounted_bond_martingale(world):
    curve = NelsonSiegel()
    model = hw(curve, 0.06, [0.01, 0.015], (1.0,)) if world == "hw" else g2(curve, world)
    state = np.zeros(model.dimension)
    np.testing.assert_allclose(
        model.bond(0.0, [0.2, 0.8, 3.0, 10.0], state), curve.discount([0.2, 0.8, 3.0, 10.0]), rtol=2e-13
    )
    states, bank = simulate(model, [0.0, 0.4, 1.4, 4.0], 18000, 81)
    for t, x, b in zip([0.4, 1.4, 4.0], states[1:], bank[1:]):
        sample = model.bond(t, 10.0, x) / b
        assert abs(sample.mean() - curve.discount(10.0)) < 4.2 * sample.std(ddof=1) / np.sqrt(len(sample))


def test_post_coupon_marks_and_stochastic_funding():
    s = swap(ORIGIN, "3Y", ORIGIN, 0.031)
    curve = NelsonSiegel()

    def bank(t):
        return np.exp(0.033 * t + 0.001 * t * t)

    for t in s.dates[1:]:

        def discount(dates):
            return curve.discount(dates) / curve.discount(t)

        before = swap_mark(s, t, bank, discount, before_payment=True)
        after = swap_mark(s, t, bank, discount)
        assert before == pytest.approx(after + coupon_cash(s, t, bank), abs=1e-13)
    book = Ledger(100.0, 1.0, np.zeros(1))
    book.accrue(1.1)
    book.trade(np.array([2.0]), np.array([11.0]), np.array([0.1]))
    assert book.discounted([11.0]) == pytest.approx(100.0 - 0.2 / 1.1)
    book.accrue(1.21)
    book.settle([0.7])
    assert book.discounted() == pytest.approx((110 - 22.2) / 1.1 + 1.4 / 1.21)


def test_key_rate_buckets_preserve_parallel_risk():
    from types import SimpleNamespace

    from quantproj.rates.hedging import key_rate_buckets

    market = SimpleNamespace(swaps=quote_swaps(ORIGIN))
    pillars = [10.1, 2.03, 5.07]
    weights = key_rate_buckets(market, pillars)
    np.testing.assert_allclose(weights.sum(axis=0), 1.0)
    nodes = np.array([s.dates[-1] for s in market.swaps])
    inside = (nodes > min(pillars)) & (nodes < max(pillars))
    # Interior nodes keep their maturity as the weighted pillar average.
    np.testing.assert_allclose((np.asarray(pillars) @ weights)[inside], nodes[inside])
    assert weights[1, 0] == 1.0 and weights[0, -1] == 1.0
