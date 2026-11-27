"""PLA desk plan and the single quote snapshot used on a roll day."""

from datetime import date

import numpy as np

from quantproj.rates.attribution import desk_plan
from quantproj.rates.calibration import observe
from quantproj.rates.curves import act360, nc1
from quantproj.rates.models import g2

ORIGIN = date(2025, 1, 2)


def test_roll_day_snapshot_reuses_the_close_ois_quotes():
    world = g2()
    old = nc1(ORIGIN, ORIGIN, world.curve.discount)
    asof = date(2026, 1, 2)
    t = act360(ORIGIN, asof)
    state = np.array([0.004, -0.002])

    def snapshot(contract):
        rng = np.random.default_rng(np.random.SeedSequence([7, 0, 251]))
        return observe(world, t, state, asof, contract, rng, 0.15, 0.30, smile_offsets=(-25, 25))

    before = snapshot(old)
    after = snapshot(nc1(asof, ORIGIN, lambda ends: world.bond(t, ends, state)))
    np.testing.assert_array_equal(before.ois, after.ois)
    np.testing.assert_array_equal(before.curve.logs, after.curve.logs)


def test_desk_plan_pairs_variants_and_shares_identical_desks():
    base = {
        "front_office": {"reversions": [0.77, 0.08], "rho": -0.7, "grid": 31, "steps": 12, "wings": True},
        "risk_g2": {"reversions": [0.75, 0.08], "rho": -0.7, "grid": 25, "steps": 8, "wings": True},
        "pla_sensitivity": [
            {"name": "daily_rho", "front_office": {"rho": None}, "risk_g2": {"rho": None}},
            {"name": "risk_atm_only", "risk_g2": {"wings": False}},
        ],
    }
    desks, pairs = desk_plan(base)
    assert pairs["main"] == ("front_office", "risk_g2")
    assert pairs["risk_atm_only"][0] == "front_office"
    assert desks[pairs["daily_rho"][0]]["rho"] is None
    assert desks[pairs["risk_atm_only"][1]]["wings"] is False
    assert len(desks) == 5
    assert desk_plan({}) == ({}, {})
