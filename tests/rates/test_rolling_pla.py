"""Exercise rollover must preserve the frozen-book 250-observation window."""

from datetime import date

import numpy as np
import pytest

from quantproj.rates.attribution import run_pla
from quantproj.rates.bermudan import g2_adi
from quantproj.rates.curves import act360, business_dates, nc1
from quantproj.rates.models import g2


@pytest.mark.slow
def test_pla_freezes_exercise_and_rolls_after_the_event():
    pytest.importorskip("jax")
    start = date(2025, 1, 2)
    asofs = business_dates(start, date(2026, 10, 2))
    times = np.array([act360(start, d) for d in asofs])
    world = g2()
    contract = nc1(start, start, world.curve.discount)
    solution = g2_adi(world, contract, n=25, steps_per_year=8)
    t1 = int(np.argmin(abs(times - contract.dates[0])))
    x = np.zeros((len(times), 1, 2))
    x[:, 0, 0] = np.linspace(0.0, 0.012, len(times))
    # A high first fixing forces a real physical-exercise and rollover branch.
    x[t1, 0] = [0.045, 0.10]
    bank = np.exp(times[:, None] * 0.03)
    warmup = next(i for i, day in enumerate(asofs) if day == date(2025, 10, 2))
    settings = {
        "mean_reversion": 0.06,
        "ois_noise_bp": 0.0,
        "normal_noise_bp": 0.0,
        "swap_tenors": ["2Y", "5Y"],
        "desk_grid": 25,
        "desk_steps": 2,
        "position_cap": 4.0,
        "adi_grid": 25,
        "adi_steps": 8,
    }
    rows, metrics, _ = run_pla(world, times, asofs, x, bank, contract, solution, warmup, settings, 4)
    assert len(rows) == 250
    assert metrics[0][6:8] == [1, 0]
    event = rows[t1 - warmup]
    assert event[3] == "exercise"
    assert np.all(np.isfinite(np.asarray(event[4:11], dtype=float)))
    assert rows[t1 - warmup + 1][3] == "none"
