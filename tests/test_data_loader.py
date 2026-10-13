"""Tests for the two supplied data sets (附件1 drying-chamber conditions,
附件2 radius shrinkage)."""

from __future__ import annotations

import numpy as np
import pytest

import tedm.config as cfg
from tedm.data_loader import load_air_conditions, load_radius


def test_attachment_1_exists():
    assert cfg.FILE_ATTACH1.exists(), f"missing data file: {cfg.FILE_ATTACH1}"


def test_attachment_2_exists():
    assert (cfg.DATA_DIR / "附件2.xlsx").exists()


def test_air_conditions_default_grid():
    t, T_air, C_air = load_air_conditions()
    assert t.shape == T_air.shape == C_air.shape
    assert t[0] == 0.0
    assert t[-1] == pytest.approx(cfg.T_END)
    assert len(t) == int(round(cfg.T_END / cfg.DT)) + 1


def test_air_conditions_respect_requested_grid():
    t, T_air, C_air = load_air_conditions(t_end=3600.0, dt=10.0)
    assert t.shape == (361,)
    assert t[1] - t[0] == pytest.approx(10.0)


def test_air_conditions_start_at_the_initial_state():
    """At t = 0 the chamber is at 28 C and holds 0.01963 kg/kg of vapour.

    Note that the air moisture content is *not* the initial moisture of the
    herb (C0 = 2.55 kg/kg) - they are different quantities on different bases.
    """
    _t, T_air, C_air = load_air_conditions()
    assert T_air[0] == pytest.approx(28.0)
    assert C_air[0] == pytest.approx(0.01963, rel=1e-4)


def test_air_conditions_are_non_negative_and_bounded():
    _t, T_air, C_air = load_air_conditions()
    assert np.all(T_air >= 28.0 - 1e-9)
    assert np.all(C_air >= 0.0)
    assert np.all(np.isfinite(T_air)) and np.all(np.isfinite(C_air))


def test_air_conditions_are_a_one_minute_series():
    """附件1 is a measured 1-minute series over 4 h (241 rows), not a set-point
    table - so the interpolation is smooth, not a staircase."""
    t, T_air, C_air = load_air_conditions(t_end=14400.0, dt=1.0)
    assert t.shape == (14401,)
    assert t[1] - t[0] == pytest.approx(1.0)

    # the signal rises from 28 C towards ~50 C with only small noise; the
    # largest single-minute jump must be small compared to the total swing
    dT = np.abs(np.diff(T_air))
    assert dT.max() < 1.0
    assert dT.max() / (T_air.max() - T_air.min()) < 0.1
    assert np.sum(dT) > 20.0          # total variation is real
    assert dT.max() > 1e-3            # and it is not constant


def test_air_conditions_reach_the_fifty_degree_plateau():
    _t, T_air, C_air = load_air_conditions(t_end=14400.0, dt=1.0)
    assert T_air[-1] == pytest.approx(50.0, abs=0.5)
    # the rise is roughly complete after about 2 h
    assert T_air[7200] > 49.0


def test_air_conditions_clamp_beyond_the_table():
    """np.interp clamps, which is the documented 'hold the plateau' behaviour.

    Problem 3 runs for ~57 h but 附件1 only covers 4 h, so the tail of that run
    is driven by the frozen 4 h end value.  This test pins that behaviour.
    """
    t_short, T_short, C_short = load_air_conditions(t_end=14400.0, dt=1.0)
    _t, T_air, C_air = load_air_conditions(t_end=96 * 3600.0, dt=3600.0)
    assert T_air[-1] == pytest.approx(T_short[-1])
    assert C_air[-1] == pytest.approx(C_short[-1])
    assert T_air[5] == pytest.approx(T_short[-1])



def test_radius_shrinks_monotonically_to_1p2_cm():
    t_data, R_data, _R_func = load_radius()
    assert np.all(np.diff(R_data) <= 1e-12)
    assert R_data[0] == pytest.approx(2.000, abs=1e-3)
    assert R_data[-1] == pytest.approx(1.198, abs=1e-3)


def test_radius_interpolant_agrees_with_the_table():
    t_data, R_data, R_func = load_radius()
    assert np.allclose(R_func(t_data), R_data, atol=1e-9)


def test_radius_interpolant_clamps_outside_the_table():
    t_data, R_data, R_func = load_radius()
    assert float(R_func(t_data[0] - 1e5)) == pytest.approx(R_data[0])
    assert float(R_func(t_data[-1] + 1e5)) == pytest.approx(R_data[-1])
