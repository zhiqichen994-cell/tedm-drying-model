"""Regression tests: the published numbers of the paper.

Every expected value below is quoted from the paper (摘要 / 表1-表6) and was
produced by the code in this repository.  If one of these tests starts failing
after a refactor, the refactor changed the physics - not the formatting.

Reference values
----------------
Q1  (30 min, 28 -> 36.79 C surface)
    T(1800 s) centre = 33.5765 C, surface = 36.7863 C
    C(1800 s) surface = 1.5104 kg/kg
Q2  (3 h, 附录3 variable properties)
    T(3 h) centre = 49.8494 C, surface = 49.9664 C
    C(3 h) centre = 1.7662,    surface = 1.0081 kg/kg
Q3  (constant 50 C, 附录3)
    drying time tau = 57.2234 h
Q4  (附录4 + shrinkage from 附件2)
    R 2.00 -> 1.20 cm, drying time tau = 50.8651 h
    fixed-radius control: tau = 129.1500 h  (-60.6 %)
"""

from __future__ import annotations

import numpy as np
import pytest

import tedm.config as cfg
from tedm.properties import APP2
from tedm.solver import solve
from tedm.solver_q2 import solve_q2
from tedm.solver_q3 import solve_q3
from tedm.solver_q4 import solve_q4
from tedm.solver_q4_fixed import solve_q4_fixed

# the paper reports 4 decimals, so 1e-3 absolute is a comfortable tolerance
TOL_4DP = 5e-4


# ----------------------------------------------------------------------
# problem 1
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def q1():
    return solve(prop=APP2, t_end=1800.0, dt=1.0)


def test_q1_temperature_endpoints(q1):
    _t, _r, T, _C, _Ta, _Ca = q1
    assert T[-1, 0] == pytest.approx(33.5765, abs=TOL_4DP)
    assert T[-1, -1] == pytest.approx(36.7863, abs=TOL_4DP)


def test_q1_surface_moisture(q1):
    _t, _r, _T, C, _Ta, _Ca = q1
    assert C[-1, -1] == pytest.approx(1.5104, abs=TOL_4DP)


def test_q1_output_shapes(q1):
    t, r, T, C, T_air, C_air = q1
    assert T.shape == C.shape == (1801, 101)
    assert t.shape == T_air.shape == C_air.shape == (1801,)
    assert r.shape == (101,)


def test_q1_profile_is_physical(q1):
    """Temperature and moisture must be monotone in r and obey the bounds."""
    _t, r, T, C, Ta, _Ca = q1
    T_end, C_end = T[-1], C[-1]
    # the surface is hottest (heating) and driest (evaporating)
    assert T_end[0] < T_end[-1]
    assert C_end[0] > C_end[-1]
    # nothing may exceed the initial moisture or fall below zero
    assert np.all(C <= cfg.C0 + 1e-12)
    assert np.all(C >= 0.0)
    # the solid can never be hotter than the air driving it
    assert T_end.max() <= float(Ta[-1]) + 1e-9
    assert np.all(T >= cfg.T0 - 1e-9)


def test_q1_moisture_monotone_decreasing_at_every_node(q1):
    _t, _r, _T, C, _Ta, _Ca = q1
    assert np.all(np.diff(C, axis=0) <= 1e-12)


def test_q1_centre_barely_dries_in_30_minutes(q1):
    """At 30 min the drying front has not reached the axis yet."""
    _t, _r, _T, C, _Ta, _Ca = q1
    assert C[-1, 0] > 2.50


# ----------------------------------------------------------------------
# problem 2
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def q2():
    return solve_q2()


def test_q2_temperature_endpoints(q2):
    _t, _r, T, _C, _Ta, _Ca = q2
    assert T[-1, 0] == pytest.approx(49.8494, abs=TOL_4DP)
    assert T[-1, -1] == pytest.approx(49.9664, abs=TOL_4DP)


def test_q2_moisture_endpoints(q2):
    _t, _r, _T, C, _Ta, _Ca = q2
    assert C[-1, 0] == pytest.approx(1.7662, abs=TOL_4DP)
    assert C[-1, -1] == pytest.approx(1.0081, abs=TOL_4DP)


def test_q2_reaches_the_50c_plateau(q2):
    """With variable properties the whole rod ends up near the 50 C plateau."""
    _t, _r, T, _C, _Ta, _Ca = q2
    assert np.all(T[-1] > 49.0)
    assert T[-1, -1] - T[-1, 0] < 0.15


def test_q2_removes_more_water_than_q1_would():
    """附录3 has a much larger D once the Arrhenius factor is included, so at
    the same physical time it removes more total water.

    Note the *surface* value alone would be misleading: the 附录2 run has a
    lower surface moisture (0.73 vs 1.01) precisely because its diffusivity
    collapses as it dries, so it skins over while its interior stays wet.  The
    volume-averaged moisture is the meaningful comparison.
    """
    from tedm.analysis.conservation import volume_average

    _t, r, _T, C1, _a, _b = solve(prop=APP2, t_end=10800.0, dt=1.0)
    _t, _r, _T, C2, _Ta, _Ca = solve_q2()
    r_face = (np.arange(C1.shape[1] - 1) + 0.5) * (cfg.R / (C1.shape[1] - 1))

    mean1 = volume_average(r_face, C1[-1], cfg.R)
    mean2 = volume_average(r_face, C2[-1], cfg.R)
    assert mean2 < mean1
    # and 附录2 really does skin over faster at the boundary
    assert C1[-1, -1] < C2[-1, -1]



# ----------------------------------------------------------------------
# problem 3
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def q3():
    return solve_q3()


def test_q3_drying_time(q3):
    assert q3[6] == pytest.approx(57.2234, abs=5e-3)


def test_q3_stops_at_the_target_moisture(q3):
    _t, _r, _T, C, _Ta, _Ca, _t_end_h, end_C = q3
    assert end_C < 0.15
    assert end_C > 0.149        # it must stop right at the threshold
    assert C[-1, 0] == pytest.approx(end_C)


def test_q3_matches_q2_at_three_hours(q3):
    """Q3 is Q2 continued, so the first 3 h must coincide."""
    _t, _r, _T, C3, _Ta, _Ca, _t_end_h, _c = q3
    _t2, _r2, _T2, C2, _a, _b = solve_q2()
    assert C3[180, 0] == pytest.approx(C2[-1, 0], abs=2e-3)
    assert C3[180, -1] == pytest.approx(C2[-1, -1], abs=2e-3)


def test_q3_time_grid_uses_60s_steps(q3):
    t = q3[0]
    assert t[1] - t[0] == pytest.approx(60.0)
    # the reported tau is interpolated inside the crossing step, so the last
    # stored sample is the *end* of that step and sits at most one dt later
    assert t[-1] > q3[6] * 3600.0
    assert t[-1] - q3[6] * 3600.0 <= 60.0
    assert (len(t) - 1) * 60.0 == pytest.approx(t[-1])



def test_q3_surface_dries_far_below_the_target(q3):
    _t, _r, _T, C, _Ta, _Ca, _t_end_h, _c = q3
    assert C[-1, -1] < 0.06


# ----------------------------------------------------------------------
# problem 4
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def q4():
    return solve_q4()


def test_q4_drying_time_with_shrinkage(q4):
    assert q4[7] == pytest.approx(50.8651, abs=5e-3)


def test_q4_radius_shrinks_from_2cm_to_around_1p2cm(q4):
    _t, _xi, _T, _C, R_result, _Ta, _Ca, _t_end_h = q4
    assert R_result[0] == pytest.approx(0.0200, abs=1e-4)
    assert R_result[-1] * 100 == pytest.approx(1.20, abs=0.01)


def test_q4_stops_at_the_target_moisture(q4):
    _t, _xi, _T, C, _R, _Ta, _Ca, _t_end_h = q4
    assert C[-1, 0] < 0.15
    assert C[-1, 0] > 0.149


def test_q4_normalised_grid_is_the_unit_interval(q4):
    _t, xi, _T, _C, _R, _Ta, _Ca, _t_end_h = q4
    assert xi[0] == 0.0
    assert xi[-1] == pytest.approx(1.0)
    assert np.all(np.diff(xi) > 0)


@pytest.mark.slow
def test_q4_fixed_radius_control():
    """Shrinkage cuts the drying time from ~129 h to ~51 h."""
    _t, _xi, _T, _C, t_end_h = solve_q4_fixed()
    assert t_end_h == pytest.approx(129.1500, abs=1e-2)


@pytest.mark.slow
def test_q4_shrinkage_saves_about_60_percent():
    tau_shrink = solve_q4()[7]
    tau_fixed = solve_q4_fixed()[4]
    saving = (tau_fixed - tau_shrink) / tau_fixed
    assert saving == pytest.approx(0.606, abs=0.01)
    assert tau_shrink < tau_fixed
