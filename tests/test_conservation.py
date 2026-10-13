"""Tests for the discrete conservation diagnostics.

A finite-volume scheme must conserve energy and mass *exactly* (to round-off)
when the diagnostic is written with the same metric as the discretisation.
These tests lock that in, so a future change to the surface treatment that
breaks the budget will fail loudly instead of producing a plausible-looking
number.

Historical note: the original ``validate.py`` reported FAILED for all three
of its checks.  The failures were metric bugs in the *check*, not in the
solver; see ``docs/limitations.md`` for the full story.
"""

from __future__ import annotations

import numpy as np
import pytest

import tedm.config as cfg
from tedm.analysis.conservation import (
    bound_check,
    cell_volumes,
    energy_budget,
    surface_exchange_series,
    volume_average,
    water_budget,
)
from tedm.model import build_grid
from tedm.properties import APP2
from tedm.solver import solve

REL_TOL = 1e-8   # backward Euler + exact telescoping => round-off only


@pytest.fixture(scope="module")
def q1():
    return solve(prop=APP2, t_end=1800.0, dt=1.0)


def test_cell_volumes_sum_to_the_cross_section():
    """pi * R^2 is the true per-unit-length volume."""
    _r, r_face, _v = build_grid()
    vol, v = cell_volumes(r_face, cfg.R)
    assert vol.sum() == pytest.approx(np.pi * cfg.R ** 2, rel=1e-12)
    assert np.allclose(vol, np.pi * v, rtol=1e-12)


def test_volume_average_of_a_constant_field():
    _r, r_face, _v = build_grid()
    field = np.full(r_face.size + 1, 3.7)
    assert volume_average(r_face, field, cfg.R) == pytest.approx(3.7, rel=1e-14)


def test_volume_average_weights_the_outer_ring_more():
    """A cylindrical average is NOT the arithmetic mean.

    For ``f = (r/R)^2`` the volume-weighted average is exactly 1/2
    (integral of f r dr over the disc), while the arithmetic mean over equally
    spaced nodes is only 1/3.  Using the wrong one is what made the original
    conservation check report a 2.0 residual.
    """
    r, r_face, _v = build_grid()
    field = (r / cfg.R) ** 2
    cyl = volume_average(r_face, field, cfg.R)
    arith = float(np.mean(field))
    assert cyl == pytest.approx(0.5, abs=5e-3)
    assert arith == pytest.approx(1.0 / 3.0, abs=1e-2)
    assert cyl - arith > 0.15


def test_energy_budget_closes_to_round_off(q1):
    t, r, T, _C, T_air, _C_air = q1
    e = energy_budget(t, r, T, T_air)
    assert e["rel_residual"] < REL_TOL
    # the rod is being heated, so both sides must be positive
    assert e["d_storage"] > 0
    assert e["q_exchanged"] > 0


def test_water_budget_closes_to_round_off(q1):
    t, r, _T, C, _T_air, C_air = q1
    m = water_budget(t, r, C, C_air)
    assert m["rel_residual"] < REL_TOL
    # moisture leaves the rod, so both sides must be negative
    assert m["d_storage"] < 0
    assert m["m_exchanged"] < 0


def test_water_budget_uses_a_single_sign_convention(q1):
    """Regression: the storage and exchange terms must not have opposite signs.

    The original check mixed a "loss" convention with a "gain" convention and
    reported a residual equal to twice the physical value.
    """
    t, r, _T, C, _T_air, C_air = q1
    m = water_budget(t, r, C, C_air)
    assert np.sign(m["d_storage"]) == np.sign(m["m_exchanged"])
    assert abs(m["abs_residual"]) < abs(m["m_exchanged"]) * REL_TOL


def test_bounds_maximum_principle(q1):
    t, r, _T, C, _T_air, C_air = q1
    b = bound_check(C, C_air, cfg.C0)
    assert b["below_initial"]
    assert b["above_air"]
    assert b["monotone_surface"]
    assert b["min"] >= min(float(np.min(C_air)), 0.0) - 1e-9
    assert b["max"] <= cfg.C0 + 1e-9


def test_energy_budget_is_also_exact_for_a_coarser_grid():
    t, r, T, _C, T_air, _C_air = solve(prop=APP2, t_end=1800.0, dt=1.0, n_cells=25)
    e = energy_budget(t, r, T, T_air)
    assert e["rel_residual"] < REL_TOL


def test_surface_exchange_series_is_additive_over_time():
    """Splitting a time integral in two must not change the total.

    This is a property of the helper itself, checked on synthetic data so that
    it does not depend on the solver.
    """
    t = np.arange(0.0, 601.0, 1.0)
    driving = np.linspace(50.0, 50.0, t.size)
    surface = 50.0 - 25.0 * (1.0 - np.exp(-t / 300.0))

    full = surface_exchange_series(t, driving, surface, cfg.H_CONV, cfg.R)
    half_a = surface_exchange_series(t[:301], driving[:301], surface[:301],
                                     cfg.H_CONV, cfg.R)
    half_b = surface_exchange_series(t[300:], driving[300:], surface[300:],
                                     cfg.H_CONV, cfg.R)
    # the split point contributes its step to both halves, so compare with the
    # tolerance of one step's worth of exchange
    one_step = cfg.H_CONV * 2 * np.pi * cfg.R * abs(driving[0] - surface[300])
    assert full == pytest.approx(half_a + half_b, abs=one_step)


def test_surface_exchange_series_sign_is_outflow_positive():
    """If the surface is warmer than the air the integral must be negative
    (heat leaving the solid)."""
    t = np.arange(0.0, 101.0, 1.0)
    driving = np.full(t.size, 20.0)
    surface = np.full(t.size, 40.0)
    val = surface_exchange_series(t, driving, surface, cfg.H_CONV, cfg.R)
    assert val < 0



@pytest.mark.slow
def test_dry_matter_drift_in_the_shrinking_model_is_reportable():
    """Problem 4 does NOT conserve dry matter.  This test documents the size
    of the drift instead of pretending it is zero; see docs/limitations.md."""
    from tedm.analysis.conservation import dry_mass_history, report_moving_boundary

    hist = dry_mass_history(N=100, dt=60.0)
    txt, summary = report_moving_boundary(hist)
    drift = summary["dry_mass_drift"]
    assert -0.20 < drift < 0.0        # the drift is real and it is negative
    assert drift == pytest.approx(-0.11, abs=0.03)
