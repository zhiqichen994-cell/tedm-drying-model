"""Structural tests for the two time-step operators in :mod:`tedm.model`.

These check the properties that must hold *regardless* of the physics: the
steady state is preserved, the discrete maximum principle holds, and the
solver does not depend on the grid size in the continuum limit.
"""

from __future__ import annotations

import numpy as np
import pytest

import tedm.config as cfg
from tedm.model import build_grid, step_moisture, step_temperature
from tedm.properties import APP2, APP3


def _field(n_cells, value):
    return np.full(n_cells + 1, float(value))


def test_temperature_step_preserves_equilibrium():
    """If the solid is already at the air temperature nothing changes."""
    r, r_face, v = build_grid(n_cells=25)
    T = _field(25, 50.0)
    C = _field(25, 1.0)
    T_new = step_temperature(T, C, r_face, v, 1.0, 50.0, 1.0, prop=APP2)
    assert np.allclose(T_new, 50.0, atol=1e-10)


def test_moisture_step_preserves_equilibrium():
    """If the solid is already at the air moisture content nothing changes."""
    r, r_face, v = build_grid(n_cells=25)
    T = _field(25, 50.0)
    C = _field(25, 1.0)
    C_new = step_moisture(C, T, r_face, v, 1.0, 1.0, prop=APP2, picard=2)
    assert np.allclose(C_new, 1.0, atol=1e-10)


def test_temperature_step_respects_the_maximum_principle():
    """With a uniform initial field inside [T0, T_air] the result must stay
    inside that interval - backward Euler is unconditionally monotone."""
    r, r_face, v = build_grid(n_cells=50)
    T = _field(50, cfg.T0)
    C = _field(50, cfg.C0)
    T_air = 60.0
    T_new = step_temperature(T, C, r_face, v, 30.0, T_air, 0.5, prop=APP2)
    assert np.all(T_new >= cfg.T0 - 1e-9)
    assert np.all(T_new <= T_air + 1e-9)


def test_moisture_step_respects_the_maximum_principle():
    r, r_face, v = build_grid(n_cells=50)
    T = _field(50, 50.0)
    C = _field(50, cfg.C0)
    C_new = step_moisture(C, T, r_face, v, 30.0, 0.5, prop=APP2, picard=2)
    assert np.all(C_new <= cfg.C0 + 1e-9)
    assert np.all(C_new >= 0.5 - 1e-9)


def test_temperature_step_is_stable_at_a_large_time_step():
    """Backward Euler must not blow up; an explicit scheme would."""
    r, r_face, v = build_grid(n_cells=100)
    T = np.linspace(cfg.T0, cfg.T0 + 5.0, 101)
    C = _field(100, cfg.C0)
    T_new = step_temperature(T, C, r_face, v, 600.0, 50.0, 0.5, prop=APP2)
    assert np.all(np.isfinite(T_new))
    assert np.all(np.abs(T_new) < 200.0)


def test_moisture_step_is_stable_at_a_large_time_step():
    r, r_face, v = build_grid(n_cells=100)
    T = _field(100, 50.0)
    C = np.linspace(cfg.C0, 1.0, 101)
    C_new = step_moisture(C, T, r_face, v, 3600.0, 0.5, prop=APP2, picard=2)
    assert np.all(np.isfinite(C_new))
    assert np.all(C_new >= -1e-6)


def test_step_operators_use_the_radius_argument():
    """The Robin coefficient scales with R, so a bigger cylinder heats slower."""
    _r1, rf1, v1 = build_grid(radius=0.02, n_cells=50)
    _r2, rf2, v2 = build_grid(radius=0.01, n_cells=50)
    T = _field(50, cfg.T0)
    C = _field(50, cfg.C0)

    small = step_temperature(T, C, rf2, v2, 60.0, 50.0, 0.5, prop=APP2,
                             radius=0.01)
    big = step_temperature(T, C, rf1, v1, 60.0, 50.0, 0.5, prop=APP2,
                           radius=0.02)
    # the small cylinder has more surface per unit volume, so it heats up more
    assert small[-1] > big[-1]


def test_extra_picard_sweeps_barely_change_the_answer():
    """The lagged diffusivity is solved by Picard sweeping.

    Two sweeps are used in production (the paper's choice).  Going to six sweeps
    must move the answer by well under 1e-5 kg/kg on a single step, i.e. the
    splitting error is negligible compared with the O(dt) backward-Euler error.
    This is the evidence that ``picard=2`` is enough.
    """
    r, r_face, v = build_grid(n_cells=50)
    T = _field(50, 50.0)
    C = np.linspace(cfg.C0, 1.0, 51)
    c2 = step_moisture(C, T, r_face, v, 60.0, 0.5, prop=APP3, picard=2)
    c6 = step_moisture(C, T, r_face, v, 60.0, 0.5, prop=APP3, picard=6)
    assert np.max(np.abs(c2 - c6)) < 1e-5


def test_picard_sweeps_are_monotone_in_accuracy():
    """More sweeps must not wander; the sequence must settle down."""
    r, r_face, v = build_grid(n_cells=50)
    T = _field(50, 50.0)
    C = np.linspace(cfg.C0, 1.0, 51)
    prev = step_moisture(C, T, r_face, v, 60.0, 0.5, prop=APP3, picard=2)
    for sweeps in (3, 4, 6):
        cur = step_moisture(C, T, r_face, v, 60.0, 0.5, prop=APP3,
                            picard=sweeps)
        assert np.max(np.abs(cur - prev)) < 1e-6
        prev = cur


def test_latent_heat_sink_cools_the_surface():
    """Turning on the evaporative sink must lower the surface temperature.

    The sink is ``L * rho_d * k_m * (C_s - C_air)`` at the surface node.  For
    the problem-1 data at C0 = 2.55 that is ~858 W/m^2, twice the concurrent
    sensible flux (h * (T_air - T_s) ~ 425 W/m^2), so the surface must drop by
    several degrees before the two balance.

    The paper deliberately leaves this term *off* (``use_latent=False``); this
    test records what turning it on would do.
    """
    r, r_face, v = build_grid(n_cells=50)
    T = _field(50, cfg.T0)
    C = _field(50, cfg.C0)
    dry = step_temperature(T, C, r_face, v, 60.0, 50.0, 0.5, prop=APP2,
                           use_latent=False)
    wet = step_temperature(T, C, r_face, v, 60.0, 50.0, 0.5, prop=APP2,
                           use_latent=True)
    assert wet[-1] < dry[-1] - 1.0          # at least a degree of cooling
    # backward Euler couples the nodes implicitly, so the whole profile shifts
    assert np.all(wet < dry)


def test_latent_heat_effect_is_grid_independent():
    """The cooling must not be an artefact of the thin outermost cell."""
    deltas = []
    for n_cells in (50, 100, 200):
        r, r_face, v = build_grid(n_cells=n_cells)
        T = _field(n_cells, cfg.T0)
        C = _field(n_cells, cfg.C0)
        d = step_temperature(T, C, r_face, v, 60.0, 50.0, 0.5, prop=APP2)
        w = step_temperature(T, C, r_face, v, 60.0, 50.0, 0.5, prop=APP2,
                             use_latent=True)
        deltas.append(w[-1] - d[-1])
    assert np.ptp(deltas) < 0.02
    assert np.all(np.asarray(deltas) < -6.0)


def test_latent_heat_is_proportional_to_the_surface_moisture():
    """Driving with drier air (C_air -> 0) must increase the sink."""
    r, r_face, v = build_grid(n_cells=50)
    T = _field(50, cfg.T0)
    C = _field(50, cfg.C0)
    mild = step_temperature(T, C, r_face, v, 60.0, 50.0, 1.0, prop=APP2,
                            use_latent=True)
    harsh = step_temperature(T, C, r_face, v, 60.0, 50.0, 0.0, prop=APP2,
                             use_latent=True)
    assert harsh[-1] < mild[-1]


def test_grid_refinement_barely_changes_one_step():
    """One backward-Euler step on 50 vs 200 cells must agree closely."""
    _r1, rf1, v1 = build_grid(n_cells=50)
    _r2, rf2, v2 = build_grid(n_cells=200)
    T50 = _field(50, cfg.T0)
    T200 = _field(200, cfg.T0)
    C50 = _field(50, cfg.C0)
    C200 = _field(200, cfg.C0)

    a = step_temperature(T50, C50, rf1, v1, 60.0, 50.0, 0.5, prop=APP2)
    b = step_temperature(T200, C200, rf2, v2, 60.0, 50.0, 0.5, prop=APP2)
    assert a[-1] == pytest.approx(b[-1], abs=0.05)
