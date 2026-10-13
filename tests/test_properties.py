"""Unit tests for the three property sets (附录2 / 附录3 / 附录4).

The point of these tests is to pin down the *physical* behaviour the model
relies on: D must decrease with C, the Kirchhoff integral must invert
correctly, and the Arrhenius factor must be temperature-increasing.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import exp1

from tedm.properties import APP2, APP3, APP4

ALL_SETS = [APP2, APP3, APP4]


@pytest.mark.parametrize("ps", ALL_SETS, ids=lambda p: p.name)
def test_rho_dry_is_consistent(ps):
    """rho_d = rho / (1 + C) by definition."""
    C = np.array([0.0, 0.5, 1.5, 2.55])
    assert np.allclose(ps.rho_dry(C), ps.rho(C) / (1.0 + C))


@pytest.mark.parametrize("ps", ALL_SETS, ids=lambda p: p.name)
def test_rho_increases_with_C(ps):
    C = np.linspace(0.05, 2.5, 40)
    if ps.rho_b > 0:
        assert np.all(np.diff(ps.rho(C)) > 0)
    else:
        assert np.allclose(ps.rho(C), ps.rho_a)


@pytest.mark.parametrize("ps", ALL_SETS, ids=lambda p: p.name)
def test_cp_and_k_are_bounded_and_monotone(ps):
    """cp = cp_a + cp_b * C/(C+1) is bounded by cp_a and cp_a + cp_b."""
    C = np.linspace(0.05, 2.5, 40)
    cp = ps.cp(C)
    k = ps.k(C)
    assert np.all(cp >= ps.cp_a - 1e-12) and np.all(cp <= ps.cp_a + ps.cp_b + 1e-12)
    assert np.all(k >= ps.k_a - 1e-12) and np.all(k <= ps.k_a + ps.k_b + 1e-12)
    if ps.cp_b > 0:
        assert np.all(np.diff(cp) > 0)
    if ps.k_b > 0:
        assert np.all(np.diff(k) > 0)


@pytest.mark.parametrize("ps", ALL_SETS, ids=lambda p: p.name)
def test_D_conc_increases_with_C(ps):
    """D_conc = A exp(-B/C) is strictly *increasing* in C.

    This is easy to get backwards.  With B > 0, ``exp(-B/C)`` grows as C grows,
    so a wetter material diffuses faster.  At C = 2.55 the 附录2 value is
    4.94e-9 m^2/s; at C = 0.05 it has collapsed to 1.3e-16 m^2/s.  That is what
    creates the sharp drying front in the profiles: as a cell dries out, its
    own diffusivity collapses and it stops feeding the surface.
    """
    C = np.linspace(0.05, 2.55, 200)
    d = ps.D_conc(C)
    assert np.all(np.diff(d) > 0)
    assert np.all(d > 0)


def test_APP2_D_conc_spans_many_orders_of_magnitude():
    """Sanity: the moisture-dependence is extreme (a 'falling rate' material).

    ``D(2.55) / D(0.15)`` is only ~266, but once the material is nearly dry the
    collapse is dramatic: ``D(2.55) / D(0.05) ~ 3.8e7``.
    """
    d_wet = float(APP2.D_conc(2.55))
    d_target = float(APP2.D_conc(0.15))
    d_very_dry = float(APP2.D_conc(0.05))
    assert d_wet / d_target == pytest.approx(266, rel=0.05)
    assert d_wet / d_very_dry > 1e7
    assert d_very_dry < 1e-15


def test_APP2_matches_config_constants():
    """附录2 must equal the constants in tedm.config (problem 1)."""
    import tedm.config as cfg

    assert APP2.rho_a == cfg.RHO
    assert APP2.cp_a == cfg.CP
    assert APP2.k_a == cfg.K_COND
    assert APP2.D_A == cfg.D0
    assert APP2.D_B == cfg.B_D
    assert APP2.arrhenius is False


def test_APP2_D_value_at_initial_moisture():
    """D(2.55) = 7e-9 * exp(-0.89/2.55) = 4.9377e-9 m^2/s."""
    expected = 7e-9 * np.exp(-0.89 / 2.55)
    assert APP2.D(2.55) == pytest.approx(expected, rel=1e-14)
    assert expected == pytest.approx(4.9377e-9, rel=1e-4)


@pytest.mark.parametrize("ps", [APP3, APP4], ids=lambda p: p.name)
def test_arrhenius_D_increases_with_temperature(ps):
    """Raising the temperature must raise D (Arrhenius, E > 0)."""
    C = 1.5
    d_low = ps.D(C, 30.0)
    d_high = ps.D(C, 70.0)
    assert d_high > d_low


@pytest.mark.parametrize("ps", [APP3, APP4], ids=lambda p: p.name)
def test_arrhenius_requires_temperature(ps):
    with pytest.raises(ValueError, match="需要 T"):
        ps.D(1.5)


@pytest.mark.parametrize("ps", ALL_SETS, ids=lambda p: p.name)
def test_Phi_is_the_antiderivative_of_D_conc(ps):
    """dPhi/dC must equal D_conc(C) (checked by central differences)."""
    C = np.array([0.4, 1.0, 2.0])
    h = 1e-7
    numeric = (ps.Phi(C + h) - ps.Phi(C - h)) / (2 * h)
    assert np.allclose(numeric, ps.D_conc(C), rtol=1e-6)


@pytest.mark.parametrize("ps", ALL_SETS, ids=lambda p: p.name)
def test_Phi_closed_form(ps):
    """Phi(C) = A * (C exp(-B/C) - B E1(B/C)) from the module docstring."""
    C = np.array([0.6, 1.3, 2.4])
    A, B = ps.D_A, ps.D_B
    expected = A * (C * np.exp(-B / C) - B * exp1(B / C))
    assert np.allclose(ps.Phi(C), expected, rtol=1e-12)


@pytest.mark.parametrize("ps", ALL_SETS, ids=lambda p: p.name)
def test_face_D_reduces_to_D_conc_when_Ci_equals_Cj(ps):
    """In the degenerate limit the integral average must fall back to D(C)."""
    C = np.array([0.5, 1.5, 2.4])
    Ti = np.array([40.0, 40.0, 40.0])
    Tj = np.array([40.0, 40.0, 40.0])
    got = ps.face_D(C, C, Ti, Tj)
    arr = np.exp(-ps.E_arr / (40.0 + 273.15)) if ps.arrhenius else 1.0
    assert np.allclose(got, ps.D_conc(C) * arr, rtol=1e-9)


@pytest.mark.parametrize("ps", ALL_SETS, ids=lambda p: p.name)
def test_face_D_is_the_integral_mean(ps):
    """(Phi(Ci) - Phi(Cj)) / (Ci - Cj) is what the FV face flux needs."""
    Ci = np.array([2.55, 1.2])
    Cj = np.array([1.0, 0.2])
    Ti = np.array([35.0, 55.0])
    Tj = np.array([40.0, 60.0])

    conc = (ps.Phi(Ci) - ps.Phi(Cj)) / (Ci - Cj)
    if ps.arrhenius:
        arr = 0.5 * (np.exp(-ps.E_arr / (Ti + 273.15))
                     + np.exp(-ps.E_arr / (Tj + 273.15)))
    else:
        arr = 1.0
    assert np.allclose(ps.face_D(Ci, Cj, Ti, Tj), conc * arr, rtol=1e-12)


@pytest.mark.parametrize("ps", [APP3, APP4], ids=lambda p: p.name)
def test_face_D_requires_temperature_when_arrhenius(ps):
    with pytest.raises(ValueError, match="需要 T"):
        ps.face_D(np.array([1.0]), np.array([0.5]))


def test_appendix_4_is_less_conductive_and_less_diffusive_than_3():
    """附录4 (shrinkage problem) uses a different herb with lower k and D."""
    C = 1.5
    assert APP4.k(C) < APP3.k(C)
    assert APP4.D(C, 50.0) < APP3.D(C, 50.0)
