"""Code verification against the closed-form Bessel series.

These tests are the scientific core of the repository: they show that the
finite-volume solver converges to the *exact* solution of the linearised
problem, and they report the observed order of accuracy.

They are marked ``slow`` (a full run is a few tens of seconds) so that the
fast regression suite stays usable.  Run them with::

    pytest -m slow
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import j0, j1, jn_zeros

from tedm.analysis.verification import (
    CONST_D,
    SAMPLE_TIMES,
    bessel_eigenvalues,
    bessel_series,
    compare_linear,
    constant_diffusion,
    spatial_order,
    temporal_order,
    verify_temperature,
)


# ----------------------------------------------------------------------
# the analytical building blocks themselves
# ----------------------------------------------------------------------
def test_eigenvalues_satisfy_the_robin_condition():
    """lam * J1(lam) = Bi * J0(lam) must hold at every returned root."""
    bi = 8e-7 * 0.02 / CONST_D
    lam = bessel_eigenvalues(bi, 20)
    assert lam.size >= 15
    assert np.all(np.diff(lam) > 0)          # ordered
    assert np.all(lam > 0)
    residual = lam * j1(lam) - bi * j0(lam)
    assert np.all(np.abs(residual) < 1e-10)


def test_eigenvalues_are_bracketed_by_the_zeros_of_J0():
    """Root k lies strictly between the k-th and (k+1)-th zero of J0.

    ``f(lam) = lam J1(lam) - Bi J0(lam)`` changes sign exactly once in each
    such interval, which is why the bracketing strategy in
    :func:`bessel_eigenvalues` is sound.  With ``edges = [0] + zeros(J0)`` the
    statement is ``edges[k] < lam[k] < edges[k+1]``.
    """
    bi = 5.0
    z = jn_zeros(0, 15)
    edges = np.concatenate(([0.0], z))
    lam = bessel_eigenvalues(bi, 12)
    assert lam.size >= 12
    for k in range(12):
        assert edges[k] < lam[k] < edges[k + 1]


def test_bessel_series_starts_at_the_initial_value():
    """At t = 0 the series must reproduce C0.

    The series converges slowly *at the boundary* (like a Fourier series at a
    jump), so the tolerance is an order of magnitude looser than the interior
    accuracy.  This is a property of the reference solution, not of the solver.
    """
    r = np.linspace(0.0, 0.02, 41)
    val = bessel_series(r, [0.0], c0=2.55, cb=0.0).ravel()
    assert np.allclose(val, 2.55, atol=3e-3)
    # away from the surface it is much tighter
    assert np.allclose(val[:-1], 2.55, atol=1e-3)



def test_bessel_series_tends_to_the_ambient_value():
    """As t -> infinity the profile must relax to Cb."""
    r = np.linspace(0.0, 0.02, 11)
    val = bessel_series(r, [1e7], c0=2.55, cb=0.0).ravel()
    assert np.allclose(val, 0.0, atol=1e-8)


def test_bessel_series_is_symmetric_about_the_axis():
    """J0 is even, so C(+r) == C(-r) - a cheap check on the basis construction."""
    t = [600.0]
    pos = bessel_series(np.array([0.01]), t, c0=2.55).ravel()
    neg = bessel_series(np.array([-0.01]), t, c0=2.55).ravel()
    assert pos[0] == pytest.approx(neg[0], rel=1e-12)


def test_bessel_series_returns_the_documented_shape():
    r = np.linspace(0.0, 0.02, 7)
    t = np.array([100.0, 200.0, 300.0])
    assert bessel_series(r, t).shape == (7, 3)


def test_bessel_series_is_monotone_in_time_for_a_drying_body():
    """The series must decay monotonically once the transient is over.

    The first samples are ignored because the truncated series oscillates at
    the level of its own truncation error (~1e-4) near t = 0.
    """
    r = np.array([0.0])
    t = np.linspace(0.0, 5000.0, 30)
    val = bessel_series(r, t, c0=2.55, cb=0.0).ravel()
    assert val[0] > val[-1]
    assert np.all(np.diff(val[3:]) < 0)



def test_constant_diffusion_property_set_is_really_constant():
    ps = constant_diffusion()
    C = np.linspace(0.1, 2.55, 20)
    d = ps.D(C)
    assert np.allclose(d, CONST_D, rtol=1e-9)
    assert ps.arrhenius is False


def test_constant_diffusion_value():
    assert CONST_D == pytest.approx(7e-9 * np.exp(-0.89 / 2.55), rel=1e-15)


# ----------------------------------------------------------------------
# the solver vs. the series
# ----------------------------------------------------------------------
@pytest.mark.slow
def test_solver_matches_the_analytical_solution_moisture():
    """Relative L2 error on the default grid must be small (paper: 5.7e-5)."""
    res = compare_linear()
    assert res["l2_relative"] < 5e-4
    assert res["linf"] < 5e-3


@pytest.mark.slow
def test_solver_matches_the_analytical_solution_temperature():
    """Relative L2 error for the energy equation (paper: 1.0e-4)."""
    res = verify_temperature()
    assert res["l2_relative"] < 5e-4
    assert res["linf"] < 2e-2


@pytest.mark.slow
def test_spatial_refinement_converges():
    """Halving the cell size must reduce the error at roughly 2nd order."""
    res = spatial_order(dt=0.5, cells=(25, 50, 100))
    errs = np.asarray(res["errors"])
    orders = np.asarray(res["order"])
    assert np.all(np.diff(errs) < 0)               # monotone improvement
    assert np.all(orders > 1.0)                    # clearly better than 1st order
    assert orders[0] > 1.5                         # clean 2nd order on the coarse pair


@pytest.mark.slow
def test_temporal_refinement_converges_at_first_order():
    """Backward Euler is O(dt); the observed order must be close to 1."""
    res = temporal_order(n_cells=400, dts=(8.0, 4.0, 2.0))
    errs = np.asarray(res["errors"])
    orders = np.asarray(res["order"])
    assert np.all(np.diff(errs) < 0)
    assert np.all(np.abs(orders - 1.0) < 0.25)


@pytest.mark.slow
def test_sample_times_are_multiples_of_the_default_step():
    dt = 1.0
    idx = np.rint(np.asarray(SAMPLE_TIMES) / dt)
    assert np.allclose(idx * dt, SAMPLE_TIMES)


@pytest.mark.slow
def test_marching_with_a_bad_dt_raises():
    with pytest.raises(ValueError, match="multiples of dt"):
        compare_linear(dt=7.0)
