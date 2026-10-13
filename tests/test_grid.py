"""Consistency checks for the finite-volume grid and the linear solver.

These are cheap and catch the two mistakes that are easiest to make and
hardest to notice: a wrong cell metric (volume weights) and a wrong
tridiagonal layout.
"""

from __future__ import annotations

import numpy as np
import pytest

import tedm.config as cfg
from tedm.model import build_grid, solve_tridiag


@pytest.mark.parametrize("n_cells", [5, 25, 100, 200])
@pytest.mark.parametrize("radius", [0.02, 0.01])
def test_volume_weights_sum_to_radius_squared(n_cells, radius):
    """sum(v) must equal R^2 so that v doubles as the cell area integral."""
    _r, _r_face, v = build_grid(radius=radius, n_cells=n_cells)
    assert v.sum() == pytest.approx(radius ** 2, rel=1e-12)


@pytest.mark.parametrize("n_cells", [5, 25, 100])
def test_volume_weights_sum_to_pi_R2_with_pi(n_cells):
    """Multiplying by pi recovers the true cross-sectional area."""
    _r, _r_face, v = build_grid(n_cells=n_cells)
    assert np.pi * v.sum() == pytest.approx(np.pi * cfg.R ** 2, rel=1e-12)


@pytest.mark.parametrize("n_cells", [5, 25, 100])
def test_grid_shapes_and_monotonicity(n_cells):
    r, r_face, v = build_grid(n_cells=n_cells)
    assert r.shape == (n_cells + 1,)
    assert r_face.shape == (n_cells,)
    assert v.shape == (n_cells + 1,)
    assert np.all(np.diff(r) > 0)
    assert np.all(v > 0)
    assert r[0] == 0.0
    assert r[-1] == pytest.approx(cfg.R)


def test_face_radii_are_cell_centres():
    """Face radius i must be the midpoint of node i and node i+1."""
    r, r_face, _v = build_grid(n_cells=20)
    assert np.allclose(r_face, 0.5 * (r[:-1] + r[1:]))


def test_centre_cell_volume_matches_half_cell_area():
    """The axis cell is a disc of radius dr/2, i.e. area (dr/2)^2."""
    R, N = cfg.R, 100
    _r, _r_face, v = build_grid(n_cells=N)
    dr = R / N
    assert v[0] == pytest.approx((dr / 2.0) ** 2, rel=1e-12)


def test_solve_tridiag_matches_dense_solve():
    rng = np.random.default_rng(0)
    n = 30
    lower = rng.random(n)
    diag = rng.random(n) + 5.0
    upper = rng.random(n)
    rhs = rng.random(n)

    dense = np.diag(diag)
    dense += np.diag(upper[:-1], 1)
    dense += np.diag(lower[1:], -1)

    x_banded = solve_tridiag(lower, diag, upper, rhs)
    x_dense = np.linalg.solve(dense, rhs)
    assert np.allclose(x_banded, x_dense, rtol=1e-12, atol=1e-12)


def test_solve_tridiag_legacy_alias():
    from tedm.model import _solve_tridiag

    assert _solve_tridiag is solve_tridiag
