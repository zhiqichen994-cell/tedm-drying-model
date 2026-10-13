"""Tests for the quantity-of-interest convergence study.

``convergence`` answers a different question from ``verification``: there is no
closed-form solution for problems 2-4, so the only available evidence is that
the quantity we actually report (drying time, end-point moisture) stops moving
as the discretisation is refined.
"""

from __future__ import annotations

import pytest

from tedm.analysis.convergence import (
    DEFAULT_MESH_CELLS,
    DEFAULT_STEPS,
    mesh_study,
    quantity,
    step_study,
)


def test_quantity_rejects_unknown_problems():
    with pytest.raises(ValueError, match="unknown problem"):
        quantity("q9")


def test_default_schedules_cover_every_problem():
    for p in ("q1", "q2", "q3", "q4"):
        assert p in DEFAULT_MESH_CELLS
        assert p in DEFAULT_STEPS
        assert len(DEFAULT_MESH_CELLS[p]) >= 3
        assert len(DEFAULT_STEPS[p]) >= 3


def test_default_mesh_schedules_are_increasing():
    for p, cells in DEFAULT_MESH_CELLS.items():
        assert list(cells) == sorted(cells), p
        assert len(set(cells)) == len(cells), p


def test_default_step_schedules_are_decreasing():
    for p, dts in DEFAULT_STEPS.items():
        assert list(dts) == sorted(dts, reverse=True), p


def test_q1_quantity_has_the_expected_keys():
    q = quantity("q1", n_cells=25, dt=10.0)
    assert set(q) == {"C_centre", "C_surface", "C_mean"}
    assert 0.0 < q["C_surface"] < q["C_mean"] < q["C_centre"] <= 2.55


def test_q1_quantity_is_grid_independent():
    """Two very different grids must give the same end-point moisture."""
    a = quantity("q1", n_cells=25, dt=10.0)
    b = quantity("q1", n_cells=100, dt=10.0)
    assert a["C_surface"] == pytest.approx(b["C_surface"], abs=3e-3)
    assert a["C_centre"] == pytest.approx(b["C_centre"], abs=3e-3)


@pytest.mark.slow
def test_q3_drying_time_mesh_convergence():
    """The reported 57.22 h must be stable to better than 0.1 % under mesh
    refinement - otherwise the headline number would be a discretisation
    artefact."""
    res = mesh_study("q3", cells=(25, 50, 100, 200))
    times = [res["results"][n]["t_end_h"] for n in res["cells"]]
    spread = max(times) - min(times)
    assert spread / (sum(times) / len(times)) < 1e-3


@pytest.mark.slow
def test_q4_drying_time_mesh_convergence():
    res = mesh_study("q4", cells=(50, 100, 200))
    times = [res["results"][n]["t_end_h"] for n in res["cells"]]
    spread = max(times) - min(times)
    assert spread / (sum(times) / len(times)) < 1e-3


@pytest.mark.slow
def test_q3_drying_time_is_finer_with_a_smaller_step():
    """Backward Euler is first-order, so refining dt must move the answer in a
    monotone direction (here: the coarse step overshoots the drying time)."""
    res = step_study("q3", dts=(240.0, 120.0, 60.0))
    times = [res["results"][d]["t_end_h"] for d in res["dts"]]
    assert times[0] > times[1] > times[2]
    assert (times[0] - times[2]) / times[2] < 0.01
