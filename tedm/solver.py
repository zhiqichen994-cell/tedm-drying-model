"""Time-integration driver for problems 1 and 2 (fixed domain, no shrinkage).

The spatial discretisation lives in :mod:`tedm.model`; this module only owns
the time loop, the output bookkeeping and the run configuration.  ``radius``
and ``n_cells`` are exposed so that the mesh/time-step studies in
:mod:`tedm.analysis.verification` and :mod:`tedm.analysis.convergence` can
drive the same code path as production runs.
"""

from __future__ import annotations

import numpy as np

import tedm.config as cfg
from tedm.data_loader import load_air_conditions
from tedm.model import build_grid, step_moisture, step_temperature
from tedm.properties import APP2


def solve(prop=APP2, t_end=None, dt=None, use_latent=False, picard=2,
          n_cells=None, radius=None):
    """March the axisymmetric heat/moisture model forward in time.

    Parameters
    ----------
    prop : PropertySet
        property set, defaults to the constant-property set from 附录2
    t_end : float, optional
        final time in seconds, defaults to ``config.T_END`` (1800 s)
    dt : float, optional
        time step in seconds, defaults to ``config.DT`` (1 s)
    use_latent : bool
        add the evaporative latent-heat sink at the surface
    picard : int
        number of Picard sweeps for the lagged moisture diffusivity
        (``2`` = production, ``6`` = near-fully-implicit, used to bound the
        splitting error)
    n_cells : int, optional
        number of radial cells, defaults to ``config.N_R`` (100)
    radius : float, optional
        cylinder radius in metres, defaults to ``config.R`` (0.02 m)

    Returns
    -------
    t : ndarray, shape (n_steps + 1,)
        time grid in seconds
    r : ndarray, shape (n_cells + 1,)
        radial node positions
    T_result : ndarray, shape (n_steps + 1, n_cells + 1)
        temperature history in degrees Celsius
    C_result : ndarray, shape (n_steps + 1, n_cells + 1)
        dry-basis moisture history in kg/kg
    T_air, C_air : ndarray, shape (n_steps + 1,)
        imposed drying-chamber conditions
    """
    R = cfg.R if radius is None else float(radius)
    N = cfg.N_R if n_cells is None else int(n_cells)

    t_end = cfg.T_END if t_end is None else float(t_end)
    dt = cfg.DT if dt is None else float(dt)
    n_steps = int(round(t_end / dt))

    t, T_air, C_air = load_air_conditions(t_end, dt)
    r, r_face, v = build_grid(radius=R, n_cells=N)

    T = np.full(N + 1, cfg.T0)
    C = np.full(N + 1, cfg.C0)

    T_result = np.zeros((n_steps + 1, N + 1))
    C_result = np.zeros((n_steps + 1, N + 1))
    T_result[0] = T
    C_result[0] = C

    for n in range(n_steps):
        T = step_temperature(T, C, r_face, v, dt,
                             T_air[n + 1], C_air[n + 1],
                             prop=prop, use_latent=use_latent, radius=R)
        C = step_moisture(C, T, r_face, v, dt,
                          C_air[n + 1], prop=prop, picard=picard, radius=R)
        T_result[n + 1] = T
        C_result[n + 1] = C

    return t, r, T_result, C_result, T_air, C_air
