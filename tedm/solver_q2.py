"""Problem 2 shortcut: variable properties (附录3) over 0-3 h.

This is a thin wrapper around :func:`tedm.solver.solve`; it exists so that the
"which property set belongs to which problem" mapping is encoded in one place
instead of being repeated in every script and notebook.
"""

from __future__ import annotations

from tedm.properties import APP3
from tedm.solver import solve

T_END_Q2 = 3.0 * 3600.0   # 3 h = 10800 s


def solve_q2(prop=APP3, use_latent=False, picard=2, dt=1.0, n_cells=None,
             radius=None, t_end=T_END_Q2):
    """Problem 2 entry point: 附录3 variable properties, 0-3 h."""
    return solve(prop=prop, t_end=t_end, dt=dt, use_latent=use_latent,
                 picard=picard, n_cells=n_cells, radius=radius)
