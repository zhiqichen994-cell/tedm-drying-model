"""Finite-volume discretisation of the axisymmetric heat / moisture problem.

Semi-discrete form (per unit length, the factor ``pi`` is dropped everywhere and
restored only when a physical budget is needed):

    rho*cp * V_i * dT_i/dt = A_{i-1/2} q_{i-1/2} - A_{i+1/2} q_{i+1/2}
    V_i * dC_i/dt          = A_{i-1/2} g_{i-1/2} - A_{i+1/2} g_{i+1/2}

with ``q = -k dT/dr``, ``g = -D dC/dr``, cell metric ``V_i = int r dr`` and face
area ``A = 2r``.  Time integration is backward Euler; the non-linear moisture
coefficient is lagged through a Picard sweep.  All interior face fluxes cancel
row-by-row, which is what makes the conservation diagnostics in
:mod:`tedm.analysis.conservation` exact.

Both ``radius`` and ``n_cells`` can be overridden per call, which is what the
mesh-refinement study in :mod:`tedm.analysis.verification` relies on.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

import tedm.config as cfg
from tedm.properties import APP2

# latent heat of vaporisation of water (J/kg); only used when use_latent=True
L_VAP = 2.26e6


# ----------------------------------------------------------------------
# grid
# ----------------------------------------------------------------------
def build_grid(radius=None, n_cells=None):
    """Cell centres, face radii and volume weights.

    ``v.sum() == radius**2``; multiply by ``pi`` for true volumes per unit
    length.
    """
    R = cfg.R if radius is None else float(radius)
    N = cfg.N_R if n_cells is None else int(n_cells)

    r = np.linspace(0.0, R, N + 1)
    r_face = (np.arange(N) + 0.5) * (R / N)
    v = np.empty(N + 1)
    v[0] = (R / N / 2.0) ** 2
    v[1:N] = r_face[1:] ** 2 - r_face[:-1] ** 2
    v[N] = R ** 2 - r_face[-1] ** 2
    return r, r_face, v


def solve_tridiag(lower, diag, upper, rhs):
    """Solve a tridiagonal system stored in three diagonals."""
    n = diag.size
    ab = np.zeros((3, n))
    ab[0, 1:] = upper[:n - 1]
    ab[1, :] = diag
    ab[2, :n - 1] = lower[1:n]
    return solve_banded((1, 1), ab, rhs)


# kept for backwards compatibility with the original scripts
_solve_tridiag = solve_tridiag


# ----------------------------------------------------------------------
# temperature step
# ----------------------------------------------------------------------
def step_temperature(T, C, r_face, v, dt, T_air, C_air,
                     prop=APP2, use_latent=False, L=L_VAP,
                     km=None, h=None, radius=None):
    """One backward-Euler step of the temperature field.

    Parameters
    ----------
    T, C : ndarray
        current temperature and dry-basis moisture content on the cell centres
    r_face : ndarray
        face radii
    v : ndarray
        volume weights from :func:`build_grid`
    dt : float
        time step
    T_air, C_air : float
        driving conditions at the *new* time level
    use_latent : bool
        add an evaporative sink ``L * rho_d * km * (C_s - C_air)`` at the surface
    """
    R = cfg.R if radius is None else float(radius)
    hv = cfg.H_CONV if h is None else float(h)
    kv = cfg.K_MASS if km is None else float(km)

    N = T.size - 1
    dr = R / N

    rho_cp = prop.rho(C) * prop.cp(C)
    k_face = 0.5 * (prop.k(C[:-1]) + prop.k(C[1:]))
    Fk = 2.0 * r_face * k_face / dr

    lower = np.zeros(N + 1)
    diag = np.zeros(N + 1)
    upper = np.zeros(N + 1)
    rhs = np.zeros(N + 1)
    m = rho_cp * v

    # centre (symmetry: no radial flux)
    diag[0] = m[0] + dt * Fk[0]
    upper[0] = -dt * Fk[0]
    rhs[0] = m[0] * T[0]

    # interior
    lower[1:N] = -dt * Fk[:N - 1]
    diag[1:N] = m[1:N] + dt * (Fk[:N - 1] + Fk[1:N])
    upper[1:N] = -dt * Fk[1:N]
    rhs[1:N] = m[1:N] * T[1:N]

    # surface: Robin condition -k dT/dr = h (T_s - T_air).  Node N sits exactly
    # on r = R, so T[N] *is* the surface value and no reconstruction is needed.
    a_surf = 2.0 * R * hv
    lower[N] = -dt * Fk[N - 1]
    diag[N] = m[N] + dt * (Fk[N - 1] + a_surf)
    rhs[N] = m[N] * T[N] + dt * a_surf * T_air

    if use_latent:
        rho_d_s = prop.rho_dry(C[N])
        j = rho_d_s * kv * (C[N] - C_air)       # kg/(m^2 s)
        rhs[N] -= dt * (2.0 * R) * L * j

    return solve_tridiag(lower, diag, upper, rhs)


# ----------------------------------------------------------------------
# moisture step
# ----------------------------------------------------------------------
def step_moisture(C, T, r_face, v, dt, C_air, prop=APP2,
                  km=None, radius=None, picard=2):
    """One backward-Euler step of the moisture field (Picard-lagged D)."""
    R = cfg.R if radius is None else float(radius)
    kv = cfg.K_MASS if km is None else float(km)

    N = C.size - 1
    dr = R / N

    C_new = C.copy()
    for _ in range(picard):
        D_face = prop.face_D(C_new[:-1], C_new[1:], T[:-1], T[1:])
        Fd = 2.0 * r_face * D_face / dr

        a_surf = 2.0 * R * kv

        lower = np.zeros(N + 1)
        diag = np.zeros(N + 1)
        upper = np.zeros(N + 1)
        rhs = np.zeros(N + 1)

        diag[0] = v[0] + dt * Fd[0]
        upper[0] = -dt * Fd[0]
        rhs[0] = v[0] * C[0]

        lower[1:N] = -dt * Fd[:N - 1]
        diag[1:N] = v[1:N] + dt * (Fd[:N - 1] + Fd[1:N])
        upper[1:N] = -dt * Fd[1:N]
        rhs[1:N] = v[1:N] * C[1:N]

        # surface: -D dC/dr = k_m (C_s - C_air)
        lower[N] = -dt * Fd[N - 1]
        diag[N] = v[N] + dt * (Fd[N - 1] + a_surf)
        rhs[N] = v[N] * C[N] + dt * a_surf * C_air

        C_new = solve_tridiag(lower, diag, upper, rhs)
    return C_new
