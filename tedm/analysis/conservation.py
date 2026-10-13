"""Discrete conservation diagnostics for the finite-volume schemes.

Why this module exists
----------------------
A finite-volume scheme is *supposed* to conserve mass and energy exactly, but
that is only true when the diagnostic uses the same metric as the scheme.  The
first version of the check used two different metrics for the two sides of the
budget (surface area without the ``2*pi`` factor on one side, storage with it on
the other) and it also summed one extra time level.  Both mistakes are fixed
here, so the residual now closes to round-off for the fixed-domain solver.

Metric used by the solver
-------------------------
``build_grid`` returns cell weights ``v`` with ``v.sum() == R**2``, i.e. the
volume *divided by* ``pi`` and by the cylinder length.  Multiplying everything
by ``pi`` recovers true areas and volumes, which is what the budgets below do.

For the fixed-domain scheme the tridiagonal matrix has row sums

    row 0      : v[0]
    row 1..N-1 : v[i]
    row N      : v[N] + dt * 2 * R * h        (energy)
                 v[N] + dt * 2 * R * k_m      (moisture)

so summing all rows telescopes every interior face and leaves only the storage
change and the surface exchange.  The identity

    sum_i v_i * (T_i^{n+1} - T_i^n) = dt * 2*R*h * (T_air[n+1] - T_N^{n+1})

is therefore satisfied *by construction* at every step; accumulating it over
all steps gives the closed budget.
"""

from __future__ import annotations

import numpy as np

from tedm.config import DT, H_CONV, K_MASS, R, RHO, CP
from tedm.model import build_grid
from tedm.data_loader import load_air_conditions, load_radius
from tedm.properties import APP2, APP4


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def cell_volumes(r_face, radius: float):
    """True cell volumes per unit length (m^2), i.e. ``pi * v``.

    ``v`` as produced by :func:`tedm.model.build_grid` is ``volume / pi``.
    """
    n = r_face.size + 1
    v = np.empty(n)
    dr = radius / (n - 1)
    v[0] = (dr / 2.0) ** 2
    v[1:n - 1] = r_face[1:] ** 2 - r_face[:-1] ** 2
    v[n - 1] = radius ** 2 - r_face[-1] ** 2
    return np.pi * v, v


def volume_average(r_face, field, radius: float):
    """Cylinder-volume weighted radial average."""
    vol, _ = cell_volumes(r_face, radius)
    return float(np.sum(vol * field) / np.sum(vol))


def surface_exchange_series(t, t_air_or_c, surface_values, coeff, radius=R):
    """Accumulated surface exchange ``sum_n dt * coeff * 2*pi*R * (X_air - X_s)``.

    ``t`` is the solver time grid; the solver uses the *new* time level of the
    driving signal, hence the ``[1:]`` slices.
    """
    dt = t[1] - t[0]
    return float(np.sum(coeff * 2.0 * np.pi * radius * (t_air_or_c[1:] - surface_values[1:]) * dt))


# ----------------------------------------------------------------------
# fixed domain (problems 1-3)
# ----------------------------------------------------------------------
def energy_budget(t, r, T_result, T_air, prop=APP2, radius=R, h=H_CONV):
    """Internal energy change vs. accumulated surface heat flow."""
    r_face = (np.arange(T_result.shape[1] - 1) + 0.5) * (radius / (T_result.shape[1] - 1))
    vol, _ = cell_volumes(r_face, radius)
    rho_cp = prop.rho(T_result) * prop.cp(T_result)

    d_storage = float(np.sum(rho_cp[-1] * vol * T_result[-1])
                      - np.sum(rho_cp[0] * vol * T_result[0]))
    q_in = surface_exchange_series(t, T_air, T_result[:, -1], h, radius)
    scale = max(abs(q_in), 1e-30)
    return {
        "d_storage": d_storage,
        "q_exchanged": q_in,
        "abs_residual": d_storage - q_in,
        "rel_residual": abs(d_storage - q_in) / scale,
    }


def water_budget(t, r, C_result, C_air, radius=R, km=K_MASS):
    """Internal moisture change vs. accumulated surface mass transfer.

    Both sides follow the "gain" convention: ``d_storage`` is the net storage
    gain and ``m_exchanged`` is the net inflow through the surface.  Since
    ``C_air < C_surface`` here the flow is outward and both quantities are
    negative.
    """
    r_face = (np.arange(C_result.shape[1] - 1) + 0.5) * (radius / (C_result.shape[1] - 1))
    vol, _ = cell_volumes(r_face, radius)

    d_storage = float(np.sum(vol * C_result[-1]) - np.sum(vol * C_result[0]))
    m_out = surface_exchange_series(t, C_air, C_result[:, -1], km, radius)
    scale = max(abs(m_out), 1e-30)
    return {
        "d_storage": d_storage,
        "m_exchanged": m_out,
        "abs_residual": d_storage - m_out,
        "rel_residual": abs(d_storage - m_out) / scale,
    }


def bound_check(C_result, C_air, C0):
    """Discrete maximum principle: concentration stays inside the data range."""
    lower = float(C_result.min())
    upper = float(C_result.max())
    floor = min(float(np.min(C_air)), 0.0)
    return {
        "min": lower,
        "max": upper,
        "below_initial": upper <= C0 + 1e-9,
        "above_air": lower >= floor - 1e-9,
        "monotone_surface": bool(np.all(np.diff(C_result[:, -1]) <= 1e-12)),
    }


def report_fixed_domain(t, r, T_result, C_result, T_air, C_air, label="problem 1"):
    e = energy_budget(t, r, T_result, T_air)
    m = water_budget(t, r, C_result, C_air)
    b = bound_check(C_result, C_air, float(C_result[0, 0]))
    lines = [
        f"[{label}] conservation diagnostics",
        "-" * 52,
        f"  energy   : d_storage = {e['d_storage']:+.6e} J/m",
        f"             exchanged = {e['q_exchanged']:+.6e} J/m",
        f"             rel.resid = {e['rel_residual']:.3e}",
        f"  moisture : d_storage = {m['d_storage']:+.6e}",
        f"             exchanged = {m['m_exchanged']:+.6e}",
        f"             rel.resid = {m['rel_residual']:.3e}",
        f"  bounds   : C in [{b['min']:.4f}, {b['max']:.4f}]",
        f"             max never exceeds initial : {b['below_initial']}",
        f"             surface monotonically dry : {b['monotone_surface']}",
    ]
    return "\n".join(lines), {"energy": e, "water": m, "bounds": b}


# ----------------------------------------------------------------------
# moving boundary (problem 4)  -- exposes a genuine modelling limitation
# ----------------------------------------------------------------------
def dry_mass_history(N=100, dt=60.0, C_target=0.15, t_max_h=96.0, prop=APP4):
    """Track dry matter, water and energy for the shrinking-domain solver.

    Returns the histories so that the *non-conservation of dry matter* can be
    quantified instead of silently assumed.  See ``docs/limitations.md``.
    """
    from tedm.solver_q4 import build_grid_xi, step_temperature_q4, step_moisture_q4
    from tedm.config import T0, C0

    t_max = t_max_h * 3600.0
    t_air, T_air, C_air = load_air_conditions(t_max, dt)
    _, _, R_func = load_radius()

    xi, xi_face, v = build_grid_xi(N)
    T = np.full(N + 1, T0)
    C = np.full(N + 1, C0)

    hist = {"t": [], "R": [], "M_dry": [], "M_water": [], "E": [], "M_dry_prescribed": []}

    def flatten(Cc, Tc, R_m):
        rho_d = prop.rho(Cc) / (1.0 + Cc)            # dry-basis bulk density
        rho_cp = prop.rho(Cc) * prop.cp(Cc)
        vol = np.pi * v * R_m ** 2                   # true volume per unit length
        return (float(np.sum(rho_d * vol)),
                float(np.sum(rho_d * Cc * vol)),
                float(np.sum(rho_cp * Tc * vol)))

    n_max = int(round(t_max / dt))
    m_dry_0, m_water_0, e_0 = flatten(C, T, R)
    hist["t"].append(0.0); hist["R"].append(R)
    hist["M_dry"].append(m_dry_0); hist["M_water"].append(m_water_0); hist["E"].append(e_0)
    # dry density implied by the uniform proportional shrinkage used in the paper
    hist["M_dry_prescribed"].append(m_dry_0)

    for n in range(n_max):
        t_new = (n + 1) * dt
        R_m = float(R_func(t_new)) / 100.0
        T = step_temperature_q4(T, C, xi_face, v, R_m, dt, T_air[n + 1], prop=prop)
        C = step_moisture_q4(C, T, xi_face, v, R_m, dt, C_air[n + 1], prop=prop)
        md, mw, e = flatten(C, T, R_m)
        hist["t"].append(t_new); hist["R"].append(R_m)
        hist["M_dry"].append(md); hist["M_water"].append(mw); hist["E"].append(e)
        hist["M_dry_prescribed"].append(m_dry_0)
        if C[0] < C_target:
            break

    for k in hist:
        hist[k] = np.asarray(hist[k])
    hist["R0"] = R
    return hist


def report_moving_boundary(hist=None):
    hist = hist if hist is not None else dry_mass_history()
    md = hist["M_dry"]
    drift = (md[-1] - md[0]) / md[0]
    rho_d0 = md[0] / (np.pi * hist["R0"] ** 2)
    rho_d_end = md[-1] / (np.pi * hist["R"][-1] ** 2)
    lines = [
        "[problem 4] moving-boundary diagnostics",
        "-" * 52,
        f"  steps                      : {md.size - 1}",
        f"  R                          : {hist['R'][0]*100:.4f} cm -> {hist['R'][-1]*100:.4f} cm",
        f"  M_dry(0)                   : {md[0]:.6e} kg/m",
        f"  M_dry(end)                 : {md[-1]:.6e} kg/m",
        f"  dry-matter drift           : {drift:+.3%}   <-- should be 0",
        f"  rho_d(0) from state        : {rho_d0:.2f} kg/m^3",
        f"  rho_d(end) from state      : {rho_d_end:.2f} kg/m^3",
        f"  rho_d implied by eq.(13)   : {rho_d0 * (hist['R0']/hist['R'][-1])**2:.2f} kg/m^3",
    ]
    return "\n".join(lines), {"dry_mass_drift": float(drift)}


# ----------------------------------------------------------------------
# CLI entry
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from tedm.solver import solve

    t, r, T_r, C_r, T_a, C_a = solve()
    txt, _ = report_fixed_domain(t, r, T_r, C_r, T_a, C_a, "problem 1")
    print(txt)
    print()
    txt, _ = report_moving_boundary()
    print(txt)
