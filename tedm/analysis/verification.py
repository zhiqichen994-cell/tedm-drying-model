"""Independent verification against a closed-form solution.

The finite-volume solver is verified against the *exact* analytical solution of
the linearised problem, which shares the same spatial operator, the same Robin
surface condition and the same centre symmetry:

    dC/dt = D * (1/r) * d/dr ( r * dC/dr ),     0 < r < R
    C(r, 0) = C0,                               dC/dr(0, t) = 0
    -D * dC/dr(R, t) = k_m * ( C(R, t) - Cb )

Substituting ``theta = C - Cb`` and ``rho = r / R`` gives the classical Bessel
series

    theta(rho, t) = sum_n A_n * J0(lam_n * rho) * exp(-lam_n^2 * Fo)
    Fo = D t / R^2,      Bi = k_m R / D
    lam_n * J1(lam_n) = Bi * J0(lam_n)
    A_n = 2 * theta0 * J1(lam_n) / ( lam_n * ( J0(lam_n)^2 + J1(lam_n)^2 ) )

The temperature solver is verified by the same series with
``D -> alpha = k / (rho cp)`` and ``k_m -> h``.

Because the series, the quadrature and the grid are all independent of the
solver code, agreement here is genuine verification rather than a consistency
check.  Observed orders of accuracy are reported for both mesh and time-step
refinement.

Run ``python -m tedm.analysis.verification`` for the report.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import j0, j1, jn_zeros

import tedm.config as cfg
from tedm.config import CP, H_CONV, K_COND, K_MASS, R, RHO
from tedm.model import build_grid, step_moisture, step_temperature
from tedm.properties import PropertySet

#: D(C) = D_A * exp(-D_B / C) with the problem-1 data evaluated at C = 2.55
CONST_D = 7e-9 * np.exp(-0.89 / 2.55)

#: sample times shared by every run so results with different dt stay comparable
SAMPLE_TIMES = (200.0, 600.0, 1000.0, 1400.0, 1800.0)


def constant_diffusion(d: float = CONST_D, rho=RHO, cp=CP, k=K_COND) -> PropertySet:
    """Property set with a *constant* diffusion coefficient (linear problem)."""
    return PropertySet(
        name="constant-D",
        rho_a=float(rho), rho_b=0.0,
        cp_a=float(cp), cp_b=0.0,
        k_a=float(k), k_b=0.0,
        D_A=float(d), D_B=1e-12,
        arrhenius=False,
    )


def bessel_eigenvalues(bi: float, n: int = 400) -> np.ndarray:
    """Positive roots of ``lam*J1(lam) = Bi*J0(lam)``.

    They are bracketed between consecutive zeros of ``J0``, where the left-hand
    side changes sign exactly once.
    """
    z = jn_zeros(0, n + 2)
    edges = np.concatenate(([0.0], z))
    f = lambda lam: lam * j1(lam) - bi * j0(lam)
    roots = []
    for a, b in zip(edges[:-1], edges[1:]):
        fa = -bi if a == 0.0 else f(a)
        fb = f(b)
        if np.sign(fa) == np.sign(fb):
            continue
        roots.append(brentq(f, a, b, xtol=1e-14, rtol=1e-15))
    return np.asarray(roots)


def bessel_series(r, t, radius=R, d=CONST_D, km=K_MASS, c0=2.55, cb=0.0, terms=600):
    """Evaluate the analytical series.  Returns shape ``(len(r), len(t))``."""
    r = np.atleast_1d(np.asarray(r, dtype=float))
    t = np.atleast_1d(np.asarray(t, dtype=float))
    lam = bessel_eigenvalues(km * radius / d, terms)
    a_n = 2.0 * (c0 - cb) * j1(lam) / (lam * (j0(lam) ** 2 + j1(lam) ** 2))
    basis = j0(np.outer(r / radius, lam))                          # (nr, K)
    with np.errstate(under="ignore"):
        decay = np.exp(-np.outer(d * t / radius ** 2, lam ** 2))    # (nt, K)
    return cb + basis @ (decay * a_n[None, :]).T


# ----------------------------------------------------------------------
def march_constant(n_cells=100, dt=1.0, t_end=1800.0, sample_times=SAMPLE_TIMES,
                   d=CONST_D, km=K_MASS, c0=2.55, radius=R):
    """March the real solver in a constant environment and return the snapshots."""
    sample_times = np.asarray(sample_times, dtype=float)
    n_steps = int(round(t_end / dt))
    idx = np.rint(sample_times / dt).astype(int)
    if not np.allclose(idx * dt, sample_times):
        raise ValueError("sample times must be multiples of dt")

    r, r_face, v = build_grid(radius=radius, n_cells=n_cells)
    prop = constant_diffusion(d)
    C = np.full(n_cells + 1, c0)
    T = np.full(n_cells + 1, 300.0)

    snaps = {0: C.copy()}
    want = set(idx.tolist())
    for n in range(n_steps):
        C = step_moisture(C, T, r_face, v, dt, 0.0, prop=prop, km=km,
                          radius=radius, picard=2)
        if (n + 1) in want:
            snaps[n + 1] = C.copy()
    return r, np.array([snaps[i] for i in idx]), prop


def compare_linear(n_cells=100, dt=1.0, t_end=1800.0, sample_times=SAMPLE_TIMES,
                   d=CONST_D, km=K_MASS, c0=2.55, radius=R):
    r, num, _ = march_constant(n_cells, dt, t_end, sample_times, d, km, c0, radius)
    times = np.asarray(sample_times, dtype=float)
    exact = bessel_series(r, times, radius=radius, d=d, km=km, c0=c0, cb=0.0).T
    err = num - exact
    l2 = float(np.sqrt(np.mean(err ** 2)))
    return {
        "r": r, "times": times, "numeric": num, "exact": exact,
        "l2": l2, "linf": float(np.max(np.abs(err))),
        "l2_relative": float(l2 / np.mean(exact)),
    }


def spatial_order(dt=0.5, cells=(25, 50, 100, 200), t_end=1800.0):
    """Observed order of accuracy in space, with a time step small enough that
    the ``O(dt)`` error no longer masks it."""
    errs = np.asarray([compare_linear(n_cells=n, dt=dt, t_end=t_end)["l2"] for n in cells])
    order = np.log2(errs[:-1] / errs[1:])
    return {"cells": list(cells), "errors": errs.tolist(), "order": order.tolist()}


def temporal_order(n_cells=400, dts=(8.0, 4.0, 2.0, 1.0), t_end=1800.0):
    """Observed order of accuracy in time on a fine spatial grid."""
    errs = np.asarray([compare_linear(n_cells=n_cells, dt=dt, t_end=t_end)["l2"]
                       for dt in dts])
    order = np.log2(errs[:-1] / errs[1:])
    return {"dts": list(dts), "errors": errs.tolist(), "order": order.tolist()}


def verify_temperature(n_cells=100, dt=0.5, t_end=1800.0, radius=R,
                       sample_times=SAMPLE_TIMES, t_amb=50.0):
    """Same idea for the energy equation: ``alpha = k/(rho cp)`` with ``h`` playing
    the role of ``k_m`` and a constant ambient temperature as the driving value."""
    sample_times = np.asarray(sample_times, dtype=float)
    n_steps = int(round(t_end / dt))
    idx = np.rint(sample_times / dt).astype(int)
    t0 = cfg.T0

    r, r_face, v = build_grid(radius=radius, n_cells=n_cells)
    prop = constant_diffusion()
    T = np.full(n_cells + 1, t0)
    C = np.full(n_cells + 1, 2.55)

    snaps = {0: T.copy()}
    want = set(idx.tolist())
    for n in range(n_steps):
        T = step_temperature(T, C, r_face, v, dt, t_amb, 0.5, prop=prop, radius=radius)
        if (n + 1) in want:
            snaps[n + 1] = T.copy()

    alpha = K_COND / (RHO * CP)
    # -k dT/dr = h (T_s - T_a)  <=>  -alpha dtheta/dr = (h/(rho cp)) theta
    km_eff = H_CONV / (RHO * CP)
    exact = bessel_series(r, sample_times, radius=radius, d=alpha, km=km_eff,
                          c0=t0, cb=t_amb).T
    num = np.array([snaps[i] for i in idx])
    err = num - exact
    l2 = float(np.sqrt(np.mean(err ** 2)))
    return {
        "l2": l2,
        "linf": float(np.max(np.abs(err))),
        "l2_relative": float(l2 / np.ptp(exact)),
        "numeric": num, "exact": exact,
    }


# ----------------------------------------------------------------------
def report():
    """Plain-text report reused by ``docs/verification.md`` and the CLI."""
    step = compare_linear()
    temp = verify_temperature()
    sp = spatial_order()
    tm = temporal_order()

    lines = [
        "=" * 68,
        "Verification against the analytical Bessel series",
        "=" * 68,
        "  base run: N = 100 cells, dt = 1 s, t = 1800 s",
        f"    moisture    relative L2 = {step['l2_relative']:.3e}",
        f"                Linf        = {step['linf']:.3e} kg/kg",
        f"    temperature relative L2 = {temp['l2_relative']:.3e}",
        f"                Linf        = {temp['linf']:.3e} K",
        "",
        "  spatial refinement (dt = 0.5 s)",
    ]
    for n, e, o in zip(sp["cells"], sp["errors"], sp["order"]):
        lines.append(f"    N  = {n:>4d}   L2 = {e:.3e}   observed order = {o:.2f}")
    lines.append("")
    lines.append("  temporal refinement (N = 400 cells)")
    for d, e, o in zip(tm["dts"], tm["errors"], tm["order"]):
        lines.append(f"    dt = {d:>5.1f} s  L2 = {e:.3e}   observed order = {o:.2f}")
    return "\n".join(lines), {"step": step, "temperature": temp, "space": sp, "time": tm}


if __name__ == "__main__":
    text, _ = report()
    print(text)
