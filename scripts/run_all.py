#!/usr/bin/env python
"""Regenerate every deliverable in ``output/`` from scratch.

Usage
-----
    python scripts/run_all.py            # solve, tabulate, plot, verify
    python scripts/run_all.py --fast     # skip the slow verification studies

Everything this script writes lands in ``output/``:

    output/result1.xlsx .. result4.xlsx   the four answer spreadsheets
    output/tables/*.xlsx                  the six tables quoted in the paper
    output/figures/*.png                  the figures quoted in the paper
    output/conservation_report.txt        discrete conservation diagnostics
    output/verification_report.txt        error vs. the analytical solution
    output/convergence_report.txt         mesh / time-step convergence

The script is deterministic: the only randomness is inside the Monte-Carlo
study, which is seeded there.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _banner(text: str) -> None:
    print()
    print("=" * 72)
    print(text)
    print("=" * 72)


def solve_and_report():
    from tedm.analysis.conservation import (
        report_fixed_domain,
        report_moving_boundary,
    )
    from tedm.config import OUTPUT_DIR
    from tedm.postprocess import extract_tables, save_result1
    from tedm.postprocess_q2 import save_result2, save_tables_q2
    from tedm.postprocess_q3 import save_result3, save_tables_q3
    from tedm.postprocess_q4 import save_result4, save_tables_q4
    from tedm.properties import APP2
    from tedm.solver import solve
    from tedm.solver_q2 import solve_q2
    from tedm.solver_q3 import solve_q3
    from tedm.solver_q4 import solve_q4
    from tedm.solver_q4_fixed import solve_q4_fixed
    from tedm.viz import q1 as viz1
    from tedm.viz import q2 as viz2
    from tedm.viz import q3 as viz3
    from tedm.viz import q4 as viz4

    summary = {}

    # ---------------- problem 1 ----------------
    _banner("Problem 1 / 问题1 : 30 min at the measured chamber conditions")
    t0 = time.time()
    t, r, T1, C1, Ta, Ca = solve(prop=APP2, t_end=1800.0, dt=1.0)
    print(f"  solved in {time.time() - t0:.1f} s")
    print(f"  T(1800 s): centre {T1[-1, 0]:.4f} C, surface {T1[-1, -1]:.4f} C")
    print(f"  C(1800 s): surface {C1[-1, -1]:.4f} kg/kg")
    save_result1(t, r, T1, C1)
    extract_tables(t, r, T1, C1)
    viz1.plot_all(t, r, T1, C1, Ta, Ca)
    txt, _ = report_fixed_domain(t, r, T1, C1, Ta, Ca, "problem 1")
    print(txt)
    summary["q1"] = {
        "T_centre_C": float(T1[-1, 0]), "T_surface_C": float(T1[-1, -1]),
        "C_surface": float(C1[-1, -1]),
    }

    # ---------------- problem 2 ----------------
    _banner("Problem 2 / 问题2 : 3 h with the 附录3 variable properties")
    t0 = time.time()
    t2, r2, T2, C2, Ta2, Ca2 = solve_q2()
    print(f"  solved in {time.time() - t0:.1f} s")
    print(f"  T(3 h): centre {T2[-1, 0]:.4f} C, surface {T2[-1, -1]:.4f} C")
    print(f"  C(3 h): centre {C2[-1, 0]:.4f}, surface {C2[-1, -1]:.4f} kg/kg")
    save_result2(t2, r2, T2, C2)
    save_tables_q2(t2, r2, T2, C2)
    viz2.plot_all_q2(t2, r2, T2, C2, Ta2, Ca2)
    summary["q2"] = {
        "T_centre_C": float(T2[-1, 0]), "T_surface_C": float(T2[-1, -1]),
        "C_centre": float(C2[-1, 0]), "C_surface": float(C2[-1, -1]),
    }

    # ---------------- problem 3 ----------------
    _banner("Problem 3 / 问题3 : dry to C < 0.15 at the 50 C plateau")
    t0 = time.time()
    t3, r3, T3, C3, _Ta3, _Ca3, tau3, end_C3 = solve_q3()
    print(f"  solved in {time.time() - t0:.1f} s")
    print(f"  drying time tau = {tau3:.4f} h, end centre C = {end_C3:.6f}")
    save_result3(t3, r3, C3, tau3)
    save_tables_q3(t3, r3, C3, tau3)
    viz3.plot_all_q3(t3, r3, T3, C3, tau3)
    summary["q3"] = {"tau_h": float(tau3), "end_C_centre": float(end_C3)}

    # ---------------- problem 4 ----------------
    _banner("Problem 4 / 问题4 : shrinkage 2.00 -> 1.20 cm, 附录4 properties")
    t0 = time.time()
    t4, xi4, T4, C4, R4, _Ta4, _Ca4, tau4 = solve_q4()
    print(f"  shrinkage run solved in {time.time() - t0:.1f} s")
    print(f"  tau = {tau4:.4f} h, R = {R4[0] * 100:.4f} -> {R4[-1] * 100:.4f} cm")

    t0 = time.time()
    t_fix, _xi_f, _T_f, C_fix, tau_fix = solve_q4_fixed()
    print(f"  fixed-radius control solved in {time.time() - t0:.1f} s")
    print(f"  tau_fixed = {tau_fix:.4f} h "
          f"(shrinkage saves {(tau_fix - tau4) / tau_fix:.1%})")

    save_result4(t4, xi4, C4, R4, tau4)
    save_tables_q4(t4, xi4, C4, R4, tau4)
    viz4.plot_all_q4(t4, xi4, T4, C4, R4, tau4,
                     C_fixed=C_fix, t_fixed=t_fix, t_end_fixed_h=tau_fix)
    txt, _ = report_moving_boundary()
    print(txt)
    summary["q4"] = {
        "tau_h": float(tau4), "tau_fixed_h": float(tau_fix),
        "R_start_cm": float(R4[0] * 100), "R_end_cm": float(R4[-1] * 100),
    }

    # ---------------- summary ----------------
    _banner("Headline results")
    for k, v in summary.items():
        print(f"  {k}: " + ", ".join(f"{a} = {b}" for a, b in v.items()))

    (OUTPUT_DIR / "summary.txt").write_text(
        "\n".join(f"{k}: " + ", ".join(f"{a}={b}" for a, b in v.items())
                  for k, v in summary.items()),
        encoding="utf-8",
    )
    return summary


def run_verification():
    from tedm.analysis.conservation import (
        report_fixed_domain,
        report_moving_boundary,
    )
    from tedm.analysis.verification import report as verify_report
    from tedm.config import OUTPUT_DIR
    from tedm.properties import APP2
    from tedm.solver import solve

    _banner("Discrete conservation diagnostics")
    t, r, T1, C1, Ta, Ca = solve(prop=APP2, t_end=1800.0, dt=1.0)
    txt_a, _ = report_fixed_domain(t, r, T1, C1, Ta, Ca, "problem 1")
    txt_b, _ = report_moving_boundary()
    text = txt_a + "\n\n" + txt_b
    print(text)
    (OUTPUT_DIR / "conservation_report.txt").write_text(text, encoding="utf-8")

    _banner("Verification against the analytical Bessel series")
    text, _ = verify_report()
    print(text)
    (OUTPUT_DIR / "verification_report.txt").write_text(text, encoding="utf-8")


def run_convergence():
    from tedm.analysis.convergence import report
    from tedm.config import OUTPUT_DIR

    _banner("Mesh and time-step convergence of the quantities of interest")
    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report()
    text = buf.getvalue()
    print(text)
    (OUTPUT_DIR / "convergence_report.txt").write_text(text, encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true",
                    help="skip the verification and convergence studies")
    ap.add_argument("--verify", action="store_true",
                    help="run only the verification and convergence studies")
    args = ap.parse_args(argv)

    from tedm.config import OUTPUT_DIR

    t_start = time.time()
    if not args.verify:
        solve_and_report()
    if not args.fast:
        run_verification()
        run_convergence()

    _banner(f"Done in {time.time() - t_start:.1f} s")
    print(f"  deliverables: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
