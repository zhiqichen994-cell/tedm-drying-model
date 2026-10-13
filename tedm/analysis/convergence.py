# ============================================================
# 收敛性检验（网格 / 时间步）
# ------------------------------------------------------------
# 与 verification.py 的区别：
#   verification.py 比较的是「真值」—— 与 Bessel 解析解逐点对比，
#                    能得到误差和观测收敛阶（需要解析解，只能做问题1）。
#   convergence.py  比较的是「关心的量」(QoI: quantity of interest)——
#                    没有解析解的问题3/4 只能做这种"自收敛"检验：
#                    加密网格 / 减小时间步，看 QoI 是否稳定。
#
# 检验口径（论文 7.x 节 + 本次补充）：
#   问题1: C_centre(1800 s), C_mean(1800 s)
#   问题2: C_centre(3 h),   C_mean(3 h)
#   问题3: 烘干时间 t_end (h)   ← 唯一判据，最该检验的量
#   问题4: 烘干时间 t_end (h)
#
# 判据：QoI 在相邻三档之间相对变化 < 1% 视为收敛。
# ============================================================

import time

import numpy as np

from tedm.properties import APP2, APP3
from tedm.solver import solve
from tedm.solver_q3 import solve_q3
from tedm.solver_q4 import solve_q4

# 各问题默认的加密序列
#   注意 dt 行：问题1/2 本身就是 1 s 量级的显式-隐式混合，1 s 已经很细，
#   所以只做 8/4/2/1；问题3/4 用 60 s 起步，做 240/120/60。
DEFAULT_MESH_CELLS = {
    "q1": (25, 50, 100, 200),
    "q2": (25, 50, 100, 200),
    "q3": (25, 50, 100, 200),
    "q4": (50, 100, 200),
}

DEFAULT_STEPS = {
    "q1": (8.0, 4.0, 2.0, 1.0),
    "q2": (8.0, 4.0, 2.0, 1.0),
    "q3": (240.0, 120.0, 60.0),
    "q4": (240.0, 120.0, 60.0),
}

# 做时间步检验时固定用的网格（够细，使空间误差不污染时间阶）
STEP_STUDY_CELLS = {"q1": 200, "q2": 200, "q3": 200, "q4": 200}


# ------------------------------------------------------------
# 关心的量
# ------------------------------------------------------------
def quantity(problem, n_cells=None, dt=None):
    """跑一次模型，返回该问题关心的量（dict）。"""
    problem = problem.lower()

    if problem == "q1":
        # 常温 30 °C 恒定风温，跑 1800 s
        _t, _r, _T, C, _Ta, _Ca = solve(
            prop=APP2, t_end=1800.0, dt=dt or 1.0, n_cells=n_cells
        )
        return {
            "C_centre": float(C[-1, 0]),
            "C_surface": float(C[-1, -1]),
            "C_mean": float(np.mean(C[-1])),
        }

    if problem == "q2":
        # 附录3 变物性 + Arrhenius，跑到 3 h = 10800 s
        _t, _r, _T, C, _Ta, _Ca = solve(
            prop=APP3, t_end=10800.0, dt=dt or 1.0, n_cells=n_cells
        )
        return {
            "C_centre": float(C[-1, 0]),
            "C_surface": float(C[-1, -1]),
            "C_mean": float(np.mean(C[-1])),
        }

    if problem == "q3":
        out = solve_q3(dt=dt or 60.0, n_cells=n_cells)
        return {"t_end_h": float(out[6]), "C_centre_end": float(out[7])}

    if problem == "q4":
        out = solve_q4(N=n_cells or 100, dt=dt or 60.0)
        return {"t_end_h": float(out[7])}

    raise ValueError(f"unknown problem: {problem!r} (use q1/q2/q3/q4)")


# ------------------------------------------------------------
# 网格收敛
# ------------------------------------------------------------
def mesh_study(problem, cells=None, dt=None):
    """固定时间步，加密网格。"""
    cells = tuple(cells or DEFAULT_MESH_CELLS[problem.lower()])
    results = {}
    timings = {}
    for n in cells:
        t0 = time.time()
        results[n] = quantity(problem, n_cells=n, dt=dt)
        timings[n] = time.time() - t0
    return {"problem": problem, "dt": dt, "cells": cells,
            "results": results, "timings": timings}


# ------------------------------------------------------------
# 时间步收敛
# ------------------------------------------------------------
def step_study(problem, dts=None, n_cells=None):
    """固定网格，减小时间步。"""
    dts = tuple(dts or DEFAULT_STEPS[problem.lower()])
    n_cells = n_cells or STEP_STUDY_CELLS[problem.lower()]
    results = {}
    timings = {}
    for d in dts:
        t0 = time.time()
        results[d] = quantity(problem, n_cells=n_cells, dt=d)
        timings[d] = time.time() - t0
    return {"problem": problem, "n_cells": n_cells, "dts": dts,
            "results": results, "timings": timings}


# ------------------------------------------------------------
# 报告
# ------------------------------------------------------------
def _spread(values):
    """极差 / 均值（相对）。"""
    v = np.asarray(list(values), dtype=float)
    spread = float(np.max(v) - np.min(v))
    rel = spread / float(np.mean(v)) * 100.0 if np.mean(v) != 0 else float("inf")
    return spread, rel


def _print_study(title, axis_name, axis_vals, results, timings):
    print(f"\n  {title}")
    keys = list(results[axis_vals[0]].keys())
    print(f"    {axis_name:>10s}  " + "".join(f"{k:>14s}" for k in keys)
          + f"{'耗时(s)':>10s}")
    for a in axis_vals:
        r = results[a]
        line = f"    {a:>10}  " + "".join(f"{r[k]:>14.6f}" for k in keys)
        line += f"{timings[a]:>10.1f}"
        print(line)
    print("    " + "-" * (12 + 14 * len(keys) + 10))
    for k in keys:
        spread, rel = _spread(results[a][k] for a in axis_vals)
        flag = "OK" if rel < 1.0 else "CHECK"
        print(f"    {k:<24s} 极差 = {spread:.6f}  相对 = {rel:.4f}%   [{flag}]")


def report(problems=("q1", "q2", "q3", "q4")):
    """跑全部收敛性检验并打印报告，返回 dict 便于写文件。"""
    print("=" * 78)
    print("收敛性检验（网格 / 时间步）")
    print("=" * 78)

    out = {}

    for p in problems:
        p = p.lower()
        print(f"\n{'─' * 78}")
        print(f"问题 {p.upper()}")
        print(f"{'─' * 78}")

        ms = mesh_study(p)
        _print_study("网格收敛（固定 dt）", "N", ms["cells"],
                     ms["results"], ms["timings"])

        ss = step_study(p)
        _print_study(f"时间步收敛（固定 N = {ss['n_cells']}）", "dt", ss["dts"],
                     ss["results"], ss["timings"])

        out[p] = {"mesh": ms, "step": ss}

    # 汇总表（只列 QoI 里最有代表性的那个）
    print(f"\n{'=' * 78}")
    print("汇总：网格收敛的相对极差（%）")
    print(f"{'=' * 78}")
    print(f"    {'问题':<8s}{'QoI':<16s}{'相对极差(%)':>14s}{'判定':>8s}")
    for p in problems:
        p = p.lower()
        ms = out[p]["mesh"]
        primary = "t_end_h" if "t_end_h" in ms["results"][ms["cells"][0]] else "C_mean"
        _, rel = _spread(ms["results"][a][primary] for a in ms["cells"])
        print(f"    {p.upper():<8s}{primary:<16s}{rel:>14.4f}"
              f"{('OK' if rel < 1.0 else 'CHECK'):>8s}")

    return out


if __name__ == "__main__":
    report()
