# ============================================================
# 问题4 灵敏度分析
# ------------------------------------------------------------
# 扰动参数：D_A, km, h, 收缩率（R_end/R_0）
# 画龙卷风图（tornado plot）
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tedm.config import FIG_DIR, T0, C0, R, H_CONV, K_MASS
from tedm.properties import APP4, PropertySet
from tedm.model import _solve_tridiag
from tedm.data_loader import load_air_conditions, load_radius

rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
rcParams['axes.unicode_minus'] = False


def _run_q4_custom(D_scale=1.0, km_scale=1.0, h_scale=1.0, shrink_scale=1.0,
                   N=100, dt=60.0, C_target=0.15, t_max_h=200.0):
    """自定义参数的求解（用于灵敏度分析）"""
    from tedm.solver_q4 import build_grid_xi, step_temperature_q4, step_moisture_q4

    # 自定义物性（只改 D_A）
    prop = PropertySet(
        name='Q4_sens',
        rho_a=APP4.rho_a, rho_b=APP4.rho_b,
        cp_a=APP4.cp_a, cp_b=APP4.cp_b,
        k_a=APP4.k_a, k_b=APP4.k_b,
        D_A=APP4.D_A * D_scale, D_B=APP4.D_B,
        arrhenius=APP4.arrhenius, E_arr=APP4.E_arr,
    )

    t_max = t_max_h * 3600.0
    t_air_arr, T_air_arr, C_air_arr = load_air_conditions(t_max, dt)
    t_r, R_data, R_func = load_radius()

    xi, xi_face, v = build_grid_xi(N)
    T = np.full(N + 1, T0)
    C = np.full(N + 1, C0)

    n_max = int(round(t_max / dt))
    t_end_h = None

    for n in range(n_max):
        t_new = (n + 1) * dt
        R_cm = float(R_func(t_new))
        # 收缩率缩放
        if shrink_scale != 1.0:
            R_cm = R * 100 - (R * 100 - R_cm) * shrink_scale
        R_m = R_cm / 100.0

        T = step_temperature_q4(T, C, xi_face, v, R_m, dt,
                                T_air_arr[n+1], prop=prop, use_latent=False)
        C = step_moisture_q4(C, T, xi_face, v, R_m, dt,
                             C_air_arr[n+1], prop=prop, picard=2)

        if C[0] < C_target:
            t_end_h = (n + 1) * dt / 3600.0
            break

    return t_end_h


def sensitivity_analysis():
    print("=" * 70)
    print("问题4 灵敏度分析")
    print("=" * 70)

    # 基准
    base = _run_q4_custom()
    print(f"基准烘干时间: {base:.4f} h\n")

    cases = [
        ("D 前因子 +20%", _run_q4_custom(D_scale=1.20)),
        ("D 前因子 -20%", _run_q4_custom(D_scale=0.80)),
        ("收缩率 +20%", _run_q4_custom(shrink_scale=1.20)),
        ("收缩率 -20%", _run_q4_custom(shrink_scale=0.80)),
    ]

    rows = []
    for name, val in cases:
        delta = (val - base) / base * 100
        rows.append({'param': name, 't_end_h': val, 'delta_pct': delta})
        print(f"  {name:>16s}: t_end = {val:.4f} h, Δ = {delta:+.2f}%")

    # 保存
    from tedm.config import OUTPUT_DIR
    lines = ["问题4 灵敏度分析报告", "=" * 40,
             f"基准: {base:.4f} h", ""]
    for r in rows:
        lines.append(f"  {r['param']}: {r['t_end_h']:.4f} h ({r['delta_pct']:+.2f}%)")
    (OUTPUT_DIR / "sensitivity_q4_report.txt").write_text(
        "\n".join(lines), encoding="utf-8")

    return base, rows


def plot_tornado(base, rows):
    names = [r['param'] for r in rows]
    deltas = [r['delta_pct'] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['red' if d > 0 else 'blue' for d in deltas]
    ax.barh(names, deltas, color=colors, alpha=0.7)
    ax.axvline(0, color='k', lw=1)
    ax.set_xlabel('烘干时间相对变化 / %')
    ax.set_title(f'问题4 灵敏度分析（基准 {base:.2f} h）')
    ax.grid(alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_q4_sensitivity_tornado.png', dpi=300,
                bbox_inches='tight')
    plt.close()
    print(f"\n[图] 龙卷风图: {FIG_DIR / 'fig_q4_sensitivity_tornado.png'}")


if __name__ == "__main__":
    base, rows = sensitivity_analysis()
    plot_tornado(base, rows)