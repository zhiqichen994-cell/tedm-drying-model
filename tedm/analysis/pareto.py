# ============================================================
# 问题3 帕累托前沿：烘干时间 vs 能耗
# ------------------------------------------------------------
# 控制变量：烘房温度 T_set ∈ {40, 45, 50, 55, 60, 65, 70} ℃
# 能耗定义：E = ∫ hA(T_air - T_s) dt / η，η=0.8
# 输出：fig_q3_11_pareto.png
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tedm.config import FIG_DIR, T0, C0, R, H_CONV, N_R
from tedm.model import build_grid, step_temperature, step_moisture
from tedm.properties import APP3

rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
rcParams['axes.unicode_minus'] = False

# 药材长度（如果 config.py 里没定义 L，就在这里手动加）
try:
    from tedm.config import L
except ImportError:
    L = 0.25   # m


def _run_with_Tset(T_set, dt=60.0, t_max_h=96.0):
    """固定烘房温度 T_set，跑模型，返回 (t_end_h, E_kWh)"""
    t_max = t_max_h * 3600.0
    r, r_face, v = build_grid()
    N = N_R

    T = np.full(N + 1, T0)
    C = np.full(N + 1, C0)

    # 表面积（m²）：2πRL（去掉两端）
    A = 2 * np.pi * R * L

    # 烘房水分浓度：用附件1 平台值
    C_air_const = 0.04986

    E_total = 0.0
    eta = 0.8

    n_max = int(round(t_max / dt))
    for n in range(n_max):
        T_air_n = T_set
        C_air_n = C_air_const

        T = step_temperature(T, C, r_face, v, dt,
                             T_air_n, C_air_n,
                             prop=APP3, use_latent=False)
        C = step_moisture(C, T, r_face, v, dt,
                          C_air_n, prop=APP3, picard=2)

        # 能耗累积
        q = H_CONV * A * (T_air_n - T[-1]) / eta
        E_total += q * dt

        if C[0] < 0.15:
            t_end_h = (n + 1) * dt / 3600.0
            E_kWh = E_total / 3.6e6
            return t_end_h, E_kWh

    return None, E_total / 3.6e6


def run_pareto():
    T_set_list = [40, 45, 50, 55, 60, 65, 70]
    taus = []
    Es = []
    valid_T = []

    print(f"开始帕累托扫描：{len(T_set_list)} 个温度点...")
    for T_set in T_set_list:
        try:
            t_end, E = _run_with_Tset(T_set)
        except Exception as e:
            print(f"  T_set = {T_set}℃ 失败: {e}")
            continue
        if t_end is not None:
            taus.append(t_end)
            Es.append(E)
            valid_T.append(T_set)
            print(f"  T_set = {T_set}℃: τ = {t_end:.2f} h, E = {E:.3f} kWh")

    return np.array(valid_T), np.array(taus), np.array(Es)


def plot_pareto(T_set_list, taus, Es):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(taus, Es, 'o-', color='steelblue', lw=2.2, ms=9,
            markerfacecolor='red', markeredgecolor='k')

    for i, T in enumerate(T_set_list):
        ax.annotate(f'{T}℃', (taus[i], Es[i]),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=10)

    # 标注折中点（50℃）
    idx_mid = list(T_set_list).index(50)
    ax.scatter(taus[idx_mid], Es[idx_mid], s=200, facecolors='none',
               edgecolors='green', linewidths=2, zorder=5,
               label='折中点（50℃）')
    ax.legend(fontsize=10)

    ax.set_xlabel('烘干时间 / h')
    ax.set_ylabel('能耗 / kWh')
    ax.set_title('问题3 烘干时间-能耗帕累托前沿')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_q3_11_pareto.png', dpi=300,
                bbox_inches='tight')
    plt.close()
    print(f"[图11] 帕累托前沿: {FIG_DIR / 'fig_q3_11_pareto.png'}")


if __name__ == "__main__":
    T_set_list, taus, Es = run_pareto()
    if len(taus) > 0:
        plot_pareto(T_set_list, taus, Es)
    else:
        print("无有效结果")