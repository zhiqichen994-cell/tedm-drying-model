# ============================================================
# 问题3 蒙特卡洛：参数不确定性对烘干时间的影响
# ------------------------------------------------------------
# 扰动参数：
#   D 前因子：±20%
#   C0：±5%
# 分布：正态
# 样本数：100（可调）
# 输出：fig_q3_10_montecarlo.png
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import gaussian_kde
from tedm.config import FIG_DIR, T0, C0 as C0_BASE, N_R
from tedm.model import build_grid, step_temperature, step_moisture
from tedm.data_loader import load_air_conditions
from tedm.properties import APP3, PropertySet

rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
rcParams['axes.unicode_minus'] = False


def _run_one(D_scale, C0_val, t_max_h=96.0, dt=60.0):
    """
    跑一次模型，返回 (t_end_h, end_center_C)
    扰动：D 前因子 * D_scale，初始 C0 * C0_val
    """
    prop = PropertySet(
        name='MC',
        rho_a=APP3.rho_a, rho_b=APP3.rho_b,
        cp_a=APP3.cp_a, cp_b=APP3.cp_b,
        k_a=APP3.k_a, k_b=APP3.k_b,
        D_A=APP3.D_A * D_scale, D_B=APP3.D_B,
        arrhenius=APP3.arrhenius, E_arr=APP3.E_arr,
    )

    t_max = t_max_h * 3600.0
    t, T_air, C_air = load_air_conditions(t_max, dt)
    r, r_face, v = build_grid()

    T = np.full(N_R + 1, T0)
    C = np.full(N_R + 1, C0_val)

    n_max = int(round(t_max / dt))
    t_end_h = None
    end_center_C = None

    for n in range(n_max):
        T = step_temperature(T, C, r_face, v, dt,
                             T_air[n+1], C_air[n+1],
                             prop=prop, use_latent=False)
        C = step_moisture(C, T, r_face, v, dt,
                          C_air[n+1], prop=prop, picard=2)

        if C[0] < 0.15:
            t_end_h = (n + 1) * dt / 3600.0
            end_center_C = C[0]
            break

    return t_end_h, end_center_C


def run_montecarlo(n_samples=100):
    np.random.seed(42)
    taus = []
    end_Cs = []

    print(f"开始蒙特卡洛：{n_samples} 样本...")
    for s in range(n_samples):
        D_scale = 1 + 0.20 * np.random.randn()
        C0_val = C0_BASE * (1 + 0.05 * np.random.randn())

        t_end_h, end_C = _run_one(D_scale, C0_val)
        if t_end_h is None:
            continue
        taus.append(t_end_h)
        end_Cs.append(end_C)

        if (s + 1) % 10 == 0:
            print(f"  完成 {s+1}/{n_samples}，当前 τ={t_end_h:.2f} h")

    return np.array(taus), np.array(end_Cs)


# ============================================================
# 新增：手绘小提琴图的辅助函数
# ============================================================
def _draw_violin(ax, data, position=0, width=0.8,
                 facecolor='steelblue', edgecolor='navy',
                 alpha=0.75, linewidth=1.5,
                 show_box=True, show_scatter=True,
                 scatter_color='black', scatter_size=12,
                 scatter_alpha=0.45):
    """
    在 ax 上画一个垂直的小提琴图。
    - data: 1D 数组
    - position: 小提琴中心在 x 轴上的位置（默认 0）
    - width: 最大宽度
    - show_box: 是否显示箱线（IQR + 中位数 + 须）
    - show_scatter: 是否叠加散点
    """
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    if len(data) < 2:
        return

    # 1. 核密度估计
    kde = gaussian_kde(data)
    y_grid = np.linspace(data.min() - 0.1 * (data.max() - data.min() + 1e-9),
                         data.max() + 0.1 * (data.max() - data.min() + 1e-9),
                         300)
    density = kde(y_grid)
    # 归一化到最大宽度
    density = density / density.max() * (width / 2.0)

    # 2. 对称填充
    ax.fill_betweenx(y_grid,
                     position - density, position + density,
                     facecolor=facecolor, edgecolor=edgecolor,
                     alpha=alpha, linewidth=linewidth, zorder=2)

    # 3. 箱线叠加（IQR + 中位数 + 1.5IQR 须）
    if show_box:
        q1, med, q3 = np.percentile(data, [25, 50, 75])
        iqr = q3 - q1
        whisker_low = max(data.min(), q1 - 1.5 * iqr)
        whisker_high = min(data.max(), q3 + 1.5 * iqr)

        box_w = width * 0.18
        # 箱体
        ax.add_patch(plt.Rectangle((position - box_w, q1),
                                   2 * box_w, q3 - q1,
                                   facecolor='white', edgecolor='black',
                                   linewidth=1.2, zorder=3))
        # 中位线
        ax.plot([position - box_w, position + box_w], [med, med],
                color='black', linewidth=2.0, zorder=4)
        # 上下须
        ax.plot([position, position], [q3, whisker_high],
                color='black', linewidth=1.0, zorder=3)
        ax.plot([position, position], [whisker_low, q1],
                color='black', linewidth=1.0, zorder=3)
        ax.plot([position - box_w * 0.7, position + box_w * 0.7],
                [whisker_high, whisker_high],
                color='black', linewidth=1.0, zorder=3)
        ax.plot([position - box_w * 0.7, position + box_w * 0.7],
                [whisker_low, whisker_low],
                color='black', linewidth=1.0, zorder=3)

    # 4. 散点叠加（jitter）
    if show_scatter:
        jitter = (np.random.rand(len(data)) - 0.5) * width * 0.35
        ax.scatter(position + jitter, data,
                   s=scatter_size, color=scatter_color,
                   alpha=scatter_alpha, zorder=5,
                   edgecolors='none')


# ============================================================
# 重写：小提琴图版本
# ============================================================
def plot_montecarlo(taus, end_Cs):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    mu = np.mean(taus)
    p95 = np.percentile(taus, 95)
    sigma = np.std(taus)

    # ---------------- (a) 烘干时间小提琴图 ----------------
    ax = axes[0]
    _draw_violin(ax, taus, position=0, width=0.9,
                 facecolor='steelblue', edgecolor='navy',
                 alpha=0.75, linewidth=1.6,
                 show_box=True, show_scatter=True,
                 scatter_color='black', scatter_size=14,
                 scatter_alpha=0.45)

    # 均值、95% 分位参考线
    ax.axhline(mu, color='r', ls='--', lw=2,
               label=f'均值 = {mu:.2f} h')
    ax.axhline(p95, color='orange', ls=':', lw=2,
               label=f'95% 分位 = {p95:.2f} h')
    # ±1σ 阴影带
    ax.axhspan(mu - sigma, mu + sigma,
               color='red', alpha=0.06, zorder=1)
    ax.text(0.45, mu + sigma, '±1σ', color='red',
            fontsize=10, va='bottom', ha='left', alpha=0.8)

    ax.set_xlim(-0.7, 0.7)
    ax.set_xticks([])
    ax.set_ylabel('烘干时间 τ / h', fontsize=12)
    ax.set_title('(a) 烘干时间分布（小提琴图）', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3, axis='y')

    # ---------------- (b) 终点中心含水率小提琴图 ----------------
    ax = axes[1]
    mu_C = np.mean(end_Cs)

    _draw_violin(ax, end_Cs, position=0, width=0.9,
                 facecolor='seagreen', edgecolor='darkgreen',
                 alpha=0.75, linewidth=1.6,
                 show_box=True, show_scatter=True,
                 scatter_color='black', scatter_size=14,
                 scatter_alpha=0.45)

    # 阈值 0.15 参考线
    ax.axhline(0.15, color='r', ls='--', lw=2,
               label='阈值 0.15')
    ax.axhline(mu_C, color='orange', ls=':', lw=2,
               label=f'均值 = {mu_C:.4f}')

    ax.set_xlim(-0.7, 0.7)
    ax.set_xticks([])
    ax.set_ylabel('终点中心含水率 / (kg/kg)', fontsize=12)
    ax.set_title('(b) 终点中心含水率分布（小提琴图）', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('问题3 参数不确定性的蒙特卡洛分析', fontsize=12, y=1.00)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(FIG_DIR / 'fig_q3_10_montecarlo.png', dpi=300,
                bbox_inches='tight')
    plt.close()
    print(f"[图10] 蒙特卡洛（小提琴图）: "
          f"{FIG_DIR / 'fig_q3_10_montecarlo.png'}")


if __name__ == "__main__":
    taus, end_Cs = run_montecarlo(n_samples=100)
    print(f"\n结果统计：")
    print(f"  烘干时间均值 = {np.mean(taus):.2f} h")
    print(f"  烘干时间标准差 = {np.std(taus):.2f} h")
    print(f"  95% 分位数 = {np.percentile(taus, 95):.2f} h")
    print(f"  终点含水率均值 = {np.mean(end_Cs):.4f}")
    plot_montecarlo(taus, end_Cs)