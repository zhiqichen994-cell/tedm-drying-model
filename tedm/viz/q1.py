# ============================================================
# 问题1 可视化模块
# ------------------------------------------------------------
# 本模块生成论文/附录所需的核心图表，共 5 张：
#   图1  温度/水分时空云图（双面板）        → 论文图6(a)(b)
#   图2  径向剖面图（双面板，多时刻）      → 论文图5(a)(b)
#   图3  热/湿渗透深度对比图               → 论文图7
#   图4  热湿时间尺度对比图                → 论文图12（问题1部分）
#   图5  中心/表面温度与水分曲线（四面板） → 论文图5补充
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tedm.config import FIG_DIR, T0, C0, R, RHO, CP, K_COND, D0, B_D

# 中文字体设置
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
rcParams['axes.unicode_minus'] = False


# ============================================================
# 图1  温度场 / 水分场 时空云图（双面板）
# ------------------------------------------------------------
# 用途：展示 0~1800 s 内，温度与水分沿半径和时间的演化
# 对应论文：图6 (a) 温度时空场  (b) 水分时空场
# 输出：output/figures/fig1_spacetime.png
# ============================================================
def plot_spacetime(t, r, T_result, C_result):
    r_cm = r * 100.0
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- (a) 温度时空云图 ---
    T_max = T_result.max()
    im1 = axes[0].pcolormesh(t / 60, r_cm, T_result.T,
                             shading='auto', cmap='inferno',
                             vmin=T0, vmax=T_max)
    axes[0].set_xlabel('时间 / min')
    axes[0].set_ylabel('到中心距离 / cm')
    axes[0].set_title('(a) 药材温度时空分布')
    c1 = plt.colorbar(im1, ax=axes[0])
    c1.set_label(r'$T$ / $^\circ$C')

    # 温度等值线
    levels_T = np.linspace(T0 + 0.5, T_max - 0.5, 5)
    cs = axes[0].contour(t / 60, r_cm, T_result.T,
                         levels=levels_T, colors='white',
                         linewidths=0.7, alpha=0.7)
    axes[0].clabel(cs, inline=True, fontsize=7, fmt='%.1f')

    # --- (b) 水分时空云图 ---
    im2 = axes[1].pcolormesh(t / 60, r_cm, C_result.T,
                             shading='auto', cmap='viridis',
                             vmin=C_result.min(), vmax=C0)
    axes[1].set_xlabel('时间 / min')
    axes[1].set_ylabel('到中心距离 / cm')
    axes[1].set_title('(b) 药材水分浓度时空分布')
    c2 = plt.colorbar(im2, ax=axes[1])
    c2.set_label(r'$C$ / (kg$\cdot$kg$^{-1}$)')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig1_spacetime.png', dpi=300)
    plt.close()
    print(f"[图1] 时空云图已保存: {FIG_DIR / 'fig1_spacetime.png'}")


# ============================================================
# 图2  径向剖面图（双面板，多时刻）
# ------------------------------------------------------------
# 用途：展示不同时刻 T(r) 和 C(r)，体现"热渗透深、湿渗透浅"
# 对应论文：图5 (a) 温度径向剖面  (b) 水分径向剖面
# 输出：output/figures/fig2_radial.png
# ============================================================
def plot_radial_profiles(t, r, T_result, C_result,
                         times_s=(100, 300, 600, 900, 1200, 1500, 1800)):
    r_cm = r * 100.0
    cmap = plt.cm.viridis(np.linspace(0, 1, len(times_s)))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for i, ts in enumerate(times_s):
        idx = int(ts)
        axes[0].plot(r_cm, T_result[idx], color=cmap[i], lw=1.5, label=f'{ts} s')
        axes[1].plot(r_cm, C_result[idx], color=cmap[i], lw=1.5, label=f'{ts} s')

    # (a) 温度
    axes[0].set_xlabel('到中心距离 $r$ / cm')
    axes[0].set_ylabel(r'温度 $T$ / $^\circ$C')
    axes[0].set_title('(a) 温度径向剖面')
    axes[0].legend(fontsize=8, title='时间')
    axes[0].grid(alpha=0.3)

    # (b) 水分
    axes[1].set_xlabel('到中心距离 $r$ / cm')
    axes[1].set_ylabel(r'水分浓度 $C$ / (kg$\cdot$kg$^{-1}$)')
    axes[1].set_title('(b) 水分浓度径向剖面')
    axes[1].legend(fontsize=8, title='时间')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig2_radial.png', dpi=300)
    plt.close()
    print(f"[图2] 径向剖面已保存: {FIG_DIR / 'fig2_radial.png'}")


# ============================================================
# 图3  热渗透深度 vs 湿渗透深度对比（修订版 v2）
# ------------------------------------------------------------
# 定义：
#   渗透深度 = 从表面向内，第一次出现显著变化的位置到表面的距离
#              = R - r_inner
#   其中 r_inner 是从中心向外扫描，第一个发生显著变化的半径
#
#   热：温度相对初始值升高超过 0.5 ℃
#   湿：水分相对初始值降低超过 0.01 kg/kg
#
# 这样得到的渗透深度从 0 平滑增长到 R，是论文期望的形状。
# 对应论文：图7 预热阶段渗透深度演化
# 输出：output/figures/fig3_penetration.png
# ============================================================
def plot_penetration_depth(t, r, T_result, C_result, T_air):
    r_cm = r * 100.0
    R_cm = R * 100.0
    n_t = len(t)
    n_r = len(r_cm)
    delta_T = np.zeros(n_t)
    delta_C = np.zeros(n_t)

    TH_T = 0.5        # ℃
    TH_C = 0.01       # kg/kg

    for i in range(n_t):
        # --- 热渗透深度 ---
        dT = T_result[i] - T0
        # 从中心向外扫，找第一个"显著升温"的索引
        idx = np.where(dT > TH_T)[0]
        if idx.size:
            r_inner = r_cm[idx[0]]              # 最靠内的"已升温"位置
            delta_T[i] = R_cm - r_inner          # 表面到这个位置的距离
        else:
            delta_T[i] = 0.0

        # --- 湿渗透深度 ---
        dC = C0 - C_result[i]
        idx = np.where(dC > TH_C)[0]
        if idx.size:
            r_inner = r_cm[idx[0]]
            delta_C[i] = R_cm - r_inner
        else:
            delta_C[i] = 0.0

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t / 60, delta_T, 'r-', lw=2, label=r'热渗透深度 $\delta_T$')
    ax.plot(t / 60, delta_C, 'b-', lw=2, label=r'湿渗透深度 $\delta_C$')
    ax.axhline(R_cm, color='gray', ls='--', lw=1,
               label=f'药材半径 {R_cm:.1f} cm')
    ax.set_xlabel('时间 / min')
    ax.set_ylabel('渗透深度 / cm')
    ax.set_title('热渗透深度与湿渗透深度对比')
    ax.set_ylim(-0.05, R_cm * 1.1)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig3_penetration.png', dpi=300)
    plt.close()
    print(f"[图3] 渗透深度对比已保存: {FIG_DIR / 'fig3_penetration.png'}")

# ============================================================
# 图4  热湿时间尺度对比（柱状图，对数纵轴）
# ------------------------------------------------------------
# 用途：解释"为什么预热阶段温度先变、水分后变"
#   τ_T = R²/α  热扩散特征时间
#   τ_C = R²/D  水分扩散特征时间
# 对应论文：图12（问题1部分）
# 输出：output/figures/fig4_time_scales.png
# ============================================================
def plot_time_scales():
    alpha = K_COND / (RHO * CP)
    D_avg = D0 * np.exp(-B_D / C0)
    tau_T = R ** 2 / alpha
    tau_C = R ** 2 / D_avg

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar([r'热扩散 $\tau_T$', r'水分扩散 $\tau_C$'],
                  [tau_T, tau_C],
                  color=['#d62728', '#1f77b4'],
                  width=0.5)
    ax.set_yscale('log')
    ax.set_ylabel(r'特征时间 $\tau$ / s')
    ax.set_title('热湿时间尺度对比')
    ax.grid(alpha=0.3, axis='y')

    for bar, val in zip(bars, [tau_T, tau_C]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val * 1.3, f'{val:.0f} s',
                ha='center', fontsize=10)

    # 附注比值
    ratio = tau_C / tau_T
    ax.text(0.5, 0.05, f'$\\tau_C / \\tau_T \\approx {ratio:.0f}$',
            transform=ax.transAxes, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', fc='lightyellow', ec='gray'))

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig4_time_scales.png', dpi=300)
    plt.close()
    print(f"[图4] 时间尺度对比已保存: {FIG_DIR / 'fig4_time_scales.png'}")


# ============================================================
# 图5  中心/表面 温度与水分曲线（四面板）
# ------------------------------------------------------------
# 用途：在统一时间轴上对比中心与表面的 T、C 演化，最直观体现"温度先变、水分后变"
# 对应论文：图5 的补充（或作为摘要图）
# 输出：output/figures/fig5_center_surface.png
# ============================================================
def plot_center_surface(t, r, T_result, C_result, T_air, C_air):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    # (a) 中心/表面温度
    axes[0, 0].plot(t / 60, T_result[:, 0], 'b-', lw=1.8, label='中心 $r=0$')
    axes[0, 0].plot(t / 60, T_result[:, -1], 'r-', lw=1.8, label='表面 $r=R$')
    axes[0, 0].plot(t / 60, T_air, 'k--', lw=1.2, alpha=0.6, label='烘房空气')
    axes[0, 0].set_xlabel('时间 / min')
    axes[0, 0].set_ylabel(r'温度 $T$ / $^\circ$C')
    axes[0, 0].set_title('(a) 中心与表面温度')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(alpha=0.3)

    # (b) 中心/表面水分
    axes[0, 1].plot(t / 60, C_result[:, 0], 'b-', lw=1.8, label='中心 $r=0$')
    axes[0, 1].plot(t / 60, C_result[:, -1], 'r-', lw=1.8, label='表面 $r=R$')
    axes[0, 1].plot(t / 60, C_air, 'k--', lw=1.2, alpha=0.6, label='烘房空气')
    axes[0, 1].set_xlabel('时间 / min')
    axes[0, 1].set_ylabel(r'水分浓度 $C$ / (kg$\cdot$kg$^{-1}$)')
    axes[0, 1].set_title('(b) 中心与表面水分浓度')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(alpha=0.3)

    # (c) 中心温度 vs 中心水分（归一化）
    T_norm = (T_result[:, 0] - T0) / (T_result[:, 0].max() - T0 + 1e-12)
    C_norm = (C0 - C_result[:, 0]) / (C0 - C_result[:, 0].min() + 1e-12)
    axes[1, 0].plot(t / 60, T_norm, 'r-', lw=1.8, label='温度（归一化）')
    axes[1, 0].plot(t / 60, C_norm, 'b-', lw=1.8, label='水分（归一化）')
    axes[1, 0].set_xlabel('时间 / min')
    axes[1, 0].set_ylabel('归一化变化量')
    axes[1, 0].set_title('(c) 中心归一化响应对比')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(alpha=0.3)

    # (d) 表面温度 vs 表面水分（归一化）
    T_norm_s = (T_result[:, -1] - T0) / (T_result[:, -1].max() - T0 + 1e-12)
    C_norm_s = (C0 - C_result[:, -1]) / (C0 - C_result[:, -1].min() + 1e-12)
    axes[1, 1].plot(t / 60, T_norm_s, 'r-', lw=1.8, label='温度（归一化）')
    axes[1, 1].plot(t / 60, C_norm_s, 'b-', lw=1.8, label='水分（归一化）')
    axes[1, 1].set_xlabel('时间 / min')
    axes[1, 1].set_ylabel('归一化变化量')
    axes[1, 1].set_title('(d) 表面归一化响应对比')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig5_center_surface.png', dpi=300)
    plt.close()
    print(f"[图5] 中心/表面曲线已保存: {FIG_DIR / 'fig5_center_surface.png'}")


# ============================================================
# 统一入口
# ------------------------------------------------------------
# 调用一次生成全部 5 张图
# ============================================================
def plot_all(t, r, T_result, C_result, T_air, C_air):
    print("=" * 50)
    print("开始生成问题1全部图表...")
    print("=" * 50)
    plot_spacetime(t, r, T_result, C_result)
    plot_radial_profiles(t, r, T_result, C_result)
    plot_penetration_depth(t, r, T_result, C_result, T_air)
    plot_time_scales()
    plot_center_surface(t, r, T_result, C_result, T_air, C_air)
    print("=" * 50)
    print("全部图表生成完成！")
    print(f"输出目录: {FIG_DIR}")
    print("=" * 50)


# ============================================================
# 命令行入口：python -m tedm.viz.q1
# ============================================================
if __name__ == "__main__":
    from tedm.properties import APP2
    from tedm.solver import solve

    t, r, T_res, C_res, T_air, C_air = solve(prop=APP2, t_end=1800.0, dt=1.0)
    plot_all(t, r, T_res, C_res, T_air, C_air)
