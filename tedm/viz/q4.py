# ============================================================
# 问题4 可视化模块（10 张图）
# ------------------------------------------------------------
# 基础图：
#   图1  物理坐标水分场时空图（带白色"已收缩"区）
#   图2  归一化坐标水分场时空图
#   图3  中心/表面/平均水分曲线（双面板）
#   图4  R(t) 收缩曲线 + 收缩速率
#   图5  烘干终点判定
# 移动边界专属：
#   图6  拉格朗日材料点轨迹
#   图7  物理坐标 vs 归一化坐标映射
#   图8  收缩 vs 固定半径对照
# 高级：
#   图9  三重耦合全景
#   图10 三维时空曲面
# ============================================================

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import rcParams
from tedm.config import FIG_DIR, C0, R

rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
rcParams['axes.unicode_minus'] = False


# ---------- 辅助：体积平均 ----------
def _volume_weights_xi(xi_face, N):
    v = np.empty(N + 1)
    v[0] = (0.5 / N) ** 2
    v[1:N] = xi_face[1:] ** 2 - xi_face[:-1] ** 2
    v[N] = 1.0 - xi_face[-1] ** 2
    return v


# ============================================================
# 图1  物理坐标水分场时空图（带白色"已收缩"区）
# ============================================================
def plot_q4_1_C_phys(t, xi, C_result, R_result):
    r_output_cm = np.round(np.arange(0.0, 2.0 + 1e-9, 0.1), 4)
    M_out = len(t)
    C_out = np.full((M_out, len(r_output_cm)), np.nan)
    for n in range(M_out):
        R_n_cm = R_result[n] * 100.0
        for j, r_cm in enumerate(r_output_cm):
            if r_cm <= R_n_cm + 1e-9:
                xi_val = r_cm / R_n_cm
                C_out[n, j] = np.interp(xi_val, xi, C_result[n])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    t_h = t / 3600
    R_cm = R_result * 100

    im = ax.pcolormesh(t_h, r_output_cm, C_out.T,
                       shading='auto', cmap='viridis',
                       vmin=0, vmax=C0)

    # 白色覆盖已收缩区
    ax.fill_between(t_h, R_cm, 2.1, color='white', alpha=1.0, zorder=2)
    ax.plot(t_h, R_cm, 'r-', lw=2.5, label='R(t) 药材表面', zorder=3)

    # C=0.15 等值线
    cs = ax.contour(t_h, r_output_cm, C_out.T, levels=[0.15],
                    colors='red', linewidths=2, linestyles='--', zorder=4)
    ax.clabel(cs, fmt='C=0.15', fontsize=10)

    ax.set_xlabel('时间 / h')
    ax.set_ylabel('物理坐标 r / cm')
    ax.set_title('问题4 物理坐标水分场演化：白色区域已不属于药材')
    ax.set_ylim(0, 2.1)
    ax.legend(loc='upper right', fontsize=10)
    plt.colorbar(im, ax=ax, label=r'$C$ / (kg$\cdot$kg$^{-1}$)')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_q4_1_C_phys.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[图1] 物理坐标水分场: {FIG_DIR / 'fig_q4_1_C_phys.png'}")


# ============================================================
# 图2  归一化坐标水分场时空图
# ============================================================
def plot_q4_2_C_xi(t, xi, C_result):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    t_h = t / 3600

    im = ax.pcolormesh(t_h, xi, C_result.T,
                       shading='auto', cmap='RdYlBu_r',
                       vmin=0, vmax=C0)
    ax.axhline(1.0, color='r', ls='--', lw=2, label='$\\xi=1$ 表面')
    ax.axhline(0.0, color='k', ls='--', lw=2, label='$\\xi=0$ 中心')

    ax.set_xlabel('时间 / h')
    ax.set_ylabel('归一化坐标 $\\xi$')
    ax.set_title('问题4 归一化坐标水分场：计算域固定为 [0, 1]')
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper right', fontsize=10)
    plt.colorbar(im, ax=ax, label=r'$C$ / (kg$\cdot$kg$^{-1}$)')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_q4_2_C_xi.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[图2] 归一化坐标水分场: {FIG_DIR / 'fig_q4_2_C_xi.png'}")


# ============================================================
# 图3  中心/表面/平均水分曲线（双面板）
# ============================================================
def plot_q4_3_center_surface(t, xi, C_result, t_end_h):
    N = len(xi) - 1
    xi_face = (np.arange(N) + 0.5) / N
    v = _volume_weights_xi(xi_face, N)
    C_avg = np.array([np.sum(v * C_result[i]) / np.sum(v)
                      for i in range(len(t))])

    j_mid = int(round(0.5 * N))   # ξ = 0.5 附近

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(t / 3600, C_result[:, 0], 'b-', lw=2, label='中心 $\\xi=0$')
    axes[0].plot(t / 3600, C_result[:, j_mid], '--', color='tab:orange',
                 lw=1.5, label='$\\xi=0.5$')
    axes[0].plot(t / 3600, C_result[:, -1], 'r-', lw=2, label='表面 $\\xi=1$')
    axes[0].plot(t / 3600, C_avg, 'g--', lw=2, label='体积平均')
    axes[0].axhline(0.15, color='k', ls=':', lw=1.5, label='阈值 0.15')
    axes[0].axvline(t_end_h, color='m', ls='-.', lw=1.5,
                    label=f'终点 {t_end_h:.2f}h')
    axes[0].set_xlabel('时间 / h')
    axes[0].set_ylabel(r'含水率 $C$ / (kg$\cdot$kg$^{-1}$)')
    axes[0].set_title('(a) 全局：中心、中间、表面、平均曲线')
    axes[0].legend(fontsize=9, loc='upper right')
    axes[0].grid(alpha=0.3)

    t_min_h = max(0, t_end_h - 20)
    mask = t / 3600 >= t_min_h
    axes[1].plot(t[mask] / 3600, C_result[mask, 0], 'b-', lw=2,
                 label='中心 $\\xi=0$')
    axes[1].plot(t[mask] / 3600, C_result[mask, -1], 'r-', lw=2,
                 label='表面 $\\xi=1$')
    axes[1].axhline(0.15, color='k', ls=':', lw=1.5, label='阈值 0.15')
    axes[1].axvline(t_end_h, color='m', ls='-.', lw=1.5,
                    label=f'终点 {t_end_h:.2f}h')
    axes[1].set_xlabel('时间 / h')
    axes[1].set_ylabel(r'含水率 $C$ / (kg$\cdot$kg$^{-1}$)')
    axes[1].set_title(f'(b) 局部放大：最后 20h')
    axes[1].legend(fontsize=9, loc='upper right')
    axes[1].grid(alpha=0.3)

    plt.suptitle('问题4 中心与表面水分浓度时程及达标阈值', fontsize=12, y=1.00)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(FIG_DIR / 'fig_q4_3_center_surface.png', dpi=300,
                bbox_inches='tight')
    plt.close()
    print(f"[图3] 中心/表面曲线: {FIG_DIR / 'fig_q4_3_center_surface.png'}")


# ============================================================
# 图4  R(t) 收缩曲线 + 收缩速率
# ============================================================
def plot_q4_4_R_evolution(t, R_result):
    R_cm = R_result * 100.0
    t_h = t / 3600

    # 收缩速率（cm/h）
    dRdt = np.gradient(R_cm, t_h)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(t_h, R_cm, 'b-', lw=2.5, label='$R(t)$ 实测')
    axes[0].axhline(R_cm[0], color='gray', ls=':', lw=1.5,
                    label=f'$R_0={R_cm[0]:.2f}$ cm')
    axes[0].axhline(R_cm[-1], color='red', ls=':', lw=1.5,
                    label=f'$R_{{end}}={R_cm[-1]:.2f}$ cm')
    axes[0].set_xlabel('时间 / h')
    axes[0].set_ylabel('半径 $R$ / cm')
    axes[0].set_title('(a) 药材半径收缩曲线')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)

    axes[1].plot(t_h, -dRdt, 'r-', lw=2.5, label='收缩速率 $-dR/dt$')
    axes[1].set_xlabel('时间 / h')
    axes[1].set_ylabel('收缩速率 / (cm$\\cdot$h$^{-1}$)')
    axes[1].set_title('(b) 收缩速率随时间变化')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)

    plt.suptitle('问题4 半径收缩演化', fontsize=12, y=1.00)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(FIG_DIR / 'fig_q4_4_R_evolution.png', dpi=300,
                bbox_inches='tight')
    plt.close()
    print(f"[图4] R 收缩曲线: {FIG_DIR / 'fig_q4_4_R_evolution.png'}")


# ============================================================
# 图5  烘干终点判定
# ============================================================
def plot_q4_5_endpoint(t, xi, C_result, t_end_h):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(t / 3600, C_result[:, 0], 'b-', lw=2, label='中心 $\\xi=0$')
    ax.axhline(0.15, color='r', ls='--', lw=1.8, label='阈值 0.15')
    ax.axvline(t_end_h, color='g', ls=':', lw=1.8,
               label=f'烘干终点 {t_end_h:.4f} h')
    ax.set_xlabel('时间 / h')
    ax.set_ylabel(r'中心含水率 $C(0,t)$ / (kg$\cdot$kg$^{-1}$)')
    ax.set_title('问题4 烘干终点判定')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_q4_5_endpoint.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[图5] 终点判定: {FIG_DIR / 'fig_q4_5_endpoint.png'}")


# ============================================================
# 图6  拉格朗日材料点轨迹
# ============================================================
def plot_q4_6_material_trajectory(t, R_result):
    R_cm = R_result * 100.0
    t_h = t / 3600

    xi_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(xi_points)))

    fig, ax = plt.subplots(figsize=(10, 6))
    for xi_p, col in zip(xi_points, colors):
        r_track = xi_p * R_cm
        ax.plot(t_h, r_track, color=col, lw=2,
                label=f'$\\xi={xi_p:.2f}$')

    ax.plot(t_h, R_cm, 'k--', lw=2.5, label='$R(t)$ 表面')
    ax.axhline(2.0, color='r', ls=':', lw=1.5, label='初始 $r=2$ cm')

    ax.set_xlabel('时间 / h')
    ax.set_ylabel('物理位置 r / cm')
    ax.set_title('问题4 拉格朗日材料点轨迹：材料随骨架收缩向内运动')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_q4_6_material_trajectory.png', dpi=300,
                bbox_inches='tight')
    plt.close()
    print(f"[图6] 材料点轨迹: {FIG_DIR / 'fig_q4_6_material_trajectory.png'}")


# ============================================================
# 图7  物理坐标 vs 归一化坐标映射
# ============================================================
def plot_q4_7_coord_mapping(t, xi, C_result, R_result):
    r_output_cm = np.round(np.arange(0.0, 2.0 + 1e-9, 0.1), 4)
    M_out = len(t)
    C_out = np.full((M_out, len(r_output_cm)), np.nan)
    for n in range(M_out):
        R_n_cm = R_result[n] * 100.0
        for j, r_cm in enumerate(r_output_cm):
            if r_cm <= R_n_cm + 1e-9:
                xi_val = r_cm / R_n_cm
                C_out[n, j] = np.interp(xi_val, xi, C_result[n])

    t_h = t / 3600
    R_cm = R_result * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左：物理坐标
    ax1 = axes[0]
    im1 = ax1.pcolormesh(t_h, r_output_cm, C_out.T,
                         shading='auto', cmap='YlGnBu_r',
                         vmin=0, vmax=C0)
    ax1.plot(t_h, R_cm, 'r-', lw=2.5, label='$R(t)$')
    ax1.fill_between(t_h, R_cm, 2.1, color='white', alpha=1.0)
    ax1.plot(t_h, R_cm, 'r-', lw=2.5)
    ax1.set_xlabel('时间 / h')
    ax1.set_ylabel('物理坐标 r / cm')
    ax1.set_title('(a) 物理坐标：域随 $R(t)$ 收缩')
    ax1.set_ylim(0, 2.1)
    ax1.legend(loc='upper right')
    plt.colorbar(im1, ax=ax1, label=r'$C$ / (kg$\cdot$kg$^{-1}$)')

    # 右：归一化坐标
    ax2 = axes[1]
    im2 = ax2.pcolormesh(t_h, xi, C_result.T,
                         shading='auto', cmap='RdYlBu_r',
                         vmin=0, vmax=C0)
    ax2.axhline(1.0, color='r', ls='--', lw=2, label='$\\xi=1$')
    ax2.axhline(0.0, color='k', ls='--', lw=2, label='$\\xi=0$')
    ax2.set_xlabel('时间 / h')
    ax2.set_ylabel('归一化坐标 $\\xi$')
    ax2.set_title('(b) 归一化坐标：计算域固定为 [0,1]')
    ax2.set_ylim(0, 1.0)
    ax2.legend(loc='upper right')
    plt.colorbar(im2, ax=ax2, label=r'$C$ / (kg$\cdot$kg$^{-1}$)')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_q4_7_coord_mapping.png', dpi=300,
                bbox_inches='tight')
    plt.close()
    print(f"[图7] 坐标映射: {FIG_DIR / 'fig_q4_7_coord_mapping.png'}")


# ============================================================
# 图8  收缩 vs 固定半径对照
# ============================================================
def plot_q4_8_shrink_vs_fixed(t, xi, C_result, C_fixed, t_fixed,
                               t_end_h, t_end_fixed_h):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    axes[0].plot(t / 3600, C_result[:, 0], 'b-', lw=2, label='收缩模型')
    axes[0].plot(t_fixed / 3600, C_fixed[:, 0], 'r--', lw=2,
                 label='固定半径模型')
    axes[0].axhline(0.15, color='k', ls=':', lw=1.5, label='阈值 0.15')
    axes[0].axvline(t_end_h, color='b', ls=':', lw=1.5, alpha=0.5)
    axes[0].axvline(t_end_fixed_h, color='r', ls=':', lw=1.5, alpha=0.5)
    axes[0].set_xlabel('时间 / h')
    axes[0].set_ylabel(r'中心含水率 $C(0,t)$ / (kg$\cdot$kg$^{-1}$)')
    axes[0].set_title(f'(a) 中心含水率对比（收缩 {t_end_h:.1f}h vs 固定 {t_end_fixed_h:.1f}h）')
    axes[0].legend(fontsize=10, loc='upper right')
    axes[0].grid(alpha=0.3)

    axes[1].plot(t / 3600, C_result[:, -1], 'b-', lw=2, label='收缩模型')
    axes[1].plot(t_fixed / 3600, C_fixed[:, -1], 'r--', lw=2,
                 label='固定半径模型')
    axes[1].axhline(0.15, color='k', ls=':', lw=1.5, label='阈值 0.15')
    axes[1].set_xlabel('时间 / h')
    axes[1].set_ylabel(r'表面含水率 $C(R,t)$ / (kg$\cdot$kg$^{-1}$)')
    axes[1].set_title('(b) 表面含水率对比')
    axes[1].legend(fontsize=10, loc='upper right')
    axes[1].grid(alpha=0.3)

    plt.suptitle('问题4 收缩 vs 固定半径 对照', fontsize=12, y=1.00)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(FIG_DIR / 'fig_q4_8_shrink_vs_fixed.png', dpi=300,
                bbox_inches='tight')
    plt.close()
    print(f"[图8] 收缩 vs 固定: {FIG_DIR / 'fig_q4_8_shrink_vs_fixed.png'}")


# ============================================================
# 图9  三重耦合全景
# ============================================================
def plot_q4_9_triple_coupling(t, xi, T_result, C_result, R_result, t_end_h):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    t_h = t / 3600

    # 水分等值线（蓝色）
    cs1 = ax.contour(t_h, xi, C_result.T,
                     levels=[0.15, 0.3, 0.6, 1.0, 1.5, 2.0, 2.4],
                     colors='blue', linewidths=1.2, alpha=0.85)
    ax.clabel(cs1, inline=True, fontsize=8, fmt='%.2f')

    # 温度等值线（红色虚线）
    cs2 = ax.contour(t_h, xi, T_result.T,
                     levels=[30, 35, 40, 45, 49],
                     colors='red', linestyles='--',
                     linewidths=1.2, alpha=0.85)
    ax.clabel(cs2, inline=True, fontsize=8, fmt='%.0f')

    # 湿潮线（ξ 坐标）
    r_f = np.zeros(len(t))
    for i in range(len(t)):
        idx = np.where(C_result[i] > 0.15)[0]
        r_f[i] = xi[idx[-1]] if len(idx) > 0 else 0
    ax.plot(t_h, r_f, 'k-', lw=2.5, label='湿潮线 (C=0.15)')

    # 阶段切换线
    ax.axvline(0.5, color='gray', ls=':', lw=1.5, alpha=0.7,
               label='预热→恒温 (0.5h)')

    # 终点
    ax.axvline(t_end_h, color='green', ls='-.', lw=2.5,
               label=f'烘干终点 {t_end_h:.2f}h')

    ax.set_xlabel('时间 / h')
    ax.set_ylabel('归一化坐标 $\\xi$')
    ax.set_title('问题4 TEDM 完整层全景：收缩边界 + 湿潮推进 + 热波耦合')
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_q4_9_triple_coupling.png', dpi=300,
                bbox_inches='tight')
    plt.close()
    print(f"[图9] 三重耦合全景: {FIG_DIR / 'fig_q4_9_triple_coupling.png'}")


# ============================================================
# 图10  三维时空曲面
# ============================================================
def plot_q4_10_3d_surface(t, xi, T_result, C_result):
    t_h = t / 3600
    sample_step = max(1, len(t_h) // 120)
    t_s = t_h[::sample_step]
    C_s = C_result[::sample_step, :]
    T_s = T_result[::sample_step, :]

    # indexing='ij' gives an (nt, nxi) grid that matches C_s directly
    X_grid, Y_grid = np.meshgrid(t_s, xi, indexing='ij')

    fig = plt.figure(figsize=(16,7))

    # 左图 含水率
    ax1 = fig.add_subplot(121, projection="3d")
    surf1 = ax1.plot_surface(
        X_grid, Y_grid, C_s,
        cmap="viridis",
        edgecolor="none",
        alpha=0.95,
        vmin=0,
        vmax=2.55,
        rstride=1,
        cstride=1
    )
    ax1.set_xlabel("时间 / h")
    ax1.set_ylabel(r"归一化坐标 $\xi$")
    ax1.set_zlabel(r"含水率 $C$ / (kg/kg)")
    ax1.set_title("(a) 含水率时空曲面")
    ax1.view_init(elev=20, azim=-60)
    ax1.set_proj_type("ortho")   # orthographic: no perspective distortion

    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.grid(True, alpha=0.3)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, pad=0.12, label="$C$")

    # 右图 温度
    ax2 = fig.add_subplot(122, projection="3d")
    surf2 = ax2.plot_surface(
        X_grid, Y_grid, T_s,
        cmap="RdYlBu_r",
        edgecolor="none",
        alpha=0.95,
        vmin=30,
        vmax=50,
        rstride=1,
        cstride=1
    )
    ax2.set_xlabel("时间 / h")
    ax2.set_ylabel(r"归一化坐标 $\xi$")
    ax2.set_zlabel(r"温度 $T$ / $^\circ$C")
    ax2.set_title("(b) 温度时空曲面")
    ax2.view_init(elev=20, azim=-60)
    ax2.set_proj_type("ortho")

    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False
    ax2.grid(True, alpha=0.3)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, pad=0.12, label=r"$T/^\circ C$")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig10_3d.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[图10] 三维曲面: {FIG_DIR / 'fig_q4_10_3d_surface.png'}")

# ============================================================
# 统一入口
# ============================================================
def plot_all_q4(t, xi, T_result, C_result, R_result, t_end_h,
                C_fixed=None, t_fixed=None, t_end_fixed_h=None):
    print("=" * 60)
    print("问题4 图表生成中...")
    print("=" * 60)

    plot_q4_1_C_phys(t, xi, C_result, R_result)
    plot_q4_2_C_xi(t, xi, C_result)
    plot_q4_3_center_surface(t, xi, C_result, t_end_h)
    plot_q4_4_R_evolution(t, R_result)
    plot_q4_5_endpoint(t, xi, C_result, t_end_h)
    plot_q4_6_material_trajectory(t, R_result)
    plot_q4_7_coord_mapping(t, xi, C_result, R_result)
    if C_fixed is not None:
        plot_q4_8_shrink_vs_fixed(t, xi, C_result, C_fixed,
                                   t_fixed, t_end_h, t_end_fixed_h)
    plot_q4_9_triple_coupling(t, xi, T_result, C_result, R_result, t_end_h)
    plot_q4_10_3d_surface(t, xi, T_result, C_result)

    print("=" * 60)
    print("问题4 全部图表生成完成！")
    print(f"输出目录: {FIG_DIR}")
    print("=" * 60)


# ============================================================
# 命令行入口：python -m tedm.viz.q4
# ============================================================
if __name__ == "__main__":
    from tedm.solver_q4 import solve_q4
    from tedm.solver_q4_fixed import solve_q4_fixed

    out = solve_q4()
    t, xi, T_res, C_res, R_res, _T_air, _C_air, t_end_h = out
    t_fixed, _xi_f, _T_f, C_fixed, _t_end_fixed = solve_q4_fixed()
    plot_all_q4(t, xi, T_res, C_res, R_res, t_end_h,
                C_fixed=C_fixed, t_fixed=t_fixed)
