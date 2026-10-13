# ============================================================
# 问题3 可视化模块（完整版）
# ------------------------------------------------------------
# 9 张图：
#   图1  长时间水分场热力图
#   图2  中心/表面/平均水分曲线（双面板：全局 + 局部放大）
#   图3  烘干终点判定图
#   图4  水分剖面瀑布图（每 6h）
#   图5  干燥速率（双面板：vs 时间 + vs 含水率）
#   图6  湿潮线演化图
#   图7  水分达标地图（二元图）
#   图8  干燥前沿到达时间图
#   图9  TEDM 退潮全景图
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tedm.config import FIG_DIR, C0

rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
rcParams['axes.unicode_minus'] = False


# ---------- 辅助：体积平均权重 ----------
def _volume_weights(r):
    N = len(r) - 1
    dr = r[1] - r[0]
    r_face = (np.arange(N) + 0.5) * dr
    v = np.empty(N + 1)
    v[0] = (dr / 2.0) ** 2
    v[1:N] = r_face[1:] ** 2 - r_face[:-1] ** 2
    v[N] = r[-1] ** 2 - r_face[-1] ** 2
    return v


# ============================================================
# 图1  长时间水分场热力图
# ============================================================
def plot_q3_C_spacetime(t, r, C_result):
    r_cm = r * 100.0
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.pcolormesh(t / 3600, r_cm, C_result.T,
                       shading='auto', cmap='viridis',
                       vmin=0, vmax=C0)
    ax.set_xlabel('时间 / h')
    ax.set_ylabel('到中心距离 / cm')
    ax.set_title('问题3 水分浓度时空分布（0~烘干结束）')
    c = plt.colorbar(im, ax=ax)
    c.set_label(r'$C$ / (kg$\cdot$kg$^{-1}$)')
    # 不调 tight_layout，用 bbox_inches='tight' 自动裁剪
    plt.savefig(FIG_DIR / 'fig_q3_1_C_long.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Q3-图1] 长时间水分热图: {FIG_DIR / 'fig_q3_1_C_long.png'}")


# ============================================================
# 图2  中心/表面/平均水分曲线（双面板）
# ------------------------------------------------------------
# 左：全局曲线（0~终点）
# 右：局部放大（最后 24h）
# ============================================================
def plot_q3_center_surface(t, r, C_result, t_end_h):
    v = _volume_weights(r)
    C_avg = np.array([np.sum(v * C_result[i]) / np.sum(v)
                      for i in range(len(t))])

    # r=1cm 中间曲线索引（N=100，Δr=0.02cm）
    j_mid = int(round(1.0 / 0.02))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---- (a) 全局 ----
    axes[0].plot(t / 3600, C_result[:, 0], 'b-', lw=2, label='中心 $r=0$')
    axes[0].plot(t / 3600, C_result[:, j_mid], '--', color='tab:orange',
                 lw=1.5, label='$r=1.0$ cm')
    axes[0].plot(t / 3600, C_result[:, -1], 'r-', lw=2, label='表面 $r=R$')
    axes[0].plot(t / 3600, C_avg, 'g--', lw=2, label='体积平均')
    axes[0].axhline(0.15, color='k', ls=':', lw=1.5, label='阈值 $C=0.15$')
    axes[0].axvline(t_end_h, color='m', ls='-.', lw=1.5,
                    label=f'终点 {t_end_h:.2f}h')
    axes[0].set_xlabel('时间 / h')
    axes[0].set_ylabel(r'水分浓度 $C$ / (kg$\cdot$kg$^{-1}$)')
    axes[0].set_title('(a) 全局：中心、表面、体积平均水分曲线')
    axes[0].legend(fontsize=9, loc='lower right',bbox_to_anchor=(0.95, 0.20))
    axes[0].grid(alpha=0.3)

    # ---- (b) 局部放大：最后 24h ----
    t_min_h = max(0, t_end_h - 24)
    mask = t / 3600 >= t_min_h
    axes[1].plot(t[mask] / 3600, C_result[mask, 0], 'b-', lw=2,
                 label='中心 $r=0$')
    axes[1].plot(t[mask] / 3600, C_result[mask, j_mid], '--',
                 color='tab:orange', lw=1.5, label='$r=1.0$ cm')
    axes[1].plot(t[mask] / 3600, C_result[mask, -1], 'r-', lw=2,
                 label='表面 $r=R$')
    axes[1].axhline(0.15, color='k', ls=':', lw=1.5, label='阈值 0.15')
    axes[1].axvline(t_end_h, color='m', ls='-.', lw=1.5,
                    label=f'终点 {t_end_h:.2f}h')
    axes[1].set_xlabel('时间 / h')
    axes[1].set_ylabel(r'水分浓度 $C$ / (kg$\cdot$kg$^{-1}$)')
    axes[1].set_title(f'(b) 局部放大：最后 24h')
    axes[1].legend(fontsize=9, loc='lower right', bbox_to_anchor=(0.95, 0.20))
    axes[1].grid(alpha=0.3)

    plt.suptitle('问题3 中心与表面水分浓度时程及达标阈值',
                 fontsize=12, y=1.00)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(FIG_DIR / 'fig_q3_2_center_surface.png', dpi=300,
                bbox_inches='tight')
    plt.close()
    print(f"[Q3-图2] 中心/表面/平均: {FIG_DIR / 'fig_q3_2_center_surface.png'}")


# ============================================================
# 图3  烘干终点判定图
# ============================================================
def plot_q3_endpoint(t, r, C_result, t_end_h):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(t / 3600, C_result[:, 0], 'b-', lw=2, label='中心 $r=0$')
    ax.axhline(0.15, color='r', ls='--', lw=1.8, label='阈值 0.15')
    ax.axvline(t_end_h, color='g', ls=':', lw=1.8,
               label=f'烘干终点 {t_end_h:.4f} h')
    ax.set_xlabel('时间 / h')
    ax.set_ylabel(r'中心含水率 $C(0,t)$ / (kg$\cdot$kg$^{-1}$)')
    ax.set_title('问题3 烘干终点判定')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_q3_3_endpoint.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Q3-图3] 烘干终点判定: {FIG_DIR / 'fig_q3_3_endpoint.png'}")


# ============================================================
# 图4  水分剖面瀑布图（每 6h）
# ============================================================
def plot_q3_profiles(t, r, C_result, t_end_h):
    r_cm = r * 100.0
    times_h = list(range(0, int(t_end_h), 6)) + [round(t_end_h, 4)]
    cmap = plt.cm.viridis(np.linspace(0, 1, len(times_h)))

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for i, h in enumerate(times_h):
        idx = min(int(h * 3600 / 60), len(t) - 1)
        ax.plot(r_cm, C_result[idx], color=cmap[i], lw=1.8,
                label=f'{h:.1f} h')

    ax.axhline(0.15, color='r', ls='--', lw=1.5, label='阈值 0.15')
    ax.set_yscale('log')          # ← 关键：改成对数纵轴
    ax.set_ylim(0.05, 3)          # ← 范围 0.05~3

    ax.set_xlabel('到中心距离 / cm')
    ax.set_ylabel(r'水分浓度 $C$ / (kg$\cdot$kg$^{-1}$)')
    ax.set_title('问题3 水分剖面瀑布图（对数纵轴）')
    ax.legend(fontsize=8, ncol=2, loc='upper right',bbox_to_anchor=(0.98, 0.95))
    ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_q3_4_profiles.png', dpi=300,
                bbox_inches='tight')
    plt.close()
    print(f"[图4] 水分剖面瀑布: {FIG_DIR / 'fig_q3_4_profiles.png'}")

# ============================================================
# 图5  干燥速率（双面板：vs 时间 + vs 含水率）
# ============================================================
def plot_q3_drying_rate(t, r, C_result, t_end_h):
    v = _volume_weights(r)
    C_avg = np.array([np.sum(v * C_result[i]) / np.sum(v)
                      for i in range(len(t))])
    dCdt = -np.gradient(C_avg, t)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---- (a) 干燥速率 vs 时间 ----
    axes[0].plot(t / 3600, dCdt, 'b-', lw=2, label=r'$-d\bar{C}/dt$')
    axes[0].axvline(0.5, color='r', ls='--', lw=1.5,
                    label='预热→恒温 (0.5h)')
    axes[0].axvline(t_end_h, color='m', ls='-.', lw=1.5,
                    label=f'终点 {t_end_h:.2f}h')
    axes[0].set_xlabel('时间 / h')
    axes[0].set_ylabel(
        r'干燥速率 $-d\bar{C}/dt$ / (kg$\cdot$kg$^{-1}\cdot$s$^{-1}$)')
    axes[0].set_title('(a) 干燥速率随时间变化')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # ---- (b) 干燥速率 vs 含水率（对数纵轴）----
    axes[1].semilogy(C_avg, dCdt, 'b-', lw=2)
    axes[1].scatter(C_avg[0], dCdt[0], color='r', s=80, zorder=5,
                    label=f'起点 $C_0={C_avg[0]:.2f}$')
    axes[1].axvline(0.15, color='r', ls=':', lw=1.5, label='阈值 0.15')
    axes[1].set_xlabel(
        r'截面平均含水率 $\bar{C}$ / (kg$\cdot$kg$^{-1}$)')
    axes[1].set_ylabel(
        r'干燥速率 $-d\bar{C}/dt$ / (kg$\cdot$kg$^{-1}\cdot$s$^{-1}$)')
    axes[1].set_title('(b) 干燥速率随含水率变化（干燥特性曲线）')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3, which='both')
    axes[1].invert_xaxis()   # 含水率从高到低

    plt.suptitle('问题3 干燥速率随时间和含水率的变化',
                 fontsize=12, y=1.00)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(FIG_DIR / 'fig_q3_5_drying_rate.png', dpi=300,
                bbox_inches='tight')
    plt.close()
    print(f"[Q3-图5] 干燥速率: {FIG_DIR / 'fig_q3_5_drying_rate.png'}")


# ============================================================
# 图6  湿潮线演化图
# ============================================================
def plot_q3_front(t, r, C_result):
    r_cm = r * 100.0
    n_t = len(t)
    r_f = np.zeros(n_t)
    for i in range(n_t):
        idx = np.where(C_result[i] > 0.15)[0]
        if idx.size:
            r_f[i] = r_cm[idx[-1]]
        else:
            r_f[i] = 0.0

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(t / 3600, r_f, 'k-', lw=2.5, label='湿潮线 $r_f(t)$')
    ax.axhline(0, color='r', ls='--', lw=1.5, label='中心 $r=0$')
    ax.set_xlabel('时间 / h')
    ax.set_ylabel('湿潮线位置 / cm')
    ax.set_title('问题3 湿潮线演化：从表面向中心推进')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_q3_6_front.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Q3-图6] 湿潮线演化: {FIG_DIR / 'fig_q3_6_front.png'}")


# ============================================================
# 图7  水分达标地图（二元图）
# ============================================================
def plot_q3_map(t, r, C_result):
    r_cm = r * 100.0
    fig, ax = plt.subplots(figsize=(8, 5))

    im = ax.pcolormesh(t / 3600, r_cm, C_result.T,
                       shading='auto', cmap='RdYlGn_r',
                       vmin=0, vmax=0.3)
    # 等值线标出 C=0.15
    cs = ax.contour(t / 3600, r_cm, C_result.T,
                    levels=[0.15], colors='black',
                    linewidths=2.0)
    ax.clabel(cs, inline=True, fontsize=10, fmt='C=%.2f')

    ax.set_xlabel('时间 / h')
    ax.set_ylabel('到中心距离 / cm')
    ax.set_title('问题3 水分达标等值线（C = 0.15）')
    c = plt.colorbar(im, ax=ax)
    c.set_label(r'$C$ / (kg$\cdot$kg$^{-1}$)')
    plt.savefig(FIG_DIR / 'fig_q3_7_map.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[图7] 水分达标等值线: {FIG_DIR / 'fig_q3_7_map.png'}")
# ============================================================
# 图8  干燥前沿到达时间图
# ============================================================
def plot_q3_first_passage(t, r, C_result):
    r_cm = r * 100.0
    n_t, n_r = C_result.shape
    t_arrive = np.full(n_r, np.nan)
    for j in range(n_r):
        idx = np.where(C_result[:, j] <= 0.15)[0]
        if idx.size:
            t_arrive[j] = t[idx[0]] / 3600.0

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(r_cm, t_arrive, 'bo-', lw=2, ms=6)
    ax.set_xlabel('到中心距离 / cm')
    ax.set_ylabel('首次达标时间 / h')
    ax.set_title('问题3 干燥前沿到达时间')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_q3_8_first_passage.png', dpi=300,
                bbox_inches='tight')
    plt.close()
    print(f"[Q3-图8] 干燥前沿到达时间: {FIG_DIR / 'fig_q3_8_first_passage.png'}")


# ============================================================
# 图9  TEDM 退潮全景图
# ============================================================
def plot_q3_tedm_panorama(t, r, T_result, C_result, t_end_h):
    r_cm = r * 100.0
    fig, ax = plt.subplots(figsize=(10, 6))

    # 温度等值线（红实线）
    T_levels = [30, 35, 40, 45, 49]
    cs1 = ax.contour(t / 3600, r_cm, T_result.T,
                     levels=T_levels, colors='red',
                     linewidths=1.3, alpha=0.85)
    ax.clabel(cs1, inline=True, fontsize=7, fmt='%.0f')

    # 水分等值线（蓝虚线）
    C_levels = [0.15, 0.5, 1.0, 1.5, 2.0, 2.4]
    cs2 = ax.contour(t / 3600, r_cm, C_result.T,
                     levels=C_levels, colors='blue',
                     linestyles='--', linewidths=1.3, alpha=0.85)
    ax.clabel(cs2, inline=True, fontsize=7, fmt='%.1f')

    # 湿潮线
    n_t = len(t)
    r_f = np.zeros(n_t)
    for i in range(n_t):
        idx = np.where(C_result[i] > 0.15)[0]
        r_f[i] = r_cm[idx[-1]] if idx.size else 0.0
    ax.plot(t / 3600, r_f, 'k-', lw=2.5, label='湿潮线 $r_f(t)$')

    # 阶段切换 + 终点
    ax.axvline(0.5, color='green', ls=':', lw=1.5,
               label='预热→恒温 (0.5h)')
    ax.axvline(t_end_h, color='magenta', ls='-.', lw=1.8,
               label=f'烘干终点 {t_end_h:.2f}h')

    # ---------- 对数时间轴 + 手动刻度 ----------
    ax.set_xscale('log')
    ax.set_xlim(0.1, t_end_h * 1.2)
    xticks = [0.1, 0.5, 1, 2, 5, 10, 20, 50]
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks])

    ax.set_xlabel('时间 / h（对数轴）')
    ax.set_ylabel('到中心距离 / cm')
    ax.set_title('问题3 TEDM 退潮全景：热波先行、湿潮滞后、中心定终点')
    ax.legend(loc='lower right', fontsize=9,bbox_to_anchor=(0.95, 0.05))
    ax.grid(alpha=0.2, which='both')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_q3_9_tedm_panorama.png', dpi=300,
                bbox_inches='tight')
    plt.close()
    print(f"[图9] TEDM 退潮全景: {FIG_DIR / 'fig_q3_9_tedm_panorama.png'}")
# ============================================================
# 统一入口
# ============================================================
def plot_all_q3(t, r, T_result, C_result, t_end_h):
    print("=" * 60)
    print("问题3 图表生成中...")
    print("=" * 60)
    plot_q3_C_spacetime(t, r, C_result)
    plot_q3_center_surface(t, r, C_result, t_end_h)
    plot_q3_endpoint(t, r, C_result, t_end_h)
    plot_q3_profiles(t, r, C_result, t_end_h)
    plot_q3_drying_rate(t, r, C_result, t_end_h)
    plot_q3_front(t, r, C_result)
    plot_q3_map(t, r, C_result)
    plot_q3_first_passage(t, r, C_result)
    plot_q3_tedm_panorama(t, r, T_result, C_result, t_end_h)
    print("=" * 60)
    print("问题3 基础图表生成完成！")
    print(f"输出目录: {FIG_DIR}")
    print("=" * 60)


# ============================================================
# 命令行入口：python -m tedm.viz.q3
# ============================================================
if __name__ == "__main__":
    from tedm.solver_q3 import solve_q3

    t, r, T_res, C_res, _T_air, _C_air, t_end_h, _end_C = solve_q3()
    plot_all_q3(t, r, T_res, C_res, t_end_h)
