# ============================================================
# 问题2 可视化模块
# ------------------------------------------------------------
# 生成 6 张图：
#   图1  温度时空云图（0~3h）
#   图2  水分时空云图（0~3h）
#   图3  中心/表面时程曲线（T + C 双轴）
#   图4  物性时空热力图（k 和 D 双面板）
#   图5  温度-水分相轨迹图（图11）
#   图6  变物性 vs 常物性对比（4 面板）
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tedm.config import FIG_DIR, T0, C0, R

rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
rcParams['axes.unicode_minus'] = False


# ============================================================
# 图1  温度时空云图（0~3h）
# ============================================================
def plot_q2_T_spacetime(t, r, T_result):
    r_cm = r * 100.0
    fig, ax = plt.subplots(figsize=(7, 5))

    im = ax.pcolormesh(t / 3600, r_cm, T_result.T,
                       shading='auto', cmap='inferno',
                       vmin=T0, vmax=T_result.max())
    ax.set_xlabel('时间 / h')
    ax.set_ylabel('到中心距离 / cm')
    ax.set_title('问题2 温度时空分布（0~3h）')
    c = plt.colorbar(im, ax=ax)
    c.set_label(r'$T$ / $^\circ$C')

    # 等值线
    levels = np.linspace(T0 + 2, T_result.max() - 1, 6)
    cs = ax.contour(t / 3600, r_cm, T_result.T,
                    levels=levels, colors='white',
                    linewidths=0.7, alpha=0.7)
    ax.clabel(cs, inline=True, fontsize=7, fmt='%.1f')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_q2_1_T_spacetime.png', dpi=300)
    plt.close()
    print(f"[Q2-图1] 温度时空云图: {FIG_DIR / 'fig_q2_1_T_spacetime.png'}")


# ============================================================
# 图2  水分时空云图（0~3h）
# ============================================================
def plot_q2_C_spacetime(t, r, C_result):
    r_cm = r * 100.0
    fig, ax = plt.subplots(figsize=(7, 5))

    im = ax.pcolormesh(t / 3600, r_cm, C_result.T,
                       shading='auto', cmap='viridis',
                       vmin=C_result.min(), vmax=C0)
    ax.set_xlabel('时间 / h')
    ax.set_ylabel('到中心距离 / cm')
    ax.set_title('问题2 水分浓度时空分布（0~3h）')
    c = plt.colorbar(im, ax=ax)
    c.set_label(r'$C$ / (kg$\cdot$kg$^{-1}$)')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_q2_2_C_spacetime.png', dpi=300)
    plt.close()
    print(f"[Q2-图2] 水分时空云图: {FIG_DIR / 'fig_q2_2_C_spacetime.png'}")


# ============================================================
# 图3  中心/表面时程曲线（T + C 双轴）
# ============================================================
def plot_q2_center_surface(t, r, T_result, C_result, T_air, C_air):
    fig, ax1 = plt.subplots(figsize=(8.5, 5))

    # 左轴：温度
    ax1.plot(t / 3600, T_result[:, 0], '-', color='tab:blue',
             lw=2, label='温度：中心 $r=0$')
    ax1.plot(t / 3600, T_result[:, -1], '-', color='tab:red',
             lw=2, label='温度：表面 $r=R$')
    ax1.plot(t / 3600, T_air, '--', color='gray',
             lw=1.2, alpha=0.7, label='烘房空气')
    ax1.set_xlabel('时间 / h')
    ax1.set_ylabel(r'温度 $T$ / $^\circ$C', color='k')
    ax1.grid(alpha=0.3)

    # 右轴：水分
    ax2 = ax1.twinx()
    ax2.plot(t / 3600, C_result[:, 0], '-', color='tab:green',
             lw=2, label='水分：中心 $r=0$')
    ax2.plot(t / 3600, C_result[:, -1], '-', color='tab:orange',
             lw=2, label='水分：表面 $r=R$')
    ax2.set_ylabel(r'水分浓度 $C$ / (kg$\cdot$kg$^{-1}$)', color='k')

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='lower right',
               bbox_to_anchor=(0.98, 0.15),  # ← 精确控制 (x, y)
               fontsize=9, framealpha=0.9)

    plt.title('问题2 中心与表面时程曲线')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_q2_3_center_surface.png', dpi=300)
    plt.close()
    print(f"[Q2-图3] 中心/表面时程: {FIG_DIR / 'fig_q2_3_center_surface.png'}")
# ============================================================
# 图4  物性时空热力图（k 和 D 双面板）
# ------------------------------------------------------------
# 横轴：时间；纵轴：径向距离
# 左面板：热传导系数 k(C)
# 右面板：水分扩散系数 D(C,T)
# 图注：物性随水分迁移在空间上形成梯度，表面物性先变，中心后变
# ============================================================
def plot_q2_property_spacetime(t, r, T_result, C_result):
    from tedm.properties import APP3

    r_cm = r * 100.0

    N_t, N_r = C_result.shape
    k_field = np.zeros_like(C_result)
    D_field = np.zeros_like(C_result)
    for i in range(N_t):
        k_field[i] = APP3.k(C_result[i])
        D_field[i] = APP3.D(C_result[i], T_result[i])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    #     (a) 热传导系数 k(r,t)
    im1 = axes[0].pcolormesh(t / 3600, r_cm, k_field.T,
                             shading='auto', cmap='plasma')
    axes[0].set_xlabel('时间 / h')
    axes[0].set_ylabel('到中心距离 / cm')
    axes[0].set_title(r'(a) 热传导系数 $k(r,t)$')
    c1 = plt.colorbar(im1, ax=axes[0])
    c1.set_label(r'$k$ / (W$\cdot$m$^{-1}\cdot$K$^{-1}$)')

    #      (b) 水分扩散系数 D(r,t)
    im2 = axes[1].pcolormesh(t / 3600, r_cm, D_field.T * 1e9,
                             shading='auto', cmap='viridis')
    axes[1].set_xlabel('时间 / h')
    axes[1].set_ylabel('到中心距离 / cm')
    axes[1].set_title(r'(b) 水分扩散系数 $D(r,t)$')
    c2 = plt.colorbar(im2, ax=axes[1])
    c2.set_label(r'$D$ / ($\times 10^{-9}$ m$^2\cdot$s$^{-1}$)')

    # ---------- 关键改动 1：suptitle 位置改成 y=0.99 ----------
    fig.suptitle('物性随水分迁移在空间上形成梯度：表面先变，中心后变',
                 fontsize=11, y=0.99)

    # ---------- 关键改动 2：tight_layout 用 rect 留出顶部空间 ----------
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    # ---------- 关键改动 3：保存用 bbox_inches='tight' 自动裁剪 ----------
    plt.savefig(FIG_DIR / 'fig_q2_4_property_spacetime.png', dpi=300,
                bbox_inches='tight')
    plt.close()
    print(f"[Q2-图4] 物性时空热力图: {FIG_DIR / 'fig_q2_4_property_spacetime.png'}")


# ============================================================
# 图5  温度-水分相轨迹图（图11）
# ------------------------------------------------------------
# 横轴：温度 T；纵轴：水分浓度 C
# 每个径向位置一条轨迹，从 (28, 2.55) 出发
# 表面：先升温，再失水；中心：几乎垂直升温
# ============================================================
def plot_q2_phase_trajectory(t, r, T_result, C_result,
                             r_list_cm=(0, 0.5, 1.0, 1.5, 2.0)):
    r_cm = r * 100.0
    fig, ax = plt.subplots(figsize=(7, 6))

    cmap = plt.cm.viridis(np.linspace(0, 1, len(r_list_cm)))
    for i, r_target in enumerate(r_list_cm):
        j = int(round(r_target / 0.02))   # N=100, Δr=0.02 cm
        T_traj = T_result[:, j]
        C_traj = C_result[:, j]
        ax.plot(T_traj, C_traj, color=cmap[i], lw=1.8,
                label=f'$r={r_target}$ cm')

    ax.set_xlabel(r'温度 $T$ / $^\circ$C')
    ax.set_ylabel(r'水分浓度 $C$ / (kg$\cdot$kg$^{-1}$)')
    ax.set_title('温度-水分相轨迹图（问题2）')
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(alpha=0.3)

    # 起始点标注
    ax.plot(T0, C0, 'k*', ms=12, label='起点 (28, 2.55)')
    ax.annotate('起点 (28, 2.55)', xy=(T0, C0),
                xytext=(T0 + 5, C0 - 0.05),
                arrowprops=dict(arrowstyle='->', color='k', lw=1),
                fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_q2_5_phase_trajectory.png', dpi=300)
    plt.close()
    print(f"[Q2-图5] 温度-水分相轨迹: {FIG_DIR / 'fig_q2_5_phase_trajectory.png'}")


# ============================================================
# 图6  变物性 vs 常物性对比（4 面板）
# ============================================================
def plot_q2_var_vs_con(t, r, T_var, C_var, T_con, C_con):
    r_cm = r * 100.0
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) 温度差（变 - 常）
    dT = T_var - T_con
    im1 = axes[0, 0].pcolormesh(t / 3600, r_cm, dT.T,
                                shading='auto', cmap='RdBu_r',
                                vmin=-abs(dT).max(), vmax=abs(dT).max())
    axes[0, 0].set_xlabel('时间 / h')
    axes[0, 0].set_ylabel('到中心距离 / cm')
    axes[0, 0].set_title(r'(a) 温度差 $T_{var} - T_{con}$ / $^\circ$C')
    plt.colorbar(im1, ax=axes[0, 0])

    # (b) 水分差（变 - 常）
    dC = C_var - C_con
    im2 = axes[0, 1].pcolormesh(t / 3600, r_cm, dC.T,
                                shading='auto', cmap='RdBu_r',
                                vmin=-abs(dC).max(), vmax=abs(dC).max())
    axes[0, 1].set_xlabel('时间 / h')
    axes[0, 1].set_ylabel('到中心距离 / cm')
    axes[0, 1].set_title(r'(b) 水分差 $C_{var} - C_{con}$ / (kg$\cdot$kg$^{-1}$)')
    plt.colorbar(im2, ax=axes[0, 1])

    # (c) 3h 时温度剖面
    axes[1, 0].plot(r_cm, T_var[10800], 'r-', lw=1.8, label='变物性')
    axes[1, 0].plot(r_cm, T_con[10800], 'b--', lw=1.8, label='常物性')
    axes[1, 0].set_xlabel('到中心距离 / cm')
    axes[1, 0].set_ylabel(r'温度 $T$ / $^\circ$C')
    axes[1, 0].set_title('(c) 3h 温度剖面')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # (d) 3h 时水分剖面
    axes[1, 1].plot(r_cm, C_var[10800], 'r-', lw=1.8, label='变物性')
    axes[1, 1].plot(r_cm, C_con[10800], 'b--', lw=1.8, label='常物性')
    axes[1, 1].set_xlabel('到中心距离 / cm')
    axes[1, 1].set_ylabel(r'水分浓度 $C$ / (kg$\cdot$kg$^{-1}$)')
    axes[1, 1].set_title('(d) 3h 水分剖面')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.suptitle('变物性（附录3） vs 常物性（附录2） 对比', fontsize=12, y=1.00)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_q2_6_var_vs_con.png', dpi=300)
    plt.close()
    print(f"[Q2-图6] 变物性 vs 常物性: {FIG_DIR / 'fig_q2_6_var_vs_con.png'}")

# ============================================================
# 图7  多物理场叠加图（王炸图，对标国奖核心亮点）
# ------------------------------------------------------------
# 温度等值线（红实线）+ 水分等值线（蓝虚线）+ 湿潮线（黑粗线）
# 一图展示：热波先行、湿潮滞后、热湿不同步
# 对应论文：图X 多物理场耦合演化
# 输出：fig_q2_7_multiphysics.png
# ============================================================
def plot_q2_multiphysics(t, r, T_result, C_result, T0=28.0, C0=2.55):
    r_cm = r * 100.0
    fig, ax = plt.subplots(figsize=(10, 6))

    # ---------- 温度等值线（红实线） ----------
    T_levels = np.linspace(T0 + 2, T_result.max() - 1, 6)
    cs1 = ax.contour(t / 3600, r_cm, T_result.T,
                     levels=T_levels, colors='red',
                     linewidths=1.3, alpha=0.85)
    ax.clabel(cs1, inline=True, fontsize=7, fmt='%.0f')

    # ---------- 水分等值线（蓝虚线） ----------
    C_levels = np.linspace(C_result.min() + 0.1, C0 - 0.1, 6)
    cs2 = ax.contour(t / 3600, r_cm, C_result.T,
                     levels=C_levels, colors='blue',
                     linestyles='--', linewidths=1.3, alpha=0.85)
    ax.clabel(cs2, inline=True, fontsize=7, fmt='%.1f')

    # ---------- 湿潮线：水分显著变化的最内位置 ----------
    # 定义：C 相对初始值下降超过 0.01 的最内半径
    eps = 0.01
    r_f = np.zeros(len(t))
    for i in range(len(t)):
        idx = np.where(C0 - C_result[i] > eps)[0]
        if idx.size:
            r_f[i] = r_cm[idx[0]]     # 最靠内的"已变"位置
        else:
            r_f[i] = r_cm[-1]         # 表面

    ax.plot(t / 3600, r_f, 'k-', lw=2.5, label='湿潮线 $r_f(t)$')

    # ---------- 阶段切换线（0.5h，烘房温度趋于恒定） ----------
    ax.axvline(0.5, color='green', ls=':', lw=1.5, alpha=0.7,
               label='预热→恒温切换')

    ax.set_xlabel('时间 / h', fontsize=11)
    ax.set_ylabel('到中心距离 / cm', fontsize=11)
    ax.set_title('问题2 多物理场耦合演化：热波先行，湿潮滞后',
                 fontsize=12)
    ax.legend(loc='center right', fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_q2_7_multiphysics.png', dpi=300,
                bbox_inches='tight')
    plt.close()
    print(f"[Q2-图7] 多物理场叠加: {FIG_DIR / 'fig_q2_7_multiphysics.png'}")


# ============================================================
# 图8  热湿耦合强度图（高级）
# ------------------------------------------------------------
# 横轴：时间；纵轴：径向距离
# 颜色：Γ(r,t) = |∂T/∂t| / |∂C/∂t|
#   红区：温度变化快（热主导）
#   蓝区：水分变化快（湿主导）
# 对应论文：图X 热湿耦合强度场
# 输出：fig_q2_8_coupling_intensity.png
# ============================================================
# ============================================================
# 图8  温度距平衡差 + 水分累积失水量（双面板）
# ------------------------------------------------------------
# 左：距平衡温度差 |T_air(t) - T(r,t)|
# 右：累积失水量 ΔC = C0 - C(r,t)
# 展示：温度趋平衡快、水分失水从表面向内推进
# ============================================================
def plot_q2_coupling_intensity(t, r, T_result, C_result, T_air=None):
    from matplotlib.colors import LogNorm
    from tedm.config import C0

    r_cm = r * 100.0

    # (a) 距平衡温度差
    if T_air is not None:
        dT = np.abs(T_result - T_air[:, None])
    else:
        dT = np.abs(T_result.max() - T_result)

    # (b) 累积失水量 ΔC = C0 - C(r,t)
    dC_cum = C0 - C_result   # 正值 = 已失水

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) 距平衡温度差（对数色标）
    im1 = axes[0].pcolormesh(t / 3600, r_cm, dT.T,
                             shading='auto', cmap='inferno',
                             norm=LogNorm(vmin=1e-2, vmax=5e1))
    axes[0].set_xlabel('时间 / h')
    axes[0].set_ylabel('到中心距离 / cm')
    axes[0].set_title(r'(a) 距平衡温度差 $|T_{air}(t) - T(r,t)|$')
    c1 = plt.colorbar(im1, ax=axes[0])
    c1.set_label(r'$|T_{air} - T|$ / $^\circ$C')

    # (b) 累积失水量（线性色标）
    im2 = axes[1].pcolormesh(t / 3600, r_cm, dC_cum.T,
                             shading='auto', cmap='viridis',
                             vmin=0, vmax=dC_cum.max())
    axes[1].set_xlabel('时间 / h')
    axes[1].set_ylabel('到中心距离 / cm')
    axes[1].set_title(r'(b) 累积失水量 $\Delta C = C_0 - C(r,t)$')
    c2 = plt.colorbar(im2, ax=axes[1])
    c2.set_label(r'$\Delta C$ / (kg$\cdot$kg$^{-1}$)')

    fig.suptitle('温度趋平衡快、水分失水从表面向内推进', fontsize=12, y=1.00)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(FIG_DIR / 'fig_q2_8_coupling_intensity.png', dpi=300,
                bbox_inches='tight')
    plt.close()
    print(f"[图8] 距平衡温度差 + 累积失水量: {FIG_DIR / 'fig_q2_8_coupling_intensity.png'}")
# ============================================================
# 图9  干燥速率曲线与阶段切换
# ------------------------------------------------------------
# 横轴：时间；纵轴：体积平均干燥速率 -dC̄/dt
# 标注：预热阶段 / 恒温阶段切换点（0.5h）
# 对应论文：图X 干燥速率曲线
# 输出：fig_q2_9_drying_rate.png
# ============================================================

def plot_q2_drying_rate(t, r, C_result):
    """体积平均干燥速率随时间的变化"""
    # 体积平均权重（除去 2πL）
    N = len(r) - 1
    dr = r[1] - r[0]
    r_face = (np.arange(N) + 0.5) * dr
    v = np.empty(N + 1)
    v[0] = (dr / 2.0) ** 2
    v[1:N] = r_face[1:] ** 2 - r_face[:-1] ** 2
    v[N] = r[-1] ** 2 - r_face[-1] ** 2

    # 体积平均含水率
    C_avg = np.array([np.sum(v * C_result[i]) / np.sum(v)
                      for i in range(len(t))])

    # 干燥速率（正值）
    dCdt = -np.gradient(C_avg, t)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(t / 3600, dCdt, 'b-', lw=2.2,
            label=r'干燥速率 $-d\bar{C}/dt$')
    ax.axvline(0.5, color='r', ls='--', lw=1.6,
               label='预热→恒温切换 (0.5 h)')
    ax.set_xlabel('时间 / h', fontsize=11)
    ax.set_ylabel(r'干燥速率 $-d\bar{C}/dt$ / (kg$\cdot$kg$^{-1}\cdot$s$^{-1}$)',
                  fontsize=11)
    ax.set_title('问题2 干燥速率曲线与阶段切换', fontsize=12)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig_q2_9_drying_rate.png', dpi=300,
                bbox_inches='tight')
    plt.close()
    print(f"[图9] 干燥速率曲线: {FIG_DIR / 'fig_q2_9_drying_rate.png'}")
# ============================================================
# 统一入口
# ============================================================
def plot_all_q2(t, r, T_result, C_result, T_air, C_air,
                T_var=None, C_var=None, T_con=None, C_con=None):
    print("=" * 60)
    print("问题2 图表生成中...")
    print("=" * 60)

    # 基础图
    plot_q2_T_spacetime(t, r, T_result)
    plot_q2_C_spacetime(t, r, C_result)
    plot_q2_center_surface(t, r, T_result, C_result, T_air, C_air)

    plot_q2_property_spacetime(t, r, T_result, C_result)
    plot_q2_phase_trajectory(t, r, T_result, C_result)

    # 亮点图
    if T_var is not None:
        plot_q2_var_vs_con(t, r, T_var, C_var, T_con, C_con)

    # 加分图（对标国奖）
    plot_q2_multiphysics(t, r, T_result, C_result)
    plot_q2_coupling_intensity(t, r, T_result, C_result)
    plot_q2_drying_rate(t, r, C_result)

    print("=" * 60)
    print("问题2 全部图表生成完成！")
    print(f"输出目录: {FIG_DIR}")
    print("=" * 60)


# ============================================================
# 命令行入口：python -m tedm.viz.q2
# ============================================================
if __name__ == "__main__":
    from tedm.solver_q2 import solve_q2

    t, r, T_res, C_res, T_air, C_air = solve_q2()
    plot_all_q2(t, r, T_res, C_res, T_air, C_air)
