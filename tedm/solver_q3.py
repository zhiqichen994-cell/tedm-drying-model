# ============================================================
# 问题3 求解器：跑到中心 C < 0.15 停止
# ------------------------------------------------------------
# 物理模型完全沿用问题2（附录3 变物性 + Arrhenius）
# 与问题2 的区别：
#   1. 时间上限改为 96h（安全上限）
#   2. 终点判据：中心 C < 0.15
#   3. 时间步 Δt = 60 s（不是 1 s）
# ============================================================

import numpy as np

import tedm.config as cfg
from tedm.model import build_grid, step_temperature, step_moisture
from tedm.data_loader import load_air_conditions
from tedm.properties import APP3


def solve_q3(prop=APP3, dt=60.0, C_target=0.15, t_max_h=96.0,
             use_latent=False, picard=2, n_cells=None, radius=None):
    """
    跑到中心含水率 < C_target 就停。

    返回：
        t:          (M+1,) 时间数组（s）
        r:          (N_R+1,) 径向节点
        T_result:   (M+1, N_R+1) 温度历史
        C_result:   (M+1, N_R+1) 水分历史
        T_air:      (M+1,) 烘房温度
        C_air:      (M+1,) 烘房水分浓度
        t_end_h:    烘干时间（h）
        end_center_C: 终点中心含水率
    """
    R = cfg.R if radius is None else float(radius)
    N = cfg.N_R if n_cells is None else int(n_cells)

    t_max = t_max_h * 3600.0
    # 先算好时间轴（附件1 之外用平台值）
    t, T_air, C_air = load_air_conditions(t_max, dt)
    r, r_face, v = build_grid(radius=R, n_cells=N)

    T = np.full(N + 1, cfg.T0)
    C = np.full(N + 1, cfg.C0)

    # 预分配（最多跑 n_max 步）
    n_max = int(round(t_max / dt))
    T_result = np.zeros((n_max + 1, N + 1))
    C_result = np.zeros((n_max + 1, N + 1))
    T_result[0] = T
    C_result[0] = C

    t_end_h = None
    end_center_C = None
    n_end = 0

    for n in range(n_max):
        T = step_temperature(T, C, r_face, v, dt,
                             T_air[n+1], C_air[n+1],
                             prop=prop, use_latent=use_latent, radius=R)
        C = step_moisture(C, T, r_face, v, dt,
                          C_air[n+1], prop=prop, picard=picard, radius=R)
        T_result[n+1] = T
        C_result[n+1] = C

        # 终点判据：中心 C < C_target
        if C[0] < C_target:
            # 线性插值精确定位
            C_prev = C_result[n, 0]
            C_curr = C[0]
            if C_prev > C_target:
                frac = (C_prev - C_target) / (C_prev - C_curr)
                t_end_h = (n * dt + frac * dt) / 3600.0
            else:
                t_end_h = (n + 1) * dt / 3600.0
            end_center_C = C[0]
            n_end = n + 1
            break

    # 截断到终点
    t = t[:n_end+1]
    T_air = T_air[:n_end+1]
    C_air = C_air[:n_end+1]
    T_result = T_result[:n_end+1]
    C_result = C_result[:n_end+1]

    return t, r, T_result, C_result, T_air, C_air, t_end_h, end_center_C


if __name__ == "__main__":
    import time
    t0 = time.time()

    t, r, T_result, C_result, T_air, C_air, t_end_h, end_C = solve_q3()

    print("=" * 60)
    print("问题3 自检")
    print("=" * 60)
    print(f"计算耗时: {time.time() - t0:.1f} s")
    print(f"烘干时间: {t_end_h:.4f} h ({t_end_h/24:.4f} 天)")
    print(f"终点中心含水率: {end_C:.6f} kg/kg")
    print(f"T_result.shape: {T_result.shape}")
    print(f"C_result.shape: {C_result.shape}")
    print(f"3h 中心水分: {C_result[int(3*3600/60), 0]:.4f}")
    print(f"3h 表面水分: {C_result[int(3*3600/60), -1]:.4f}")