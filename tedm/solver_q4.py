# ============================================================
# 问题4 求解器：移动边界（尺寸收缩）
# ------------------------------------------------------------
# 物理：附录4 物性 + R(t) 收缩（附件2）
# 方法：归一化坐标 ξ = r/R(t)，有限体积 + Kirchhoff + 后向 Euler
# 判据：中心 C < 0.15 停止
# ============================================================

import numpy as np
from tedm.config import T0, C0, R, H_CONV, K_MASS
from tedm.model import _solve_tridiag
from tedm.properties import APP4
from tedm.data_loader import load_air_conditions, load_radius


def build_grid_xi(N):
    """归一化坐标网格 ξ ∈ [0, 1]"""
    xi = np.linspace(0, 1, N + 1)
    xi_face = (np.arange(N) + 0.5) / N
    # 控制体权重（不含 2πR²，单独处理）
    v = np.empty(N + 1)
    v[0] = (0.5 / N) ** 2
    v[1:N] = xi_face[1:] ** 2 - xi_face[:-1] ** 2
    v[N] = 1.0 - xi_face[-1] ** 2
    return xi, xi_face, v


def step_temperature_q4(T, C, xi_face, v, R_now, dt, T_air,
                        prop=APP4, use_latent=False):
    """归一化坐标下的温度步（后向 Euler）。"""
    N = T.size - 1
    dxi = 1.0 / N

    rho_cp = prop.rho(C) * prop.cp(C)
    k_face = 0.5 * (prop.k(C[:-1]) + prop.k(C[1:]))

    # 物理体积（归一化权重 × R²）
    v_phys = v * R_now * R_now

    # 归一化坐标下界面导热：2·xi_face·k/dxi（不含 1/R²）
    Fk = 2.0 * xi_face * k_face / dxi

    lower = np.zeros(N + 1)
    diag = np.zeros(N + 1)
    upper = np.zeros(N + 1)
    rhs = np.zeros(N + 1)
    m = rho_cp * v_phys             # ← 用 v_phys

    diag[0] = m[0] + dt * Fk[0]
    upper[0] = -dt * Fk[0]
    rhs[0] = m[0] * T[0]

    lower[1:N] = -dt * Fk[:N-1]
    diag[1:N] = m[1:N] + dt * (Fk[:N-1] + Fk[1:N])
    upper[1:N] = -dt * Fk[1:N]
    rhs[1:N] = m[1:N] * T[1:N]

    a_surf = 2.0 * R_now * H_CONV
    lower[N] = -dt * Fk[N-1]
    diag[N] = m[N] + dt * (Fk[N-1] + a_surf)
    rhs[N] = m[N] * T[N] + dt * a_surf * T_air

    return _solve_tridiag(lower, diag, upper, rhs)

def step_moisture_q4(C, T, xi_face, v, R_now, dt, C_air,
                     prop=APP4, picard=2):
    """归一化坐标下的水分步（Kirchhoff 界面 D）。"""
    N = C.size - 1
    dxi = 1.0 / N
    a_surf = 2.0 * R_now * K_MASS

    # 物理体积（归一化权重 × R²）
    v_phys = v * R_now * R_now

    C_new = C.copy()
    for _ in range(picard):
        D_face = prop.face_D(C_new[:-1], C_new[1:], T[:-1], T[1:])
        Fd = 2.0 * xi_face * D_face / dxi     # 不含 1/R²

        lower = np.zeros(N + 1)
        diag = np.zeros(N + 1)
        upper = np.zeros(N + 1)
        rhs = np.zeros(N + 1)

        diag[0] = v_phys[0] + dt * Fd[0]      # ← 用 v_phys
        upper[0] = -dt * Fd[0]
        rhs[0] = v_phys[0] * C[0]

        lower[1:N] = -dt * Fd[:N-1]
        diag[1:N] = v_phys[1:N] + dt * (Fd[:N-1] + Fd[1:N])
        upper[1:N] = -dt * Fd[1:N]
        rhs[1:N] = v_phys[1:N] * C[1:N]

        lower[N] = -dt * Fd[N-1]
        diag[N] = v_phys[N] + dt * (Fd[N-1] + a_surf)
        rhs[N] = v_phys[N] * C[N] + dt * a_surf * C_air

        C_new = _solve_tridiag(lower, diag, upper, rhs)
    return C_new


def solve_q4(N=100, dt=60.0, C_target=0.15, t_max_h=96.0,
             use_latent=False, picard=2, verbose=False):
    """
    问题4：含尺寸收缩的移动边界求解。

    返回：
        t:           (M+1,) 时间数组（s）
        xi:          (N+1,) 归一化坐标
        T_result:    (M+1, N+1) 归一化温度历史
        C_result:    (M+1, N+1) 归一化水分历史
        R_result:    (M+1,) 半径历史（m）
        T_air_arr:   (M+1,) 烘房温度
        C_air_arr:   (M+1,) 烘房水分浓度
        t_end_h:     烘干时间（h）
    """
    # 读附件1、附件2
    t_max = t_max_h * 3600.0
    t_air_arr, T_air_arr, C_air_arr = load_air_conditions(t_max, dt)
    t_r, R_data, R_func = load_radius()  # R 单位 cm

    # 网格
    xi, xi_face, v = build_grid_xi(N)

    T = np.full(N + 1, T0)
    C = np.full(N + 1, C0)

    n_max = int(round(t_max / dt))
    T_result = np.zeros((n_max + 1, N + 1))
    C_result = np.zeros((n_max + 1, N + 1))
    R_result = np.zeros(n_max + 1)
    T_result[0] = T
    C_result[0] = C

    R0_m = R  # 初始半径 0.02 m = 2 cm（config.R 以米为单位）
    R_result[0] = R0_m

    t_end_h = None
    n_end = 0

    for n in range(n_max):

        t_new = (n + 1) * dt
        R_cm = float(R_func(t_new))
        R_m = R_cm / 100.0
        R_result[n + 1] = R_m

        T = step_temperature_q4(T, C, xi_face, v, R_m, dt,
                                T_air_arr[n+1], prop=APP4, use_latent=False)
        C = step_moisture_q4(C, T, xi_face, v, R_m, dt,
                             C_air_arr[n+1], prop=APP4, picard=picard)

        T_result[n+1] = T
        C_result[n+1] = C

        if verbose and n % 500 == 0:
            print(f"n={n}, t={(n + 1) * dt / 3600:.2f}h, "
                  f"C[0]={C[0]:.4f}, C[-1]={C[-1]:.4f}, R={R_m * 100:.3f}cm")

        if C[0] < C_target:
            C_prev = C_result[n, 0]
            C_curr = C[0]
            if C_prev > C_target:
                frac = (C_prev - C_target) / (C_prev - C_curr)
                t_end_h = (n * dt + frac * dt) / 3600.0
            else:
                t_end_h = (n + 1) * dt / 3600.0
            n_end = n + 1
            break

    # 截断
    t_result = t_air_arr[:n_end + 1]
    T_result = T_result[:n_end + 1]
    C_result = C_result[:n_end + 1]
    R_result = R_result[:n_end + 1]
    T_air_arr = T_air_arr[:n_end + 1]
    C_air_arr = C_air_arr[:n_end + 1]

    return t_result, xi, T_result, C_result, R_result, T_air_arr, C_air_arr, t_end_h


if __name__ == "__main__":
    import time
    t0 = time.time()
    t, xi, T_r, C_r, R_r, T_a, C_a, t_end_h = solve_q4()
    print(f"计算耗时: {time.time()-t0:.1f} s")
    print(f"烘干时间: {t_end_h:.4f} h ({t_end_h/24:.4f} 天)")
    print(f"R 范围: {R_r[0]*100:.4f} → {R_r[-1]*100:.4f} cm")
    print(f"T_result.shape: {T_r.shape}")
    print(f"3h 中心 C: {C_r[int(3*3600/60), 0]:.4f}")