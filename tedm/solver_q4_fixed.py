# ============================================================
# 问题4 对照模型：固定半径（R=0.02m，物性用附录4）
# ============================================================

import numpy as np
from tedm.solver_q4 import build_grid_xi, step_temperature_q4, step_moisture_q4
from tedm.config import T0, C0, R, H_CONV, K_MASS
from tedm.properties import APP4
from tedm.data_loader import load_air_conditions


def solve_q4_fixed(N=100, dt=60.0, C_target=0.15, t_max_h=200.0, verbose=False):
    """Control experiment: same 附录4 physics but R held at 2 cm.

    Used to quantify how much of the drying time is bought by shrinkage
    (problem 4 compares 50.8651 h against this run's 129.1500 h).

    Note: the stopping time is quantised to whole steps (``(n+1)*dt``), so the
    reported value carries a +-dt/2 = +-0.0083 h uncertainty.  The
    shrinkage run in :func:`tedm.solver_q4.solve_q4` interpolates within the
    crossing step instead.
    """
    t_max = t_max_h * 3600.0
    t_air_arr, T_air_arr, C_air_arr = load_air_conditions(t_max, dt)

    xi, xi_face, v = build_grid_xi(N)
    T = np.full(N + 1, T0)
    C = np.full(N + 1, C0)

    n_max = int(round(t_max / dt))
    T_result = np.zeros((n_max + 1, N + 1))
    C_result = np.zeros((n_max + 1, N + 1))
    T_result[0] = T
    C_result[0] = C

    t_end_h = None
    n_end = 0
    R_fixed = R   # 0.02 m

    for n in range(n_max):
        T = step_temperature_q4(T, C, xi_face, v, R_fixed, dt,
                                T_air_arr[n+1], prop=APP4, use_latent=False)
        C = step_moisture_q4(C, T, xi_face, v, R_fixed, dt,
                             C_air_arr[n+1], prop=APP4, picard=2)
        T_result[n+1] = T
        C_result[n+1] = C

        if verbose and n % 1000 == 0:
            print(f"n={n}, t={(n+1)*dt/3600:.2f}h, C[0]={C[0]:.4f}")

        if C[0] < C_target:
            t_end_h = (n + 1) * dt / 3600.0
            n_end = n + 1
            break

    t_result = t_air_arr[:n_end + 1]
    T_result = T_result[:n_end + 1]
    C_result = C_result[:n_end + 1]

    return t_result, xi, T_result, C_result, t_end_h


if __name__ == "__main__":
    import time
    t0 = time.time()
    t, xi, T_r, C_r, t_end_h = solve_q4_fixed()
    print(f"计算耗时: {time.time()-t0:.1f} s")
    print(f"固定半径烘干时间: {t_end_h:.4f} h ({t_end_h/24:.4f} 天)")