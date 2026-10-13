# ============================================================
# 问题4 结果输出：result4.xlsx + 表6
# ============================================================

import numpy as np
import pandas as pd
from tedm.config import OUTPUT_DIR, TABLE_DIR, DECIMALS


def save_result4(t, xi, C_result, R_result, t_end_h):
    """写 output/result4.xlsx（物理坐标 + 表面列）"""
    file_path = OUTPUT_DIR / "result4.xlsx"
    r_output_cm = np.round(np.arange(0.0, 2.0 + 1e-9, 0.1), 4)

    M_out = len(t)
    C_out = np.full((M_out, len(r_output_cm)), np.nan)

    for n in range(M_out):
        R_n_m = R_result[n]
        R_n_cm = R_n_m * 100.0
        for j, r_cm in enumerate(r_output_cm):
            if r_cm <= R_n_cm + 1e-9:
                xi_val = r_cm / R_n_cm
                C_out[n, j] = np.interp(xi_val, xi, C_result[n])

    # 加"药材表面"列
    C_surface = C_result[:, -1]   # 归一化坐标下表面
    cols = [f"{r:.1f}" for r in r_output_cm] + ["药材表面"]
    data_full = np.column_stack([C_out, C_surface])

    df = pd.DataFrame(np.round(data_full, DECIMALS), columns=cols)
    df.insert(0, "时间", t)
    header = ["时间\\到药材中心的距离"] + cols
    df.columns = header

    with pd.ExcelWriter(file_path) as writer:
        df.to_excel(writer, sheet_name="水分浓度", index=False)
    print(f"已保存: {file_path}")


def save_tables_q4(t, xi, C_result, R_result, t_end_h):
    """写 output/tables/表6_水分浓度.xlsx"""
    r_output_cm = [0, 0.5, 1.0, 1.5, 2.0]
    times_h = list(range(0, int(t_end_h), 6)) + [round(t_end_h, 4)]

    rows = []
    for h in times_h:
        idx = min(int(h * 3600 / 60), len(t) - 1)
        R_n_cm = R_result[idx] * 100.0
        row = []
        for r_cm in r_output_cm:
            if r_cm <= R_n_cm + 1e-9:
                xi_val = r_cm / R_n_cm
                row.append(np.interp(xi_val, xi, C_result[idx]))
            else:
                row.append(np.nan)
        # 表面列
        row.append(C_result[idx, -1])
        rows.append(row)

    cols = [f"{r}" for r in r_output_cm] + ["药材表面"]
    df = pd.DataFrame(np.round(rows, DECIMALS), index=times_h, columns=cols)
    df.to_excel(TABLE_DIR / "表6_水分浓度.xlsx")
    print(f"已保存: {TABLE_DIR / '表6_水分浓度.xlsx'}")
    return df


if __name__ == "__main__":
    from tedm.solver_q4 import solve_q4
    t, xi, T_r, C_r, R_r, T_a, C_a, t_end_h = solve_q4()
    save_result4(t, xi, C_r, R_r, t_end_h)
    df = save_tables_q4(t, xi, C_r, R_r, t_end_h)
    print("\n=== 表6 ===")
    print(df)