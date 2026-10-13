# ============================================================
# 问题3 结果输出：result3.xlsx + 表5
# ============================================================

import numpy as np
import pandas as pd
from tedm.config import OUTPUT_DIR, TABLE_DIR, DECIMALS


def _interp_to_01cm(r, field):
    r_cm = r * 100.0
    target = np.round(np.arange(0.0, 2.0 + 1e-9, 0.1), 4)
    return np.interp(target, r_cm, field)


def save_result3(t, r, C_result, t_end_h):
    """写 output/result3.xlsx（只有水分浓度工作表）"""
    file_path = OUTPUT_DIR / "result3.xlsx"
    cols = [f"{ri:.1f}" for ri in np.round(np.arange(0.0, 2.0 + 1e-9, 0.1), 4)]

    C_01 = np.array([_interp_to_01cm(r, C_result[i]) for i in range(C_result.shape[0])])

    df_C = pd.DataFrame(np.round(C_01, DECIMALS), columns=cols)
    df_C.insert(0, "时间", t)

    header = ["时间\\到药材中心的距离"] + cols
    df_C.columns = header

    with pd.ExcelWriter(file_path) as writer:
        df_C.to_excel(writer, sheet_name="水分浓度", index=False)
    print(f"已保存: {file_path}")


def save_tables_q3(t, r, C_result, t_end_h):
    """写 output/tables/表5_水分浓度.xlsx（每 6h + 结束时刻）"""
    r_cm = r * 100.0

    # 6h 间隔 + 烘干终点
    times_h = list(range(0, int(t_end_h), 6)) + [round(t_end_h, 4)]
    times_s = [h * 3600 for h in times_h]

    target_r = [0, 0.5, 1.0, 1.5, 2.0]

    rows = []
    for ts in times_s:
        idx = int(round(ts / 60))
        idx = min(idx, len(t) - 1)
        vals = np.interp(target_r, r_cm, C_result[idx])
        rows.append(vals)

    df = pd.DataFrame(np.round(rows, DECIMALS), index=times_h, columns=target_r)
    df.to_excel(TABLE_DIR / "表5_水分浓度.xlsx")
    print(f"已保存: {TABLE_DIR / '表5_水分浓度.xlsx'}")
    return df


if __name__ == "__main__":
    from tedm.solver_q3 import solve_q3
    t, r, T_result, C_result, T_air, C_air, t_end_h, end_C = solve_q3()
    save_result3(t, r, C_result, t_end_h)
    df = save_tables_q3(t, r, C_result, t_end_h)
    print("\n=== 表5 ===")
    print(df)