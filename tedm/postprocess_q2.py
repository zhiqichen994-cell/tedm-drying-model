# ============================================================
# 问题2 结果输出：result2.xlsx + 表3、表4
# ------------------------------------------------------------
# result2.xlsx 格式（参照附件3模板）：
#   - 工作表"温度"、"水分浓度"
#   - A列为时间（0~10800s，每秒一行）
#   - 第1行为距离（0~2cm，每0.1cm一列）
# 表3、表4：3h 内每 0.5h，5 个径向位置
# ============================================================

import numpy as np
import pandas as pd
from tedm.config import TABLE_DIR, OUTPUT_DIR, DECIMALS
from tedm.properties import APP3
from tedm.solver import solve


def _interp_to_01cm(r, field):
    r_cm = r * 100.0
    target = np.round(np.arange(0.0, 2.0 + 1e-9, 0.1), 4)
    return np.interp(target, r_cm, field)


def save_result2(t, r, T_result, C_result):
    """写 output/result2.xlsx，格式对齐附件3模板"""
    file_path = OUTPUT_DIR / "result2.xlsx"
    cols = [f"{ri:.1f}" for ri in np.round(np.arange(0.0, 2.0 + 1e-9, 0.1), 4)]

    T_01 = np.array([_interp_to_01cm(r, T_result[i]) for i in range(T_result.shape[0])])
    C_01 = np.array([_interp_to_01cm(r, C_result[i]) for i in range(C_result.shape[0])])

    df_T = pd.DataFrame(np.round(T_01, DECIMALS), columns=cols)
    df_T.insert(0, "时间", t)
    df_C = pd.DataFrame(np.round(C_01, DECIMALS), columns=cols)
    df_C.insert(0, "时间", t)

    header = ["时间\\到药材中心的距离"] + cols
    df_T.columns = header
    df_C.columns = header

    with pd.ExcelWriter(file_path) as writer:
        df_T.to_excel(writer, sheet_name="温度", index=False)
        df_C.to_excel(writer, sheet_name="水分浓度", index=False)
    print(f"已保存: {file_path}")


def save_tables_q2(t, r, T_result, C_result):
    """写 output/tables/表3_温度.xlsx、表4_水分浓度.xlsx"""
    r_cm = r * 100.0
    times_s = [1800, 3600, 5400, 7200, 9000, 10800]
    times_h = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    target_r = [0, 0.5, 1.0, 1.5, 2.0]

    T_rows, C_rows = [], []
    for ti in times_s:
        T_rows.append(np.interp(target_r, r_cm, T_result[ti]))
        C_rows.append(np.interp(target_r, r_cm, C_result[ti]))

    df_T = pd.DataFrame(np.round(T_rows, DECIMALS), index=times_h, columns=target_r)
    df_C = pd.DataFrame(np.round(C_rows, DECIMALS), index=times_h, columns=target_r)

    df_T.to_excel(TABLE_DIR / "表3_温度.xlsx")
    df_C.to_excel(TABLE_DIR / "表4_水分浓度.xlsx")
    print(f"已保存: {TABLE_DIR / '表3_温度.xlsx'}")
    print(f"已保存: {TABLE_DIR / '表4_水分浓度.xlsx'}")
    return df_T, df_C


if __name__ == "__main__":
    t, r, T_result, C_result, T_air, C_air = solve(
        prop=APP3, t_end=3 * 3600.0, dt=1.0, use_latent=False)
    save_result2(t, r, T_result, C_result)
    df_T, df_C = save_tables_q2(t, r, T_result, C_result)
    print("\n=== 表3 温度 ===")
    print(df_T)
    print("\n=== 表4 水分浓度 ===")
    print(df_C)