import numpy as np
import pandas as pd
from tedm.config import (FILE_RESULT1, DECIMALS, TABLE_TIMES, TABLE_R,
                        TABLE_DIR)


def _interp_to_01cm(r, field):
    """把任意节点场插值到 0.0, 0.1, ..., 2.0 cm（21 个点）"""
    r_cm = r * 100.0
    target = np.round(np.arange(0.0, 2.0 + 1e-9, 0.1), 4)
    return np.interp(target, r_cm, field)


def save_result1(t, r, T_result, C_result):
    """按附件3模板保存 result1.xlsx（A列时间，第1行距离 0~2cm）"""
    cols = [f"{ri:.1f}" for ri in np.round(np.arange(0.0, 2.0 + 1e-9, 0.1), 4)]

    # 每行插值到 0.1 cm 网格
    T_01 = np.array([_interp_to_01cm(r, T_result[i]) for i in range(T_result.shape[0])])
    C_01 = np.array([_interp_to_01cm(r, C_result[i]) for i in range(C_result.shape[0])])

    df_T = pd.DataFrame(np.round(T_01, DECIMALS), columns=cols)
    df_T.insert(0, "时间", t)
    df_C = pd.DataFrame(np.round(C_01, DECIMALS), columns=cols)
    df_C.insert(0, "时间", t)

    header = ["时间\\到药材中心的距离"] + cols
    df_T.columns = header
    df_C.columns = header

    with pd.ExcelWriter(FILE_RESULT1) as writer:
        df_T.to_excel(writer, sheet_name="温度", index=False)
        df_C.to_excel(writer, sheet_name="水分浓度", index=False)
    print(f"已保存: {FILE_RESULT1}")


def extract_tables(t, r, T_result, C_result):
    """提取论文表1、表2：7 个时刻 × 5 个距离，用插值取到精确刻度"""
    r_cm = r * 100.0
    time_idx = [int(tt) for tt in TABLE_TIMES]  # 每秒一个点，索引=秒数

    T_rows = []
    C_rows = []
    for idx in time_idx:
        T_rows.append(np.interp(TABLE_R, r_cm, T_result[idx]))
        C_rows.append(np.interp(TABLE_R, r_cm, C_result[idx]))

    df_T = pd.DataFrame(np.round(T_rows, DECIMALS),
                        index=TABLE_TIMES, columns=TABLE_R)
    df_C = pd.DataFrame(np.round(C_rows, DECIMALS),
                        index=TABLE_TIMES, columns=TABLE_R)

    df_T.to_excel(TABLE_DIR / "表1_温度.xlsx")
    df_C.to_excel(TABLE_DIR / "表2_水分浓度.xlsx")
    print("表1、表2 已保存")
    return df_T, df_C