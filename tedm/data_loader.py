import numpy as np
import pandas as pd
from tedm.config import FILE_ATTACH1, T_END, DT,DATA_DIR


def load_air_conditions(t_end=None, dt=None):
    t_end = t_end if t_end is not None else T_END
    dt = dt if dt is not None else DT
    df = pd.read_excel(FILE_ATTACH1)
    t_data = df.iloc[:, 0].values.astype(float)
    T_data = df.iloc[:, 1].values.astype(float)
    C_data = df.iloc[:, 2].values.astype(float)
    t = np.arange(0, t_end + dt, dt)
    T_air = np.interp(t, t_data, T_data)
    C_air = np.interp(t, t_data, C_data)
    return t, T_air, C_air

def load_radius():
    """
    读取附件2：半径 R(t)，单位 cm。
    返回：
        t_data:   (M,) 时间 s
        R_data:   (M,) 半径 cm
        R_func:   插值函数 R(t)，单位 cm；超出范围取末值
    """
    df = pd.read_excel(DATA_DIR / "附件2.xlsx")
    t_data = df.iloc[:, 0].values.astype(float)
    R_data = df.iloc[:, 1].values.astype(float)

    def R_func(t):
        t = np.asarray(t, dtype=float)
        return np.interp(t, t_data, R_data)  # 超出范围自动 clamp

    return t_data, R_data, R_func