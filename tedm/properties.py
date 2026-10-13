# ============================================================
# 三套物性（附录2 / 附录3 / 附录4）
# ------------------------------------------------------------
# 每套提供：
#   rho(C), cp(C), k(C)   —— 密度、比热、导热
#   rho_dry(C)            —— 干物质密度 = rho/(1+C)
#   D(C, T_degC)          —— 水分扩散系数（含 Arrhenius）
#   Phi(C)                —— Kirchhoff 势 ∫D(C)dC
#   face_D(Ci, Cj, Ti, Tj) —— 界面扩散系数（Kirchhoff 积分平均）
# ============================================================

import numpy as np
from scipy.special import exp1


class PropertySet:
    def __init__(self, name, rho_a, rho_b, cp_a, cp_b, k_a, k_b,
                 D_A, D_B, arrhenius, E_arr=3850.0):
        self.name = name
        self.rho_a, self.rho_b = rho_a, rho_b
        self.cp_a, self.cp_b = cp_a, cp_b
        self.k_a, self.k_b = k_a, k_b
        self.D_A, self.D_B = D_A, D_B
        self.arrhenius = arrhenius
        self.E_arr = E_arr

    # ---------- 基本物性 ----------
    def rho(self, C):
        return self.rho_a + self.rho_b * C

    def cp(self, C):
        return self.cp_a + self.cp_b * C / (C + 1.0)

    def k(self, C):
        return self.k_a + self.k_b * C / (C + 1.0)

    def rho_dry(self, C):
        """干物质密度 rho_d = rho / (1+C)"""
        return self.rho(C) / (1.0 + C)

    # ---------- 扩散系数 ----------
    def D_conc(self, C):
        """仅含水率部分: A * exp(-B/C)"""
        return self.D_A * np.exp(-self.D_B / C)

    def D(self, C, T_degC=None):
        """完整 D(C, T): 含 Arrhenius 温度项（T 输入 ℃，内部转 K）"""
        d = self.D_conc(C)
        if self.arrhenius:
            if T_degC is None:
                raise ValueError(f"{self.name}: D 需要 T")
            d = d * np.exp(-self.E_arr / (np.asarray(T_degC, dtype=float) + 273.15))
        return d

    # ---------- Kirchhoff 势 ----------
    def Phi(self, C):
        """Phi(C) = ∫ D_conc(C) dC 的解析原函数"""
        A, B = self.D_A, self.D_B
        C = np.asarray(C, dtype=float)
        return A * (C * np.exp(-B / C) - B * exp1(B / C))

    def face_D(self, Ci, Cj, Ti=None, Tj=None, tol=1e-9):
        """界面扩散系数：Kirchhoff 积分平均 + Arrhenius 算术平均"""
        Ci = np.asarray(Ci, dtype=float)
        Cj = np.asarray(Cj, dtype=float)
        dC = Ci - Cj
        small = np.abs(dC) < tol
        dC_safe = np.where(small, 1.0, dC)

        C_mid = 0.5 * (Ci + Cj)
        D_mid = self.D_conc(C_mid)
        D_face_conc = np.where(small, D_mid,
                               (self.Phi(Ci) - self.Phi(Cj)) / dC_safe)

        if self.arrhenius:
            if Ti is None or Tj is None:
                raise ValueError(f"{self.name}: face_D 需要 T")
            arr_i = np.exp(-self.E_arr / (np.asarray(Ti, dtype=float) + 273.15))
            arr_j = np.exp(-self.E_arr / (np.asarray(Tj, dtype=float) + 273.15))
            return D_face_conc * 0.5 * (arr_i + arr_j)
        return D_face_conc


# ============================================================
# 附录2：问题1（常数物性）
# ============================================================
APP2 = PropertySet(
    name='附录2',
    rho_a=820.0, rho_b=0.0,
    cp_a=2600.0, cp_b=0.0,
    k_a=0.36, k_b=0.0,
    D_A=7e-9, D_B=0.89,
    arrhenius=False,
)

# ============================================================
# 附录3：问题2、3（变物性 + Arrhenius）
# ============================================================
APP3 = PropertySet(
    name='附录3',
    rho_a=650.0, rho_b=128.0,
    cp_a=1450.0, cp_b=2736.0,
    k_a=0.21, k_b=0.38,
    D_A=2.4e-3, D_B=0.45,
    arrhenius=True, E_arr=3850.0,
)

# ============================================================
# 附录4：问题4（变物性 + Arrhenius）
# ============================================================
APP4 = PropertySet(
    name='附录4',
    rho_a=760.0, rho_b=90.0,
    cp_a=1850.0, cp_b=2150.0,
    k_a=0.12, k_b=0.20,
    D_A=4.2e-4, D_B=0.30,
    arrhenius=True, E_arr=3850.0,
)


if __name__ == "__main__":
    C = 2.55
    T = 50.0
    print("=== properties.py 自检 ===")
    print(f"APP2: rho={APP2.rho(C):.1f}, cp={APP2.cp(C):.1f}, k={APP2.k(C):.4f}")
    print(f"      D(C)={APP2.D(C):.4e}")
    print(f"APP3: rho={APP3.rho(C):.1f}, cp={APP3.cp(C):.2f}, k={APP3.k(C):.4f}")
    print(f"      D(C,T)={APP3.D(C,T):.4e}")
    print(f"APP4: rho={APP4.rho(C):.1f}, cp={APP4.cp(C):.2f}, k={APP4.k(C):.4f}")
    print(f"      D(C,T)={APP4.D(C,T):.4e}")