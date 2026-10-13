from pathlib import Path

# ========== 路径 ==========
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

for d in [OUTPUT_DIR, FIG_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

FILE_ATTACH1 = DATA_DIR / "附件1.xlsx"
FILE_TEMPLATE = DATA_DIR / "result1.xlsx"     # 模板
FILE_RESULT1 = OUTPUT_DIR / "result1.xlsx"    # 输出

# ========== 几何 ==========
R = 0.02          # 半径 m (2 cm)
L = 0.25          # 长度 m (25 cm)，问题1一维径向不需要，但记着

# ========== 物性（问题1：常数）==========
RHO = 820.0       # kg/m^3
CP = 2600.0       # J/(kg·K)
K_COND = 0.36     # W/(m·K)
H_CONV = 25.0     # W/(m^2·K)
K_MASS = 8e-7     # m/s

# 水分扩散系数 D(C) = D0 * exp(-B/C)
D0 = 7e-9         # m^2/s
B_D = 0.89

# ========== 初始条件 ==========
T0 = 28.0         # ℃
C0 = 2.55         # kg/kg

# ========== 数值参数 ==========
DR = R / 100.0
DT = 1.0
T_END = 1800.0
N_R = 100
N_T = int(round(T_END / DT))  # 1800

# ========== 输出 ==========
DECIMALS = 4
TABLE_TIMES = [100, 300, 600, 900, 1200, 1500, 1800]   # 表1、表2时间
TABLE_R = [0, 0.5, 1, 1.5, 2]                           # 表1、表2距离 cm

#验证代码
if __name__ == "__main__":
    print("=== config.py 自检 ===")
    print("N_R =", N_R)
    print("N_T =", N_T)
    print("R =", R, "m")
    print("T0 =", T0, "℃")
    print("C0 =", C0, "kg/kg")
    print("FILE_ATTACH1 =", FILE_ATTACH1)
    print("附件1存在？", FILE_ATTACH1.exists())
    print("输出目录 =", OUTPUT_DIR)
    print("输出目录存在？", OUTPUT_DIR.exists())