# -*- coding: utf-8 -*-
"""
任务 01 — 模拟 LightTools 照度数据：均匀性分析 + 可视化
================================================================
功能：
  1) 用 numpy 生成 100x100 网格照度分布（高斯圆形光斑 + 随机噪声）
  2) 用 pandas 导出 -> data/illuminance_data.csv
  3) 读回 CSV，计算 9 点均匀性与整体均匀性
  4) matplotlib 输出伪彩色热力图 + 径向剖面 -> output/heatmap.png

安全约定：只写 data/ 与 output/，不触碰任何原始设计文件。

用法：
    python scripts/01_illuminance_uniformity.py
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")            # 无窗口后端，保证在纯终端/后台也能出图
import matplotlib.pyplot as plt
from matplotlib import font_manager


# ================================================================
# 0. 全局配置
# ================================================================
@dataclass
class Config:
    """所有可调参数集中在这里，改参数不用翻代码"""

    # --- 路径（相对脚本所在项目的根目录）---
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: str = "data"
    output_dir: str = "output"
    csv_name: str = "illuminance_data.csv"
    png_name: str = "heatmap.png"
    report_name: str = "uniformity_report.txt"

    # --- 网格与光斑 ---
    grid_n: int = 100               # 100x100 网格
    pitch_mm: float = 0.5           # 相邻采样点间距 (mm)，仅用于坐标轴标注
    peak_illuminance: float = 1000.0  # 峰值照度 (lux)，模拟值
    spot_radius_frac: float = 0.35  # 1/e^2 半径 = 0.35 * grid_n
    gauss_order: float = 2.0        # I = I0 * exp(-gauss_order * (r/R)^2)

    # --- 噪声 ---
    seed: int = 42
    noise_rel_sigma: float = 0.020  # 乘性噪声 2%（模拟探测器/光源波动）
    noise_abs_sigma: float = 0.005  # 加性噪声 0.5% 峰值（模拟暗电流）

    # --- 统计窗口 ---
    core_frac: float = 0.50         # 中心有效区半径 = 0.5R（近轴区，配光评价常看这里）

    # --- 9 点采样 ---
    nine_point_frac: float = 0.50   # 9 点在 ±0.5R 处，对角点 r=0.71R 仍在孔径内

    # --- 绘图 ---
    dpi: int = 150

    # 派生属性
    csv_path: Path = field(init=False)
    png_path: Path = field(init=False)
    report_path: Path = field(init=False)

    def __post_init__(self):
        self.csv_path = self.project_root / self.data_dir / self.csv_name
        self.png_path = self.project_root / self.output_dir / self.png_name
        self.report_path = self.project_root / self.output_dir / self.report_name


# ================================================================
# 1. 中文字体（不做这一步，图上的中文会变成方块）
# ================================================================
def setup_chinese_font() -> str:
    """挑选本机可用的中文字体，返回实际使用的字体名"""
    candidates = [
        "Microsoft YaHei", "微软雅黑", "SimHei", "黑体",
        "Noto Sans CJK SC", "Source Han Sans SC", "Arial Unicode MS",
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False   # 负号正常显示
            return name
    # 全部没有也不崩，只是中文会显示为方块
    print("[警告] 未找到中文字体，图内中文可能显示为方块（不影响数值结果）")
    return "DejaVu Sans"


# ================================================================
# 2. 生成照度数据
# ================================================================
def generate_illuminance(cfg: Config) -> np.ndarray:
    """
    生成模拟照度分布：
        圆形高斯光斑（中心亮、边缘暗） + 乘性噪声 + 加性噪声，截断为非负
    返回 shape=(grid_n, grid_n) 的二维数组，单位 lux
    """
    n = cfg.grid_n
    radius = cfg.spot_radius_frac * n

    # 以网格中心为原点的坐标（单位：网格格数）
    axis = np.arange(n) - (n - 1) / 2.0
    xx, yy = np.meshgrid(axis, axis)
    rr = np.hypot(xx, yy)

    # 理想光斑：高斯分布，r=R 处降至峰值 13.5%
    base = cfg.peak_illuminance * np.exp(-cfg.gauss_order * (rr / radius) ** 2)

    # 噪声：乘性（与信号成正比）+ 加性（与信号无关）
    rng = np.random.default_rng(cfg.seed)
    shot = base * rng.normal(0.0, cfg.noise_rel_sigma, size=base.shape)
    dark = rng.normal(0.0, cfg.noise_abs_sigma * cfg.peak_illuminance, size=base.shape)

    data = np.clip(base + shot + dark, 0.0, None)   # 照度不可能为负
    return data


# ================================================================
# 3. 存取 CSV
# ================================================================
def save_csv(data: np.ndarray, path: Path) -> None:
    """保存为无表头矩阵 CSV —— 与 LightTools / 探测器导出的格式一致"""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(np.round(data, 4))
    df.to_csv(path, index=False, header=False)
    print(f"[OK] 数据已保存 : {path}  ({data.shape[0]}x{data.shape[1]})")


def load_csv(path: Path) -> np.ndarray:
    """读回 CSV 并做完整性校验"""
    if not path.exists():
        raise FileNotFoundError(f"CSV 不存在：{path}")
    df = pd.read_csv(path, header=None)
    arr = df.to_numpy(dtype=float)

    if arr.size == 0:
        raise ValueError("CSV 内容为空")
    if np.isnan(arr).any():
        raise ValueError(f"CSV 含 NaN，共 {int(np.isnan(arr).sum())} 个，请检查数据源")
    if (arr < 0).any():
        raise ValueError("CSV 含负照度值，物理上不成立")
    return arr


# ================================================================
# 4. 均匀性指标
# ================================================================
def aperture_mask(shape: tuple[int, int], radius: float) -> np.ndarray:
    """圆形孔径掩膜：只统计光斑内部，排除外部暗区（否则均匀性恒为 0）"""
    n_row, n_col = shape
    ax_r = np.arange(n_row) - (n_row - 1) / 2.0
    ax_c = np.arange(n_col) - (n_col - 1) / 2.0
    cc, rr = np.meshgrid(ax_c, ax_r)
    return np.hypot(cc, rr) <= radius


def nine_point_values(data: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """
    传统 3x3 九点法：在光斑中心及 ±frac*R 处取 9 个点
    返回 (9 个照度值, 9 个采样点坐标(列, 行))
    """
    n = cfg.grid_n
    radius = cfg.spot_radius_frac * n
    center = (n - 1) / 2.0
    off = cfg.nine_point_frac * radius

    vals, pts = [], []
    for dy in (-off, 0.0, off):
        for dx in (-off, 0.0, off):
            col = int(round(center + dx))
            row = int(round(center + dy))
            col = min(max(col, 0), n - 1)      # 边界保护
            row = min(max(row, 0), n - 1)
            vals.append(float(data[row, col]))
            pts.append((col, row))
    return np.array(vals), np.array(pts)


def _window_stats(data: np.ndarray, mask: np.ndarray) -> dict:
    """统计单个窗口内的照度特征与三种均匀性口径"""
    spot = data[mask]
    i_max, i_min, i_mean = float(spot.max()), float(spot.min()), float(spot.mean())
    i_std = float(spot.std(ddof=1)) if spot.size > 1 else 0.0
    return {
        "pixels": int(mask.sum()),
        "peak": i_max,
        "min": i_min,
        "mean": i_mean,
        "std": i_std,
        "U_min_max": i_min / i_max if i_max > 0 else 0.0,
        "U_min_mean": i_min / i_mean if i_mean > 0 else 0.0,
        "U_mean_max": i_mean / i_max if i_max > 0 else 0.0,
        "CV": i_std / i_mean if i_mean > 0 else 0.0,
    }


def compute_metrics(data: np.ndarray, cfg: Config) -> dict:
    """
    计算两套统计窗口 + 9 点均匀性。

    为什么要两套窗口：均匀性数值强依赖"统计到哪里"。
      全光斑 (r<=R)  —— 含边缘滚降，数值低，反映"整片光斑"表现
      中心有效区(0.5R) —— 近轴均匀性，数值高，反映"工作视场内"表现
    只报一个数会误导，行业口径混乱正源于此。
    """
    radius = cfg.spot_radius_frac * cfg.grid_n
    masks = {
        "full": aperture_mask(data.shape, radius),
        "core": aperture_mask(data.shape, radius * cfg.core_frac),
    }
    stats = {k: _window_stats(data, v) for k, v in masks.items()}

    nine_vals, nine_pts = nine_point_values(data, cfg)
    n_max, n_min, n_mean = (float(nine_vals.max()), float(nine_vals.min()),
                            float(nine_vals.mean()))

    return {
        "masks": masks,
        "radius": radius,
        "full": stats["full"],
        "core": stats["core"],
        "nine_vals": nine_vals,
        "nine_pts": nine_pts,
        "nine": {
            "min": n_min,
            "max": n_max,
            "mean": n_mean,
            "U_min_max": n_min / n_max if n_max > 0 else 0.0,
            "U_min_mean": n_min / n_mean if n_mean > 0 else 0.0,
        },
        "peak": float(data.max()),
    }


# ================================================================
# 5. 绘图
# ================================================================
def plot_heatmap(data: np.ndarray, m: dict, cfg: Config, font: str) -> None:
    """左：2D 伪彩色热力图；右：径向照度剖面"""
    n = cfg.grid_n
    extent_mm = [(-(n - 1) / 2) * cfg.pitch_mm, ((n - 1) / 2) * cfg.pitch_mm] * 2

    fig = plt.figure(figsize=(13.8, 5.8), dpi=cfg.dpi, constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.12)

    # ---------- 左：热力图 ----------
    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(
        data, origin="lower", cmap="inferno", extent=extent_mm, interpolation="nearest"
    )
    cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03)
    cb.set_label("照度 Illuminance (lux)", fontsize=10)

    # 叠加 9 个采样点
    pts_mm_x = (m["nine_pts"][:, 0] - (n - 1) / 2.0) * cfg.pitch_mm
    pts_mm_y = (m["nine_pts"][:, 1] - (n - 1) / 2.0) * cfg.pitch_mm
    ax1.plot(pts_mm_x, pts_mm_y, "o", ms=7, mfc="none", mec="#00E5FF", mew=1.4,
             label="9 点采样位置")

    # 两个统计窗口边界
    radius_mm = m["radius"] * cfg.pitch_mm
    core_mm = radius_mm * cfg.core_frac
    theta = np.linspace(0, 2 * np.pi, 400)
    ax1.plot(radius_mm * np.cos(theta), radius_mm * np.sin(theta),
             "--", lw=1.1, color="#00E5FF", alpha=0.85, label="全光斑区 r ≤ R")
    ax1.plot(core_mm * np.cos(theta), core_mm * np.sin(theta),
             ":", lw=1.6, color="#2ECC71", alpha=0.95, label=f"中心有效区 r ≤ {cfg.core_frac:.2f}R")

    ax1.set_xlabel("X (mm)", fontsize=10)
    ax1.set_ylabel("Y (mm)", fontsize=10)
    ax1.set_title(
        f"照度分布伪彩色图    峰值 {m['peak']:.0f} lux\n"
        f"U(min/max)  全光斑 {m['full']['U_min_max']*100:.1f}%   |   "
        f"中心区 {m['core']['U_min_max']*100:.1f}%",
        fontsize=10.5, pad=9,
    )
    ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.75)
    ax1.set_aspect("equal")

    # ---------- 右：径向剖面 ----------
    ax2 = fig.add_subplot(gs[0, 1])
    center = (n - 1) / 2
    row_mid = data[int(round(center)), :]
    x_mm = (np.arange(n) - center) * cfg.pitch_mm

    ax2.plot(x_mm, row_mid, lw=1.2, color="#C0392B", label="水平中心剖面（含噪声）")
    # 理论无噪声曲线作对照
    r_mm = np.abs(x_mm)
    theory = cfg.peak_illuminance * np.exp(
        -cfg.gauss_order * (r_mm / radius_mm) ** 2
    )
    ax2.plot(x_mm, theory, "--", lw=1.4, color="#2C3E50", alpha=0.8, label="理论高斯")

    ax2.axhline(m["full"]["mean"], ls=":", lw=1.1, color="#27AE60",
                label=f"全光斑均值 {m['full']['mean']:.0f} lux")
    for xv in (-radius_mm, radius_mm):
        ax2.axvline(xv, ls="-.", lw=1.0, color="#00E5FF", alpha=0.7)
    for xv in (-core_mm, core_mm):
        ax2.axvline(xv, ls=":", lw=1.2, color="#2ECC71", alpha=0.85)

    ax2.set_xlabel("径向位置 (mm)", fontsize=10)
    ax2.set_ylabel("照度 (lux)", fontsize=10)
    ax2.set_title("径向照度剖面", fontsize=11, pad=10)
    ax2.grid(alpha=0.25, ls=":")
    ax2.legend(fontsize=8.5, loc="upper right")
    ax2.set_xlim(x_mm.min(), x_mm.max())

    fig.suptitle("模拟 LightTools 照度数据 · 均匀性分析", fontsize=13)

    cfg.png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 热力图已保存: {cfg.png_path}")


# ================================================================
# 6. 报告
# ================================================================
def build_report(data: np.ndarray, m: dict, cfg: Config) -> str:
    """生成文本报告（同时打印并落盘）"""
    lines = []
    add = lines.append
    pitch = cfg.pitch_mm
    r_mm = m["radius"] * pitch
    c_mm = r_mm * cfg.core_frac
    f, c = m["full"], m["core"]

    add("=" * 72)
    add("照度均匀性分析报告")
    add("=" * 72)
    add(f"数据文件    : {cfg.csv_path}")
    add(f"网格        : {data.shape[0]} x {data.shape[1]}   采样间距 {pitch} mm")
    add("")
    add("-- 统计窗口 --")
    add(f"  全光斑区   : r <= {r_mm:.2f} mm  (1/e² 半径, {f['pixels']} 个采样点)")
    add(f"  中心有效区 : r <= {c_mm:.2f} mm  ({cfg.core_frac:.2f}R, {c['pixels']} 个采样点)")
    add("")
    add(f"{'指标':<16}{'全光斑区':>14}{'中心有效区':>16}")
    add("-" * 72)
    add(f"{'峰值 Imax':<16}{f['peak']:>14.2f}{c['peak']:>16.2f}")
    add(f"{'最小 Imin':<16}{f['min']:>14.2f}{c['min']:>16.2f}")
    add(f"{'平均 Imean':<16}{f['mean']:>14.2f}{c['mean']:>16.2f}")
    add(f"{'标准差':<16}{f['std']:>14.2f}{c['std']:>16.2f}")
    add(f"{'变异系数 CV':<14}{f['CV'] * 100:>14.2f}%{c['CV'] * 100:>15.2f}%")
    add("")
    add("-- 整体均匀性（单位: %）--")
    add(f"{'U(min/max)':<16}{f['U_min_max'] * 100:>14.2f}{c['U_min_max'] * 100:>16.2f}   最严格")
    add(f"{'U(min/mean)':<16}{f['U_min_mean'] * 100:>14.2f}{c['U_min_mean'] * 100:>16.2f}   较严苛")
    add(f"{'U(mean/max)':<16}{f['U_mean_max'] * 100:>14.2f}{c['U_mean_max'] * 100:>16.2f}   最宽松")
    add("")
    add(f"-- 9 点均匀性（3x3 网格, 采样点位于 ±{cfg.nine_point_frac:.2f}R）--")
    vals = m["nine_vals"].reshape(3, 3)
    for r_i in range(3):
        add("   " + "".join(f"{v:>10.1f}" for v in vals[r_i]) + "   lux")
    add(f"  9 点 U(min/max)  = {m['nine']['U_min_max'] * 100:6.2f} %")
    add(f"  9 点 U(min/mean) = {m['nine']['U_min_mean'] * 100:6.2f} %")
    add("")
    add("-- 判定参考 --")
    add("  整体 U(min/max): 全光斑 >= 50% 为一般照明常规要求；中心区 >= 85% 为配光优良")
    add(f"    -> 本组数据: 全光斑 {f['U_min_max'] * 100:.1f}% "
        f"({'通过' if f['U_min_max'] >= 0.50 else '不通过'})，"
        f"中心区 {c['U_min_max'] * 100:.1f}% "
        f"({'通过' if c['U_min_max'] >= 0.85 else '不通过'})")
    add("  提示: 高斯型光斑在孔径边缘必然衰减到 13.5%（1/e² 定义），")
    add("        故全光斑均匀性偏低属物理必然；要提升需改用平顶/超高斯配光。")
    add("=" * 72)
    return "\n".join(lines)


# ================================================================
# 7. 主流程
# ================================================================
def main() -> int:
    cfg = Config()

    try:
        # --- 准备 ---
        for d in (cfg.data_dir, cfg.output_dir):
            (cfg.project_root / d).mkdir(parents=True, exist_ok=True)
        font = setup_chinese_font()
        print(f"[OK] 中文字体 : {font}")

        # --- 1) 生成 ---
        data = generate_illuminance(cfg)
        print(f"[OK] 已生成照度数据: {data.shape}, "
              f"范围 {data.min():.1f} ~ {data.max():.1f} lux")

        # --- 2) 存 CSV ---
        save_csv(data, cfg.csv_path)

        # --- 3) 读回 + 计算 ---
        loaded = load_csv(cfg.csv_path)
        if loaded.shape != data.shape:
            raise ValueError(f"读回形状不一致: {loaded.shape} vs {data.shape}")
        max_dev = float(np.abs(loaded - np.round(data, 4)).max())
        print(f"[OK] CSV 读回校验通过（最大写入偏差 {max_dev:.2e} lux）")

        metrics = compute_metrics(loaded, cfg)

        # --- 4) 出图 + 报告 ---
        plot_heatmap(loaded, metrics, cfg, font)

        report = build_report(loaded, metrics, cfg)
        print()
        print(report)
        cfg.report_path.write_text(report, encoding="utf-8")
        print(f"\n[OK] 报告已保存: {cfg.report_path}")
        return 0

    except FileNotFoundError as e:
        print(f"[错误] 文件路径问题: {e}", file=sys.stderr)
        return 2
    except PermissionError as e:
        print(f"[错误] 无写入权限（文件是否被 Excel 占用？）: {e}", file=sys.stderr)
        return 3
    except ValueError as e:
        print(f"[错误] 数据校验失败: {e}", file=sys.stderr)
        return 4
    except Exception:
        print("[错误] 未预期异常，完整堆栈如下：", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
