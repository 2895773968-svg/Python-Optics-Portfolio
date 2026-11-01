# -*- coding: utf-8 -*-
"""
任务 03 — Zemax 二维参数扫描（曲率半径 × 透镜间距）
================================================================================
在 02 单参数扫描基础上扩展为二维网格扫描，输出矩阵 CSV + 3D 曲面 / 2D 等高线。

依赖同目录 zemax_engine.py（后端共享层）。切真机：
    pip install zospy
    python scripts/03_zemax_2d_sweep.py --backend real --lens "D:/design/test.zmx"

默认参数：
    曲率半径  47 ~ 53 mm，7 点
    透镜间距   2 ~ 10 mm，9 点
    => 7 x 9 = 63 个参数组合

【面号提醒】间距改的是 spacing_surface_index 那一个面的 Thickness：
    该面为镜片前表面 -> Thickness 是玻璃中心厚度
    该面为镜片后表面 -> Thickness 是该面之后的空气间隔（通常你要的是这个，默认 4）

输出：
    output/zemax_2d_sweep_matrix.csv   矩阵式（行=半径, 列=间距）
    output/zemax_2d_sweep_long.csv     长表式（每行一个组合，便于后续统计）
    output/zemax_2d_sweep.png          3D 曲面 + 2D 等高线
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zemax_engine import EngineConfig, create_engine   # noqa: E402


# ================================================================
# 扫描网格配置
# ================================================================
@dataclass
class GridConfig:
    radii: np.ndarray
    spacings: np.ndarray

    @property
    def n_cells(self) -> int:
        return len(self.radii) * len(self.spacings)


@dataclass
class Cell:
    radius: float
    spacing: float
    mtf_at: float | None = None
    mtf_mean: float | None = None
    status: str = "ok"
    note: str = ""


# ================================================================
# 扫描
# ================================================================
def run_2d_sweep(engine, ecfg: EngineConfig, grid: GridConfig) -> list[Cell]:
    """
    双层循环扫描。每个格点独立捕获异常：
    单点失败记 NaN 继续跑，不会因为一个坏点丢掉整张图。
    """
    cells: list[Cell] = []
    t_start = time.perf_counter()

    for i, r in enumerate(grid.radii, 1):
        row_vals, row_fail = [], 0
        for s in grid.spacings:
            cell = Cell(radius=float(r), spacing=float(s))
            try:
                engine.set_radius(float(r))
                engine.set_spacing(float(s))
                mtf_at, mtf_mean = engine.extract_metrics(ecfg)
                cell.mtf_at = round(mtf_at, 4)
                cell.mtf_mean = round(mtf_mean, 4)
                row_vals.append(mtf_at)
            except Exception as e:
                cell.status = "failed"
                cell.note = f"{type(e).__name__}: {e}"
                row_fail += 1
            cells.append(cell)

        if row_vals:
            best_i = int(np.argmax(row_vals))
            print(f"  [R {i}/{len(grid.radii)}] R={r:6.2f} mm  "
                  f"最优 S={grid.spacings[best_i]:5.2f} mm  "
                  f"MTF={row_vals[best_i]:.4f}"
                  + (f"  ({row_fail} 点失败)" if row_fail else ""))
        else:
            print(f"  [R {i}/{len(grid.radii)}] R={r:6.2f} mm  全部失败 ({row_fail} 点)")

    print(f"  扫描完成，{grid.n_cells} 个组合，耗时 {time.perf_counter() - t_start:.2f} s")
    return cells


# ================================================================
# 导出
# ================================================================
def to_matrix(cells: list[Cell], grid: GridConfig) -> pd.DataFrame:
    """长表 -> 矩阵：行=曲率半径，列=间距，空值填 NaN"""
    df = pd.DataFrame([{
        "radius_mm": c.radius,
        "spacing_mm": c.spacing,
        "mtf_at_target": c.mtf_at,
        "mtf_mean": c.mtf_mean,
        "status": c.status,
        "note": c.note,
    } for c in cells])

    matrix = df.pivot(index="radius_mm", columns="spacing_mm", values="mtf_at_target")
    matrix = matrix.reindex(index=np.round(grid.radii, 6), columns=np.round(grid.spacings, 6))
    matrix.index.name = "radius_mm"
    matrix.columns.name = "spacing_mm"
    return df, matrix


def export(df_long: pd.DataFrame, matrix: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    p_long = out_dir / "zemax_2d_sweep_long.csv"
    p_mat = out_dir / "zemax_2d_sweep_matrix.csv"

    df_long.to_csv(p_long, index=False, encoding="utf-8-sig")

    # 矩阵里加一行一列表头说明，方便直接看；数值统一保留 4 位
    matrix.round(4).to_csv(p_mat, encoding="utf-8-sig")

    print(f"[OK] 长表已导出: {p_long}")
    print(f"[OK] 矩阵已导出: {p_mat}")
    return p_mat, p_long


# ================================================================
# 绘图
# ================================================================
def setup_chinese_font() -> str:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"):
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    print("[警告] 未找到中文字体，图内中文可能显示为方块")
    return "DejaVu Sans"


def plot_2d(matrix: pd.DataFrame, grid: GridConfig, ecfg: EngineConfig,
            backend: str, out_path: Path) -> None:
    radii = matrix.index.to_numpy(dtype=float)
    spacings = matrix.columns.to_numpy(dtype=float)
    Z = matrix.to_numpy(dtype=float)

    if np.all(np.isnan(Z)):
        print("[警告] 矩阵全为 NaN，跳过绘图")
        return

    X, Y = np.meshgrid(spacings, radii)          # X=间距, Y=半径
    finite = Z[np.isfinite(Z)]
    zmin, zmax = float(finite.min()), float(finite.max())

    # 全局最优
    flat_idx = np.nanargmax(Z)
    ri, ci = np.unravel_index(flat_idx, Z.shape)
    best_r, best_s, best_m = radii[ri], spacings[ci], Z[ri, ci]

    # 实测最优脊线：每个半径下取 MTF 最大的间距
    ridge_s, ridge_r = [], []
    for k in range(Z.shape[0]):
        row = Z[k, :]
        if np.all(np.isnan(row)):
            continue
        j = int(np.nanargmax(row))
        ridge_s.append(spacings[j])
        ridge_r.append(radii[k])

    fig = plt.figure(figsize=(15.0, 6.2), dpi=150, constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.0])

    # ---------------- 左：3D 曲面 ----------------
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    surf = ax1.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none",
                            alpha=0.95, antialiased=True, rstride=1, cstride=1)
    # 底部投影等高线，帮助读出"山谷"位置
    ax1.contourf(X, Y, Z, zdir="z", offset=zmin - (zmax - zmin) * 0.28,
                 levels=12, cmap="viridis", alpha=0.55)
    ax1.scatter([best_s], [best_r], [best_m], color="#E74C3C", s=110,
                marker="*", depthshade=False, zorder=10, label="全局最优")
    ax1.set_xlabel("透镜间距 S (mm)", fontsize=10, labelpad=8)
    ax1.set_ylabel("曲率半径 R (mm)", fontsize=10, labelpad=8)
    ax1.set_zlabel(f"MTF @ {ecfg.freq_target:g} lp/mm", fontsize=10, labelpad=8)
    ax1.set_title("MTF 响应曲面", fontsize=11.5, pad=6)
    ax1.view_init(elev=24, azim=-126)
    ax1.legend(loc="upper left", fontsize=9)
    fig.colorbar(surf, ax=ax1, shrink=0.62, pad=0.08, label="MTF")

    # ---------------- 右：2D 等高线 ----------------
    ax2 = fig.add_subplot(gs[0, 1])
    cf = ax2.contourf(X, Y, Z, levels=14, cmap="viridis")
    cs = ax2.contour(X, Y, Z, levels=7, colors="white", linewidths=0.7, alpha=0.55)
    ax2.clabel(cs, inline=True, fontsize=7, fmt="%.2f")
    fig.colorbar(cf, ax=ax2, fraction=0.046, pad=0.03,
                 label=f"MTF @ {ecfg.freq_target:g} lp/mm")

    # 最优脊线（曲率-间距耦合的证据）
    ax2.plot(ridge_s, ridge_r, "o--", color="#FF6B6B", lw=1.6, ms=5,
             label="各半径下的最优间距（脊线）")
    ax2.plot([best_s], [best_r], "*", color="#E74C3C", ms=20, mec="white",
             mew=0.8, label="全局最优", zorder=10)

    # 最优值 90% 等值线围出的容差区
    ax2.contour(X, Y, Z, levels=[best_m * 0.90], colors="#F39C12",
                linewidths=1.6, linestyles="--")
    ax2.annotate(f"R*={best_r:g} mm\nS*={best_s:g} mm\nMTF={best_m:.4f}",
                 xy=(best_s, best_r), xytext=(26, 18), textcoords="offset points",
                 fontsize=9, color="#1B2631",
                 bbox=dict(boxstyle="round,pad=0.4", fc="#FFF9E6", ec="#B7950B", alpha=0.95),
                 arrowprops=dict(arrowstyle="->", color="#7F8C8D", lw=1.1))

    ax2.set_xlabel("透镜间距 S (mm)", fontsize=11)
    ax2.set_ylabel("曲率半径 R (mm)", fontsize=11)
    ax2.set_title("MTF 等高线图（橙虚线 = 最优值 90% 容差边界）", fontsize=11.5, pad=10)
    ax2.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    tag = "MOCK 仿真" if backend == "mock" else "真机实测"
    fig.suptitle(f"二维参数扫描：曲率半径 × 透镜间距    [{tag}]   {grid.n_cells} 个组合",
                 fontsize=13)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 曲面图已保存: {out_path}")


# ================================================================
# 报告
# ================================================================
def report(df: pd.DataFrame, matrix: pd.DataFrame, ecfg: EngineConfig,
           grid: GridConfig, backend: str) -> None:
    ok = df[df["status"] == "ok"].dropna(subset=["mtf_at_target"])
    print()
    print("=" * 74)
    print("二维扫描结论")
    print("=" * 74)
    print(f"后端        : {backend}   |   组合数: {len(df)}   |   成功: {len(ok)}")

    if ok.empty:
        print("无可用的有效结果，请检查失败原因")
        return

    print(f"面号        : 曲率=第{ecfg.surface_index}面, 间距=第{ecfg.spacing_surface_index}面")
    print(f"MTF 范围    : {ok['mtf_at_target'].min():.4f} ~ {ok['mtf_at_target'].max():.4f}")
    print("-" * 74)

    top = ok.sort_values("mtf_at_target", ascending=False).head(5)
    print("Top 5 参数组合：")
    print(f"  {'排名':<5}{'曲率R(mm)':>12}{'间距S(mm)':>12}{'MTF@目标频率':>14}")
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        print(f"  {rank:<5}{row['radius_mm']:>12.2f}{row['spacing_mm']:>12.2f}"
              f"{row['mtf_at_target']:>14.4f}")

    best = top.iloc[0]
    print("-" * 74)
    print(f"★ 全局最优  : R = {best['radius_mm']:g} mm,  S = {best['spacing_mm']:g} mm, "
          f"MTF = {best['mtf_at_target']:.4f}")

    # 局部敏感性：最优点的四个邻居能掉多少
    z = matrix.to_numpy(dtype=float)
    ri = int(np.where(np.isclose(matrix.index.to_numpy(float), best["radius_mm"]))[0][0])
    ci = int(np.where(np.isclose(matrix.columns.to_numpy(float), best["spacing_mm"]))[0][0])
    neighbors = []
    for dr, dc, label in ((-1, 0, "R-1格"), (1, 0, "R+1格"), (0, -1, "S-1格"), (0, 1, "S+1格")):
        rr, cc = ri + dr, ci + dc
        if 0 <= rr < z.shape[0] and 0 <= cc < z.shape[1] and np.isfinite(z[rr, cc]):
            drop = (best["mtf_at_target"] - z[rr, cc]) / best["mtf_at_target"] * 100
            neighbors.append(f"{label} 下降 {drop:.1f}%")
    if neighbors:
        print(f"  邻点衰减  : {' | '.join(neighbors)}")

    # 耦合判断：脊线是否倾斜
    ridge = []
    for k in range(z.shape[0]):
        if np.all(np.isnan(z[k, :])):
            continue
        ridge.append(matrix.columns.to_numpy(float)[int(np.nanargmax(z[k, :]))])
    if len(ridge) >= 3 and len(grid.spacings) > 1:
        tilt = ridge[-1] - ridge[0]
        span = grid.radii[-1] - grid.radii[0]
        step = float(grid.spacings[1] - grid.spacings[0])
        if abs(tilt) > 0.5 * step:
            print(f"  参数耦合  : 强耦合 —— 曲率变化 {span:g} mm 时，最优间距平移 {tilt:+.1f} mm")
            print("              不能单独定曲率或单独定间距，必须联合优化（见图中脊线）")
        else:
            print(f"  参数耦合  : 弱耦合 —— 最优间距基本不随曲率变化（漂移 {tilt:+.1f} mm）")

    if backend == "mock":
        print("-" * 74)
        print("[注意] 本次为 Mock 仿真结果，用于验证流程与图表正确性，不可作为设计依据。")
    print("=" * 74)


# ================================================================
# 主流程
# ================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Zemax 二维参数扫描（曲率 × 间距）")
    p.add_argument("--backend", choices=["auto", "mock", "real"], default="auto")
    p.add_argument("--lens", type=str, default=None, help=".zmx 绝对路径")
    p.add_argument("--radii", type=str, default="47,53,7",
                   help="曲率: 起,止,点数 或 逗号列表")
    p.add_argument("--spacings", type=str, default="2,10,9",
                   help="间距: 起,止,点数 或 逗号列表")
    p.add_argument("--surface", type=int, default=3, help="曲率所在面号 (默认3)")
    p.add_argument("--spacing-surface", type=int, default=4, help="间距所在面号 (默认4)")
    return p.parse_args()


def parse_range(spec: str) -> np.ndarray:
    """'47,53,7' -> linspace; '47,49,51' -> 直接取值"""
    parts = [float(v) for v in spec.split(",") if v.strip()]
    if len(parts) == 3 and parts[2] == int(parts[2]) and parts[2] >= 2:
        return np.round(np.linspace(parts[0], parts[1], int(parts[2])), 6)
    if len(parts) < 2:
        raise ValueError(f"参数至少需要 2 个值: {spec}")
    return np.round(np.array(parts), 6)


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "output"

    ecfg = EngineConfig(
        use_real=args.backend,
        lens_file=Path(args.lens) if args.lens else None,
        surface_index=args.surface,
        spacing_surface_index=args.spacing_surface,
        work_dir=out_dir / "working_copy",
    )

    engine = None
    try:
        try:
            grid = GridConfig(radii=parse_range(args.radii),
                              spacings=parse_range(args.spacings))
        except ValueError as e:
            print(f"[错误] 网格参数解析失败: {e}", file=sys.stderr)
            return 2

        setup_chinese_font()

        print("=" * 74)
        print("Zemax 二维参数扫描：曲率半径 × 透镜间距")
        print("=" * 74)
        print(f"曲率半径   : {grid.radii[0]:g} ~ {grid.radii[-1]:g} mm, {len(grid.radii)} 点")
        print(f"透镜间距   : {grid.spacings[0]:g} ~ {grid.spacings[-1]:g} mm, {len(grid.spacings)} 点")
        print(f"组合总数   : {grid.n_cells}")
        print(f"作用面     : 曲率=第{ecfg.surface_index}面, 间距=第{ecfg.spacing_surface_index}面")

        engine, note = create_engine(ecfg)
        print(f"当前后端   : {engine.name}  ({note})")
        print("-" * 74)

        cells = run_2d_sweep(engine, ecfg, grid)
        df_long, matrix = to_matrix(cells, grid)
        export(df_long, matrix, out_dir)
        plot_2d(matrix, grid, ecfg, engine.name, out_dir / "zemax_2d_sweep.png")
        report(df_long, matrix, ecfg, grid, engine.name)

        n_fail = int((df_long["status"] != "ok").sum())
        return 0 if n_fail == 0 else 5

    except Exception:
        print("[错误] 未预期异常：", file=sys.stderr)
        traceback.print_exc()
        return 1
    finally:
        if engine is not None:
            try:
                engine.close()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
