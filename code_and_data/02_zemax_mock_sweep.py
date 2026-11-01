# -*- coding: utf-8 -*-
"""
任务 02 — Zemax ZOS-API 批量扫描框架（Mock / 真机 双后端）
================================================================================
业务逻辑：批量修改 Lens 第 3 面的曲率半径 → 读取 MTF → 导出 CSV + 折线图

【架构】业务层与后端完全解耦
    SweepRunner  ──>  OpticsEngine (抽象接口)
                        ├── MockEngine          本机无软件时使用（物理模型仿真）
                        ├── ZospyEngine         真机通道 A：zospy 封装（推荐）
                        └── RawZOSAPIEngine     真机通道 B：pythonnet 直连 ZOS-API

【切到真机三步】
    1. pip install zospy pythonnet
    2. 把下面 Config.use_real 改成 "real"（或命令行加 --backend real）
    3. 填 Config.lens_file 为你的 .zmx 绝对路径
    未装 zospy 时脚本不会崩：会打印降级原因，自动用 Mock 跑完全部流程，
    CSV 里 backend 列会标 "mock"，避免把仿真数据误当真机结果。

【安全】真机模式绝不改动原始文件
    open() 会先把 .zmx 复制到 output/working_copy/ 再在副本上改半径。

【真机适配清单】以下 3 处依赖你的 OpticStudio 版本，已在代码中用 [核对] 标出
    1. ZOSAPI_NetHelper.dll 所在目录（Config.zosapi_dir）
    2. MTF 分析结果的取数接口（GetDataSeries / DataSeries）
    3. 面号索引：ZOS-API 中 LDE 第 0 面是物面，故"第 3 面"= GetSurfaceAt(3)

用法：
    python scripts/02_zemax_mock_sweep.py                     # 自动探测，无软件则 Mock
    python scripts/02_zemax_mock_sweep.py --backend mock      # 强制 Mock
    python scripts/02_zemax_mock_sweep.py --backend real --lens "D:/design/test.zmx"
"""

from __future__ import annotations

import argparse
import shutil
import sys
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


# ================================================================
# 0. 配置
# ================================================================
@dataclass
class Config:
    # ---- 后端选择: "auto" | "mock" | "real" ----
    use_real: str = "auto"

    # ---- 真机参数（切真机时改这里）----
    lens_file: Path | None = None
    zosapi_dir: str = r"C:\Program Files\Zemax\OpticStudio"   # [核对] 你的安装目录
    surface_index: int = 3            # 第 3 面（LDE 第 0 面为物面）

    # ---- 扫描参数 ----
    radii: tuple[float, ...] = (47.0, 48.5, 50.0, 51.5, 53.0)   # 5 个曲率半径 (mm)
    nominal_radius: float = 50.0      # 设计名义值，Mock 模型以此为基准
    freq_target: float = 30.0         # 评价空间频率 (lp/mm)
    max_freq: float = 50.0            # MTF 曲线最大频率 (lp/mm)
    sampling: str = "64x64"           # 光瞳采样

    # ---- Mock 物理参数 ----
    f_number: float = 2.8
    wavelength_mm: float = 0.0005876  # d 线 587.6 nm
    defocus_gain: float = 220.0       # 曲率偏差 → 离焦波前误差(波) 的增益
    mock_noise_rel: float = 0.005     # MTF 读数抖动 0.5%
    seed: int = 2026

    # ---- 路径 ----
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: str = "output"
    csv_name: str = "zemax_sweep.csv"
    png_name: str = "zemax_sweep.png"

    def __post_init__(self):
        self.data_dir_path = self.project_root / self.data_dir
        self.csv_path = self.data_dir_path / self.csv_name
        self.png_path = self.data_dir_path / self.png_name
        self.work_dir = self.data_dir_path / "working_copy"


# ================================================================
# 1. 结果数据结构
# ================================================================
@dataclass
class SweepPoint:
    """单次扫描点的结果（成功或失败都记录下来，便于批量排查）"""
    radius: float
    mtf_at_freq: float | None = None
    mtf_mean: float | None = None
    status: str = "ok"                 # ok | failed | skipped
    backend: str = "mock"
    note: str = ""
    elapsed_s: float = 0.0


# ================================================================
# 2. 后端抽象接口 —— 业务层只认这个接口
# ================================================================
class OpticsEngine(ABC):
    """光学引擎抽象基类。真机 / Mock 只要实现这四个方法，业务层无需改动。"""

    name = "base"

    @abstractmethod
    def open(self) -> None:
        """建立连接并载入镜头副本"""

    @abstractmethod
    def set_radius(self, radius: float) -> None:
        """把目标面的曲率半径改成 radius (mm)"""

    @abstractmethod
    def get_mtf(self) -> tuple[np.ndarray, np.ndarray]:
        """返回 (频率数组 lp/mm, MTF 数组)，长度一致"""

    @abstractmethod
    def close(self) -> None:
        """释放资源"""

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    # ---- 共用的指标提取，两个后端结果口径一致 ----
    def extract_metrics(self, cfg: Config) -> tuple[float, float]:
        freqs, mtf = self.get_mtf()
        if freqs.size == 0 or freqs.size != mtf.size:
            raise ValueError(f"MTF 数据异常: freq={freqs.size}, mtf={mtf.size}")
        mtf_at = float(np.interp(cfg.freq_target, freqs, mtf))
        band = (freqs >= 0) & (freqs <= cfg.max_freq)
        span = float(freqs[band].max() - freqs[band].min())
        if span <= 0:
            raise ValueError("MTF 频率轴跨度为零，无法计算带内均值")
        # numpy 2.x 把 trapz 改名为 trapezoid，这里做双向兼容
        trapz = getattr(np, "trapezoid", None) or np.trapz
        mtf_mean = float(trapz(mtf[band], freqs[band]) / span)
        return mtf_at, mtf_mean


# ================================================================
# 3. Mock 后端 —— 用物理模型生成可信的 MTF，而非乱数
# ================================================================
class MockEngine(OpticsEngine):
    """
    模拟 ZOS-API 行为，但 MTF 不是随机数，而是按真实光学规律算出来的：

      1) 衍射极限 MTF（圆形光阑）：MTF_dl = (2/π)(arccos x − x√(1−x²)),  x = ν/νc
         截止频率 νc = 1/(λ·F#)
      2) 离焦 MTF（几何近似）：MTF_defocus ≈ 1 − 2·W20·λ·ν
         曲率半径偏离最优值 → 产生离焦波前误差 W20（单位：波长）
         W20 = defocus_gain · (c − c_best)/c_best,  c = 1/R
      3) 乘一个基础像差因子 + 微小读数噪声

    这样得到的曲线峰值落在最优半径附近，两侧单调下降 —— 与真实镜头行为一致，
    可以直接用来验证业务逻辑和绘图代码，不会产生"假数据掩盖真问题"的错觉。
    """

    name = "mock"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        # 最优半径略偏离名义值，模拟真实镜头的最优点不在整数上
        self.best_radius = cfg.nominal_radius + 0.2
        self._current = cfg.nominal_radius
        self._opened = False

    def open(self) -> None:
        self._opened = True

    def close(self) -> None:
        self._opened = False

    def set_radius(self, radius: float) -> None:
        if not self._opened:
            raise RuntimeError("引擎未打开，请先调用 open()")
        if radius is None or radius == 0:
            raise ValueError(f"曲率半径非法: {radius}")
        if radius < 0:
            # 负半径=凹面，Mock 模型只覆盖正透镜面，主动报错演示异常路径
            raise ValueError(f"曲率半径 {radius} mm 为负值，Mock 模型仅支持正透镜面")
        self._current = float(radius)

    def get_mtf(self, n_pts: int = 201) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        nu_c = 1.0 / (cfg.wavelength_mm * cfg.f_number)          # 截止频率 lp/mm
        freqs = np.linspace(0.0, cfg.max_freq, n_pts)
        x = np.clip(freqs / nu_c, 0.0, 1.0)

        # 衍射极限
        mtf_dl = (2.0 / np.pi) * (np.arccos(x) - x * np.sqrt(np.maximum(0.0, 1.0 - x * x)))

        # 离焦量：用曲率(1/R)相对偏差，比直接用 ΔR 更贴近物理
        c_now, c_best = 1.0 / self._current, 1.0 / self.best_radius
        w20 = cfg.defocus_gain * (c_now - c_best) / c_best       # 单位：波长

        mtf_defocus = np.clip(1.0 - 2.0 * abs(w20) * cfg.wavelength_mm * freqs, 0.0, 1.0)

        base_aberration = 0.95                                    # 残余像差因子

        mtf = mtf_dl * mtf_defocus * base_aberration
        noise = self.rng.normal(0.0, cfg.mock_noise_rel, size=mtf.shape)
        mtf = np.clip(mtf * (1.0 + noise), 0.0, 1.0)
        return freqs, mtf


# ================================================================
# 4. 真机后端 A —— zospy（推荐，封装好、代码短）
# ================================================================
class ZospyEngine(OpticsEngine):
    """
    zospy 通道。需 pip install zospy。
    [核对] 若你的 zospy < 2.0，把 connect() 换成 create_new_application()。
    """

    name = "zospy"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.zp = None
        self.oss = None
        self._working_file: Path | None = None

    def open(self) -> None:
        import zospy as zp                    # 失败由上层工厂捕获

        if self.cfg.lens_file is None:
            raise FileNotFoundError("Config.lens_file 未设置，无法连真机")

        # 安全第一：先复制，绝不动原始文件
        self.cfg.work_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._working_file = self.cfg.work_dir / f"{self.cfg.lens_file.stem}_{stamp}.zmx"
        shutil.copy2(self.cfg.lens_file, self._working_file)

        self.zp = zp
        self.zos = zp.ZOS()
        try:
            self.oss = self.zos.connect()               # zospy >= 2.0
        except AttributeError:
            self.oss = self.zos.create_new_application()  # zospy < 2.0 兼容

        self.oss.load(str(self._working_file))

    def set_radius(self, radius: float) -> None:
        surface = self.oss.LDE.GetSurfaceAt(self.cfg.surface_index)
        surface.Radius = float(radius)
        # zospy 会自动触发重算；若报错可显式调用:
        # self.oss.LDE.GetSurfaceAt(...) 后 self.oss.Update()

    def get_mtf(self) -> tuple[np.ndarray, np.ndarray]:
        """[核对] zospy 2.x 的 MTF 接口；不同版本返回结构不同，必要时改这里"""
        analysis = self.zp.analyses.mtf.FFTMTF(
            fov=1, wavelength=1, max_frequency=self.cfg.max_freq,
            sampling=self.cfg.sampling,
        )
        result = analysis.run(self.oss)
        df = result.Data                              # DataFrame: index=频率, 列=各视场

        freqs = np.asarray(df.index, dtype=float)
        mtf = np.asarray(df.iloc[:, 0], dtype=float)  # 取第 1 视场
        return freqs, mtf

    def close(self) -> None:
        try:
            if self.oss is not None:
                self.oss.close()                      # 关闭副本，原始文件从未被打开写入
        except Exception:
            pass


# ================================================================
# 5. 真机后端 B —— 原生 ZOS-API（pythonnet 直连，跨版本最稳）
# ================================================================
class RawZOSAPIEngine(OpticsEngine):
    """
    不依赖 zospy，直接 pythonnet 加载 ZOSAPI_NetHelper.dll。
    适用场景：公司电脑装了 OpticStudio 但 pip 装不了 zospy（内网/权限限制）。
    """

    name = "zos-api"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.app = None
        self.system = None
        self._working_file: Path | None = None

    def open(self) -> None:
        import clr                                    # pythonnet

        api_dir = Path(self.cfg.zosapi_dir)           # [核对] 安装目录
        helper = api_dir / "ZOSAPI_NetHelper.dll"
        if not helper.exists():
            raise FileNotFoundError(
                f"未找到 {helper}，请把 Config.zosapi_dir 改成本机 OpticStudio 安装目录"
            )
        if self.cfg.lens_file is None:
            raise FileNotFoundError("Config.lens_file 未设置，无法连真机")

        sys.path.append(str(api_dir))
        clr.AddReference(str(helper))
        import ZOSAPI_NetHelper

        if not ZOSAPI_NetHelper.ZOSAPI_Initializer.Initialize(str(api_dir)):
            raise RuntimeError("ZOSAPI_Initializer.Initialize 失败，检查授权(License)是否可用")

        import ZOSAPI
        self._ZOSAPI = ZOSAPI

        conn = ZOSAPI.ZOSAPI_Connection()
        self.app = conn.CreateNewApplication()
        if self.app is None:                          # 已有实例在跑时改为连接
            self.app = conn.ConnectToApplication()
        if self.app is None:
            raise RuntimeError("无法创建/连接 OpticStudio 实例")

        self.system = self.app.PrimarySystem

        # 安全第一：复制副本再改
        self.cfg.work_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._working_file = self.cfg.work_dir / f"{self.cfg.lens_file.stem}_{stamp}.zmx"
        shutil.copy2(self.cfg.lens_file, self._working_file)

        if not self.system.LoadFile(str(self._working_file), False):
            raise RuntimeError(f"LoadFile 失败: {self._working_file}")

    def set_radius(self, radius: float) -> None:
        # LDE.GetSurfaceAt(i) 返回 ILDESurface，Radius 为可写属性
        surface = self.system.LDE.GetSurfaceAt(self.cfg.surface_index)
        surface.Radius = float(radius)

    def get_mtf(self) -> tuple[np.ndarray, np.ndarray]:
        """[核对] MTF 分析取数：不同版本结果接口略有差异"""
        ZOSAPI = self._ZOSAPI
        analysis = self.system.Analyses.New_Analysis(
            ZOSAPI.Analysis.AnalysisIDM.FftMtf
        )
        settings = analysis.GetSettings()
        settings.MaximumFrequency = float(self.cfg.max_freq)
        # settings.SampleSize = ZOSAPI.Analysis.SampleSizes.S_64x64   # 视版本而定

        analysis.ApplyAndWaitForCompletion()
        results = analysis.GetResults()

        n_series = results.NumberOfDataSeries
        if n_series == 0:
            raise RuntimeError("MTF 分析返回 0 条曲线（可能镜头未设置视场/波长）")

        series = results.GetDataSeries(0)             # 第 1 视场
        freqs = np.asarray(list(series.XData.Data), dtype=float)
        mtf = np.asarray(list(series.YData.Data), dtype=float)
        analysis.Close()
        return freqs, mtf

    def close(self) -> None:
        try:
            if self.app is not None:
                self.app.CloseApplication()           # 只关副本
        except Exception:
            pass


# ================================================================
# 6. 后端工厂 —— 优雅降级的核心
# ================================================================
def create_engine(cfg: Config) -> tuple[OpticsEngine, str]:
    """
    按优先级选后端，任何一步失败都降级到 Mock，且返回降级原因字符串。
    返回值: (引擎实例, 说明信息)
    """
    want_real = cfg.use_real.lower() in ("real", "auto", "true", "1")
    if cfg.use_real.lower() == "mock":
        eng = MockEngine(cfg)
        eng.open()
        return eng, "用户强制指定 Mock"

    if not want_real:
        eng = MockEngine(cfg)
        eng.open()
        return eng, "配置未启用真机"

    reasons = []

    # ---- 通道 A: zospy ----
    try:
        import zospy  # noqa: F401
        eng = ZospyEngine(cfg)
        eng.open()
        return eng, "真机已连接（zospy 通道）"
    except ImportError as e:
        reasons.append(f"zospy 未安装 ({e})")
    except Exception as e:
        reasons.append(f"zospy 通道失败: {type(e).__name__}: {e}")

    # ---- 通道 B: 原生 ZOS-API ----
    try:
        eng = RawZOSAPIEngine(cfg)
        eng.open()
        return eng, "真机已连接（原生 ZOS-API 通道）"
    except ImportError as e:
        reasons.append(f"pythonnet 未安装 ({e})")
    except Exception as e:
        reasons.append(f"原生 ZOS-API 通道失败: {type(e).__name__}: {e}")

    # ---- 降级 ----
    detail = " | ".join(reasons)
    print("[降级] 真机不可用，自动切换 Mock 后端")
    for r in reasons:
        print(f"        - {r}")
    eng = MockEngine(cfg)
    eng.open()
    return eng, f"降级为 Mock:{detail}"


# ================================================================
# 7. 业务层 —— 完全不关心后端是谁
# ================================================================
def run_sweep(engine: OpticsEngine, cfg: Config) -> list[SweepPoint]:
    """批量扫描：每个点独立捕获异常，单点失败不中断整批"""
    import time

    results: list[SweepPoint] = []
    total = len(cfg.radii)

    for i, r in enumerate(cfg.radii, 1):
        point = SweepPoint(radius=r, backend=engine.name)
        t0 = time.perf_counter()
        try:
            engine.set_radius(r)
            mtf_at, mtf_mean = engine.extract_metrics(cfg)
            point.mtf_at_freq = round(mtf_at, 4)
            point.mtf_mean = round(mtf_mean, 4)
            point.status = "ok"
            print(f"  [{i}/{total}] R={r:7.2f} mm  MTF@{cfg.freq_target:g}lp={mtf_at:.4f}"
                  f"  均值={mtf_mean:.4f}")
        except Exception as e:
            point.status = "failed"
            point.note = f"{type(e).__name__}: {e}"
            print(f"  [{i}/{total}] R={r:7.2f} mm  失败 -> {point.note}")
        finally:
            point.elapsed_s = round(time.perf_counter() - t0, 3)
            results.append(point)
    return results


# ================================================================
# 8. 导出 CSV
# ================================================================
def export_csv(points: list[SweepPoint], cfg: Config) -> pd.DataFrame:
    df = pd.DataFrame([{
        "radius_mm": p.radius,
        "mtf_at_target_freq": p.mtf_at_freq,
        f"freq_lp_per_mm": cfg.freq_target,
        "mtf_mean_0_to_max": p.mtf_mean,
        "status": p.status,
        "backend": p.backend,
        "elapsed_s": p.elapsed_s,
        "note": p.note,
    } for p in points])

    cfg.data_dir_path.mkdir(parents=True, exist_ok=True)
    # 加 UTF-8 BOM，Excel 双击打开不乱码（工程交付常见坑）
    df.to_csv(cfg.csv_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 结果已导出: {cfg.csv_path}")
    return df


# ================================================================
# 9. 绘图
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


def plot_sweep(df: pd.DataFrame, cfg: Config, engine_note: str) -> None:
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        print("[警告] 无有效扫描点，跳过绘图")
        return

    fig, ax1 = plt.subplots(figsize=(9.2, 5.6), dpi=150, constrained_layout=True)

    ax1.plot(ok["radius_mm"], ok["mtf_at_target_freq"], "o-", lw=2.0, ms=8,
             color="#C0392B", label=f"MTF @ {cfg.freq_target:g} lp/mm")
    ax1.set_xlabel("第 3 面曲率半径 R (mm)", fontsize=11)
    ax1.set_ylabel(f"MTF @ {cfg.freq_target:g} lp/mm", fontsize=11, color="#C0392B")
    ax1.tick_params(axis="y", labelcolor="#C0392B")
    ax1.grid(alpha=0.28, ls=":")

    # 次坐标轴：频带平均 MTF
    ax2 = ax1.twinx()
    ax2.plot(ok["radius_mm"], ok["mtf_mean_0_to_max"], "s--", lw=1.5, ms=6,
             color="#2C5F8A", alpha=0.9, label=f"MTF 均值 (0~{cfg.max_freq:g} lp/mm)")
    ax2.set_ylabel(f"平均 MTF (0~{cfg.max_freq:g} lp/mm)", fontsize=11, color="#2C5F8A")
    ax2.tick_params(axis="y", labelcolor="#2C5F8A")

    # 标注峰值
    idx = ok["mtf_at_target_freq"].idxmax()
    best_r = float(ok.loc[idx, "radius_mm"])
    best_m = float(ok.loc[idx, "mtf_at_target_freq"])
    ax1.axvline(best_r, ls="-.", lw=1.2, color="gray", alpha=0.75)
    ax1.annotate(f"最优 R = {best_r:g} mm\nMTF = {best_m:.4f}",
                 xy=(best_r, best_m), xytext=(-95, -58), textcoords="offset points",
                 fontsize=9.5, color="#1B2631",
                 bbox=dict(boxstyle="round,pad=0.4", fc="#FFF9E6", ec="#B7950B", alpha=0.95),
                 arrowprops=dict(arrowstyle="->", color="#7F8C8D", lw=1.1))

    backend_label = ok["backend"].iloc[0]
    tag = "MOCK 仿真数据（非真机结果）" if backend_label == "mock" else "真机 ZOS-API 实测"
    ax1.set_title(f"曲率半径扫描 · MTF 响应    [{tag}]", fontsize=12.5, pad=12)

    # 合并两个轴的图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower center",
               fontsize=9.5, framealpha=0.9)

    fig.text(0.5, -0.02, f"后端: {engine_note}", ha="center", fontsize=8.5, color="gray")

    cfg.png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 折线图已保存: {cfg.png_path}")


# ================================================================
# 10. 主流程
# ================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Zemax 曲率半径批量扫描（Mock/真机双后端）")
    p.add_argument("--backend", choices=["auto", "mock", "real"], default="auto",
                   help="auto=有软件连真机,否则Mock; mock=强制仿真; real=强制真机")
    p.add_argument("--lens", type=str, default=None, help="镜头文件 .zmx 绝对路径")
    p.add_argument("--radii", type=str, default=None,
                   help="逗号分隔的半径列表, 如 47,48.5,50,51.5,53")
    p.add_argument("--surface", type=int, default=None, help="面号(默认3)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = Config()
    cfg.use_real = args.backend
    if args.lens:
        cfg.lens_file = Path(args.lens)
    if args.radii:
        try:
            cfg.radii = tuple(float(v) for v in args.radii.split(",") if v.strip())
        except ValueError:
            print(f"[错误] --radii 解析失败: {args.radii}", file=sys.stderr)
            return 2
    if args.surface:
        cfg.surface_index = args.surface

    engine = None
    try:
        setup_chinese_font()

        print("=" * 70)
        print("Zemax 曲率半径批量扫描")
        print("=" * 70)
        print(f"目标面     : 第 {cfg.surface_index} 面")
        print(f"扫描半径   : {', '.join(f'{r:g}' for r in cfg.radii)} mm")
        print(f"评价指标   : MTF @ {cfg.freq_target:g} lp/mm + 0~{cfg.max_freq:g} lp/mm 均值")

        engine, note = create_engine(cfg)
        print(f"当前后端   : {engine.name}  ({note})")
        print("-" * 70)

        points = run_sweep(engine, cfg)

        df = export_csv(points, cfg)
        plot_sweep(df, cfg, note)

        # ---- 汇总 ----
        n_ok = int((df["status"] == "ok").sum())
        n_fail = len(df) - n_ok
        print("-" * 70)
        print(f"完成: 成功 {n_ok} / {len(df)} 点" + (f"，失败 {n_fail} 点" if n_fail else ""))
        if n_ok:
            best = df[df["status"] == "ok"].sort_values("mtf_at_target_freq", ascending=False).iloc[0]
            print(f"最优半径: {best['radius_mm']:g} mm  "
                  f"MTF@{cfg.freq_target:g}lp = {best['mtf_at_target_freq']:.4f}")
            if engine.name == "mock":
                print("\n[注意] 本次为 Mock 仿真结果，仅验证流程正确性，不可作为设计依据。")
                print("       切真机: --backend real --lens \"你的.zmx路径\"")
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
