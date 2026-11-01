# -*- coding: utf-8 -*-
"""
zemax_engine.py — Zemax 后端共享层（Mock / zospy / 原生 ZOS-API）
================================================================================
被 03_zemax_2d_sweep.py 引用。单文件便携版见 02_zemax_mock_sweep.py（引擎逻辑内嵌）。

设计要点：
  1. OpticsEngine 抽象出 4 个动作：open / set_radius / set_spacing / get_mtf
     业务层（扫描循环、出表、绘图）完全不感知后端是谁。
  2. create_engine() 三级降级：zospy → 原生 ZOS-API → Mock，全程不崩。
  3. 真机一律先复制 .zmx 副本再操作，原始设计文件永不被写入。

面号语义（很容易搞错，务必确认）：
  surface_index         改曲率的面号，LDE 第 0 面为物面
  spacing_surface_index 改"间距"的面号 —— Thickness 属于面属性：
      若该面是镜片前表面，Thickness = 玻璃中心厚度
      若该面是镜片后表面，Thickness = 该面之后的空气间隔  <- 通常你要的是这个
"""

from __future__ import annotations

import shutil
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np


# ================================================================
# 配置
# ================================================================
@dataclass
class EngineConfig:
    # ---- 后端 ----
    use_real: str = "auto"                     # auto | mock | real
    lens_file: Path | None = None
    zosapi_dir: str = r"C:\Program Files\Zemax\OpticStudio"   # [核对]

    # ---- 面号 ----
    surface_index: int = 3                     # 曲率半径所在面
    spacing_surface_index: int = 4             # 间距(Thickness)所在面

    # ---- 分析 ----
    freq_target: float = 30.0                  # 评价频率 lp/mm
    max_freq: float = 50.0                     # MTF 曲线上限
    sampling: str = "64x64"

    # ---- Mock 物理参数 ----
    nominal_radius: float = 50.0
    nominal_spacing: float = 6.0
    f_number: float = 2.8
    wavelength_mm: float = 0.0005876
    gain_space: float = 2.5         # 间距偏离最优关系(mm) -> 离焦波前(波)
    gain_ho: float = 120.0          # 曲率相对偏差 -> 高阶像差波前(波)
    focus_comp: float = 1.0         # 焦点补偿: 每 1mm 曲率变化对应的最优间距位移
    mock_noise_rel: float = 0.005
    seed: int = 2026

    # ---- 路径 ----
    work_dir: Path | None = None

    def __post_init__(self):
        if self.work_dir is None:
            self.work_dir = Path.cwd() / "output" / "working_copy"


# ================================================================
# 抽象接口
# ================================================================
class OpticsEngine(ABC):
    name = "base"

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def set_radius(self, radius: float) -> None: ...

    @abstractmethod
    def set_spacing(self, spacing: float) -> None: ...

    @abstractmethod
    def get_mtf(self) -> tuple[np.ndarray, np.ndarray]: ...

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def extract_metrics(self, cfg: EngineConfig) -> tuple[float, float]:
        """统一指标口径：目标频率处 MTF + 频带平均 MTF"""
        freqs, mtf = self.get_mtf()
        if freqs.size == 0 or freqs.size != mtf.size:
            raise ValueError(f"MTF 数据异常: freq={freqs.size}, mtf={mtf.size}")
        mtf_at = float(np.interp(cfg.freq_target, freqs, mtf))
        band = (freqs >= 0) & (freqs <= cfg.max_freq)
        span = float(freqs[band].max() - freqs[band].min())
        if span <= 0:
            raise ValueError("MTF 频率轴跨度为零")
        trapz = getattr(np, "trapezoid", None) or np.trapz
        mtf_mean = float(trapz(mtf[band], freqs[band]) / span)
        return mtf_at, mtf_mean


# ================================================================
# Mock 后端
# ================================================================
class MockEngine(OpticsEngine):
    """
    MockEngine 的 MTF 由两个物理量决定，不是随机数：

      离焦波前（只由"间距偏离最优关系"决定）:
          W20 = gain_space · (S − S_best(R))
          S_best(R) = S0 + focus_comp·(Rb − R)
          含义：曲率变化引起的焦移可由间距补偿，二者存在一条最优关系；
          偏离这条关系才产生离焦。这条关系在二维图上就是那条斜脊线。
      高阶像差（曲率偏离引入，间距无法补偿）:
          W_ho = gain_ho·|(c−c_b)/c_b|,  c = 1/R
      两者按 RMS 平方和叠加（独立像差的标准处理）:
          W_rms = sqrt(W20² + W_ho²)
      离焦 MTF 几何近似 + 衍射极限 + 残余像差因子：
          MTF(ν) = MTF_dl(ν)·max(0, 1 − 2·W_rms·λ·ν)·0.95

    [注意] 各增益按可视化友好区间缩放，绝对量级不代表真实公差灵敏度，仅用于验证流程。
    """

    name = "mock"

    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.r_best = cfg.nominal_radius + 0.2
        self.s_best0 = cfg.nominal_spacing
        self._R = cfg.nominal_radius
        self._S = cfg.nominal_spacing
        self._opened = False

    def open(self) -> None:
        self._opened = True

    def close(self) -> None:
        self._opened = False

    def _check(self, v: float, label: str) -> float:
        if not self._opened:
            raise RuntimeError("引擎未打开，请先调用 open()")
        if v is None or not np.isfinite(v):
            raise ValueError(f"{label} 非法: {v}")
        return float(v)

    def set_radius(self, radius: float) -> None:
        radius = self._check(radius, "曲率半径")
        if radius <= 0:
            raise ValueError(f"曲率半径 {radius} mm 非正，Mock 模型仅支持正透镜面")
        self._R = radius

    def set_spacing(self, spacing: float) -> None:
        spacing = self._check(spacing, "透镜间距")
        if spacing < 0:
            raise ValueError(f"间距 {spacing} mm 为负，物理上不成立")
        self._S = spacing

    def best_spacing_at(self, radius: float) -> float:
        """给定曲率下的最优间距（焦点补偿关系）"""
        return self.s_best0 + self.cfg.focus_comp * (self.r_best - radius)

    def get_mtf(self, n_pts: int = 201) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        nu_c = 1.0 / (cfg.wavelength_mm * cfg.f_number)
        freqs = np.linspace(0.0, cfg.max_freq, n_pts)
        x = np.clip(freqs / nu_c, 0.0, 1.0)

        mtf_dl = (2.0 / np.pi) * (np.arccos(x) - x * np.sqrt(np.maximum(0.0, 1.0 - x * x)))

        rel_c = (1.0 / self._R - 1.0 / self.r_best) / (1.0 / self.r_best)   # 曲率相对偏差
        # 离焦：间距偏离"当前曲率下的最优间距"多少；偏离量才产生离焦
        w20 = cfg.gain_space * (self._S - self.best_spacing_at(self._R))
        # 高阶像差：曲率偏离引入，无法靠间距补偿
        w_ho = cfg.gain_ho * abs(rel_c)
        w_rms = float(np.hypot(w20, w_ho))

        mtf_defocus = np.clip(1.0 - 2.0 * w_rms * cfg.wavelength_mm * freqs, 0.0, 1.0)
        mtf = mtf_dl * mtf_defocus * 0.95
        mtf = np.clip(mtf * (1.0 + self.rng.normal(0.0, cfg.mock_noise_rel, mtf.shape)), 0.0, 1.0)
        return freqs, mtf


# ================================================================
# 真机后端 A：zospy
# ================================================================
class ZospyEngine(OpticsEngine):
    name = "zospy"

    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.zp = None
        self.oss = None

    def _prepare_copy(self) -> Path:
        if self.cfg.lens_file is None:
            raise FileNotFoundError("lens_file 未设置")
        self.cfg.work_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = self.cfg.work_dir / f"{self.cfg.lens_file.stem}_{stamp}.zmx"
        shutil.copy2(self.cfg.lens_file, dst)     # 安全第一：只改副本
        return dst

    def open(self) -> None:
        import zospy as zp

        working = self._prepare_copy()
        self.zp = zp
        self.zos = zp.ZOS()
        try:
            self.oss = self.zos.connect()                  # zospy >= 2.0
        except AttributeError:
            self.oss = self.zos.create_new_application()   # zospy < 2.0
        self.oss.load(str(working))

    def set_radius(self, radius: float) -> None:
        self.oss.LDE.GetSurfaceAt(self.cfg.surface_index).Radius = float(radius)

    def set_spacing(self, spacing: float) -> None:
        self.oss.LDE.GetSurfaceAt(self.cfg.spacing_surface_index).Thickness = float(spacing)

    def get_mtf(self) -> tuple[np.ndarray, np.ndarray]:
        """[核对] zospy 版本差异点"""
        analysis = self.zp.analyses.mtf.FFTMTF(
            fov=1, wavelength=1, max_frequency=self.cfg.max_freq, sampling=self.cfg.sampling
        )
        df = analysis.run(self.oss).Data
        return np.asarray(df.index, dtype=float), np.asarray(df.iloc[:, 0], dtype=float)

    def close(self) -> None:
        try:
            if self.oss is not None:
                self.oss.close()
        except Exception:
            pass


# ================================================================
# 真机后端 B：原生 ZOS-API
# ================================================================
class RawZOSAPIEngine(OpticsEngine):
    name = "zos-api"

    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.app = None
        self.system = None
        self._ZOSAPI = None

    def open(self) -> None:
        import clr

        api_dir = Path(self.cfg.zosapi_dir)               # [核对]
        helper = api_dir / "ZOSAPI_NetHelper.dll"
        if not helper.exists():
            raise FileNotFoundError(f"未找到 {helper}，请修改 zosapi_dir")
        if self.cfg.lens_file is None:
            raise FileNotFoundError("lens_file 未设置")

        sys.path.append(str(api_dir))
        clr.AddReference(str(helper))
        import ZOSAPI_NetHelper

        if not ZOSAPI_NetHelper.ZOSAPI_Initializer.Initialize(str(api_dir)):
            raise RuntimeError("ZOSAPI 初始化失败，检查授权是否可用")

        import ZOSAPI
        self._ZOSAPI = ZOSAPI
        conn = ZOSAPI.ZOSAPI_Connection()
        self.app = conn.CreateNewApplication() or conn.ConnectToApplication()
        if self.app is None:
            raise RuntimeError("无法创建/连接 OpticStudio 实例")
        self.system = self.app.PrimarySystem

        # 复制副本
        self.cfg.work_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        working = self.cfg.work_dir / f"{self.cfg.lens_file.stem}_{stamp}.zmx"
        shutil.copy2(self.cfg.lens_file, working)

        if not self.system.LoadFile(str(working), False):
            raise RuntimeError(f"LoadFile 失败: {working}")

    def set_radius(self, radius: float) -> None:
        self.system.LDE.GetSurfaceAt(self.cfg.surface_index).Radius = float(radius)

    def set_spacing(self, spacing: float) -> None:
        self.system.LDE.GetSurfaceAt(self.cfg.spacing_surface_index).Thickness = float(spacing)

    def get_mtf(self) -> tuple[np.ndarray, np.ndarray]:
        """[核对] MTF 结果取数接口"""
        ZOSAPI = self._ZOSAPI
        analysis = self.system.Analyses.New_Analysis(ZOSAPI.Analysis.AnalysisIDM.FftMtf)
        analysis.GetSettings().MaximumFrequency = float(self.cfg.max_freq)
        analysis.ApplyAndWaitForCompletion()
        results = analysis.GetResults()
        if results.NumberOfDataSeries == 0:
            raise RuntimeError("MTF 返回 0 条曲线（检查视场/波长设置）")
        series = results.GetDataSeries(0)
        freqs = np.asarray(list(series.XData.Data), dtype=float)
        mtf = np.asarray(list(series.YData.Data), dtype=float)
        analysis.Close()
        return freqs, mtf

    def close(self) -> None:
        try:
            if self.app is not None:
                self.app.CloseApplication()
        except Exception:
            pass


# ================================================================
# 工厂：三级降级
# ================================================================
def create_engine(cfg: EngineConfig) -> tuple[OpticsEngine, str]:
    mode = (cfg.use_real or "auto").lower()
    if mode == "mock":
        eng = MockEngine(cfg)
        eng.open()
        return eng, "用户强制指定 Mock"
    if mode not in ("real", "auto", "true", "1"):
        eng = MockEngine(cfg)
        eng.open()
        return eng, "配置未启用真机"

    reasons = []
    try:
        import zospy  # noqa: F401
        eng = ZospyEngine(cfg)
        eng.open()
        return eng, "真机已连接（zospy 通道）"
    except ImportError as e:
        reasons.append(f"zospy 未安装 ({e})")
    except Exception as e:
        reasons.append(f"zospy 通道失败: {type(e).__name__}: {e}")

    try:
        eng = RawZOSAPIEngine(cfg)
        eng.open()
        return eng, "真机已连接（原生 ZOS-API 通道）"
    except ImportError as e:
        reasons.append(f"pythonnet 未安装 ({e})")
    except Exception as e:
        reasons.append(f"原生 ZOS-API 通道失败: {type(e).__name__}: {e}")

    print("[降级] 真机不可用，自动切换 Mock 后端")
    for r in reasons:
        print(f"        - {r}")
    eng = MockEngine(cfg)
    eng.open()
    return eng, "降级为 Mock: " + " | ".join(reasons)
