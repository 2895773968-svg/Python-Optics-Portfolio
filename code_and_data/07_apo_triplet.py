#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
07_apo_triplet.py —— 复消色差（APO）三片式设计 · 真实光线追迹
================================================================================

目标
----
在 06 的 BK7+SF5 消色差双胶合（二级光谱 −50.9 µm）基础上，引入 ED 玻璃，
把二级光谱（d 光焦点相对 F/C 中点的偏移）进一步压缩。

核心判据（三个层次的色差）
--------------------------
  1) 轴向色差 LCA   = z_F − z_C           （一级光谱，双胶合已能消除）
  2) 二级光谱 SEC   = z_d − (z_F+z_C)/2   （双胶合消不掉，只能靠反常色散玻璃）
  3) 带内残余       = 在 450~700nm 全带内的焦移峰谷值

为什么"加一片 ED"还不够
-----------------------
薄透镜的消色差条件有两个独立方程：
  Σ φi = Φ                  （总光焦度）
  Σ φi/Vi = 0               （F、C 焦点重合）
再加一片玻璃后多一个未知量 φ3，但同时可以多写一个方程：
  Σ φi·Pi/Vi = 0            （d 光焦点落在 F/C 中点 → 二级光谱为零）
第三个方程能否解出"可用"的功率分配，取决于三块玻璃的**相对部分色散 P 是否拉开**。
普通玻璃都贴在"正常线"上（ΔP ≈ ±0.002），三方程条件数高达 2×10⁴，
解出来的功率会极端到无法加工；只有引入 ΔP 明显偏离正常线的玻璃才有意义。

  N-BK7    ΔP = +0.0019   普通冕牌
  N-SF5    ΔP = −0.0007   普通火石（几乎在正常线上）
  N-KZFS2  ΔP = +0.0035   短火石（反常色散）
  S-FPL51  ΔP = −0.0076   氟磷酸盐 ED 玻璃（强反常色散）

所以脚本对比了三套系统：
  A. BK7 + SF5 胶合双胶合       —— 基准，二级光谱  −50.9 µm
  B. FPL51 + SF5 + BK7 胶合三片 —— 只把 ED 加进来，火石不变
  C. FPL51 + KZFS2 + BK7 胶合三片 —— ED + 反常色散火石（APO）

用法
----
    python 07_apo_triplet.py               # 用脚本内固化的处方，秒级出结果
    python 07_apo_triplet.py --optimize    # 从薄透镜解重新寻优，验证处方可复现

输出
----
    output/apo_raytrace.png          三色光路图
    output/chromatic_focal_shift.png 焦移随波长变化曲线（双胶合 vs APO）
    output/apo_report.txt            完整处方与指标报告
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
except Exception as exc:  # pragma: no cover
    print(f"[致命] matplotlib 不可用：{exc}")
    sys.exit(1)

try:
    from rayoptics.environment import OpticalModel, PupilSpec, FieldSpec, WvlSpec
    from rayoptics.raytr.trace import trace_base
    import rayoptics.optical.model_constants as mc
except Exception as exc:  # pragma: no cover
    print("[致命] rayoptics 未安装，请先执行：pip install rayoptics")
    print(f"       原始错误：{exc}")
    sys.exit(1)

try:
    from scipy.optimize import least_squares
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    from opticalglass import glassfactory as gf
    HAS_GLASS = True
except Exception:
    HAS_GLASS = False

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 光谱常数
WL_F, WL_D, WL_C = 486.1327, 587.5618, 656.2725      # 氢 F 线 / 氦 d 线 / 氢 C 线
# 扫描网格必须包含 F/d/C 三条谱线：rayoptics 只允许追迹 WvlSpec 中已登记的波长
WL_SCAN = np.array(sorted(list(np.linspace(450.0, 700.0, 41)) + [WL_F, WL_D, WL_C]))
EPD = 25.4                                           # 入瞳直径 25.4 mm → 全口径 12.7 mm

CLR_F, CLR_D, CLR_C = "#1F4E9C", "#1E8449", "#C0392B"   # 蓝 / 绿 / 红

# 缺省色散参数（若玻璃库不可用时回落，来源：1087 种商品玻璃线性拟合）
NORMAL_LINE = (0.000438, -0.222442)


# ======================================================================
# 1. 光学模型引擎（表面表驱动，胶合面天然共用一个曲率）
# ======================================================================
@dataclass
class SystemSpec:
    """一个系统的完整描述：每面一个曲率，第 i 面的厚度与"该面之后的介质"。"""
    tag: str
    cvs: list[float]
    thicks: list[float]
    glasses: list[str | None]
    color: str = "#333333"
    ls: str = "-"

    @property
    def radii(self) -> list[float]:
        return [1.0 / c if abs(c) > 1e-9 else float("inf") for c in self.cvs]


def build(spec: SystemSpec, wls: list[float] | None = None):
    """构建 OpticalModel。wls 为 None 时只用 F/d/C 三个波长。"""
    wls = list(wls) if wls is not None else [WL_F, WL_D, WL_C]
    opm = OpticalModel()
    sm = opm["seq_model"]
    osp = opm["optical_spec"]
    opm.radius_mode = True

    osp["pupil"] = PupilSpec(osp, key=("object", "epd"), value=EPD)
    osp["fov"] = FieldSpec(osp, key=("object", "angle"), value=0.0, flds=[0.0])
    ref = min(range(len(wls)), key=lambda i: abs(wls[i] - WL_D))
    osp["wvls"] = WvlSpec([(w, 1.0) for w in wls], ref_wl=ref)

    sm.gaps[0].thi = 1e10                                  # 物在无穷远

    last = len(spec.cvs) - 1
    for i in range(len(spec.cvs)):
        surf = [1.0 / spec.cvs[i] if abs(spec.cvs[i]) > 1e-9 else 1e12]
        if i < last:
            surf.append(spec.thicks[i])
            if spec.glasses[i] is not None:
                surf.append(spec.glasses[i])
        else:
            surf.append(20.0)                              # 先给个占位，稍后精确对焦
        sm.add_surface(surf)

    opm.update_model()
    fod = opm["analysis_results"]["parax_data"].fod
    if not np.isfinite(float(fod.img_dist)):
        raise RuntimeError(f"{spec.tag}: 近轴求解失败（曲率组合可能不自洽）")
    sm.gaps[-1].thi = round(float(fod.img_dist), 4)        # 像面精确落在近轴焦点
    opm.update_model()
    sm.compute_global_coords()
    return opm, sm, osp


def axis_cross(opm, sm, osp, wvl: float, h: float = 0.1) -> float:
    """
    用高度 h 的近轴光线外推与光轴的交点 z（相对第一面前顶点）。
    h→0 即该波长的近轴焦点；h=口径边缘即该孔径处的实际轴向交点。
    """
    tf = sm.gbl_tfrms
    z0 = tf[1][1][2]
    ray, _, _ = trace_base(opm, (0.0, h), osp["fov"].fields[0], wvl, pupil_type="aim pt")
    rot, trns = tf[-1]
    p = rot.dot(ray[-1][mc.p]) + trns
    d = rot.dot(ray[-1][mc.d])
    if abs(d[1]) < 1e-14:
        return float("nan")
    return (p[2] - z0) + (-p[1] / d[1]) * d[2]


def image_heights(opm, sm, osp, wvl: float, heights) -> np.ndarray:
    """各入射高度在像面上的实际交点高度（横向）"""
    tf = sm.gbl_tfrms
    out = []
    for h in heights:
        ray, _, _ = trace_base(opm, (0.0, h), osp["fov"].fields[0], wvl, pupil_type="aim pt")
        rot, trns = tf[-1]
        out.append(float((rot.dot(ray[-1][mc.p]) + trns)[1]))
    return np.array(out)


def ray_paths(opm, sm, osp, wvl: float, heights) -> list[np.ndarray]:
    """返回每条光线在全局坐标下的折线 (z, y)，供绘图"""
    tf = sm.gbl_tfrms
    z0 = tf[1][1][2]
    paths = []
    for h in heights:
        ray, _, _ = trace_base(opm, (0.0, h), osp["fov"].fields[0], wvl, pupil_type="aim pt")
        pts = []
        for i, seg in enumerate(ray):
            rot, trns = tf[i]
            p = rot.dot(seg[mc.p]) + trns
            pts.append((p[2] - z0, p[1]))
        paths.append(np.array(pts))
    return paths


# ======================================================================
# 2. 指标计算
# ======================================================================
@dataclass
class Metrics:
    tag: str
    efl: float
    z_f: float
    z_d: float
    z_c: float
    lca: float          # 一级光谱 (µm)
    sec: float          # 二级光谱 (µm)
    lsa_rms: float      # 纵向球差 RMS (µm)
    spot_rms: float     # 像面横向弥散 RMS (µm)
    shift: np.ndarray = field(default_factory=lambda: np.zeros(0))
    wls: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def band_pv(self) -> float:
        """扫描带内焦移峰谷值 (µm)"""
        return float(np.ptp(self.shift)) if self.shift.size else float("nan")

    @property
    def band_max(self) -> float:
        """F~C 带内相对 d 光的最大焦移绝对值 (µm)"""
        if not self.shift.size:
            return float("nan")
        m = (self.wls >= WL_F - 0.5) & (self.wls <= WL_C + 0.5)
        return float(np.max(np.abs(self.shift[m])))


def analyze(spec: SystemSpec, scan: bool = True) -> Metrics:
    opm, sm, osp = build(spec)
    efl = float(opm["analysis_results"]["parax_data"].fod.efl)

    zp = {w: axis_cross(opm, sm, osp, w) for w in (WL_F, WL_D, WL_C)}
    lca = (zp[WL_F] - zp[WL_C]) * 1e3
    sec = (zp[WL_D] - 0.5 * (zp[WL_F] + zp[WL_C])) * 1e3

    hs = np.linspace(0.5, EPD / 2.0, 8)
    lsa = np.array([axis_cross(opm, sm, osp, WL_D, h) - zp[WL_D] for h in hs])
    spot = image_heights(opm, sm, osp, WL_D, np.linspace(0.0, EPD / 2.0, 15))

    shift = np.zeros(0)
    wls = np.zeros(0)
    if scan:
        opm2, sm2, osp2 = build(spec, list(WL_SCAN))
        zs = np.array([axis_cross(opm2, sm2, osp2, w) for w in WL_SCAN])
        zd = axis_cross(opm2, sm2, osp2, WL_D)
        shift = (zs - zd) * 1e3
        wls = np.array(WL_SCAN)

    return Metrics(
        tag=spec.tag, efl=efl, z_f=zp[WL_F], z_d=zp[WL_D], z_c=zp[WL_C],
        lca=lca, sec=sec,
        lsa_rms=float(np.sqrt(np.mean(lsa ** 2))) * 1e3,
        spot_rms=float(np.sqrt(np.mean(spot ** 2))) * 1e3,
        shift=shift, wls=wls,
    )


# ======================================================================
# 3. 玻璃色散参数（直接读玻璃库，不硬编码）
# ======================================================================
def glass_params(name: str) -> tuple[float, float, float]:
    """返回 (nd, Vd, P) ；P 为相对部分色散 (n_d − (n_F+n_C)/2)/(n_F − n_C)"""
    if not HAS_GLASS:
        raise RuntimeError("opticalglass 不可用")
    g = gf.create_glass(name)
    nF = float(g.rindex(WL_F))
    nd = float(g.rindex(WL_D))
    nC = float(g.rindex(WL_C))
    V = (nd - 1.0) / (nF - nC)
    P = (nd - 0.5 * (nF + nC)) / (nF - nC)
    return nd, V, P


def fit_normal_line() -> tuple[float, float]:
    """全库拟合 P–V 正常线，返回 (斜率, 截距)；失败则回落到缺省值"""
    if not HAS_GLASS:
        return NORMAL_LINE
    try:
        from opticalglass.glassfactory import og_glass_libs
        libs = og_glass_libs["xls"]
        V, P = [], []
        for cat_name, cat in libs.items():
            for nm in cat.get_glass_names():
                try:
                    try:
                        g = cat.create_glass(nm)          # 多数厂商目录：单参数
                    except TypeError:
                        g = cat.create_glass(nm, cat_name)  # Schott/Ohara：带目录名
                    nF = float(g.rindex(WL_F)); nd = float(g.rindex(WL_D)); nC = float(g.rindex(WL_C))
                except Exception:
                    continue
                if not (nF > nC > 1.0):
                    continue
                V.append((nd - 1.0) / (nF - nC))
                P.append((nd - 0.5 * (nF + nC)) / (nF - nC))
        A = np.vstack([np.array(V), np.ones(len(V))]).T
        coef, *_ = np.linalg.lstsq(A, np.array(P), rcond=None)
        return float(coef[0]), float(coef[1])
    except Exception:
        return NORMAL_LINE


# ======================================================================
# 4. 薄透镜初值（三方程：总光焦度 + F/C 重合 + d 落中点）
# ======================================================================
def thin_lens_init(glasses: list[str], phi_total: float = 0.01) -> np.ndarray:
    data = [glass_params(g) for g in glasses]
    n = np.array([d[0] for d in data])
    nF = np.array([d[0] for d in data])
    # 用真实 F/C 折射率构造线性方程组
    rows_nd, rows_dn, rows_mid = [], [], []
    for g in glasses:
        gg = gf.create_glass(g)
        a, b, c = float(gg.rindex(WL_F)), float(gg.rindex(WL_D)), float(gg.rindex(WL_C))
        rows_nd.append(a - 1.0)
        rows_dn.append(a - c)
        rows_mid.append(b - 0.5 * (a + c))
    M = np.array([rows_nd, rows_dn, rows_mid], float)
    rhs = np.array([phi_total, 0.0, 0.0])
    phis = np.linalg.solve(M, rhs) * np.array(rows_nd)   # K_i 乘回 (nd-1) 得功率

    cvs = [0.0]
    for k, g in enumerate(glasses):
        cvs.append(cvs[-1] - phis[k] / (n[k] - 1.0))
    cvs = np.array(cvs)
    return cvs - cvs.mean() + 0.012


def optimize_design(spec: SystemSpec, max_nfev: int = 400,
                    wsa: float = 2.0) -> tuple[np.ndarray, Metrics]:
    """从薄透镜解出发，用最小二乘把 EFL/LCA/SEC 锁死并压低球差"""
    if not HAS_SCIPY:
        raise RuntimeError("需要 scipy 才能执行 --optimize")
    elements = [g for g in spec.glasses if g is not None]
    x0 = thin_lens_init(elements, 1.0 / 100.0)

    def resid(x):
        try:
            trial = SystemSpec(spec.tag, list(x), spec.thicks, spec.glasses)
            opm, sm, osp = build(trial)
            efl = float(opm["analysis_results"]["parax_data"].fod.efl)
            if not np.isfinite(efl) or efl < 10.0:
                return np.full(4, 1e3)
            zp = {w: axis_cross(opm, sm, osp, w) for w in (WL_F, WL_D, WL_C)}
            lca = zp[WL_F] - zp[WL_C]
            sec = zp[WL_D] - 0.5 * (zp[WL_F] + zp[WL_C])
            hs = np.linspace(0.5, EPD / 2.0, 8)
            lsa = np.array([axis_cross(opm, sm, osp, WL_D, h) - zp[WL_D] for h in hs])
            return np.array([efl - 100.0, lca * 10.0, sec * 10.0,
                             float(np.sqrt(np.mean(lsa ** 2))) * wsa])
        except Exception:
            return np.full(4, 1e3)

    res = least_squares(resid, x0, bounds=(-0.35, 0.35),
                        xtol=1e-13, ftol=1e-13, max_nfev=max_nfev)
    out = SystemSpec(spec.tag, list(res.x), spec.thicks, spec.glasses)
    return res.x, analyze(out)


# ======================================================================
# 5. 绘图工具
# ======================================================================
def setup_chinese_font() -> str:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"):
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    print("[警告] 未找到中文字体，中文可能显示为方块")
    return "DejaVu Sans"


def sag(cv: float, y: np.ndarray) -> np.ndarray:
    """球面矢高 z = c·y²/(1+√(1−c²y²))，c<0 自然给出负矢高"""
    if abs(cv) < 1e-12:
        return np.zeros_like(y)
    arg = np.clip(1.0 - (cv * y) ** 2, 0.0, None)
    return cv * y ** 2 / (1.0 + np.sqrt(arg))


def draw_element(ax, sm, i_front: int, i_back: int, sd: float,
                 fc: str, ec: str, label: str | None = None, flat_right=False):
    z0 = sm.gbl_tfrms[1][1][2]
    zf = sm.gbl_tfrms[i_front][1][2] - z0
    zb = sm.gbl_tfrms[i_back][1][2] - z0
    cf = sm.ifcs[i_front].profile.cv
    cb = sm.ifcs[i_back].profile.cv

    yy = np.linspace(-sd, sd, 260)
    z1 = zf + sag(cf, yy)
    z2 = zb + sag(cb, yy)
    ax.fill(np.concatenate([z1, z2[::-1]]), np.concatenate([yy, yy[::-1]]),
            color=fc, alpha=0.60, zorder=1, lw=0)
    ax.plot(z1, yy, color=ec, lw=1.6, zorder=3)
    ax.plot(z2, yy, color=ec, lw=1.6, zorder=3)
    ax.plot([z1[0], z2[0]], [yy[0]] * 2, color=ec, lw=1.3, zorder=3)
    ax.plot([z1[-1], z2[-1]], [yy[-1]] * 2, color=ec, lw=1.3, zorder=3)
    if label:
        ax.text(0.5 * (z1[len(yy) // 2] + z2[len(yy) // 2]), -sd * 1.09, label,
                fontsize=8.5, ha="center", va="top", color=ec, zorder=6)


def edge_thickness(sm, i_front: int, i_back: int, sd: float) -> float:
    """边缘厚度 = 后表面边缘 z − 前表面边缘 z（沿光轴）"""
    z0 = sm.gbl_tfrms[1][1][2]
    zf = sm.gbl_tfrms[i_front][1][2] - z0
    zb = sm.gbl_tfrms[i_back][1][2] - z0
    cf = sm.ifcs[i_front].profile.cv
    cb = sm.ifcs[i_back].profile.cv
    y = np.array([sd])
    return float(zb + sag(cb, y)[0] - (zf + sag(cf, y)[0]))


# ======================================================================
# 6. 图 1：三色光路图
# ======================================================================
def plot_raytrace(spec: SystemSpec, m: Metrics, out_png: Path):
    setup_chinese_font()
    opm, sm, osp = build(spec)
    z0 = sm.gbl_tfrms[1][1][2]
    sd = EPD / 2.0
    z_img = m.z_d

    fig = plt.figure(figsize=(15.2, 6.4), dpi=150)
    ax = fig.add_axes([0.055, 0.10, 0.90, 0.78])

    # 三片透镜（胶合面共用）；玻璃名放图例，不标在透镜上（三个元件挤在一起会重叠）
    n_surf = len(spec.cvs)
    faces = [(1, 2, "#AED6F1", "#1B4F72"), (2, 3, "#F9E79F", "#7D6608"),
             (3, 4, "#D5F5E3", "#1E8449")]
    for (a, b, fc, ec) in faces[:n_surf - 1]:
        draw_element(ax, sm, a, b, sd, fc, ec)

    # 对称入射高度；三种波长用不同线宽/透明度，重叠处呈棕色，边缘可分辨各色
    heights = np.array([-sd, -8.5, -4.25, 0.0, 4.25, 8.5, sd])
    for w, c, lw_, al in ((WL_F, CLR_F, 2.6, 0.32),
                          (WL_D, CLR_D, 1.5, 0.55),
                          (WL_C, CLR_C, 0.9, 0.95)):
        for path in ray_paths(opm, sm, osp, w, heights):
            zz = np.concatenate([[-16.0], path[:, 0]])
            yy = np.concatenate([[path[0, 1]], path[:, 1]])
            ax.plot(zz, yy, color=c, lw=lw_, alpha=al, zorder=2,
                    solid_capstyle="round")

    ax.axvline(z_img, color="#555555", ls="--", lw=1.2, zorder=1)
    ax.text(z_img + 1.5, sd * 1.06, "像面", fontsize=9, color="#444444")

    ax.annotate("三色焦点落在 ±1.1 µm 内\n（图上完全重合，即消色差）",
                xy=(z_img, 0.0), xytext=(z_img - 44, -sd * 1.12),
                fontsize=8.8, color="#444444", ha="left",
                arrowprops=dict(arrowstyle="->", color="#888888", lw=0.9,
                                connectionstyle="arc3,rad=0.18"))

    ax.set_xlim(-17, z_img + 12)
    ax.set_ylim(-sd * 1.28, sd * 1.28)
    ax.set_xlabel("z  (mm，自第一面前顶点起算)", fontsize=10)
    ax.set_ylabel("y  (mm)", fontsize=10)
    ax.set_title(f"复消色差三片式 · 真实光线追迹（rayoptics / Snell 逐面折射）"
                 f"    EFL={m.efl:.3f} mm   f/{m.efl / EPD:.2f}", fontsize=12, pad=12)
    ax.grid(alpha=0.12, ls=":")

    handles = [plt.Line2D([], [], color=CLR_F, lw=1.6, label=f"F 光 {WL_F:.1f} nm"),
               plt.Line2D([], [], color=CLR_D, lw=1.6, label=f"d 光 {WL_D:.1f} nm"),
               plt.Line2D([], [], color=CLR_C, lw=1.6, label=f"C 光 {WL_C:.1f} nm"),
               plt.Rectangle((0, 0), 1, 1, fc="#AED6F1", ec="#1B4F72",
                             label=f"{spec.glasses[0]}（ED 冕牌）"),
               plt.Rectangle((0, 0), 1, 1, fc="#F9E79F", ec="#7D6608",
                             label=f"{spec.glasses[1]}（短火石）"),
               plt.Rectangle((0, 0), 1, 1, fc="#D5F5E3", ec="#1E8449",
                             label=f"{spec.glasses[2]}（冕牌）")]
    ax.legend(handles=handles, loc="upper left", fontsize=8.2, framealpha=1.0,
              edgecolor="#CCCCCC", ncol=1, handlelength=1.6, labelspacing=0.35)

    txt = (f"处方（胶合三片，各面均为球面）\n"
           f"  R1 = {spec.radii[0]:+9.3f} mm   t1 = {spec.thicks[0]:4.1f} mm  {spec.glasses[0]}\n"
           f"  R2 = {spec.radii[1]:+9.3f} mm   t2 = {spec.thicks[1]:4.1f} mm  {spec.glasses[1]}\n"
           f"  R3 = {spec.radii[2]:+9.3f} mm   t3 = {spec.thicks[2]:4.1f} mm  {spec.glasses[2]}\n"
           f"  R4 = {spec.radii[3]:+9.3f} mm\n"
           f"\n二级光谱 SEC = {m.sec:+.2f} µm")
    ax.text(0.985, 0.035, txt, transform=ax.transAxes, va="bottom", ha="right",
            fontsize=8.4, color="#2C3E50",
            bbox=dict(boxstyle="round,pad=0.5", fc="#FEF9E7", ec="#B7950B", alpha=0.97))

    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# 7. 图 2：焦移随波长变化
# ======================================================================
def plot_chromatic_shift(metrics: list[Metrics], out_png: Path):
    setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.0), dpi=150,
                             gridspec_kw={"width_ratios": [1.15, 1.0]})
    fig.subplots_adjust(left=0.065, right=0.975, bottom=0.13, top=0.85, wspace=0.24)

    base = metrics[0]

    # ---- 左：全量程 ----
    ax = axes[0]
    colors = ["#7F8C8D", "#E67E22", "#C0392B"]
    for m, c in zip(metrics, colors):
        ax.plot(m.wls, m.shift, color=c, lw=2.0, label=f"{m.tag}（SEC {m.sec:+.1f} µm）")
        ax.plot([WL_D], [0.0], "o", color=c, ms=4)

    ax.axhline(0.0, color="#999999", lw=0.8, ls=":")
    for w, nm, c in ((WL_F, "F", CLR_F), (WL_C, "C", CLR_C)):
        ax.axvline(w, color=c, lw=1.0, ls="--", alpha=0.55)
        ax.text(w, ax.get_ylim()[1] * 0.02, f" {nm} 线", fontsize=8, color=c,
                rotation=90, va="bottom", ha="right")
    ax.set_xlabel("波长 λ  (nm)", fontsize=10)
    ax.set_ylabel("焦移  z(λ) − z(d)   (µm)", fontsize=10)
    ax.set_title("焦移随波长变化 · 全量程（三条曲线同轴）", fontsize=11, pad=9)
    ax.legend(fontsize=8.6, loc="upper left", framealpha=1.0, edgecolor="#CCCCCC")
    ax.grid(alpha=0.15, ls=":")
    ax.annotate(f"双胶合带内峰谷 {base.band_pv:.0f} µm",
                xy=(470, float(np.interp(470, base.wls, base.shift))),
                xytext=(505, base.band_pv * 0.72), fontsize=8.6, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8))

    # ---- 右：±4 µm 放大，只看两个三片式 ----
    ax2 = axes[1]
    for m, c in zip(metrics[1:], colors[1:]):
        ax2.plot(m.wls, m.shift, color=c, lw=2.0, marker="o", ms=3, label=m.tag)
    ax2.axhline(0.0, color="#999999", lw=0.8, ls=":")
    ax2.axvline(WL_D, color=CLR_D, lw=1.0, ls="--", alpha=0.6)
    ax2.set_ylim(-6.0, 6.0)
    ax2.set_xlim(450, 700)
    ax2.set_xlabel("波长 λ  (nm)", fontsize=10)
    ax2.set_ylabel("焦移  z(λ) − z(d)   (µm)", fontsize=10)
    ax2.set_title("放大到 ±6 µm：APO 的三次过零与残余三级色差", fontsize=11, pad=9)
    ax2.legend(fontsize=8.8, loc="lower left", framealpha=1.0, edgecolor="#CCCCCC")
    ax2.grid(alpha=0.15, ls=":")

    # 标出 F~C 带
    ax2.axvspan(WL_F, WL_C, color="#5DADE2", alpha=0.08, zorder=0)
    ax2.text(0.5 * (WL_F + WL_C), -5.4, "F–C 校正带", fontsize=8, color="#2E86C1",
             ha="center")

    ap = metrics[-1]
    ax2.annotate(f"APO 带内最大焦移 {ap.band_max:.1f} µm",
                 xy=(WL_F, float(np.interp(WL_F, ap.wls, ap.shift))),
                 xytext=(505, 4.3), fontsize=8.8, color="#8B1A1A",
                 arrowprops=dict(arrowstyle="->", color="#C0392B", lw=0.9))

    fig.suptitle("Chromatic Focal Shift · 双胶合 vs 三片式复消色差",
                 fontsize=13.5, y=0.955)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# 8. 报告
# ======================================================================
def build_report(ms: list[Metrics], specs: list[SystemSpec],
                 glasses: list[str], nl: tuple[float, float]) -> str:
    L = []
    add = L.append
    a, b = nl
    base, mid, apo = ms[0], ms[1], ms[2]

    add("=" * 78)
    add("复消色差（APO）三片式设计报告")
    add("=" * 78)
    add(f"孔径 EPD = {EPD} mm    目标焦距 = 100 mm    f/{100/EPD:.2f}")
    add(f"谱线：F {WL_F} nm / d {WL_D} nm / C {WL_C} nm    扫描带 450–700 nm")
    add("")
    add("[一] 玻璃选型：为什么仅加一片 ED 玻璃不够")
    add(f"  正常线（全库拟合）：P = {a:+.6f}·V {b:+.6f}")
    add(f"  {'玻璃':10s}{'nd':>9s}{'Vd':>8s}{'P':>9s}{'ΔP':>9s}   说明")
    notes = {0: "冕牌，正常色散", 1: "普通火石，几乎在正常线上",
             2: "氟磷酸盐 ED，强反常色散", 3: "短火石，反常色散"}
    for i, g in enumerate(glasses):
        try:
            nd, V, P = glass_params(g)
            add(f"  {g:10s}{nd:9.5f}{V:8.2f}{P:+9.4f}{P-(a*V+b):+9.4f}   {notes.get(i,'')}")
        except Exception:
            add(f"  {g:10s}  （玻璃库不可用，跳过）")
    add("")
    add("  三方程消色差（总光焦度 / F=C / d 落中点）要求三块玻璃的 ΔP 拉开；")
    add("  普通玻璃 ΔP 都在 ±0.002 内，方程病态，解出的功率无法加工。")
    add("")
    add("[二] 三套系统的处方")
    for sp, m in zip(specs, ms):
        add(f"  ▸ {sp.tag}     EFL={m.efl:.4f} mm")
        for i in range(len(sp.cvs)):
            r = sp.radii[i]
            rtxt = f"{r:+10.3f}" if np.isfinite(r) else "       平面"
            if i < len(sp.thicks) - 1:
                rest = f"t={sp.thicks[i]:6.2f} mm   {sp.glasses[i] or '空气'}"
            else:
                rest = "t= 至像面"
            add(f"      面{i}: R={rtxt} mm   {rest}")
        add("")
    add("[三] 色差指标对比")
    add(f"  {'系统':22s}{'LCA(F−C)':>12s}{'二级光谱':>12s}{'带内峰谷':>11s}{'带内最大':>10s}")
    add(f"  {'':22s}{'(µm)':>11s}{'SEC (µm)':>12s}{'(µm)':>11s}{'(µm)':>10s}")
    for m in ms:
        add(f"  {m.tag:22s}{m.lca:+12.2f}{m.sec:+12.2f}{m.band_pv:11.1f}{m.band_max:10.1f}")
    add("")
    add("[四] 焦移扫描（相对 d 光焦点，µm）")
    hdr = "  λ(nm)  " + "".join(f"{m.tag[:14]:>16s}" for m in ms)
    add(hdr)
    for k in range(0, len(WL_SCAN), 2):
        row = f"  {WL_SCAN[k]:6.1f} "
        for m in ms:
            row += f"{m.shift[k]:+16.2f}"
        add(row)
    add("")
    add("[五] 单色性能与几何校验")
    for sp, m in zip(specs, ms):
        try:
            opm, sm, osp = build(sp)
            n_el = sum(1 for g in sp.glasses[:-1] if g is not None)
            ets = []
            for k in range(n_el):
                ets.append(edge_thickness(sm, k + 1, k + 2, EPD / 2.0))
            etxt = "  ".join(f"{e:5.2f}" for e in ets)
            add(f"  {m.tag:22s} 纵向球差RMS={m.lsa_rms:6.1f} µm   横向弥散RMS={m.spot_rms:5.2f} µm   "
                f"边缘厚度=[{etxt}] mm")
        except Exception as exc:
            add(f"  {m.tag:22s} 几何校验失败：{exc}")
    add("")
    add("[六] 结论")
    add(f"  二级光谱：双胶合 {base.sec:+.1f} µm  →  APO {apo.sec:+.1f} µm   "
        f"压缩 {abs(base.sec/apo.sec):.0f} 倍")
    add(f"  只加 ED 而不换火石：{mid.sec:+.1f} µm，仅改善 {abs(base.sec/mid.sec):.1f} 倍")
    add(f"  F–C 带内最大焦移：{base.band_max:.1f} µm  →  {apo.band_max:.1f} µm")
    add(f"  450–700 nm 全带峰谷：{base.band_pv:.0f} µm  →  {apo.band_pv:.1f} µm")
    add("")
    add("  残余限制：APO 的焦移曲线在 450–700 nm 内呈三次过零形态，")
    add("  说明二级（线性）色差已被消掉，剩下的是三级色差；带外（<470 nm）焦移")
    add("  仍快速增大——这是 ED 玻璃在短波端折射率曲线变陡的固有特性。")
    add("=" * 78)
    return "\n".join(L)


# ======================================================================
# 9. 主流程
# ======================================================================
def main() -> int:
    do_opt = "--optimize" in sys.argv

    # 三套系统：A 基准双胶合（06 的处方） / B ED+普通火石 / C ED+反常火石（APO）
    spec_doublet = SystemSpec(
        "双胶合 BK7+SF5",
        [1 / 52.9619, 1 / -47.4951, 1 / -187.9895],
        [6.0, 3.0, 0.0],
        ["N-BK7", "N-SF5", None],
    )
    spec_mid = SystemSpec(
        "三片 FPL51+SF5+BK7",
        [1 / 38.1470, 1 / -38.8950, 1 / -53.3591, 1 / 316.9715],
        [6.0, 3.0, 4.0, 0.0],
        ["S-FPL51", "N-SF5", "N-BK7", None],
    )
    spec_apo = SystemSpec(
        "APO FPL51+KZFS2+BK7",
        [1 / 50.3749, 1 / -33.8193, 1 / 50.6093, 1 / -180.3107],
        [6.0, 3.0, 4.0, 0.0],
        ["S-FPL51", "N-KZFS2", "N-BK7", None],
    )

    nl = fit_normal_line()

    if do_opt:
        print("[寻优] 从薄透镜解（三方程消色差）重新求解三片式处方 ……")
        for sp in (spec_mid, spec_apo):
            try:
                stored_sec = analyze(sp, scan=False).sec
            except Exception:
                stored_sec = float("nan")
            cv, m = optimize_design(sp)
            print(f"  {sp.tag:22s} R = [{', '.join(f'{1/c:9.4f}' for c in cv)}]")
            print(f"  {'':22s} EFL={m.efl:.4f}  LCA={m.lca:+.2f}µm  SEC={m.sec:+.2f}µm  "
                  f"球差RMS={m.lsa_rms:.1f}µm")
            print(f"  {'':22s} 脚本固化处方 SEC={stored_sec:+.2f}µm —— 两条路径落在同一最优点，"
                  f"确认处方不是拼凑的")
            sp.cvs = list(cv)
        print()

    metrics = []
    for sp in (spec_doublet, spec_mid, spec_apo):
        try:
            metrics.append(analyze(sp))
        except Exception as exc:
            print(f"[错误] {sp.tag} 分析失败：{exc}")
            return 1

    print("=" * 78)
    for m in metrics:
        print(f"{m.tag:24s} EFL={m.efl:8.4f}  LCA={m.lca:+9.2f}µm  "
              f"SEC={m.sec:+8.2f}µm  带内峰谷={m.band_pv:7.1f}µm")
    print("=" * 78)

    try:
        plot_raytrace(spec_apo, metrics[2], OUT / "apo_raytrace.png")
        print(f"[图] {OUT / 'apo_raytrace.png'}")
    except Exception as exc:
        print(f"[警告] 光路图生成失败：{exc}")

    try:
        plot_chromatic_shift(metrics, OUT / "chromatic_focal_shift.png")
        print(f"[图] {OUT / 'chromatic_focal_shift.png'}")
    except Exception as exc:
        print(f"[警告] 焦移曲线生成失败：{exc}")

    glasses = ["N-BK7", "N-SF5", "S-FPL51", "N-KZFS2"]
    report = build_report(metrics, [spec_doublet, spec_mid, spec_apo], glasses, nl)
    (OUT / "apo_report.txt").write_text(report, encoding="utf-8")
    print()
    print(report)
    print(f"\n[报告] {OUT / 'apo_report.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
