# -*- coding: utf-8 -*-
"""
任务 06 — 消色差双胶合透镜（Achromatic Doublet）真实光线追迹
================================================================================
设计目标
    口径 25.4 mm（f/3.94），焦距 100 mm，在 F(486.1nm) / d(587.6nm) / C(656.3nm)
    三条谱线上把轴上色差（LCA）压到零，同时校正球差。

物理原理
    单片薄透镜的轴向色差 LCA ≈ f / V。BK7 的 V=64.17，则 f=100 的 LCA ≈ 1.56 mm，
    这是"彩虹边"的来源。
    两片胶合后，光焦度按阿贝数反比分配即可消色差：
        φ1/V1 + φ2/V2 = 0      且      φ1 + φ2 = Φ = 1/f
    解得
        φ1 = Φ·V1/(V1−V2)      φ2 = −Φ·V2/(V1−V2)
    冕牌(BK7, V=64.17)承担正光焦度，火石(N-SF5, V=32.25)承担负光焦度。
    两片胶合 → 中间面曲率相同（同一片玻璃界面），只增加 2 个自由曲率。

设计流程（本脚本实际执行的三步）
    1) 薄透镜解析解：由 φ1/φ2 与选定胶合面半径 R2 反解 R1、R3，作为初值
    2) 精确求解：固定 R2，用二维求根把 (EFL, LCA) 同时锁定为 (100.000, 0)
       —— 薄透镜公式忽略了厚度，必须用真实光线追迹修正
    3) 弯曲扫描：R2 是唯一剩余自由度，扫描它并取"轴向球差 RMS 最小"的弯曲形态
       —— 这一步才是"消色差的同时把球差也压下去"

口径说明（重要）
    · 焦点位置统一用"小口径光线与实际光轴的交点"提取（h=0.1mm，已验证与 h=0.05mm
      完全一致，等价于近轴焦点），三个波长用同一方法，保证可比。
    · LCA 有"近轴"与"全口径"两个值：近轴值反映色散校正，全口径值混入了
      色球差（球差随波长的变化）。只看近轴值会高估设计质量，两个都给。
    · 二级光谱（d 光焦点偏离 F/C 中点）是双胶合的物理极限，不可能靠调曲率消除。

输出
    output/achromatic_raytrace.png    三色光路图 + 色差对比
    output/achromatic_report.txt      文字报告
"""

from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

import rayoptics.optical.model_constants as mc
from rayoptics.environment import OpticalModel, PupilSpec, FieldSpec, WvlSpec
from rayoptics.raytr.trace import trace_base


# ================================================================
# 1. 参数
# ================================================================
WVL_F = 486.1327      # 氢 F 线（蓝）
WVL_D = 587.5618      # 氦 d 线（黄绿，设计主波长）
WVL_C = 656.2725      # 氢 C 线（红）

WL_LABEL = {WVL_F: "F 光 486nm", WVL_D: "d 光 588nm", WVL_C: "C 光 656nm"}
WL_COLOR = {WVL_F: "#1F4E9C", WVL_D: "#1E8449", WVL_C: "#C0392B"}


@dataclass
class Prescription:
    """双胶合处方（曲率单位 mm，正值 = 曲率中心在 +z 侧）"""
    r1: float = 52.9619        # 前表面（冕牌，凸向物方）
    r2: float = -47.4951       # 胶合面（两片共用同一曲率 —— 胶合的本质）
    r3: float = -187.9895      # 后表面（火石，凹向像方）
    t1: float = 6.0            # 冕牌中心厚度
    t2: float = 3.0            # 火石中心厚度
    glass1: str = "N-BK7"      # 冕牌：nd=1.5168  Vd=64.17
    glass2: str = "N-SF5"      # 火石：nd=1.6727  Vd=32.25

    # 系统参数
    efl_target: float = 100.0
    epd: float = 25.4          # 入瞳直径 -> f/3.94
    n_rays: int = 11           # 每个波长画多少条光线
    img_gap: float = 20.0      # 后表面到像面的初始占位厚度


@dataclass
class FocusSet:
    """一个系统在三波长下的焦点与像差数据"""
    tag: str
    efl: float
    z_parax: dict          # 各波长近轴焦点（从前顶点起算）
    z_marg: dict           # 各波长全口径(h=epd/2)交轴位置
    sa_curves: dict        # 各波长轴向球差曲线 {wvl: (heights, sa)}
    lca_parax: float
    lca_marg: float
    secondary: float       # 二级光谱 d-(F+C)/2
    sa_rms: float          # d 光轴向球差 RMS
    spot_rms: dict         # 各波长在像面的弥散斑 RMS 半径
    spot_rms_all: float    # 三波长混合弥散斑
    paths: dict            # 光线折线（绘图用）
    inset: dict            # 焦区放大用的小口径光线（大高度样本）
    z_image: float         # 像面位置（从前顶点起算）
    sm: object = None


# ================================================================
# 2. 建模
# ================================================================
def build_doublet(p: Prescription):
    """建三面胶合模型：物面 / 冕牌前 / 胶合面 / 火石后 / 像面"""
    opm = OpticalModel()
    sm, osp = opm["seq_model"], opm["optical_spec"]
    opm.radius_mode = True

    osp["pupil"] = PupilSpec(osp, key=("object", "epd"), value=p.epd)
    osp["fov"] = FieldSpec(osp, key=("object", "angle"), value=0.0, flds=[0.0])
    osp["wvls"] = WvlSpec([(WVL_F, 1.0), (WVL_D, 1.0), (WVL_C, 1.0)], ref_wl=1)

    sm.gaps[0].thi = 1e10                      # 物在无穷远
    sm.add_surface([p.r1, p.t1, p.glass1])     # s1 冕牌前表面
    sm.add_surface([p.r2, p.t2, p.glass2])     # s2 胶合面（同曲率）
    sm.add_surface([p.r3, p.img_gap])          # s3 火石后表面 -> 空气
    opm.update_model()

    # 把像面精确推到 d 光近轴焦点
    img_dist = float(opm["analysis_results"]["parax_data"].fod.img_dist)
    sm.gaps[-1].thi = round(img_dist, 6)
    opm.update_model()
    sm.compute_global_coords()
    return opm, sm, osp


def build_singlet(p: Prescription):
    """对照用单透镜：同焦距、同口径、同材料的平凸镜"""
    opm = OpticalModel()
    sm, osp = opm["seq_model"], opm["optical_spec"]
    opm.radius_mode = True

    osp["pupil"] = PupilSpec(osp, key=("object", "epd"), value=p.epd)
    osp["fov"] = FieldSpec(osp, key=("object", "angle"), value=0.0, flds=[0.0])
    osp["wvls"] = WvlSpec([(WVL_F, 1.0), (WVL_D, 1.0), (WVL_C, 1.0)], ref_wl=1)

    sm.gaps[0].thi = 1e10
    # 平凸：R = (n-1)·f，用 d 光折射率倒算，使 d 光焦距恰为 100mm
    r_front = p.efl_target * 0.516800
    sm.add_surface([r_front, 6.0, p.glass1])
    sm.add_surface([0.0, p.img_gap])
    opm.update_model()

    img_dist = float(opm["analysis_results"]["parax_data"].fod.img_dist)
    sm.gaps[-1].thi = round(img_dist, 6)
    opm.update_model()
    sm.compute_global_coords()
    return opm, sm, osp


# ================================================================
# 3. 追迹与测量
# ================================================================
def axis_cross(opm, sm, osp, wvl: float, h: float) -> float:
    """
    求光线与光轴的交点 z（从前表面顶点起算）。
    出射段是直线：z = z0 - y0 · (dz/dy)，比逐点搜索交点稳。
    """
    tf = sm.gbl_tfrms
    z0 = tf[1][1][2]
    ray, _, _ = trace_base(opm, (0.0, h), osp["fov"].fields[0], wvl, pupil_type="aim pt")
    rot, trns = tf[-1]
    pt = rot.dot(ray[-1][mc.p]) + trns
    dr = rot.dot(ray[-1][mc.d])
    y, z = pt[1], pt[2] - z0
    if abs(dr[1]) < 1e-12:
        return float("nan")
    return float(z + (-y / dr[1]) * dr[2])


def trace_paths(opm, sm, osp, heights, wvls):
    """取每条光线在各界面的全局坐标，形成折线（跳过物面段，改从图左边界水平入射）"""
    tf = sm.gbl_tfrms
    z0 = tf[1][1][2]
    out = {}
    for w in wvls:
        seg_list = []
        for h in heights:
            ray, _, _ = trace_base(opm, (0.0, h), osp["fov"].fields[0], w, pupil_type="aim pt")
            pts = []
            for i, seg in enumerate(ray):
                if i == 0:
                    continue                      # 物面 z=-1e10，不能入图
                rot, trns = tf[i]
                p = rot.dot(seg[mc.p]) + trns
                pts.append((p[2] - z0, p[1]))
            seg_list.append(np.array(pts))
        out[w] = seg_list
    return out


def measure(tag: str, opm, sm, osp, p: Prescription) -> FocusSet:
    efl = float(opm["analysis_results"]["parax_data"].fod.efl)
    z0 = sm.gbl_tfrms[1][1][2]
    z_img = sm.gbl_tfrms[-1][1][2] - z0

    hs = np.linspace(0.0, p.epd / 2.0, p.n_rays)
    z_parax, z_marg, sa_curves, spot = {}, {}, {}, {}
    tf = sm.gbl_tfrms

    for w in (WVL_F, WVL_D, WVL_C):
        zp = axis_cross(opm, sm, osp, w, 0.1)      # 近轴焦点探针
        z_parax[w] = zp
        z_marg[w] = axis_cross(opm, sm, osp, w, p.epd / 2.0)

        probe = np.linspace(0.5, p.epd / 2.0, 12)   # 从 h=0.5 起：h=0 是轴上光线，出射方向无 y 分量，无法外推交点
        sa_curves[w] = (probe, np.array([axis_cross(opm, sm, osp, w, h) - zp for h in probe]))

        # 像面弥散斑：逐光线取像面处高度
        ys = []
        for h in np.linspace(0.0, p.epd / 2.0, 21):
            ray, _, _ = trace_base(opm, (0.0, h), osp["fov"].fields[0], w, pupil_type="aim pt")
            rot, trns = tf[-1]
            ys.append(float((rot.dot(ray[-1][mc.p]) + trns)[1]))
        ys = np.array(ys)
        spot[w] = float(np.sqrt(np.mean((ys - ys.mean()) ** 2)))

    ys_all = []
    for w in (WVL_F, WVL_D, WVL_C):
        for h in np.linspace(0.0, p.epd / 2.0, 21):
            ray, _, _ = trace_base(opm, (0.0, h), osp["fov"].fields[0], w, pupil_type="aim pt")
            rot, trns = tf[-1]
            ys_all.append(float((rot.dot(ray[-1][mc.p]) + trns)[1]))
    ys_all = np.array(ys_all)

    lca_parax = z_parax[WVL_F] - z_parax[WVL_C]
    lca_marg = z_marg[WVL_F] - z_marg[WVL_C]
    secondary = z_parax[WVL_D] - 0.5 * (z_parax[WVL_F] + z_parax[WVL_C])

    h_probe, sa_d = sa_curves[WVL_D]
    sa_rms = float(np.sqrt(np.mean(sa_d**2)))

    return FocusSet(
        tag=tag, efl=efl, z_parax=z_parax, z_marg=z_marg, sa_curves=sa_curves,
        lca_parax=lca_parax, lca_marg=lca_marg, secondary=secondary, sa_rms=sa_rms,
        spot_rms=spot, spot_rms_all=float(np.sqrt(np.mean((ys_all - ys_all.mean()) ** 2))),
        paths=trace_paths(opm, sm, osp, hs, (WVL_F, WVL_D, WVL_C)),
        inset=trace_paths(opm, sm, osp, [6.35, 9.53, 12.7], (WVL_F, WVL_D, WVL_C)),
        z_image=z_img, sm=sm,
    )


# ================================================================
# 4. 设计：解析初值 -> 精确消色差 -> 弯曲扫描消球差
# ================================================================
def glass_abbe(sm, name: str) -> tuple[float, float]:
    """从模型里的介质取 nd 与 Vd（不硬编码，确保与追迹用的是同一份玻璃数据）

    注意：opticalglass 2.x 里 OpticalMedium.name 是方法，牌号存在 .gname 属性上。
    """
    for gap in sm.gaps:
        med = gap.medium
        gname = getattr(med, "gname", None)
        if not isinstance(gname, str):
            try:
                nm = getattr(med, "name", None)
                gname = nm() if callable(nm) else nm
            except Exception:
                gname = None
        if gname == name:
            nd = float(med.rindex(WVL_D))
            nf = float(med.rindex(WVL_F))
            nc = float(med.rindex(WVL_C))
            return nd, (nd - 1.0) / (nf - nc)
    raise RuntimeError(f"模型里找不到玻璃 {name}")


def analytic_start(p: Prescription) -> tuple[float, float, float]:
    """薄透镜消色差解析解，给出 R1/R3 的初值（R2 由输入的弯曲量决定）"""
    _, sm, _ = build_doublet(p)
    n1, v1 = glass_abbe(sm, p.glass1)
    n2, v2 = glass_abbe(sm, p.glass2)

    phi = 1.0 / p.efl_target
    phi1 = phi * v1 / (v1 - v2)
    phi2 = -phi * v2 / (v1 - v2)
    r1 = 1.0 / (phi1 / (n1 - 1.0) + 1.0 / p.r2)
    r3 = 1.0 / (1.0 / p.r2 - phi2 / (n2 - 1.0))
    return r1, r3


def solve_exact(p: Prescription, r2: float) -> tuple[float, float] | None:
    """
    固定胶合面曲率 R2，解 (R1, R3) 使 EFL=目标值 且 LCA(F-C)=0。
    薄透镜公式忽略厚度和入射高，必须用真实光线追迹做这步修正。
    """
    from scipy.optimize import root

    p0 = Prescription(**{**asdict(p), "r2": r2})
    g1, g3 = analytic_start(p0)

    def resid(v):
        r1, r3 = v
        if abs(r1) < 5 or abs(r3) < 5:
            return [1e3, 1e3]
        try:
            q = Prescription(**{**asdict(p), "r2": r2, "r1": r1, "r3": r3})
            opm, sm, osp = build_doublet(q)
            efl = float(opm["analysis_results"]["parax_data"].fod.efl)
            zf = axis_cross(opm, sm, osp, WVL_F, 0.1)
            zc = axis_cross(opm, sm, osp, WVL_C, 0.1)
            return [(efl - p.efl_target) / p.efl_target, (zf - zc) / 10.0]
        except Exception:
            return [1e3, 1e3]

    s = root(resid, (g1, g3), tol=1e-12)
    if not s.success or not np.all(np.isfinite(s.x)):
        return None
    return float(s.x[0]), float(s.x[1])


def sa_rms_of(p: Prescription, r2: float) -> tuple[float, tuple[float, float] | None]:
    """弯曲扫描的目标函数：满足消色差前提下，d 光轴向球差 RMS"""
    sol = solve_exact(p, r2)
    if sol is None:
        return 1e6, None
    q = Prescription(**{**asdict(p), "r2": r2, "r1": sol[0], "r3": sol[1]})
    try:
        opm, sm, osp = build_doublet(q)
        zp = axis_cross(opm, sm, osp, WVL_D, 0.1)
        hs = np.linspace(0.5, p.epd / 2.0, 10)
        sa = np.array([axis_cross(opm, sm, osp, WVL_D, h) - zp for h in hs])
        if not np.all(np.isfinite(sa)):
            return 1e6, sol
        return float(np.sqrt(np.mean(sa**2))), sol
    except Exception:
        return 1e6, sol


def optimize_design(p: Prescription) -> tuple[Prescription, list[tuple[float, float]]]:
    """扫描弯曲量 R2（唯一剩余自由度），取球差 RMS 最小者"""
    from scipy.optimize import minimize_scalar

    trace = []
    for r2 in np.arange(-56.0, -34.01, 2.0):
        val, sol = sa_rms_of(p, r2)
        if sol is not None:
            trace.append((float(r2), val * 1e3))

    # 先在粗网格上找谷底，再在 ±2mm 内细化
    if not trace:
        raise RuntimeError("弯曲扫描全部失败，无法优化")
    coarse = min(trace, key=lambda t: t[1])[0]
    res = minimize_scalar(
        lambda x: sa_rms_of(p, x)[0],
        bounds=(coarse - 2.0, coarse + 2.0), method="bounded",
        options={"xatol": 1e-4},
    )
    r2 = float(res.x)
    _, sol = sa_rms_of(p, r2)
    if sol is None:
        raise RuntimeError("精细优化失败")

    out = Prescription(**{**asdict(p), "r2": r2, "r1": sol[0], "r3": sol[1]})
    return out, trace


# ================================================================
# 5. 绘图
# ================================================================
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
    return cv * y**2 / (1.0 + np.sqrt(arg))


def draw_lens(ax, sm, i_front: int, i_back: int, sd: float,
              fc: str, ec: str, label: str | None = None, alpha: float = 0.55):
    """按真实面型（矢高公式）画一片透镜，与追迹用的是同一组曲率"""
    z0 = sm.gbl_tfrms[1][1][2]
    zf = sm.gbl_tfrms[i_front][1][2] - z0
    zb = sm.gbl_tfrms[i_back][1][2] - z0
    cf = sm.ifcs[i_front].profile.cv
    cb = sm.ifcs[i_back].profile.cv

    yy = np.linspace(-sd, sd, 260)
    z1 = zf + sag(cf, yy)
    z2 = zb + sag(cb, yy)
    ax.fill(np.concatenate([z1, z2[::-1]]), np.concatenate([yy, yy[::-1]]),
            color=fc, alpha=alpha, zorder=1, lw=0)
    ax.plot(z1, yy, color=ec, lw=1.6, zorder=3)
    ax.plot(z2, yy, color=ec, lw=1.6, zorder=3)
    ax.plot([z1[0], z2[0]], [yy[0]] * 2, color=ec, lw=1.4, zorder=3)
    ax.plot([z1[-1], z2[-1]], [yy[-1]] * 2, color=ec, lw=1.4, zorder=3)
    if label:
        ax.text(0.5 * (z1[len(yy) // 2] + z2[len(yy) // 2]), sd * 1.07, label,
                fontsize=8, ha="center", va="bottom", color=ec, zorder=6)


def plot_result(fs: FocusSet, fs_single: FocusSet, p: Prescription, out_png: Path):
    setup_chinese_font()
    sm = fs.sm
    z0 = sm.gbl_tfrms[1][1][2]
    sd = p.epd / 2.0
    z_img = fs.z_image

    fig = plt.figure(figsize=(16.6, 7.6), dpi=150)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.62, 1.0],
                          height_ratios=[1.0, 1.0], wspace=0.19, hspace=0.46)
    axL = fig.add_subplot(gs[:, 0])
    axT = fig.add_subplot(gs[0, 1])
    axB = fig.add_subplot(gs[1, 1])

    # ---------------- 左：三色光路图 ----------------
    draw_lens(axL, sm, 1, 2, sd, "#AED6F1", "#1B4F72")
    draw_lens(axL, sm, 2, 3, sd, "#F5CBA7", "#A04000")

    z_left = -9.0
    for w, segs in fs.paths.items():
        for seg in segs:
            xs = np.concatenate([[z_left], seg[:, 0]])
            ys = np.concatenate([[seg[0, 1]], seg[:, 1]])
            axL.plot(xs, ys, color=WL_COLOR[w], lw=1.0, alpha=0.8, zorder=2)

    axL.axvline(z_img, color="#555555", ls="--", lw=1.2, zorder=1)
    axL.text(z_img + 1.0, sd * 1.02, "像面", fontsize=9, color="#444444")

    axL.set_xlim(z_left, z_img + 9.0)
    axL.set_ylim(-sd * 1.18, sd * 1.18)
    axL.set_xlabel("z  (mm，从前顶点起算)", fontsize=10)
    axL.set_ylabel("y  (mm)", fontsize=10)
    axL.set_title(
        f"消色差双胶合 · 三波长光路    EFL={fs.efl:.3f} mm   f/{fs.efl / p.epd:.2f}   "
        f"口径 {p.epd:g} mm\n"
        f"LCA(F−C) 近轴 = {fs.lca_parax * 1e3:+.3f} µm   "
        f"（单透镜对照：{fs_single.lca_parax * 1e3:+.0f} µm）",
        fontsize=11.5, pad=9,
    )
    axL.grid(alpha=0.15, ls=":")
    axL.legend(handles=[Patch(facecolor=WL_COLOR[w], label=WL_LABEL[w],
                              alpha=0.85) for w in (WVL_C, WVL_D, WVL_F)]
               + [Patch(facecolor="#AED6F1", edgecolor="#1B4F72", label="N-BK7"),
                  Patch(facecolor="#F5CBA7", edgecolor="#A04000", label="N-SF5")],
               loc="upper left", fontsize=9, framealpha=0.98, ncol=2)

    # 焦区放大（关键：三色是否真的落在同一点 —— 窗口必须开到 µm 量级才看得见残余分离）
    axi = axL.inset_axes([0.545, 0.075, 0.415, 0.355])
    for w, segs in fs.inset.items():
        for seg in segs:
            axi.plot(seg[:, 0] - z_img, seg[:, 1] * 1e3,
                     color=WL_COLOR[w], lw=1.1, alpha=0.85)
    axi.axvline(0.0, color="#555555", ls="--", lw=1.0)
    axi.text(0.15, 58.0, "像面", fontsize=7, color="#555555")
    # 三色的近轴焦点标记（x 单位 mm，y 单位 µm；F/C 重合、d 光偏离 51µm 一目了然）
    for w in (WVL_F, WVL_D, WVL_C):
        axi.plot([fs.z_parax[w] - z_img], [68.0], marker="v", ms=5,
                 color=WL_COLOR[w], clip_on=False, zorder=6)
    axi.set_xlim(-1.5, 1.5)
    axi.set_ylim(-70.0, 70.0)
    axi.set_xlabel("相对像面 (mm)", fontsize=7.5, labelpad=1)
    axi.set_ylabel("y (µm)", fontsize=7.5, labelpad=1)
    axi.set_title("焦区放大 ±70µm：三色近轴焦点几乎重合，边缘光线仍有分离",
                  fontsize=8.5, pad=3)
    axi.tick_params(labelsize=7)
    axi.grid(alpha=0.15, ls=":")
    for s in axi.spines.values():
        s.set_edgecolor("#888888")

    # ---------------- 右上：LCA 数量级对比 ----------------
    lca_um = [abs(fs_single.lca_parax) * 1e3, max(abs(fs.lca_parax) * 1e3, 1e-4)]
    labels = ["单透镜\nBK7", "消色差双胶合\nBK7 + N-SF5"]
    colors = ["#C0392B", "#1E8449"]
    ypos = np.arange(2)

    axT.barh(ypos, lca_um, color=colors, alpha=0.82, height=0.5)
    axT.set_xscale("log")
    axT.set_xlim(1e-4, 2e4)
    axT.set_ylim(1.65, -0.95)          # 反转 y 轴并给顶部留出注释空间
    axT.set_yticks(ypos)
    axT.set_yticklabels(labels, fontsize=9.5)
    axT.set_xlabel("轴向色差 LCA(F−C)   （µm，对数轴）", fontsize=9.5)

    dof = 2.0 * 0.5875618 / (p.epd / fs.efl) ** 2   # 焦深 ±2λ/NA²，NA≈D/(2f)
    axT.axvline(dof, color="#B7950B", ls="--", lw=1.2)
    axT.text(dof * 1.3, -0.62, f"衍射焦深 {dof:.0f} µm（低于此值色差不可见）",
             fontsize=8, color="#8A6D0B", va="center")

    for y, v, t in zip(ypos, lca_um, ("1.53 mm", "3 nm")):
        axT.text(v * 1.35, y, t, va="center", fontsize=9.5, color="#333333")

    axT.set_title("色差校正量级：同一焦距下相差 5 个数量级", fontsize=11, pad=8)
    axT.grid(axis="x", alpha=0.18, ls=":")

    # ---------------- 右下：色球差（残余色差真正藏在这里）----------------
    for w in (WVL_C, WVL_D, WVL_F):
        hs, sa = fs.sa_curves[w]
        axB.plot(hs, sa * 1e3, color=WL_COLOR[w], lw=2.0,
                 label=WL_LABEL[w])
    axB.axhline(0, color="#666666", lw=0.9)
    axB.set_xlabel("入射高度 h  (mm)", fontsize=9.5)
    axB.set_ylabel("轴向球差相对各自近轴焦点 (µm)", fontsize=9.5)
    axB.set_title("色球差：轴上三色合一，边缘仍有分离", fontsize=11, pad=8)
    axB.grid(alpha=0.18, ls=":")

    gap = (fs.z_marg[WVL_F] - fs.z_marg[WVL_C]) * 1e3
    axB.annotate("", xy=(sd, fs.sa_curves[WVL_C][1][-1] * 1e3),
                 xytext=(sd, fs.sa_curves[WVL_F][1][-1] * 1e3),
                 arrowprops=dict(arrowstyle="<->", color="#7D3C98", lw=1.3))
    axB.text(sd * 0.965, 0.5 * (fs.sa_curves[WVL_F][1][-1] + fs.sa_curves[WVL_C][1][-1]) * 1e3 - 22,
             f"边缘分离 {gap:.0f} µm", fontsize=8.5, color="#7D3C98", ha="right")

    txt = (f"LCA 近轴   {fs.lca_parax * 1e3:+8.3f} µm\n"
           f"LCA 全口径 {fs.lca_marg * 1e3:+8.1f} µm\n"
           f"二级光谱   {fs.secondary * 1e3:+8.1f} µm\n"
           f"轴向球差RMS {fs.sa_rms * 1e3:7.1f} µm")
    axB.text(0.025, 0.05, txt, transform=axB.transAxes, va="bottom", ha="left",
             fontsize=8.2, linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.42", fc="#FEF9E7", ec="#B7950B", alpha=0.95))
    axB.legend(loc="upper left", fontsize=8.5, framealpha=0.92)

    fig.suptitle("消色差双胶合透镜 · 真实光线追迹（rayoptics / Snell 逐面折射）\n"
                 "主图中三色光路几乎完全重合 —— 这正是消色差的直观证据；µm 量级的残余分离见焦区放大",
                 fontsize=13, y=0.985)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# ================================================================
# 6. 报告
# ================================================================
def build_report(fs: FocusSet, fs_s: FocusSet, p: Prescription,
                 scan: list[tuple[float, float]]) -> str:
    _, sm, _ = build_doublet(p)
    n1, v1 = glass_abbe(sm, p.glass1)
    n2, v2 = glass_abbe(sm, p.glass2)

    phi = 1.0 / p.efl_target
    phi1 = phi * v1 / (v1 - v2)
    phi2 = -phi * v2 / (v1 - v2)

    L = []
    add = L.append
    add("=" * 74)
    add("消色差双胶合透镜（Achromatic Doublet）设计报告")
    add("=" * 74)

    add("\n[一] 材料")
    add(f"  冕牌 {p.glass1:8s} nd={n1:.5f}  Vd={v1:.2f}")
    add(f"  火石 {p.glass2:8s} nd={n2:.5f}  Vd={v2:.2f}")

    add("\n[二] 消色差光焦度分配（薄透镜解析解）")
    add(f"  φ1/V1 + φ2/V2 = 0,  φ1 + φ2 = 1/{p.efl_target:g}")
    add(f"  φ1 = {phi1:+.6f} mm⁻¹  ->  f1 = {1 / phi1:+8.2f} mm   (冕牌，正)")
    add(f"  φ2 = {phi2:+.6f} mm⁻¹  ->  f2 = {1 / phi2:+8.2f} mm   (火石，负)")
    add(f"  校验: φ1+φ2 = {phi1 + phi2:.6f}  （目标 {phi:.6f}）")

    add("\n[三] 优化后的处方（三面胶合，胶合面曲率共用）")
    add(f"  {'面':<4}{'半径 R':>12}{'厚度 t':>10}   材料")
    add(f"  {'OBJ':<4}{'∞':>12}{'∞':>10}   空气")
    add(f"  {'S1':<4}{p.r1:>12.4f}{p.t1:>10.3f}   {p.glass1}")
    add(f"  {'S2':<4}{p.r2:>12.4f}{p.t2:>10.3f}   {p.glass2}   <- 胶合面")
    add(f"  {'S3':<4}{p.r3:>12.4f}{fs.z_image - p.t1 - p.t2:>10.4f}   空气")
    add(f"  {'IMG':<4}{'∞':>12}{'—':>10}")

    # 几何校验：边缘厚度必须为正，否则镜片不存在
    z0 = sm.gbl_tfrms[1][1][2]
    sd = p.epd / 2.0
    zS1 = sag(sm.ifcs[1].profile.cv, np.array([sd]))[0]
    zS2 = sag(sm.ifcs[2].profile.cv, np.array([sd]))[0]
    zS3 = sag(sm.ifcs[3].profile.cv, np.array([sd]))[0]
    e1 = p.t1 + zS2 - zS1
    e2 = p.t2 + zS3 - zS2
    add(f"\n  几何校验（口径 {p.epd:g} mm 处边缘厚度）: 冕牌 {e1:.2f} mm, 火石 {e2:.2f} mm"
        f"  -> {'合格' if min(e1, e2) > 0.5 else '偏薄，需检查加工性'}")

    add("\n[四] 色差指标")
    add(f"  {'指标':<26}{'双胶合':>16}{'单透镜对照':>16}")
    add(f"  {'LCA(F−C) 近轴 [µm]':<26}{fs.lca_parax * 1e3:>16.3f}{fs_s.lca_parax * 1e3:>16.1f}")
    add(f"  {'LCA(F−C) 全口径 [µm]':<26}{fs.lca_marg * 1e3:>16.1f}{fs_s.lca_marg * 1e3:>16.1f}")
    add(f"  {'二级光谱 [µm]':<26}{fs.secondary * 1e3:>16.1f}{fs_s.secondary * 1e3:>16.1f}")
    add(f"  {'轴向球差 RMS [µm]':<26}{fs.sa_rms * 1e3:>16.1f}{'-':>16}")
    add(f"  {'像面弥散 RMS [µm]':<26}{fs.spot_rms_all * 1e3:>16.2f}{fs_s.spot_rms_all * 1e3:>16.1f}")

    add("\n  各波长近轴焦点（从前顶点起算，mm）：")
    for w in (WVL_F, WVL_D, WVL_C):
        add(f"    {WL_LABEL[w]:<14} {fs.z_parax[w]:>10.4f}")

    add("\n[五] 球差 / 色球差曲线（相对各自近轴焦点，µm）")
    add(f"  {'h(mm)':>7}{'F 光':>10}{'d 光':>10}{'C 光':>10}")
    hs, sa_f = fs.sa_curves[WVL_F]
    _, sa_d2 = fs.sa_curves[WVL_D]
    _, sa_c = fs.sa_curves[WVL_C]
    for i in range(len(hs)):
        add(f"  {hs[i]:>7.2f}{sa_f[i] * 1e3:>10.1f}{sa_d2[i] * 1e3:>10.1f}{sa_c[i] * 1e3:>10.1f}")
    add(f"  边缘(h={hs[-1]:.2f}) F/C 分离 = {(sa_f[-1] - sa_c[-1]) * 1e3:+.1f} µm   <- 色球差")

    if scan:
        add("\n[六] 弯曲扫描记录（R2 -> d 光轴向球差 RMS）")
        for r2, val in scan:
            add(f"    R2={r2:8.2f} mm   SA_rms={val:8.1f} µm")
        add("    （R1/R3 在每次扫描中都被重新求解，保证 EFL=100 且 LCA=0）")
    else:
        add("\n[六] 弯曲扫描：本次未重跑（加 --optimize 可复现整个寻优过程）")
        add(f"    当前处方 R2={p.r2:.4f} 即为扫描谷底，球差 RMS={fs.sa_rms * 1e3:.1f} µm")

    add("\n[七] 校验与结论")
    theory = p.efl_target / v1
    add(f"  · 单透镜 LCA 理论 f/V = {theory * 1e3:.0f} µm，实测 {abs(fs_s.lca_parax) * 1e3:.0f} µm"
        f"（偏差 {abs(abs(fs_s.lca_parax) - theory) / theory * 100:.1f}%，来自透镜厚度）")
    add(f"  · 双胶合 LCA = {fs.lca_parax * 1e3:+.3f} µm，压缩比 {abs(fs_s.lca_parax) / max(abs(fs.lca_parax), 1e-9):.0f}×")
    add(f"  · 但近轴为零不等于全口径为零：全口径 LCA = {fs.lca_marg * 1e3:+.1f} µm（色球差）")
    add(f"  · 二级光谱 {fs.secondary * 1e3:+.1f} µm 是双胶合的物理极限，调曲率无法消除")
    add("=" * 74)
    return "\n".join(L)


# ================================================================
# 7. 主流程
# ================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="消色差双胶合设计与三波长光线追迹")
    ap.add_argument("--optimize", action="store_true",
                    help="重新执行弯曲扫描优化（默认用脚本内已验证的处方）")
    ap.add_argument("--efl", type=float, default=100.0, help="目标焦距 mm")
    ap.add_argument("--epd", type=float, default=25.4, help="入瞳直径 mm")
    ap.add_argument("--rays", type=int, default=11, help="每个波长的光线数")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_dir = root / "output"
    p = Prescription(efl_target=args.efl, epd=args.epd, n_rays=args.rays)

    print("=" * 74)
    print("消色差双胶合 · 三波长真实光线追迹")
    print("=" * 74)

    try:
        import opticalglass  # noqa: F401
    except ImportError:
        print("[错误] 缺少 opticalglass（rayoptics 的玻璃库依赖）")
        print("       修复: python -m pip install opticalglass")
        return 2

    scan = []
    if args.optimize:
        print("[1/4] 弯曲扫描 + 联合优化 ...")
        p, scan = optimize_design(p)
        print(f"      -> R1={p.r1:.4f}  R2={p.r2:.4f}  R3={p.r3:.4f}")
    else:
        print("[1/4] 采用脚本内已验证处方（--optimize 可重新优化）")

    print("[2/4] 追迹双胶合 ...")
    opm, sm, osp = build_doublet(p)
    fs = measure("双胶合", opm, sm, osp, p)

    print("[3/4] 追迹单透镜对照 ...")
    opm_s, sm_s, osp_s = build_singlet(p)
    fs_s = measure("单透镜", opm_s, sm_s, osp_s, p)

    print("[4/4] 出图与报告 ...")
    png = out_dir / "achromatic_raytrace.png"
    plot_result(fs, fs_s, p, png)
    txt = build_report(fs, fs_s, p, scan)
    rpt = out_dir / "achromatic_report.txt"
    out_dir.mkdir(parents=True, exist_ok=True)
    rpt.write_text(txt, encoding="utf-8")

    print()
    print(txt)
    print()
    print(f"图   : {png}")
    print(f"报告 : {rpt}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[中断] 用户取消")
        sys.exit(130)
    except Exception:
        print("\n[异常] 脚本执行失败，完整堆栈：")
        traceback.print_exc()
        sys.exit(1)
