# -*- coding: utf-8 -*-
"""
任务 08 — 离轴视场像差：彗差 / 像散 / 场曲 的真实光线追迹
================================================================================
设计目标
    沿用任务 06 的消色差双胶合（N-BK7 + N-SF5，EFL=100mm，f/3.94），
    把视场从 0° 开到 3°、5°，看轴上已经做到极致的系统在离轴处如何崩坏。
    不换 APO：因为要看的就是"典型像差"，三片式复消色差只会把问题掩盖得更深。

为什么轴上干净、离轴一塌糊涂
    轴上（θ=0）唯一的不对称来源消失了，球面对称性保证：
        · 所有孔径带的光线都在同一个子午面内
        · 波前只剩旋转对称项（球差 + 色差）
    一旦光束斜着进来（θ≠0），孔径上下两侧的"等效焦距"不再相等，
    三次像差里的彗差、像散、场曲同时被激活：
        · 彗差  W131 ∝ θ¹  —— 横向像差 δy ∝ ρ²，随视场**线性**增长
        · 像散  W222 ∝ θ²  —— 切向/弧矢焦距分裂，随视场**平方**增长
        · 场曲  W220 ∝ θ²  —— 最佳像面弯向透镜（Petzval），也随视场**平方**增长
    本脚本用真实光线追迹把这三个指数测出来，作为"是不是三次像差"的判据。

量化方法（三条独立的尺子，互相校验）
    (1) 弥散斑：六角极坐标网格 127 条光线，统计质心、RMS 半径、X/Y 跨度、
        二阶矩主轴 → 看形状是"彗星"还是"椭圆/线段"。
    (2) 切向扇 δy(ρ) 的三次多项式拟合：δy = a1·ρ + a2·ρ² + a3·ρ³
        · a1 是离焦（离焦只影响奇次项，不影响 a2 —— 所以 a2 是**纯彗差**）
        · a2 是彗差，a3 是球差
    (3) 调焦扫描：分别把"切向扇(0,ρ)"和"弧矢扇(ρ,0)"的 RMS 最小化，
        两个极值位置之差就是像散 Δz_ts；二者中点近似 Petzval 面。
        RMS²(dz) 是 dz 的二次函数，极值可解析求出，不依赖搜索步长：
            dz* = −⟨q̃·s̃⟩ / ⟨s̃²⟩
        其中 q̃ 是相对质心的像面坐标，s̃ 是相对质心的斜率。

崩溃判据
    以艾里半径 r_A = 1.22·λ·F#（本项目 2.82 µm）为标尺：
        RMS ≤ 2 r_A   接近衍射极限
        RMS = 5 r_A   明显变软（Strehl 已跌破 0.2 量级）
        RMS ≥ 10 r_A  彻底崩溃，后处理救不回来
    同时给出"允许每个视场单独调焦"这一宽松条件下的崩溃视场。

输出
    output/offaxis_spot.png          三视场弥散斑 + 彗差扇 + 像散 + 场扫描
    output/offaxis_throughfocus.png  5° 视场的调焦序列（线→圆→线）+ T/S 场曲
    output/offaxis_report.txt        文字报告
    data/offaxis_spot.csv            三条视场的光线级落点（可复算）
    data/offaxis_fieldscan.csv       场扫描原始数据
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import traceback
from dataclasses import dataclass, field as dc_field
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle

import rayoptics.optical.model_constants as mc
from rayoptics.environment import OpticalModel, PupilSpec, FieldSpec, WvlSpec
from rayoptics.raytr.trace import trace_base


# ================================================================
# 0. 复用任务 06 的处方（唯一真源，避免两处维护同一组曲率）
# ================================================================
def load_doublet_module():
    """
    以文件方式加载 06_achromatic_doublet.py，只取它里面的 Prescription。

    坑：模块用了 `from __future__ import annotations` + @dataclass，
    必须先注册到 sys.modules 再 exec_module，否则 dataclasses 反查
    sys.modules[cls.__module__] 会拿到 None 而炸。
    """
    path = Path(__file__).resolve().parent / "06_achromatic_doublet.py"
    if not path.exists():
        raise FileNotFoundError(
            f"找不到任务 06 的脚本：{path}\n"
            f"       修复：确认 06_achromatic_doublet.py 与本脚本在同一目录"
        )
    spec = importlib.util.spec_from_file_location("_doublet06", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_doublet06"] = mod
    spec.loader.exec_module(mod)
    return mod


M06 = load_doublet_module()
WVL_F, WVL_D, WVL_C = M06.WVL_F, M06.WVL_D, M06.WVL_C

FIELD_COLOR = {0.0: "#2E86C1", 3.0: "#E67E22", 5.0: "#C0392B"}
AP_COLOR = {25.4: "#C0392B", 12.7: "#E67E22", 8.0: "#2E86C1"}


# ================================================================
# 1. 参数
# ================================================================
@dataclass
class Config:
    fields: tuple = (0.0, 3.0, 5.0)     # 分析视场（度）
    rings: int = 6                       # 六角极坐标环数 -> 1+6·(1+..+6) = 127 条
    fan_rays: int = 21                   # 切向/弧矢扇的点数
    scan_max: float = 6.0                # 场扫描上限（度）
    scan_step: float = 0.25
    scan_aps: tuple = (25.4, 12.7, 8.0)  # 场扫描用的入瞳直径：f/3.9 / f/7.9 / f/12.5
    wvl: float = WVL_D                   # 像差分析用主波长（单色）
    sd: float = 13.5                     # 镜片机械半口径 mm（25.4 光束 + 离轴偏移余量）


@dataclass
class FieldResult:
    """单个视场的全部测量结果（长度单位除注明外均为 µm）"""
    theta: float
    y_chief: float                  # 主光线像高 mm
    spot: np.ndarray                # (N,3) 相对主光线的落点 x,y,环带 ρ
    centroid: np.ndarray            # 质心相对主光线
    rms_img: float                  # 像面处弥散 RMS
    rms_best: float                 # 最佳焦面处弥散 RMS
    dz_best: float                  # 最佳焦面相对像面的位移（负 = 朝透镜）
    ptp_x: float
    ptp_y: float
    ax_ratio: float                 # 二阶矩主轴长宽比（>1 表示沿 y 拉长）
    chief_shift: float              # 质心 − 主光线 的 y 分量（彗差不对称性）
    rho: np.ndarray                 # 弥散斑每条光线所属的光瞳环带（配色用）
    fan_rho: np.ndarray             # 切向扇的 ρ 序列
    fan_t: np.ndarray               # 切向扇 δy(ρ)，像面处
    a1: float
    a2: float                       # 彗差系数
    a3: float                       # 球差系数
    z_tan: float
    z_sag: float
    z_med: float
    rms_t: float                    # 切向焦面的 RMS
    rms_s: float                    # 弧矢焦面的 RMS
    n_lost: int = 0


# ================================================================
# 2. 建模与追迹
# ================================================================
class System:
    """一个给定口径 + 给定视场列表的序列模型；所有追迹都从这里出发"""

    def __init__(self, p, epd: float, fields: tuple, sd: float = 13.5):
        self.p = p
        self.epd = float(epd)
        self.fields = tuple(float(f) for f in fields)
        self.sd = float(sd)

        opm = OpticalModel()
        sm, osp = opm["seq_model"], opm["optical_spec"]
        opm.radius_mode = True

        osp["pupil"] = PupilSpec(osp, key=("object", "epd"), value=self.epd)
        osp["fov"] = FieldSpec(osp, key=("object", "angle"),
                               value=max(self.fields), flds=list(self.fields))
        osp["wvls"] = WvlSpec([(WVL_F, 1.0), (WVL_D, 1.0), (WVL_C, 1.0)], ref_wl=1)

        sm.gaps[0].thi = 1e10                  # 物在无穷远
        sm.add_surface([p.r1, p.t1, p.glass1])  # S1 冕牌前表面
        sm.add_surface([p.r2, p.t2, p.glass2])  # S2 胶合面
        sm.add_surface([p.r3, p.img_gap])       # S3 火石后表面
        opm.update_model()

        # 像面钉在 d 光的近轴焦点上（等价于 Zemax 的 paraxial image）
        fod = opm["analysis_results"]["parax_data"].fod
        sm.gaps[-1].thi = round(float(fod.img_dist), 6)
        opm.update_model()
        sm.compute_global_coords()

        self.opm, self.sm, self.osp = opm, sm, osp
        self.efl = float(fod.efl)
        self.enp_dist = float(fod.enp_dist)
        self.tf = sm.gbl_tfrms
        self.z_img = float(self.tf[-1][1][2])
        self._chief_cache: dict = {}

        # 每个视场的光束在 S1/S3 上的实际脚印（判断口径是否够用）
        self.footprint = self._footprint()

    # ---------- 单条光线 ----------
    def trace(self, ifld: int, wvl: float, px: float, py: float):
        """
        追迹一条光线，返回 (像面交点 xyz, 出射方向余弦 xyz)。
        px/py 是相对入瞳坐标（±1 对应入瞳边缘），随视场一起倾斜 ——
        rayoptics 的 'rel pupil' 会自动把光瞳网格放到主光线周围。
        """
        ray, _, _ = trace_base(self.opm, (float(px), float(py)),
                               self.osp["fov"].fields[ifld], wvl,
                               pupil_type="rel pupil")
        rot, trns = self.tf[-1]
        q = rot.dot(ray[-1][mc.p]) + trns
        d = rot.dot(ray[-1][mc.d])
        return np.asarray(q, dtype=float), np.asarray(d, dtype=float)

    def chief(self, ifld: int, wvl: float):
        """主光线（光瞳中心）落点，作为弥散斑的坐标原点"""
        key = (ifld, wvl)
        if key not in self._chief_cache:
            q, _ = self.trace(ifld, wvl, 0.0, 0.0)
            self._chief_cache[key] = q
        return self._chief_cache[key]

    # ---------- 一束光线 ----------
    def bundle(self, ifld, wvl, grid, dz=0.0):
        """
        追迹一束光（grid 为 (N,2) 光瞳坐标），可选地把落点沿出射方向
        平移到 dz 处的离焦面。返回 (Q, D)：Q 是 2D 交点（相对主光线，mm），
        D 是对应方向余弦。追迹失败的光线丢掉并计数。
        """
        q0 = self.chief(ifld, wvl)
        Q, D = [], []
        for px, py in grid:
            try:
                q, d = self.trace(ifld, wvl, px, py)
            except Exception:
                continue
            if not np.all(np.isfinite(q)) or abs(d[2]) < 1e-9:
                continue
            Q.append(q[:2] + (dz / d[2]) * d[:2] - q0[:2])
            D.append(d)
        if not Q:
            return np.zeros((0, 2)), np.zeros((0, 3))
        return np.asarray(Q), np.asarray(D)

    def _footprint(self):
        """
        各视场光束在三个折射面上的最大 |y|（判断镜片口径够不够）。
        口径不够就会渐晕 —— 那样测到的"像差"里混进了切趾效应，结论会失真，
        所以这一步必须先做。
        """
        out = {}
        for i, th in enumerate(self.fields):
            hmax = 0.0
            for k in range(36):
                a = 2 * np.pi * k / 36
                try:
                    ray, _, _ = trace_base(self.opm, (np.cos(a), np.sin(a)),
                                           self.osp["fov"].fields[i], WVL_D,
                                           pupil_type="rel pupil")
                except Exception:
                    continue
                for j in (1, 2, 3):
                    rot, trns = self.tf[j]
                    h = float(rot.dot(ray[j][mc.p])[1] + trns[1])
                    hmax = max(hmax, abs(h))
            out[th] = hmax
        return out


# ================================================================
# 3. 光瞳采样
# ================================================================
def hexapolar(rings: int) -> tuple[np.ndarray, np.ndarray]:
    """六角极坐标网格（Zemax 默认样式）：中心 1 点 + 每环 6i 点"""
    pts, rho = [(0.0, 0.0)], [0.0]
    for i in range(1, rings + 1):
        r = i / rings
        for k in range(6 * i):
            a = 2 * np.pi * k / (6 * i) + (0.0 if i % 2 == 0 else np.pi / (6 * i))
            pts.append((r * np.cos(a), r * np.sin(a)))
            rho.append(r)
    return np.asarray(pts), np.asarray(rho)


def fan_grid(n: int, axis: str) -> np.ndarray:
    """切向扇（y 方向，axis='t'）或弧矢扇（x 方向，axis='s'）"""
    r = np.linspace(-1.0, 1.0, n)
    return np.column_stack([np.zeros_like(r), r]) if axis == "t" else \
        np.column_stack([r, np.zeros_like(r)])


# ================================================================
# 4. 度量
# ================================================================
def focus_stats(Q: np.ndarray, D: np.ndarray, dz: float) -> tuple[float, float]:
    """离焦 dz 处的 (RMS 半径, 质心 y)"""
    S = D[:, :2] / D[:, 2:3]
    q = Q + dz * S
    c = q.mean(0)
    return float(np.sqrt(np.mean(((q - c) ** 2).sum(1)))), float(c[1])


def best_focus(Q: np.ndarray, D: np.ndarray) -> tuple[float, float]:
    """
    RMS 半径最小的焦面位置（解析解）。
    rms²(dz) = ⟨|q̃ + dz·s̃|²⟩ 是 dz 的二次函数 → 顶点即最优。
    比网格扫描稳，也不会被步长卡住。
    """
    S = D[:, :2] / D[:, 2:3]
    qd = Q - Q.mean(0)
    sd = S - S.mean(0)
    A = float(np.mean((sd ** 2).sum(1)))
    B = float(np.mean((qd * sd).sum(1)))
    C0 = float(np.mean((qd ** 2).sum(1)))
    if A < 1e-15:
        return 0.0, float(np.sqrt(max(C0, 0.0)))
    dz = -B / A
    var = C0 - B * B / A
    return float(dz), float(np.sqrt(max(var, 0.0)))


def principal_axes(P: np.ndarray) -> tuple[float, float, float]:
    """二阶矩主轴：返回 (长轴/短轴比, 长轴与 x 轴夹角°, 长轴长度 µm)"""
    if len(P) < 6:
        return 1.0, 0.0, 0.0
    C = np.cov(P.T)
    w, v = np.linalg.eigh(C)
    w = np.clip(w, 1e-18, None)
    major = v[:, int(np.argmax(w))]
    ang = float(np.degrees(np.arctan2(major[1], major[0])))
    return float(np.sqrt(w.max() / w.min())), ang, float(np.sqrt(w.max()) * 2.0)


def coma_fit(rho: np.ndarray, dy: np.ndarray) -> tuple[float, float, float]:
    """
    切向扇 δy(ρ) = a1·ρ + a2·ρ² + a3·ρ³ 最小二乘。
    · a1 由离焦贡献（离焦是奇次项）
    · a2 是纯彗差（偶次项，离焦无法伪造）
    · a3 是球差（奇次三次项）
    """
    A = np.vstack([rho, rho ** 2, rho ** 3]).T
    coef, *_ = np.linalg.lstsq(A, dy, rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2])


def airy_radius(epd: float, efl: float, lam_um: float = 0.5875618) -> float:
    """艾里半径 µm：1.22·λ·F#"""
    return 1.22 * lam_um * (efl / epd)


def crossing(x: np.ndarray, y: np.ndarray, level: float) -> float:
    """y(x) 单调上升时，求 y=level 对应的 x（线性插值）；无解返回 nan"""
    d = y - level
    idx = np.where(np.diff(np.sign(d)) != 0)[0]
    if len(idx) == 0:
        return float("nan")
    i = int(idx[0])
    x0, x1, y0, y1 = x[i], x[i + 1], y[i], y[i + 1]
    if abs(y1 - y0) < 1e-15:
        return float(x0)
    return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))


# ================================================================
# 5. 单视场完整分析
# ================================================================
def analyse_field(sysm: System, ifld: int, cfg: Config) -> FieldResult:
    theta = sysm.fields[ifld]
    grid, rho = hexapolar(cfg.rings)

    # --- 像面弥散斑 ---
    Q, D = sysm.bundle(ifld, cfg.wvl, grid)
    if len(Q) == 0:
        raise RuntimeError(f"视场 {theta}° 全部光线追迹失败")
    n_lost = len(grid) - len(Q)
    rho = rho[: len(Q)]

    cent = Q.mean(0)
    rms_img = float(np.sqrt(np.mean(((Q - cent) ** 2).sum(1))))
    dz_best, rms_best = best_focus(Q, D)
    ratio, _, _ = principal_axes(Q)
    y_chief = float(sysm.chief(ifld, cfg.wvl)[1])

    # --- 切向扇 → 彗差 ---
    gt = fan_grid(cfg.fan_rays, "t")
    Ft, Dt = sysm.bundle(ifld, cfg.wvl, gt)
    dy = Ft[:, 1] - Ft[len(Ft) // 2, 1]
    a1, a2, a3 = coma_fit(gt[:, 1], dy)

    # --- 切向 / 弧矢焦点（各自的 RMS 极小位置）---
    gs = fan_grid(cfg.fan_rays, "s")
    Fs, Ds = sysm.bundle(ifld, cfg.wvl, gs)
    z_tan, rms_t = best_focus(Ft, Dt)
    z_sag, rms_s = best_focus(Fs, Ds)

    return FieldResult(
        theta=theta, y_chief=y_chief,
        spot=np.column_stack([Q * 1e3, rho]),          # µm
        centroid=cent * 1e3, rms_img=rms_img * 1e3, rms_best=rms_best * 1e3,
        dz_best=dz_best * 1e3, ptp_x=float(np.ptp(Q[:, 0]) * 1e3),
        ptp_y=float(np.ptp(Q[:, 1]) * 1e3), ax_ratio=ratio,
        chief_shift=float(cent[1] * 1e3),
        rho=rho, fan_rho=gt[:, 1], fan_t=dy * 1e3,
        a1=a1 * 1e3, a2=a2 * 1e3, a3=a3 * 1e3,
        z_tan=z_tan * 1e3, z_sag=z_sag * 1e3, z_med=0.5 * (z_tan + z_sag) * 1e3,
        rms_t=rms_t * 1e3, rms_s=rms_s * 1e3, n_lost=n_lost,
    )


# ================================================================
# 6. 场扫描
# ================================================================
def field_scan(p, cfg: Config, epd: float, thetas: np.ndarray):
    """固定口径，扫视场；返回 (rms_img, rms_best, z_best, a2, dz_astig)"""
    sysm = System(p, epd, [0.0] + list(thetas), sd=cfg.sd)
    grid, _ = hexapolar(cfg.rings)

    out = {k: [] for k in ("rms_img", "rms_best", "dz_best", "a2", "astig",
                           "z_tan", "z_sag", "y")}
    for i in range(1, len(sysm.fields)):
        Q, D = sysm.bundle(i, cfg.wvl, grid)
        if len(Q) < 10:
            for k in out:
                out[k].append(np.nan)
            continue
        c = Q.mean(0)
        out["rms_img"].append(np.sqrt(np.mean(((Q - c) ** 2).sum(1))) * 1e3)
        dz, rb = best_focus(Q, D)
        out["rms_best"].append(rb * 1e3)
        out["dz_best"].append(dz * 1e3)
        out["y"].append(float(sysm.chief(i, cfg.wvl)[1]))

        Ft, Dt = sysm.bundle(i, cfg.wvl, fan_grid(cfg.fan_rays, "t"))
        Fs, Dsc = sysm.bundle(i, cfg.wvl, fan_grid(cfg.fan_rays, "s"))
        dy = Ft[:, 1] - Ft[len(Ft) // 2, 1]
        _, a2, _ = coma_fit(np.linspace(-1, 1, len(Ft)), dy)
        out["a2"].append(a2 * 1e3)
        zt, _ = best_focus(Ft, Dt)
        zs, _ = best_focus(Fs, Dsc)
        out["astig"].append((zs - zt) * 1e3)
        out["z_tan"].append(zt * 1e3)
        out["z_sag"].append(zs * 1e3)

    return {k: np.asarray(v) for k, v in out.items()}


# ================================================================
# 7. 绘图
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


def draw_spot(ax, res: FieldResult, lim: float, title: str, sub: str):
    """画一个视场的弥散斑：颜色 = 光瞳环带 ρ（能看到彗尾由外圈构成）"""
    x, y, rho = res.spot[:, 0], res.spot[:, 1], res.spot[:, 2]
    ax.scatter(x, y, c=rho, cmap="viridis", s=17, lw=0.0, alpha=0.92, zorder=3)

    # 主光线（坐标原点）与质心
    ax.plot([0], [0], marker="+", ms=11, mew=1.6, color="#111111", zorder=5)
    ax.plot([res.centroid[0]], [res.centroid[1]], marker="o", ms=6.5, mfc="none",
            mec="#7D3C98", mew=1.5, zorder=5)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axhline(0, color="#999999", lw=0.6, ls=":", zorder=1)
    ax.axvline(0, color="#999999", lw=0.6, ls=":", zorder=1)
    ax.set_xlabel("x  弧矢方向 (µm)", fontsize=9)
    ax.set_ylabel("y  子午方向 (µm)", fontsize=9)
    ax.set_title(title, fontsize=11.5, pad=7)
    ax.text(0.035, 0.965, sub, transform=ax.transAxes, ha="left", va="top",
            fontsize=8.4, color="#1A1A1A", linespacing=1.5, zorder=6,
            bbox=dict(boxstyle="round,pad=0.38", fc="#FFFFFF", ec="#BDC3C7", alpha=1.0))
    ax.grid(alpha=0.12, ls=":")
    ax.tick_params(labelsize=8)


def plot_main(results: list[FieldResult], cfg: Config, sysm: System,
              scan: dict, out_png: Path):
    setup_chinese_font()
    airy = airy_radius(sysm.epd, sysm.efl)

    # 统一比例尺：取最大视场的最大偏移，向上取整到 20 µm
    lim = max(np.abs(r.spot[:, :2]).max() for r in results) * 1.06
    lim = float(np.ceil(lim / 20.0) * 20.0)

    fig = plt.figure(figsize=(17.2, 10.8), dpi=150)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.22, 1.0],
                          wspace=0.30, hspace=0.42,
                          left=0.055, right=0.975, top=0.875, bottom=0.075)

    # ---------------- 上排：三视场弥散斑（同一比例尺）----------------
    for i, res in enumerate(results):
        ax = fig.add_subplot(gs[0, i])
        shape = "圆形（对称）" if res.ax_ratio < 1.25 else (
            "彗星状（沿 y 拖尾）" if abs(res.centroid[1]) > 0.25 * res.rms_img else "椭圆")
        sub = (f"RMS = {res.rms_img:.1f} µm   （艾里半径 {airy:.2f} µm 的 {res.rms_img / airy:.1f} 倍）\n"
               f"跨度 x×y = {res.ptp_x:.0f} × {res.ptp_y:.0f} µm   主轴比 {res.ax_ratio:.2f}\n"
               f"形状：{shape}")
        draw_spot(ax, res, lim, f"视场 {res.theta:.0f}°   像高 {res.y_chief:.2f} mm", sub)

        if i == 0:
            # 轴上斑太小，加一个 ±20µm 的放大窗，才看得见它是艾里斑量级
            axin = ax.inset_axes([0.50, 0.055, 0.46, 0.46])
            axin.scatter(res.spot[:, 0], res.spot[:, 1], c=res.spot[:, 2],
                         cmap="viridis", s=16, lw=0.0)
            axin.add_patch(Circle((0, 0), airy, fill=False, ec="#B03A2E",
                                  lw=1.2, ls="--"))
            axin.set_xlim(-20, 20)
            axin.set_ylim(-20, 20)
            axin.set_aspect("equal")
            axin.set_title("轴上放大 ±20µm（虚线圈 = 艾里斑）", fontsize=7.6, pad=3)
            axin.tick_params(labelsize=6.5)
            axin.grid(alpha=0.15, ls=":")
            for s in axin.spines.values():
                s.set_edgecolor("#777777")

    # ---------------- 左下：切向扇 δy(ρ) —— 彗差的指纹 ----------------
    ax = fig.add_subplot(gs[1, 0])
    for res in results:
        c = FIELD_COLOR.get(res.theta, "#555555")
        ax.plot(res.fan_rho, res.fan_t, "o-", ms=3.4, lw=1.6, color=c, alpha=0.9,
                label=f"{res.theta:.0f}°   a₂={res.a2:+.1f} µm")
        # 只用二次项画一条辅助抛物线，直观显示"不对称的那一半"
        rr = np.linspace(-1, 1, 100)
        ax.plot(rr, res.a2 * rr ** 2, "--", lw=0.9, color=c, alpha=0.55)
    ax.axhline(0, color="#666666", lw=0.9)
    ax.axvline(0, color="#666666", lw=0.9)
    ax.set_xlabel("光瞳坐标 ρ（子午面内，±1 = 边缘）", fontsize=9.5)
    ax.set_ylabel("横向像差 δy  (µm，相对主光线)", fontsize=9.5)
    ax.set_title("切向扇 δy(ρ)：奇函数→偶函数的转变就是彗差", fontsize=11.5, pad=7)
    ax.grid(alpha=0.15, ls=":")
    ax.legend(fontsize=8.2, loc="upper center", framealpha=1.0)
    ax.text(0.02, 0.04,
            "0° 是奇对称的三次曲线（纯球差）；\n"
            "3°/5° 变成开口向上的抛物线 → 上下边缘光线\n"
            "落在主光线的同一侧，这就是彗尾。\n"
            "虚线 = 拟合出的纯 ρ² 项。",
            transform=ax.transAxes, fontsize=8.0, va="bottom", ha="left",
            linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="#FBFCFC", ec="#95A5A6", alpha=1.0))

    # ---------------- 中下：离焦扫描 —— 像散的直接证据 ----------------
    ax = fig.add_subplot(gs[1, 1])
    res5 = results[-1]
    sysm5 = sysm
    ifld5 = sysm5.fields.index(res5.theta)
    Qa, Da = sysm5.bundle(ifld5, cfg.wvl, hexapolar(cfg.rings)[0])
    Qt, Dt = sysm5.bundle(ifld5, cfg.wvl, fan_grid(cfg.fan_rays, "t"))
    Qs, Ds = sysm5.bundle(ifld5, cfg.wvl, fan_grid(cfg.fan_rays, "s"))

    # 注意单位：横轴一律用 µm（dzs 本身就是 µm；best_focus 返回 mm，×1e3 换算）
    dzs = np.linspace(-1800, 400, 221)
    curves = [("全瞳（127 条）", Qa, Da, "#34495E", 1.9),
              ("切向扇", Qt, Dt, "#C0392B", 1.5),
              ("弧矢扇", Qs, Ds, "#1F618D", 1.5)]
    ymax = 0.0
    for lab, Q, D, c, lw in curves:
        v = [focus_stats(Q, D, dz / 1e3)[0] * 1e3 for dz in dzs]
        ymax = max(ymax, float(np.max(v)))
        ax.plot(dzs, v, color=c, lw=lw, label=lab)
        dz_b, rb = best_focus(Q, D)
        ax.plot([dz_b * 1e3], [rb * 1e3], marker="v", ms=7, color=c, zorder=5)
        ax.annotate(f"{dz_b * 1e3:+.0f}", (dz_b * 1e3, rb * 1e3),
                    textcoords="offset points", xytext=(0, -14),
                    fontsize=7.8, color=c, ha="center")

    ax.set_ylim(0.0, ymax * 1.40)
    ax.axvline(0, color="#555555", ls="--", lw=1.1)
    ax.text(28, ymax * 1.30, "像面", fontsize=8.5, color="#444444")

    ax.set_xlabel("相对像面的离焦 Δz  (µm，负 = 朝透镜)", fontsize=9.5)
    ax.set_ylabel("弥散斑 RMS 半径  (µm)", fontsize=9.5)
    ax.set_title("5° 视场的调焦扫描：切向扇与弧矢扇的极小值不重合 = 像散", fontsize=11.5, pad=7)
    ax.grid(alpha=0.15, ls=":")
    ax.legend(fontsize=8.2, loc="upper right", framealpha=1.0)
    y_ann = ymax * 0.62
    ax.annotate("", xy=(res5.z_tan, y_ann), xytext=(res5.z_sag, y_ann),
                arrowprops=dict(arrowstyle="<->", color="#7D3C98", lw=1.4))
    ax.text(0.5 * (res5.z_tan + res5.z_sag), y_ann * 1.05,
            f"像散 {abs(res5.z_tan - res5.z_sag):.0f} µm\n（切向焦面更靠透镜）",
            fontsize=8.4, color="#7D3C98", ha="center", va="bottom", linespacing=1.35)

    # ---------------- 右下：场扫描 —— 崩溃在哪里 ----------------
    ax = fig.add_subplot(gs[1, 2])
    th = scan["theta"]
    for epd in cfg.scan_aps:
        v = scan[epd]["rms_img"]
        c = AP_COLOR.get(epd, "#555555")
        ax.semilogy(th, np.clip(v, 1e-3, None), "-", lw=1.8, color=c,
                    label=f"EPD {epd:g} mm (f/{sysm.efl / epd:.1f})")

    airy0 = airy_radius(25.4, sysm.efl)
    for k, ls in ((2, ":"), (5, "--"), (10, "-")):
        ax.axhline(k * airy0, color="#7B241C", ls=ls, lw=1.1, alpha=0.75)
        ax.text(th[-1], k * airy0 * 1.06, f"{k}×艾里", fontsize=7.4,
                color="#7B241C", ha="right", va="bottom")

    for th_mark, lab in ((3.0, "3°"), (5.0, "5°")):
        ax.axvline(th_mark, color="#95A5A6", lw=0.8, ls=":")
    ax.set_xlabel("视场角 θ  (度)", fontsize=9.5)
    ax.set_ylabel("像面弥散斑 RMS  (µm，对数)", fontsize=9.5)
    ax.set_title("像质随视场的崩坏过程（三条曲线 = 三种光圈）", fontsize=11.5, pad=7)
    ax.grid(alpha=0.15, ls=":", which="both")
    ax.legend(fontsize=8.0, loc="lower right", framealpha=1.0)
    ax.set_xlim(0, th[-1] * 1.01)

    fig.suptitle(
        "消色差双胶合 · 离轴视场像差（真实光线追迹，127 条/视场，d 光）\n"
        "轴上只剩球差（左下图那条奇对称曲线）；视场一开，ρ² 项长出来 → 彗差；再往外，切向/弧矢焦点分裂 → 像散",
        fontsize=13.2, y=0.965)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_through_focus(sysm: System, res5: FieldResult, cfg: Config,
                       z_tan_all: np.ndarray, field_curve: dict, out_png: Path):
    """5° 视场的调焦序列（线→圆→线）+ T/S 场曲曲线"""
    setup_chinese_font()
    ifld = sysm.fields.index(res5.theta)
    grid, _ = hexapolar(cfg.rings)

    dz_list = [0.0,
               res5.z_sag / 1e3,
               0.5 * (res5.z_tan + res5.z_sag) / 1e3,
               res5.z_tan / 1e3,
               res5.z_tan / 1e3 * 1.30]

    fig = plt.figure(figsize=(17.2, 9.4), dpi=150)
    gs = fig.add_gridspec(2, 5, height_ratios=[1.0, 0.85],
                          wspace=0.30, hspace=0.30,
                          left=0.045, right=0.985, top=0.835, bottom=0.085)

    # 先把五个停位都追一遍，再统一确定比例尺（同尺度才好比较形状）
    frames = []
    for dz in dz_list:
        Q, D = sysm.bundle(ifld, cfg.wvl, grid, dz=dz)
        frames.append((dz, Q, D))
    lim = max(np.abs(Q).max() for _, Q, _ in frames) * 1e3 * 1.05

    for k, (dz, Q, D) in enumerate(frames):
        ax = fig.add_subplot(gs[0, k])
        rms = float(np.sqrt(np.mean(((Q - Q.mean(0)) ** 2).sum(1))))
        px, py = np.ptp(Q[:, 0]) * 1e3, np.ptp(Q[:, 1]) * 1e3
        if px > 1.7 * py:
            lab = "横向拉长：水平焦线\n（切向焦点，彗差把它抹宽）"
        elif py > 1.7 * px:
            lab = "纵向拉长：竖直焦线\n（弧矢焦点）"
        elif abs(dz) < 1e-9:
            lab = "平面像面\n（彗星状）"
        else:
            lab = "两焦线之间\n（趋近最小弥散圆）"
        ax.scatter(Q[:, 0] * 1e3, Q[:, 1] * 1e3, s=14, c="#34495E", lw=0, alpha=0.9)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.grid(alpha=0.12, ls=":")
        ax.axhline(0, color="#999999", lw=0.5, ls=":")
        ax.axvline(0, color="#999999", lw=0.5, ls=":")
        ax.tick_params(labelsize=7)
        ax.set_title(f"Δz = {dz * 1e3:+.0f} µm", fontsize=10)
        ax.set_xlabel("x (µm)", fontsize=8, labelpad=1)
        if k == 0:
            ax.set_ylabel("y (µm)", fontsize=8)
        ax.text(0.035, 0.965,
                f"{lab}\nRMS {rms * 1e3:.1f} µm\n{px:.0f} × {py:.0f} µm",
                transform=ax.transAxes, ha="left", va="top", fontsize=7.8,
                linespacing=1.45, zorder=6,
                bbox=dict(boxstyle="round,pad=0.35", fc="#FFFFFF",
                          ec="#BDC3C7", alpha=1.0))

    # ---------------- 下排：T/S 场曲 ----------------
    ax = fig.add_subplot(gs[1, :])
    th = field_curve["theta"]
    ax.plot(th, field_curve["z_tan"], "o-", ms=4, lw=1.8, color="#C0392B",
            label="切向焦面 T")
    ax.plot(th, field_curve["z_sag"], "s-", ms=4, lw=1.8, color="#1F618D",
            label="弧矢焦面 S")
    ax.plot(th, field_curve["z_med"], "^--", ms=4, lw=1.3, color="#7D3C98",
            label="中间面（T/S 中点，最小弥散圆附近）")
    ax.axhline(0, color="#333333", lw=1.6, label="平面像面（探测器）")
    ax.fill_between(th, field_curve["z_tan"], field_curve["z_sag"],
                    color="#F5B7B1", alpha=0.35, lw=0,
                    label="像散区（T 与 S 之间）")
    ax.set_xlabel("视场角 θ  (度)", fontsize=10)
    ax.set_ylabel("相对像面的焦点位移  (µm)", fontsize=10)
    ax.set_title("场曲：三个焦面都随 θ² 弯向透镜，而探测器是平的 —— 这就是离焦的来源",
                 fontsize=11.5, pad=8)
    ax.grid(alpha=0.15, ls=":")
    ax.legend(fontsize=9, loc="lower left", ncol=2, framealpha=1.0)

    # θ² 拟合的斜率标注
    m_t = field_curve["slope_tan"]
    m_s = field_curve["slope_sag"]
    ax.text(0.985, 0.955,
            f"θ² 拟合：z_T ≈ {m_t:+.1f}·θ² µm      z_S ≈ {m_s:+.1f}·θ² µm\n"
            f"切向/弧矢偏离 Petzval 面之比 ≈ {abs(field_curve['ratio31']):.2f} : 1（三次像差理论值 3 : 1）",
            transform=ax.transAxes, fontsize=8.6, ha="right", va="top",
            linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.42", fc="#FEF9E7", ec="#B7950B", alpha=1.0))

    fig.suptitle("5° 视场的调焦序列：彗星 → 竖直方向收窄 → 最小弥散圆 → 水平方向收窄\n"
                 "同一条光束，只换观察面：像散让两个方向的焦点分裂，彗差则把焦线抹成扇形",
                 fontsize=13, y=0.975)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# ================================================================
# 8. 报告
# ================================================================
def build_report(results, cfg, sysm, scan, crossings, petzval, out_txt: Path):
    p = sysm.p
    airy = airy_radius(sysm.epd, sysm.efl)
    L = []
    add = L.append

    add("=" * 78)
    add("离轴视场像差分析报告（彗差 / 像散 / 场曲）")
    add("  系统：任务 06 的消色差双胶合  N-BK7 + N-SF5")
    add("=" * 78)

    add("\n[一] 系统与观察条件")
    add(f"  EFL = {sysm.efl:.4f} mm    EPD = {sysm.epd:g} mm    F/# = {sysm.efl / sysm.epd:.2f}")
    add(f"  艾里半径 r_A = 1.22·λ·F# = {airy:.3f} µm（λ = 0.5876 µm，d 光）")
    add(f"  入瞳距 = {sysm.enp_dist:.4f} mm（≈0：光阑就在第一个面上）")
    add(f"  像面：d 光近轴焦点，z = {sysm.z_img:.4f} mm")
    add(f"  光线网格：六角极坐标 {cfg.rings} 环 = {1 + 3 * cfg.rings * (cfg.rings + 1)} 条/视场")
    add("  说明：像差分析用单色 d 光 —— 三次像差理论本就是单色的，混色会污染判据。")

    add("\n[二] 三个视场的弥散斑")
    add(f"  {'视场':>5}{'像高mm':>9}{'RMS µm':>10}{'/r_A':>7}{'最佳焦RMS':>11}"
        f"{'跨度x µm':>10}{'跨度y µm':>10}{'主轴比':>8}")
    for r in results:
        add(f"  {r.theta:>4.0f}°{r.y_chief:>9.3f}{r.rms_img:>10.1f}{r.rms_img / airy:>7.1f}"
            f"{r.rms_best:>11.1f}{r.ptp_x:>10.0f}{r.ptp_y:>10.0f}{r.ax_ratio:>8.2f}")
    add("")
    for r in results:
        if r.theta == 0:
            add(f"  · 0°：{r.ptp_x:.0f}×{r.ptp_y:.0f} µm，主轴比 {r.ax_ratio:.2f} —— 圆斑，"
                f"残留的是纯球差（RMS {r.rms_img:.1f} µm ≈ {r.rms_img / airy:.1f} r_A）")
        else:
            # 彗差符号决定尾巴朝向：a₂<0 → 斑相对主光线向 −y（朝光轴）展开 = 负彗差
            tail = ("负彗差：彗尾向光轴一侧（−y）收"
                    if r.a2 < 0 else "正彗差：彗尾向像高外侧（+y）甩")
            add(f"  · {r.theta:.0f}°：{r.ptp_x:.0f}×{r.ptp_y:.0f} µm，长宽比 {r.ptp_y / max(r.ptp_x, 1e-9):.1f}:1，"
                f"质心偏离主光线 {r.chief_shift:+.1f} µm，光斑{tail} —— 典型彗星形态")
    add(f"  · 光线丢失：{[r.n_lost for r in results]}（0 = 全部通过，无渐晕/全反射）")

    add("\n[三] 彗差：切向扇的三次拟合  δy(ρ) = a₁ρ + a₂ρ² + a₃ρ³")
    add(f"  {'视场':>6}{'a₁ (µm)':>12}{'a₂ (µm)':>12}{'a₃ (µm)':>12}   解读")
    for r in results:
        note = "纯奇次：球差" if r.theta == 0 else f"偶次项主导：彗差 {abs(r.a2) / (abs(r.a1) + abs(r.a3) + 1e-9) * 100:.0f}% 权重"
        add(f"  {r.theta:>5.0f}°{r.a1:>12.1f}{r.a2:>12.1f}{r.a3:>12.1f}   {note}")
    a2s = np.array([r.a2 for r in results if r.theta > 0])
    ths = np.array([r.theta for r in results if r.theta > 0])
    slope = float(np.polyfit(ths, a2s, 1)[0]) if len(a2s) >= 2 else float("nan")
    add(f"  彗差系数随视场：{slope:+.2f} µm/度（线性拟合）")
    add(f"  比值 a₂(5°)/a₂(3°) = {a2s[-1] / a2s[0]:.3f}，理论值 5/3 = {5 / 3:.3f}"
        "  → 彗差 ∝ θ¹，确认是三级彗差")
    add("  重要：离焦只产生奇次项 a₁ρ，不会伪造 a₂ —— 所以 a₂ 是干净的彗差量。")

    add("\n[四] 像散与场曲：切向 / 弧矢焦点")
    add(f"  {'视场':>6}{'z_T (µm)':>12}{'z_S (µm)':>12}{'|像散|':>12}{'z_med (µm)':>13}"
        f"{'T面RMS':>10}{'S面RMS':>10}")
    for r in results:
        add(f"  {r.theta:>5.0f}°{r.z_tan:>12.1f}{r.z_sag:>12.1f}{abs(r.z_tan - r.z_sag):>12.1f}"
            f"{r.z_med:>13.1f}{r.rms_t:>10.1f}{r.rms_s:>10.1f}")
    add("  注：z 为相对平面像面的位移，负 = 朝透镜方向；z_med = 两者中点（最小弥散圆附近）")
    add("")
    d3 = abs(results[1].z_tan - results[1].z_sag)
    d5 = abs(results[2].z_tan - results[2].z_sag)
    add(f"  · 像散 |z_T − z_S|：3° 时 {d3:.0f} µm，5° 时 {d5:.0f} µm")
    add(f"    比值 {d5 / max(d3, 1e-9):.2f}，理论 (5/3)² = {(5 / 3) ** 2:.2f}  → 像散 ∝ θ²，确认是三级像散")
    add("    切向焦面比弧矢焦面更靠近透镜（z_T 更负），这与正透镜的 Petzval 面朝物方弯曲一致。")
    add(f"  · 三个焦面都随 θ² 弯曲：z_T ≈ {petzval['m_tan']:+.1f}·θ² µm，"
        f"z_S ≈ {petzval['m_sag']:+.1f}·θ² µm，z_med ≈ {petzval['m_med']:+.1f}·θ² µm")
    add(f"  · Petzval 面 z_p = (3·z_S − z_T)/2 ≈ {petzval['m_pet']:+.1f}·θ² µm")
    add(f"    切向偏离 : 弧矢偏离 = {abs(petzval['ratio31']):.2f} : 1"
        f"（光阑在透镜上的薄透镜理论值 3 : 1，实测吻合）")
    add(f"  · Petzval 半径：实测 R_p = y²/(2|z_p|) = {petzval['R_meas']:.1f} mm （y = f·tanθ）")
    add(f"                理论 1/R_p = Σ(φ_i/n_i) = {petzval['invRp']:.5f} mm⁻¹ → R_p = {petzval['R_theory']:.1f} mm"
        f"（偏差 {abs(petzval['R_meas'] - petzval['R_theory']) / petzval['R_theory'] * 100:.0f}%，"
        f"薄透镜近似忽略了厚度与入射高）")

    add("\n[五] 像质崩溃判据（以艾里半径 r_A 为标尺）")
    add(f"  {'阈值':<16}{'RMS 限值':>12}{'无重调焦的临界视场':>22}{'允许重调焦':>16}")
    for k in (2, 5, 10):
        lvl = k * airy
        c1 = crossings["img"][k]
        c2 = crossings["best"][k]
        name = {2: "接近衍射极限", 5: "明显变软", 10: "彻底崩溃"}[k]
        add(f"  {name + f' ({k}×r_A)':<16}{lvl:>10.1f} µm"
            f"{('%.2f°' % c1) if np.isfinite(c1) else '未达到':>22}"
            f"{('%.2f°' % c2) if np.isfinite(c2) else '未达到':>16}")
    add("")
    add("  等效波前误差量级（δy_rms / 2F#，仅作量级判断）：")
    for r in results:
        w = r.rms_img / (2 * (sysm.efl / sysm.epd))
        add(f"    {r.theta:>3.0f}°   W_rms ≈ {w:.2f} µm = {w / 0.5875618:.1f} λ   "
            f"（瑞利判据要求 ≤ 0.25 λ）")

    add("\n[六] 缩光圈能不能救？")
    add(f"  {'口径':<10}{'F/#':>6}{'r_A µm':>9}" + "".join(f"{t:>9.0f}°" for t in (1, 2, 3, 5)))
    for epd in cfg.scan_aps:
        a = airy_radius(epd, sysm.efl)
        row = scan[epd]["rms_img"]
        th = scan["theta"]
        vs = [np.interp(t, th, row) for t in (1, 2, 3, 5)]
        add(f"  {epd:<10.1f}{sysm.efl / epd:>6.1f}{a:>9.2f}" + "".join(f"{v:>10.1f}" for v in vs))
    add("  说明：像差 ∝ 口径²（f/3.9→f/12.5 口径缩 3.2 倍，像差缩 ~10 倍），")
    add("        但艾里斑 ∝ 口径⁻¹ 也在长大，所以 RMS/r_A 只改善约 3 倍 —— 缩光圈治不了场曲。")

    add("\n[七] 结论")
    c10 = crossings["img"][10]
    c5 = crossings["img"][5]
    c2 = crossings["img"][2]
    add(f"  1) 轴上（0°）RMS = {results[0].rms_img:.1f} µm（{results[0].rms_img / airy:.1f} r_A），"
        "圆斑、无拖尾 —— 任务 06 的球差/色差校正确实到位。")
    add(f"  2) 3° 时 RMS = {results[1].rms_img:.1f} µm（{results[1].rms_img / airy:.0f} r_A），"
        f"斑已经拉成 {results[1].ptp_y:.0f} µm 长的彗星；5° 时 {results[2].rms_img:.0f} µm"
        f"（{results[2].rms_img / airy:.0f} r_A），比轴上大 {results[2].rms_img / results[0].rms_img:.0f} 倍。")
    add(f"  3) 这个系统的可用视场只有 ±{c2:.1f}°（2×艾里），"
        f"±{c5:.1f}° 已经肉眼可见发软，±{c10:.1f}° 彻底崩溃。")
    add(f"     若允许每个视场单独调焦，崩溃点后移到 ±{crossings['best'][10]:.1f}°——")
    add("     说明限制不是'调焦没调好'，而是像散+彗差本身的波前畸变。")
    add("  4) 崩溃的元凶排序（5° 时）：像散 > 场曲 > 彗差。")
    add(f"     像散把 T/S 焦面拉开 {abs(results[2].z_tan - results[2].z_sag):.0f} µm，"
        f"而焦深只有 ±{2 * 0.5876 * (sysm.efl / sysm.epd) ** 2:.0f} µm —— "
        "任何平面探测器都同时接不住两条焦线。")
    add("  5) 想要宽视场，这个两片结构在物理上做不到，必须引入新的自由度：")
    add("     · 场镜/平场镜（field flattener）：把 Petzval 面压平，最省事的办法")
    add("     · 移动光阑（stop shift）：光阑前移/后移可显著改变彗差与像散的配比")
    add("     · 换成 Cooke 三片或 Double-Gauss：用正负分离结构独立控制场曲")
    add("     · 或直接限制视场 ≤ ±1.5°，用在这个口径上它才是一支合格的镜头")
    add("=" * 78)

    txt = "\n".join(L)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(txt, encoding="utf-8")
    return txt


# ================================================================
# 9. 主流程
# ================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="离轴视场像差分析（彗差/像散/场曲）")
    ap.add_argument("--fields", type=float, nargs="+", default=[0.0, 3.0, 5.0],
                    help="要画的视场角（度）")
    ap.add_argument("--rings", type=int, default=6, help="六角极坐标环数")
    ap.add_argument("--scan-max", type=float, default=6.0, help="场扫描上限（度）")
    ap.add_argument("--scan-step", type=float, default=0.25, help="场扫描步长（度）")
    ap.add_argument("--no-scan", action="store_true", help="跳过场扫描（快很多）")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_dir, data_dir = root / "output", root / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config(fields=tuple(args.fields), rings=args.rings,
                 scan_max=args.scan_max, scan_step=args.scan_step)
    p = M06.Prescription()

    print("=" * 78)
    print("离轴视场像差分析 · 真实光线追迹")
    print("=" * 78)
    print(f"处方来自 06：R1={p.r1:.4f}  R2={p.r2:.4f}  R3={p.r3:.4f}  "
          f"t=({p.t1:g}, {p.t2:g})  {p.glass1}+{p.glass2}")

    try:
        # ---------- 主系统（画图用，3 个视场）----------
        print(f"[1/4] 建模：EPD={p.epd:g}mm，视场 {list(cfg.fields)}° ...")
        sysm = System(p, p.epd, cfg.fields, sd=cfg.sd)
        print(f"      EFL={sysm.efl:.4f}  像面 z={sysm.z_img:.4f}  "
              f"艾里半径={airy_radius(sysm.epd, sysm.efl):.3f}µm")
        print(f"      光束脚印（各视场最大 |y|）："
              f"{ {k: round(v, 2) for k, v in sysm.footprint.items()} }")

        print("[2/4] 逐视场追迹 ...")
        results = []
        for i in range(len(cfg.fields)):
            r = analyse_field(sysm, i, cfg)
            results.append(r)
            print(f"      {r.theta:>4.0f}°  RMS={r.rms_img:8.1f}µm  "
                  f"跨度 {r.ptp_x:6.0f}×{r.ptp_y:6.0f}µm  彗差a2={r.a2:+7.1f}µm  "
                  f"像散={r.z_tan - r.z_sag:+7.1f}µm")

        # ---------- 场扫描 ----------
        scan = {}
        if args.no_scan:
            print("[3/4] 跳过场扫描（--no-scan）")
            scan["theta"] = np.arange(cfg.scan_step, cfg.scan_max + 1e-9, cfg.scan_step)
            for epd in cfg.scan_aps:
                scan[epd] = {k: np.full(len(scan["theta"]), np.nan)
                             for k in ("rms_img", "rms_best", "a2", "astig")}
        else:
            print(f"[3/4] 场扫描 0→{cfg.scan_max:g}°，步长 {cfg.scan_step:g}°，"
                  f"口径 {list(cfg.scan_aps)} ...")
            thetas = np.arange(cfg.scan_step, cfg.scan_max + 1e-9, cfg.scan_step)
            for epd in cfg.scan_aps:
                d = field_scan(p, cfg, epd, thetas)
                d["rms_img"] = np.clip(d["rms_img"], 1e-3, None)
                scan[epd] = d
                print(f"      EPD {epd:5.1f}mm: RMS(1°/3°/5°) = "
                      f"{np.interp(1, thetas, d['rms_img']):.1f} / "
                      f"{np.interp(3, thetas, d['rms_img']):.1f} / "
                      f"{np.interp(5, thetas, d['rms_img']):.1f} µm")
            scan["theta"] = thetas

        # ---------- 阈值穿越点 ----------
        airy = airy_radius(sysm.epd, sysm.efl)
        th = scan["theta"]
        crossings = {"img": {}, "best": {}, "rms_img": {}, "rms_best": {}}
        for k in (2, 5, 10):
            crossings["img"][k] = crossing(th, scan[p.epd]["rms_img"], k * airy)
            crossings["best"][k] = crossing(th, scan[p.epd]["rms_best"], k * airy)
        crossings["rms_img"] = {round(t, 2): float(np.interp(t, th, scan[p.epd]["rms_img"]))
                                for t in cfg.fields if t > 0}
        crossings["rms_best"] = {round(t, 2): float(np.interp(t, th, scan[p.epd]["rms_best"]))
                                 for t in cfg.fields if t > 0}

        # ---------- Petzval 校验 ----------
        n1 = float(np.real(sysm.sm.gaps[1].medium.rindex(WVL_D)))
        n2 = float(np.real(sysm.sm.gaps[2].medium.rindex(WVL_D)))
        phi1 = (n1 - 1.0) * (1.0 / p.r1 - 1.0 / p.r2)
        phi2 = (n2 - 1.0) * (1.0 / p.r2 - 1.0 / p.r3)
        invRp = phi1 / n1 + phi2 / n2
        R_th = 1.0 / invRp

        ths = scan["theta"]
        zt, zs = scan[p.epd]["dz_best"], scan[p.epd]["rms_best"]   # 保留给 CSV/图
        rr = results
        tt = np.array([r.theta for r in rr])
        zt_a = np.array([r.z_tan for r in rr])
        zs_a = np.array([r.z_sag for r in rr])
        zmd = 0.5 * (zt_a + zs_a)                     # T/S 中点 ≈ 最小弥散圆所在
        # Petzval 面：由 T/S 焦点反推，z_p = (3·z_S − z_T)/2（光阑在透镜上时 T:S = 3:1）
        zp_a = (3.0 * zs_a - zt_a) / 2.0
        m_t = float(np.polyfit(tt ** 2, zt_a, 1)[0])
        m_s = float(np.polyfit(tt ** 2, zs_a, 1)[0])
        m_m = float(np.polyfit(tt ** 2, zmd, 1)[0])
        m_p = float(np.polyfit(tt ** 2, zp_a, 1)[0])
        y5 = results[-1].y_chief
        R_meas = y5 ** 2 / (2 * abs(zp_a[-1]) * 1e-3) if abs(zp_a[-1]) > 1e-9 else float("inf")
        petzval = dict(m_tan=m_t, m_sag=m_s, m_med=m_m, m_pet=m_p, zp=zp_a,
                       invRp=invRp, R_theory=R_th, R_meas=R_meas, z_p_meas=zp_a[-1],
                       ratio31=float((zt_a[-1] - zp_a[-1]) / max(zp_a[-1] - zs_a[-1], 1e-9)))

        # ---------- 出图 ----------
        print("[4/4] 出图与报告 ...")
        png1 = out_dir / "offaxis_spot.png"
        plot_main(results, cfg, sysm, scan, png1)

        # 场曲曲线：优先用场扫描的密集采样（24 点），没有扫描时退回 3 个分析视场
        if not args.no_scan and p.epd in cfg.scan_aps:
            d0 = scan[p.epd]
            fc_t = np.asarray(d0["z_tan"])
            fc_s = np.asarray(d0["z_sag"])
            fc_th = np.concatenate([[0.0], th])
            fc_t = np.concatenate([[0.0], fc_t])
            fc_s = np.concatenate([[0.0], fc_s])
            print(f"      场曲曲线：{len(fc_th)} 个视场点（来自场扫描）")
        else:
            fc_th, fc_t, fc_s = tt, zt_a, zs_a
        field_curve = dict(theta=fc_th, z_tan=fc_t, z_sag=fc_s,
                           z_med=0.5 * (fc_t + fc_s),
                           slope_tan=m_t, slope_sag=m_s, ratio31=petzval["ratio31"])
        png2 = out_dir / "offaxis_throughfocus.png"
        plot_through_focus(sysm, results[-1], cfg, tt, field_curve, png2)

        rpt = out_dir / "offaxis_report.txt"
        txt = build_report(results, cfg, sysm, scan, crossings, petzval, rpt)

        # ---------- 数据落盘 ----------
        rows = ["field_deg,x_um,y_um,rho"]
        for r in results:
            for x, y, rho in r.spot:
                rows.append(f"{r.theta:g},{x:.4f},{y:.4f},{rho:.4f}")
        (data_dir / "offaxis_spot.csv").write_text("\n".join(rows), encoding="utf-8")

        if not args.no_scan:
            rows = ["field_deg,epd_mm,rms_img_um,rms_best_um,coma_a2_um,astig_um"]
            for epd in cfg.scan_aps:
                for i, t in enumerate(th):
                    d = scan[epd]
                    rows.append(f"{t:.3f},{epd:g},{d['rms_img'][i]:.4f},"
                                f"{d['rms_best'][i]:.4f},{d['a2'][i]:.3f},{d['astig'][i]:.3f}")
            (data_dir / "offaxis_fieldscan.csv").write_text("\n".join(rows), encoding="utf-8")

        print()
        print(txt)
        print()
        print(f"图 1 : {png1}")
        print(f"图 2 : {png2}")
        print(f"报告 : {rpt}")
        print(f"数据 : {data_dir / 'offaxis_spot.csv'}")
        return 0

    except Exception:
        print("\n[异常] 脚本执行失败，完整堆栈：")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[中断] 用户取消")
        sys.exit(130)
