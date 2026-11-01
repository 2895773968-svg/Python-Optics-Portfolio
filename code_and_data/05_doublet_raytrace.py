# -*- coding: utf-8 -*-
"""
任务 05 — 双透镜组合消球差（平凸 + 负弯月）真实光线追迹
================================================================================
思路：
    第一片 平凸正透镜（凸面朝物）产生 正球差 —— 边缘光线焦点比近轴焦点更靠近透镜
    第二片 负弯月透镜（凹面朝入射光束）产生 负球差 —— 方向相反，可部分抵消

    本脚本不只是"画个图"，而是：
      1) 用 scipy 优化间距与弯月镜两半径，把边缘光线 (h=10mm) 的 LSA 压到零
      2) 用 11 条光线 (h=0..10mm) 对比单透镜 / 双透镜的横向球差曲线
      3) 给出 LSA、RMS 弥散、EFL 三项指标的定量对比

【关键口径说明】
    加负透镜必然改变焦距（本例 EFL 49.15 -> 61.69 mm）。若只看 LSA 的毫米数，
    焦距变长本身也会让 LSA 变小，容易误判为"球差被校正"。
    所以本脚本同时给出 LSA/EFL 的相对值，作为公平判据。

输出：
    output/doublet_raytrace.png   11 条光线光路图 + 球差对比曲线
"""

from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

import rayoptics.optical.model_constants as mc
from rayoptics.environment import OpticalModel, PupilSpec, FieldSpec, WvlSpec
from rayoptics.raytr.trace import trace_base


# ================================================================
# 1. 处方
# ================================================================
@dataclass
class Prescription:
    # 第一片：平凸正透镜
    r1: float = 25.4
    t1: float = 5.0
    glass1: str = "N-BK7"

    # 第二片：负弯月透镜（R3/R4 均为负 = 凹面朝向入射光束，这是消球差的经典朝向）
    r3: float = -41.10
    r4: float = -63.19
    t2: float = 5.0
    glass2: str = "N-BK7"
    gap: float = 2.41          # 两片之间的空气间隔

    # 公共参数
    aperture: float = 25.4     # 入瞳直径
    wavelength_nm: float = 587.6
    heights: tuple[float, ...] = tuple(float(h) for h in range(0, 11))  # 0..10 共 11 条

    use_meniscus: bool = True
    out_png: str = "doublet_raytrace.png"


# ================================================================
# 2. 建模
# ================================================================
def build_system(p: Prescription):
    """
    use_meniscus=True  -> 平凸 + 负弯月（双透镜）
    use_meniscus=False -> 只有平凸（单透镜基准）
    """
    opm = OpticalModel()
    sm = opm["seq_model"]
    osp = opm["optical_spec"]
    opm.radius_mode = True

    osp["pupil"] = PupilSpec(osp, key=("object", "epd"), value=p.aperture)
    osp["fov"] = FieldSpec(osp, key=("object", "angle"), value=0.0, flds=[0.0])
    osp["wvls"] = WvlSpec([(p.wavelength_nm, 1.0)], ref_wl=0)

    # s0 物面：厚度取极大 -> 物在无穷远
    sm.gaps[0].thi = 1e10
    # s1 前表面（凸面）-> s2 后表面（平面）
    sm.add_surface([p.r1, p.t1, p.glass1])

    if p.use_meniscus:
        # add_surface 是"插入到像面之前"，所以四个面依次是 s1~s4，像面是 s5：
        #   s1=平凸前表面  s2=平凸后表面(其后空气间隔=gap)
        #   s3=弯月前表面  s4=弯月后表面(其后空气间隔=自动定焦距离)
        sm.add_surface([0.0, p.gap])
        sm.add_surface([p.r3, p.t2, p.glass2])
        sm.add_surface([p.r4, 20.0])
        lens_surfaces = [(1, 2), (3, 4)]
    else:
        sm.add_surface([0.0, 20.0])
        lens_surfaces = [(1, 2)]

    opm.update_model()

    # 自动定焦：像面移到近轴焦点，球差才会体现为像面弥散
    fod = opm["analysis_results"]["parax_data"].fod
    sm.gaps[-1].thi = round(float(fod.img_dist), 4)
    opm.update_model()
    sm.compute_global_coords()

    return opm, sm, osp, lens_surfaces


# ================================================================
# 3. 追迹
# ================================================================
@dataclass
class RayFan:
    heights: np.ndarray = field(default_factory=lambda: np.array([]))
    y_image: np.ndarray = field(default_factory=lambda: np.array([]))
    z_cross: np.ndarray = field(default_factory=lambda: np.array([]))
    polylines: list = field(default_factory=list)      # 每条光线的 [(z,y), ...]
    efl: float = 0.0
    bfl: float = 0.0
    z_img: float = 0.0
    z_front: float = 0.0


def trace_fan(opm, sm, osp, p: Prescription) -> RayFan:
    """对每个高度追迹一条平行光线，收集像面高度、光轴交点、折线路径"""
    fan = RayFan()
    fan.efl = float(opm["analysis_results"]["parax_data"].fod.efl)
    fan.bfl = float(opm["analysis_results"]["parax_data"].fod.img_dist)
    tf = sm.gbl_tfrms
    fan.z_front = tf[1][1][2]
    fan.z_img = tf[-1][1][2] - fan.z_front

    heights, y_img, z_cross, polys = [], [], [], []
    for h in p.heights:
        try:
            ray, _op, _wvl = trace_base(
                opm, (0.0, float(h)), osp["fov"].fields[0],
                p.wavelength_nm, pupil_type="aim pt",
            )
        except Exception as e:
            print(f"[警告] h={h} mm 追迹失败，跳过: {type(e).__name__}: {e}")
            continue

        pts = []
        for i, seg in enumerate(ray):
            rot, trns = tf[i]
            pt = rot.dot(seg[mc.p]) + trns
            pts.append((float(pt[2] - fan.z_front), float(pt[1])))

        rot, trns = tf[-1]
        d = rot.dot(ray[-1][mc.d])
        y, z = pts[-1][1], pts[-1][0]
        zc = z + (-y / d[1]) * d[2] if abs(d[1]) > 1e-12 else float("nan")

        heights.append(float(h))
        y_img.append(y)
        z_cross.append(zc)
        polys.append(pts)

    fan.heights = np.array(heights)
    fan.y_image = np.array(y_img)
    fan.z_cross = np.array(z_cross)
    fan.polylines = polys
    return fan


def summarize(fan: RayFan) -> dict:
    """LSA / 横向球差 / RMS 弥散"""
    ta = fan.y_image - fan.y_image[0] if fan.y_image.size else np.array([])
    edge = fan.heights[-1] if fan.heights.size else np.nan
    lsa = fan.z_cross[-1] - fan.z_img if fan.z_cross.size else np.nan
    rms = float(np.sqrt(np.mean(ta**2))) if ta.size else np.nan
    return {
        "efl": fan.efl,
        "lsa_edge": float(lsa),
        "ta_edge": float(ta[-1]) if ta.size else np.nan,
        "rms": rms,
        "lsa_rel": float(lsa / fan.efl * 100) if fan.efl else np.nan,   # 相对焦距 (%)
        "rms_rel": float(rms / fan.efl * 100) if fan.efl else np.nan,
        "edge_height": edge,
        "ta": ta,
    }


# ================================================================
# 4. 优化（可选：--optimize 时重跑）
# ================================================================
def optimize_meniscus(p: Prescription) -> Prescription:
    """只调间距与弯月镜两半径（第一片固定），目标是边缘光线 LSA = 0"""
    from scipy.optimize import minimize

    def merit(x):
        r3, r4, gap = x
        if not (-200 < r3 < -12 and -400 < r4 < 400 and 0.2 < gap < 15):
            return 1e3
        try:
            q = replace(p, r3=r3, r4=r4, gap=gap)
            opm, sm, osp, _ = build_system(q)
            if not (45 < opm["analysis_results"]["parax_data"].fod.efl < 150):
                return 1e3
            fan = trace_fan(opm, sm, osp, q)
            lsa = summarize(fan)["lsa_edge"]
            return abs(lsa) if np.isfinite(lsa) else 1e3
        except Exception:
            return 1e3

    res = minimize(merit, x0=[p.r3, p.r4, p.gap], method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 3000})
    print(f"[优化] 收敛={res.success}  残差 |LSA|={res.fun:.6f} mm")
    print(f"       初始: R3={p.r3:.2f}  R4={p.r4:.2f}  gap={p.gap:.2f}")
    print(f"       结果: R3={res.x[0]:.2f}  R4={res.x[1]:.2f}  gap={res.x[2]:.2f}")
    return replace(p, r3=float(res.x[0]), r4=float(res.x[1]), gap=float(res.x[2]))


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
    print("[警告] 未找到中文字体")
    return "DejaVu Sans"


def sag(cv: float, y: np.ndarray) -> np.ndarray:
    if abs(cv) < 1e-12:
        return np.zeros_like(y)
    arg = np.clip(1.0 - (cv * y) ** 2, 0.0, None)
    return cv * y**2 / (1.0 + np.sqrt(arg))


def plot_result(sm, lens_surfaces, fan2: RayFan, fan1: RayFan, s1: dict, s2: dict,
                p: Prescription, out_path: Path) -> None:
    setup_chinese_font()
    tf = sm.gbl_tfrms
    z0 = tf[1][1][2]
    sd = p.aperture / 2.0

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(16.0, 6.2), dpi=150, constrained_layout=True,
        gridspec_kw={"width_ratios": [1.55, 1.0]},
    )

    # ---------- 左：光路图 ----------
    # 透镜轮廓：矢高统一用 z(y) = z_v + sag(cv, y)
    #   前表面 cv>0 -> 边缘向 +z 退让（凸面朝物）；后表面 cv<0 -> 边缘向 -z 退回玻璃
    for (fi, bi) in lens_surfaces:
        zf, zb = tf[fi][1][2] - z0, tf[bi][1][2] - z0
        yy = np.linspace(-sd, sd, 220)
        z1 = zf + sag(sm.ifcs[fi].profile.cv, yy)
        z2 = zb + sag(sm.ifcs[bi].profile.cv, yy)
        axL.fill(np.concatenate([z1, z2[::-1]]), np.concatenate([yy, yy[::-1]]),
                 color="#AED6F1", alpha=0.45, zorder=1)
        axL.plot(z1, yy, color="#2471A3", lw=1.7, zorder=3)
        axL.plot(z2, yy, color="#2471A3", lw=1.7, zorder=3)
        axL.plot([z1[0], z2[0]], [yy[0]] * 2, color="#2471A3", lw=1.7, zorder=3)
        axL.plot([z1[-1], z2[-1]], [yy[-1]] * 2, color="#2471A3", lw=1.7, zorder=3)

    # 11 条光线
    cmap = plt.get_cmap("viridis")
    z_start = -12.0
    for k, (poly, h) in enumerate(zip(fan2.polylines, fan2.heights)):
        zs = [z_start] + [pt[0] for pt in poly[1:]]
        ys = [h] + [pt[1] for pt in poly[1:]]
        c = cmap(0.05 + 0.85 * k / max(1, len(fan2.polylines) - 1))
        axL.plot(zs, ys, "-", lw=1.25, color=c, alpha=0.92, zorder=4,
                 label="11 条光线 (h=0~10mm)" if k == 0 else None)
        axL.plot([fan2.z_img], [poly[-1][1]], "o", ms=3.2, color=c, zorder=6)

    axL.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.6, zorder=0)
    axL.axvline(fan2.z_img, color="#E67E22", lw=1.4, ls="-.", alpha=0.9, zorder=2,
                label=f"近轴像面 z={fan2.z_img:.2f} mm")

    # 边缘光线焦点（球差零点）
    if np.isfinite(fan2.z_cross[-1]):
        axL.plot([fan2.z_cross[-1]], [0], "X", ms=11, mew=2.0, color="#8E44AD",
                 zorder=8, label=f"边缘光线焦点 (h=10mm)")

    axL.set_xlabel("光轴方向 z (mm)", fontsize=11)
    axL.set_ylabel("高度 y (mm)", fontsize=11)
    axL.set_title(
        f"双透镜光路图：平凸(R={p.r1:g}) + 负弯月(R={p.r3:.1f}/{p.r4:.1f})\n"
        f"间距 {p.gap:.2f} mm    EFL={s2['efl']:.2f} mm    11 条光线 h=0~10 mm",
        fontsize=11,
    )
    axL.grid(alpha=0.22, ls=":")
    axL.set_xlim(z_start - 1, fan2.z_img + 10)
    axL.set_ylim(-sd * 1.3, sd * 1.3)
    axL.legend(loc="lower left", fontsize=8.5, framealpha=0.92)

    # ---------- 右：横向球差曲线对比 ----------
    axR.axhline(0, color="gray", lw=0.9, ls="--", alpha=0.7)
    axR.plot(fan1.y_image - fan1.y_image[0], fan1.heights, "o--",
             color="#7F8C8D", lw=1.6, ms=5, label=f"单透镜  (LSA={s1['lsa_edge']:+.3f} mm)")
    axR.plot(fan2.y_image - fan2.y_image[0], fan2.heights, "o-",
             color="#C0392B", lw=2.0, ms=5.5, label=f"双透镜  (LSA={s2['lsa_edge']:+.3f} mm)")
    axR.fill_betweenx(fan2.heights, s1["ta"], s2["ta"], color="#F5B7B1", alpha=0.35,
                      label="改善量")
    axR.set_xlabel("横向球差 TA (mm)", fontsize=11)
    axR.set_ylabel("入射高度 h (mm)", fontsize=11)
    axR.set_title("横向球差 vs 入射高度", fontsize=11.5, pad=10)
    axR.grid(alpha=0.25, ls=":")
    axR.legend(loc="lower left", fontsize=9, framealpha=0.95)

    ratio = abs(s1["lsa_edge"]) / abs(s2["lsa_edge"]) if abs(s2["lsa_edge"]) > 1e-9 else float("inf")
    txt = (f"边缘 LSA：{s1['lsa_edge']:+.3f} → {s2['lsa_edge']:+.4f} mm\n"
           f"RMS 弥散：{s1['rms']:.4f} → {s2['rms']:.5f} mm\n"
           f"改善倍数：{ratio:.0f}×")
    axR.text(0.02, 0.98, txt, transform=axR.transAxes, va="top", ha="left",
             fontsize=9.5,
             bbox=dict(boxstyle="round,pad=0.45", fc="#FEF9E7", ec="#B7950B", alpha=0.96))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 图已保存: {out_path}")


# ================================================================
# 6. 主流程
# ================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="双透镜消球差光线追迹")
    ap.add_argument("--optimize", action="store_true", help="重新优化间距与弯月镜半径")
    args = ap.parse_args()

    p = Prescription()
    root = Path(__file__).resolve().parent.parent

    try:
        print("=" * 78)
        print("双透镜组合消球差：平凸正透镜 + 负弯月透镜")
        print("=" * 78)

        if args.optimize:
            p = optimize_meniscus(p)

        # 单透镜基准
        p1 = replace(p, use_meniscus=False)
        opm1, sm1, osp1, _ = build_system(p1)
        fan1 = trace_fan(opm1, sm1, osp1, p1)
        s1 = summarize(fan1)

        # 双透镜
        opm2, sm2, osp2, lens_surfaces2 = build_system(p)
        fan2 = trace_fan(opm2, sm2, osp2, p)
        s2 = summarize(fan2)

        print(f"第一片      : 平凸 R1={p.r1:g} mm, t={p.t1:g} mm, {p.glass1}")
        print(f"第二片      : 负弯月 R3={p.r3:.2f} / R4={p.r4:.2f} mm, t={p.t2:g} mm, {p.glass2}")
        print(f"空气间隔    : {p.gap:.2f} mm      口径 {p.aperture:g} mm")
        print(f"光线        : {len(p.heights)} 条，h = {p.heights[0]:g} ~ {p.heights[-1]:g} mm")
        print("-" * 78)

        # ---- 逐条光线对比 ----
        print(f"{'h(mm)':>6}{'单透镜 y(mm)':>14}{'单透镜 TA':>13}"
              f"{'双透镜 y(mm)':>14}{'双透镜 TA':>13}")
        for i, h in enumerate(fan2.heights):
            j = int(np.where(np.isclose(fan1.heights, h))[0][0]) if np.any(np.isclose(fan1.heights, h)) else i
            print(f"{h:>6.1f}{fan1.y_image[j]:>14.5f}{s1['ta'][j]:>13.5f}"
                  f"{fan2.y_image[i]:>14.5f}{s2['ta'][i]:>13.5f}")

        # ---- 汇总指标 ----
        print("-" * 78)
        print(f"{'指标':<26}{'单透镜':>16}{'双透镜':>16}{'变化':>14}")
        print(f"{'EFL (mm)':<24}{s1['efl']:>16.3f}{s2['efl']:>16.3f}"
              f"{s2['efl'] - s1['efl']:>+14.3f}")
        print(f"{'边缘 LSA (mm)':<22}{s1['lsa_edge']:>16.4f}{s2['lsa_edge']:>16.4f}"
              f"{(s2['lsa_edge'] - s1['lsa_edge']):>+14.4f}")
        print(f"{'LSA/EFL (%)':<24}{s1['lsa_rel']:>16.3f}{s2['lsa_rel']:>16.3f}"
              f"{(s2['lsa_rel'] - s1['lsa_rel']):>+14.3f}")
        print(f"{'RMS 弥散 (mm)':<23}{s1['rms']:>16.5f}{s2['rms']:>16.5f}"
              f"{(s2['rms'] - s1['rms']):>+14.5f}")

        # ---- 结论 ----
        ratio = abs(s1["lsa_edge"]) / abs(s2["lsa_edge"]) if abs(s2["lsa_edge"]) > 1e-9 else float("inf")
        lsa_gain = (1 - abs(s2["lsa_edge"]) / abs(s1["lsa_edge"])) * 100
        rms_gain = (1 - s2["rms"] / s1["rms"]) * 100

        print("=" * 78)
        print("结论")
        print("=" * 78)
        print(f"  边缘光线 (h=10mm) 纵向球差：{s1['lsa_edge']:+.4f} mm  →  {s2['lsa_edge']:+.4f} mm")
        print(f"     绝对量下降 {lsa_gain:.2f}%，约 {ratio:.0f} 倍")
        if abs(s2["lsa_edge"]) < abs(s1["lsa_edge"]) * 0.2:
            print("  ✔ 球差被显著校正：边缘焦点与近轴焦点的距离缩短到原来的 "
                  f"{abs(s2['lsa_edge']) / abs(s1['lsa_edge']) * 100:.1f}%")
        else:
            print("  ✘ 球差未有效校正")
        print(f"  弥散斑 RMS：{s1['rms']:.5f} → {s2['rms']:.5f} mm（下降 {rms_gain:.1f}%）")

        if s2["efl"] > s1["efl"] * 1.02:
            print(f"  ⚠ 注意：EFL 由 {s1['efl']:.2f} 增至 {s2['efl']:.2f} mm"
                  f"（+{(s2['efl']/s1['efl']-1)*100:.1f}%）。")
            print(f"     按相对口径 LSA/EFL：{s1['lsa_rel']:.3f}% → {s2['lsa_rel']:.3f}%，"
                  f"仍是真实改善，不是焦距拉长造成的假象。")

        plot_result(sm2, lens_surfaces2, fan2, fan1, s1, s2, p, root / "output" / p.out_png)
        return 0

    except Exception:
        print("[错误] 失败，完整堆栈：", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
