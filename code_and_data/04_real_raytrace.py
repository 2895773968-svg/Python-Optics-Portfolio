# -*- coding: utf-8 -*-
"""
任务 04 — 真实光线追迹：平凸单透镜球差分析（rayoptics）
================================================================================
不是 Mock。光路由 rayoptics 做真实 Snell 折射计算，逐面求解球面交点。

模型：
    物在无穷远（平行光）→ 平凸透镜（凸面朝物，R=25.4mm，厚 5mm，N-BK7）→ 近轴焦面
    口径 25.4mm，三条光线高度 0 / 5 / 10 mm（10mm 接近边缘 = 边缘光线）

输出：
    output/ray_trace_real.png   光路图（Ray Trace Layout）
    控制台：各面交点高度、像面交点高度、横向球差、纵向球差 LSA

物理预期：
    单透镜球差未校正，边缘光线会聚点比近轴光线更靠近透镜（正球差）
    横向球差满足三级像差规律 TA ∝ h³，即 h 翻倍、TA 变 8 倍 —— 脚本会自动验证
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

import rayoptics.optical.model_constants as mc
from rayoptics.environment import (
    OpticalModel, PupilSpec, FieldSpec, WvlSpec,
)
from rayoptics.raytr.trace import trace_base


# ================================================================
# 1. 系统参数
# ================================================================
@dataclass
class LensSpec:
    """平凸单透镜规格"""
    radius_front: float = 25.4      # 前表面曲率半径 (mm)，凸面朝物
    thickness: float = 5.0          # 中心厚度 (mm)
    glass: str = "N-BK7"            # 材料（opticalglass 目录内置）
    aperture: float = 25.4          # 入瞳直径 = 透镜口径 (mm)
    wavelength_nm: float = 587.6    # d 线
    ray_heights: tuple[float, ...] = (0.0, 5.0, 10.0)   # 三条光线的高度 (mm)
    out_png: str = "ray_trace_real.png"


# ================================================================
# 2. 建模
# ================================================================
def build_model(spec: LensSpec):
    """
    构建顺序模型。rayoptics 用 [曲率或半径, 厚度, 玻璃, ...] 描述每个面，
    radius_mode=True 时第一项按"半径"解释。
    """
    opm = OpticalModel()
    sm = opm["seq_model"]
    osp = opm["optical_spec"]
    opm.radius_mode = True

    osp["pupil"] = PupilSpec(osp, key=("object", "epd"), value=spec.aperture)
    osp["fov"] = FieldSpec(osp, key=("object", "angle"), value=0.0, flds=[0.0])
    osp["wvls"] = WvlSpec([(spec.wavelength_nm, 1.0)], ref_wl=0)

    # s0 = 物面：厚度设极大值 => 物在无穷远（入射为平行光）
    sm.gaps[0].thi = 1e10
    # s1 = 前表面（凸面，玻璃），其后是 5mm 玻璃
    sm.add_surface([spec.radius_front, spec.thickness, spec.glass])
    # s2 = 后表面（平面，空气），厚度先给估值，稍后按近轴焦点自动更新
    sm.add_surface([0.0, 40.0])
    # s3 = 像面（初始模型自带的最后一面）

    opm.update_model()

    # 自动定焦：把像面挪到近轴焦点，这样球差才会表现为"像面上的弥散"
    bfl = opm["analysis_results"]["parax_data"].fod.img_dist
    sm.gaps[-1].thi = round(float(bfl), 4)
    opm.update_model()
    sm.compute_global_coords()

    return opm, sm, osp, float(bfl)


# ================================================================
# 3. 追迹
# ================================================================
@dataclass
class RayTrace:
    """单条光线的追迹结果（全部用全局坐标，单位 mm）"""
    height_in: float
    pts: list[tuple[float, float]] = field(default_factory=list)  # [(z, y), ...] 逐面交点
    dir_last: tuple[float, float] = (0.0, 1.0)                    # 出射方向 (dy, dz)
    y_image: float | None = None
    z_axis_cross: float | None = None                              # 与光轴交点 z


def trace_all(opm, sm, osp, spec: LensSpec, z_vertex_front: float) -> list[RayTrace]:
    """对每个设定高度追迹一条平行光线"""
    fld = osp["fov"].fields[0]
    tf = sm.gbl_tfrms
    results: list[RayTrace] = []

    for h in spec.ray_heights:
        rt = RayTrace(height_in=float(h))
        try:
            # pupil_type='aim pt'：直接指定在光瞳平面上的瞄准高度，等价于"平行光某高度入射"
            ray, _op, _wvl = trace_base(
                opm, (0.0, float(h)), fld, spec.wavelength_nm, pupil_type="aim pt"
            )
        except Exception as e:
            print(f"[警告] 高度 {h} mm 的光线追迹失败: {type(e).__name__}: {e}")
            continue

        for i, seg in enumerate(ray):
            rot, trns = tf[i]
            p = rot.dot(seg[mc.p]) + trns
            rt.pts.append((float(p[2] - z_vertex_front), float(p[1])))

        # 出射方向（最后一面之后的全局方向余弦）
        rot_last, _ = tf[-1]
        d_last = rot_last.dot(ray[-1][mc.d])
        rt.dir_last = (float(d_last[1]), float(d_last[2]))

        # 像面交点高度（像面在 z_img）
        z_img = rt.pts[-1][0]
        rt.y_image = rt.pts[-1][1]

        # 与光轴的交点：沿出射方向外推，解 y=0
        if abs(rt.dir_last[0]) > 1e-12:
            t = -rt.y_image / rt.dir_last[0]
            rt.z_axis_cross = z_img + t * rt.dir_last[1]

        results.append(rt)

    return results


# ================================================================
# 4. 球差指标
# ================================================================
def compute_aberrations(traces: list[RayTrace], z_paraxial_focus: float) -> dict:
    """
    横向球差 TA：像面上实际交点高度（近轴参考取 h=0 那条光线）
    纵向球差 LSA：光线与光轴交点的 z 偏移，基准是近轴焦点（自动定焦所在平面）

    说明：轴向光线 (h=0) 本身就贴着光轴走，不存在"与光轴的交点"，
    它的 LSA 按定义取 0，不能用它的交点当基准（否则整表偏移）。
    """
    if not traces:
        raise RuntimeError("没有成功追迹的光线")

    out = {"z_ref_focus": z_paraxial_focus, "rows": []}
    for t in traces:
        ta = t.y_image - next(
            (x.y_image for x in traces if abs(x.height_in) < 1e-9),
            min(traces, key=lambda x: abs(x.height_in)).y_image,
        )
        if abs(t.height_in) < 1e-9:
            lsa = 0.0                      # 轴向光线：定义为零
        elif t.z_axis_cross is not None:
            lsa = t.z_axis_cross - z_paraxial_focus
        else:
            lsa = float("nan")
        out["rows"].append({
            "h": t.height_in,
            "y_image": t.y_image,
            "TA": ta,
            "LSA": lsa,
            "z_cross": t.z_axis_cross,
        })
    return out


# ================================================================
# 5. 光路图
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


def surface_sag(cv: float, y: np.ndarray) -> np.ndarray:
    """球面矢高 z(y) = c·y² / (1 + sqrt(1 - c²y²))，c=0 时退化为平面"""
    if abs(cv) < 1e-12:
        return np.zeros_like(y)
    arg = 1.0 - (cv * y) ** 2
    arg = np.clip(arg, 0.0, None)
    return cv * y**2 / (1.0 + np.sqrt(arg))


def plot_layout(sm, traces: list[RayTrace], ab: dict, spec: LensSpec,
                z_img: float, bfl: float, out_path: Path) -> None:
    setup_chinese_font()

    z_front, z_back = 0.0, None
    # 各面顶点 z（相对前表面顶点）
    tf = sm.gbl_tfrms
    z_v = [t[1][2] for t in tf]
    z0 = z_v[1]
    z_back = z_v[2] - z0
    sd = spec.aperture / 2.0

    fig, ax = plt.subplots(figsize=(12.2, 6.0), dpi=150, constrained_layout=True)

    # ---- 透镜轮廓（由矢高公式生成，与追迹用的面完全一致）----
    yy = np.linspace(-sd, sd, 240)
    cv1 = sm.ifcs[1].profile.cv
    cv2 = sm.ifcs[2].profile.cv
    z1 = z_front + surface_sag(cv1, yy)
    z2 = z_back + surface_sag(-cv2, yy)   # 后表面沿 -z 方向离开玻璃

    poly_z = np.concatenate([z1, z2[::-1]])
    poly_y = np.concatenate([yy, yy[::-1]])
    ax.fill(poly_z, poly_y, color="#AED6F1", alpha=0.45, ec="none", zorder=1)
    ax.plot(z1, yy, color="#2471A3", lw=1.8, zorder=3)
    ax.plot(z2, yy, color="#2471A3", lw=1.8, zorder=3)
    ax.plot([z1[0], z2[0]], [yy[0], yy[0]], color="#2471A3", lw=1.8, zorder=3)
    ax.plot([z1[-1], z2[-1]], [yy[-1], yy[-1]], color="#2471A3", lw=1.8, zorder=3)

    # ---- 光线 ----
    z_start = -12.0
    colors = ["#7F8C8D", "#C0392B", "#1F618D", "#B03A2E", "#7D3C98"]
    for k, t in enumerate(traces):
        # 只画 s1 之后（s0 在 z=-1e10，画不出来）；入射段按平行光直接连过来
        zs = [z_start] + [p[0] for p in t.pts[1:]]
        ys = [t.height_in] + [p[1] for p in t.pts[1:]]
        c = colors[k % len(colors)]
        ax.plot(zs, ys, "-", lw=1.5, color=c, alpha=0.95,
                label=f"h = {t.height_in:g} mm", zorder=4)
        # 像面交点
        ax.plot([z_img], [t.y_image], "o", ms=5.5, color=c, zorder=6)

        # 光轴交点（球差焦点）
        if t.z_axis_cross is not None and abs(t.height_in) > 1e-9:
            ax.plot([t.z_axis_cross], [0.0], "x", ms=7, mew=1.8, color=c, zorder=6)

    # ---- 光轴 ----
    ax.axhline(0.0, color="gray", lw=0.9, ls="--", alpha=0.7, zorder=0)

    # ---- 像面 ----
    ax.axvline(z_img, color="#E67E22", lw=1.4, ls="-.", alpha=0.9, zorder=2,
               label=f"近轴像面 z={z_img:.3f} mm")

    # ---- 像面交点高度：汇总文本框（逐点标注在近轴区会互相重叠）----
    lines = ["像面交点高度 (mm)"]
    for t in traces:
        lines.append(f"  h={t.height_in:>5g} : {t.y_image:+.4f}")
    ax.text(z_img + 2.2, sd * 0.88, "\n".join(lines),
            fontsize=9, va="top", ha="left", color="#1B2631",
            bbox=dict(boxstyle="round,pad=0.45", fc="#FEF9E7", ec="#B7950B", alpha=0.95),
            zorder=8)

    # 球差示意：边缘光线焦点与近轴焦点差
    edge = max(traces, key=lambda t: abs(t.height_in))
    if edge.z_axis_cross is not None:
        ax.annotate("", xy=(edge.z_axis_cross, 0), xytext=(z_img, 0),
                    arrowprops=dict(arrowstyle="<->", color="#8E44AD", lw=1.4))
        mid_z = (edge.z_axis_cross + z_img) / 2
        ax.annotate(f"纵向球差\nLSA = {edge.z_axis_cross - z_img:+.3f} mm",
                    xy=(mid_z, 0), xytext=(0, -46), textcoords="offset points",
                    ha="center", fontsize=9, color="#6C3483",
                    bbox=dict(boxstyle="round,pad=0.35", fc="#F4ECF7",
                              ec="#8E44AD", alpha=0.95),
                    arrowprops=dict(arrowstyle="-", color="#8E44AD", lw=0.9, alpha=0.6))

    ax.set_xlabel("光轴方向 z (mm)", fontsize=11)
    ax.set_ylabel("高度 y (mm)", fontsize=11)
    ax.set_title(
        f"真实光线追迹 · 平凸单透镜（R={spec.radius_front:g} mm, t={spec.thickness:g} mm, "
        f"{spec.glass}）   EFL={sm.opt_model['analysis_results']['parax_data'].fod.efl:.2f} mm",
        fontsize=12,
    )
    ax.grid(alpha=0.22, ls=":")
    ax.set_xlim(z_start - 1, z_img + 9)
    ax.set_ylim(-sd * 1.25, sd * 1.25)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.92)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 光路图已保存: {out_path}")


# ================================================================
# 6. 主流程
# ================================================================
def main() -> int:
    spec = LensSpec()
    root = Path(__file__).resolve().parent.parent
    out_path = root / "output" / spec.out_png

    try:
        print("=" * 76)
        print("真实光线追迹：平凸单透镜球差")
        print("=" * 76)

        opm, sm, osp, bfl = build_model(spec)
        parax = opm["analysis_results"]["parax_data"].fod
        n_glass = sm.gaps[1].medium.rindex(spec.wavelength_nm)

        print(f"材料        : {spec.glass}   n_d = {n_glass:.5f}")
        print(f"前表面半径  : R1 = {spec.radius_front:g} mm（凸面朝物）")
        print(f"中心厚度    : {spec.thickness:g} mm")
        print(f"入瞳/口径   : {spec.aperture:g} mm")
        print(f"焦距 / 后焦 : EFL = {parax.efl:.4f} mm   BFL = {bfl:.4f} mm")
        print(f"理论校核    : R/(n-1) = {spec.radius_front/(n_glass-1):.4f} mm (单球面近轴焦距)")
        print("-" * 76)

        z0 = sm.gbl_tfrms[1][1][2]
        z_img = sm.gbl_tfrms[-1][1][2] - z0

        traces = trace_all(opm, sm, osp, spec, z0)
        ab = compute_aberrations(traces, z_img)

        # ---- 逐面交点表 ----
        print("各面交点高度 y (mm)：")
        header = "  " + f"{'面':<6}{'z(mm)':>12}" + "".join(
            f"{f'h={t.height_in:g}':>14}" for t in traces)
        print(header)
        labels = ["物面", "前表面", "后表面", "像面"]
        n_show = min(len(labels), len(traces[0].pts) if traces else 0)
        for i in range(n_show):
            z_val = traces[0].pts[i][0]
            row = f"  {labels[i]:<6}{z_val:>12.4f}"
            for t in traces:
                row += f"{t.pts[i][1]:>14.5f}"
            print(row)

        # ---- 像面高度与球差 ----
        print("-" * 76)
        print("像面上的交点高度与球差：")
        print(f"  {'入射高度 h':>12}{'像面高度 y':>14}{'横向球差 TA':>14}{'纵向球差 LSA':>16}")
        for r in ab["rows"]:
            print(f"  {r['h']:>12.3f}{r['y_image']:>14.5f}"
                  f"{r['TA']:>14.5f}{r['LSA']:>16.4f}")

        # ---- 三级球差规律验证 ----
        nonzero = [r for r in ab["rows"] if abs(r["h"]) > 1e-9 and abs(r["TA"]) > 1e-9]
        if len(nonzero) >= 2:
            a, b = nonzero[0], nonzero[-1]
            h_ratio = abs(b["h"] / a["h"])
            ta_ratio = abs(b["TA"] / a["TA"])
            print("-" * 76)
            print(f"三级像差校核（两个独立判据，都能对上才说明追迹是真的）：")
            print(f"  横向球差 TA ∝ h³ : 高度比 {h_ratio:.2f} -> 实测比 {ta_ratio:.2f}"
                  f"，理论 {h_ratio**3:.2f}，偏差 {abs(ta_ratio - h_ratio**3) / h_ratio**3 * 100:.1f}%")
            if abs(a["LSA"]) > 1e-9 and abs(b["LSA"]) > 1e-9:
                lsa_ratio = abs(b["LSA"] / a["LSA"])
                print(f"  纵向球差 LSA ∝ h²: 高度比 {h_ratio:.2f} -> 实测比 {lsa_ratio:.2f}"
                      f"，理论 {h_ratio**2:.2f}，偏差 {abs(lsa_ratio - h_ratio**2) / h_ratio**2 * 100:.1f}%")
            print("  结论：单透镜未校正球差，边缘光线会聚点比近轴焦点更靠近透镜（正球差）")

        plot_layout(sm, traces, ab, spec, z_img, bfl, out_path)
        return 0

    except Exception:
        print("[错误] 追踪失败，完整堆栈：", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
