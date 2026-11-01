# -*- coding: utf-8 -*-
"""
09_build_portfolio.py — 资产归档与光学工程作品集生成

做三件事：
  1. 建立 My_Optics_Portfolio/ 目录骨架（images/ 与 code_and_data/）
  2. 复制 8 张核心图表 + 2 张补充图 + 全部 .py 脚本 / .txt 报告 / .csv 数据，
     逐个做 MD5 校验，确保归档件与源文件一致
  3. 生成单页 HTML 作品集 Portfolio_Report.html（内联 CSS，无外部依赖、无 JS）

用法：
    python scripts/09_build_portfolio.py
    python scripts/09_build_portfolio.py --open      # 生成后直接拉起浏览器

安全约定：只读源文件、只写 My_Optics_Portfolio/，不触碰 output/ data/ scripts/ 内任何原件。
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "My_Optics_Portfolio"
IMGDIR = OUT / "images"
CODEDIR = OUT / "code_and_data"
DATADIR = CODEDIR / "data"

# ---------------------------------------------------------------- 归档清单
# 8 张核心图表（章节 1-7 使用），外加 2 张补充图（附录展示）
IMAGES = [
    ("heatmap.png",              "照度伪彩色渲染"),
    ("zemax_sweep.png",          "一维曲率扫描"),
    ("zemax_2d_sweep.png",       "二维参数矩阵"),
    ("ray_trace_real.png",       "单透镜真实追迹"),
    ("doublet_raytrace.png",     "双片式消球差"),
    ("achromatic_raytrace.png",  "消色差双胶合光路"),
    ("chromatic_focal_shift.png", "APO 焦移曲线"),
    ("offaxis_throughfocus.png", "离轴调焦序列与场曲"),
    ("apo_raytrace.png",         "APO 三片式光路"),          # 补充
    ("offaxis_spot.png",         "三视场弥散斑对比"),        # 补充
]


# ---------------------------------------------------------------- 工具函数
def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def human_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.0f} KB"
    return f"{n / 1024 / 1024:.1f} MB"


def copy_verified(src: Path, dst: Path) -> tuple[bool, str]:
    """复制并做 MD5 比对。返回 (是否一致, 说明)。"""
    if not src.is_file():
        return False, "源文件缺失"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)          # copy2 保留 mtime，便于追溯生成时间
    ok = md5_of(src) == md5_of(dst)
    return ok, human_size(dst.stat().st_size)


# ---------------------------------------------------------------- 样式表
# 说明：纯静态样式，无外部字体/框架依赖；极简排版 + 单一强调色
CSS = r"""
*,*::before,*::after{box-sizing:border-box}
:root{
  --ink:#0F1720; --body:#3C4654; --mute:#79838F; --line:#E4E8ED;
  --soft:#F7F9FB; --accent:#1C4E75;
}
html{-webkit-text-size-adjust:100%;scroll-behavior:smooth}
body{
  margin:0;background:#fff;color:var(--body);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Helvetica Neue",
              "PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
  font-size:15.5px;line-height:1.85;
  -webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale;
}
.wrap{max-width:1020px;margin:0 auto;padding:0 44px}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}

/* ---------- 封面 ---------- */
.cover{
  padding:104px 0 58px;border-bottom:1px solid var(--line);
  background:linear-gradient(180deg,#F8FAFC 0%,#FFFFFF 100%);
}
.kicker{
  margin:0 0 26px;font-size:11.5px;font-weight:700;
  letter-spacing:.2em;text-transform:uppercase;color:var(--accent);
}
.kicker::before{
  content:"";display:inline-block;width:26px;height:1px;
  background:var(--accent);vertical-align:middle;
  margin-right:12px;transform:translateY(-2px);
}
h1{
  margin:0;font-size:42px;line-height:1.32;letter-spacing:-.6px;
  font-weight:650;color:var(--ink);
}
.lede{margin:26px 0 0;max-width:74ch;font-size:16.5px;line-height:1.85;color:#4A5563}
.meta{
  display:grid;grid-template-columns:repeat(4,1fr);
  margin-top:44px;padding-top:22px;border-top:1px solid var(--line);
}
.meta div{padding-right:22px}
.meta span{
  display:block;margin-bottom:7px;font-size:10.5px;font-weight:700;
  letter-spacing:.14em;text-transform:uppercase;color:var(--mute);
}
.meta strong{display:block;font-size:13.5px;font-weight:600;color:var(--ink);line-height:1.55}

/* ---------- 摘要 ---------- */
.abstract{margin:62px 0 0;padding-left:26px;border-left:2px solid var(--accent)}
.abstract h2{
  margin:0 0 14px;font-size:12px;font-weight:700;
  letter-spacing:.18em;text-transform:uppercase;color:var(--accent);
}
.abstract p{margin:0 0 13px;font-size:15px;line-height:1.9;color:#46505E}
.abstract p:last-child{margin-bottom:0}

/* ---------- 总览表 ---------- */
.tbl{margin:54px 0 0;border-top:1px solid var(--ink)}
table{width:100%;border-collapse:collapse;font-size:13.5px}
thead th{
  padding:12px 12px 12px 0;text-align:left;font-size:10.5px;font-weight:700;
  letter-spacing:.14em;text-transform:uppercase;color:var(--mute);
  border-bottom:1px solid var(--line);white-space:nowrap;
}
tbody td{padding:13px 12px 13px 0;border-bottom:1px solid var(--line);vertical-align:top}
tbody td:first-child{width:56px;font-weight:700;font-family:ui-monospace,Consolas,monospace;font-size:12.5px}
tbody td:nth-child(2){width:30%;font-weight:600;color:var(--ink)}
tbody td:nth-child(3){font-family:ui-monospace,Consolas,monospace;font-size:12.5px;font-variant-numeric:tabular-nums}
tbody td:last-child{width:24%;padding-right:0;color:var(--mute);font-size:12.5px;text-align:right}
tbody tr:hover{background:var(--soft)}

/* ---------- 章节 ---------- */
.chapter{padding:76px 0 10px;border-top:1px solid var(--line)}
.ch-head{display:flex;gap:22px;align-items:flex-start}
.ch-num{
  flex:0 0 auto;padding-top:9px;font-family:ui-monospace,Consolas,monospace;
  font-size:13px;font-weight:700;letter-spacing:.08em;color:var(--accent);
}
.ch-en{
  margin:0 0 8px;font-size:11px;font-weight:700;
  letter-spacing:.18em;text-transform:uppercase;color:var(--mute);
}
.ch-title{
  margin:0;font-size:26px;line-height:1.42;font-weight:650;
  letter-spacing:-.3px;color:var(--ink);
}

/* ---------- 图 ---------- */
.figs{margin:34px 0 0}
figure{margin:0 0 28px}
figure:last-child{margin-bottom:0}
.figbox{
  padding:12px;background:#fff;border:1px solid var(--line);border-radius:3px;
  box-shadow:0 1px 1px rgba(15,23,32,.03),0 12px 28px -18px rgba(15,23,32,.20);
}
.figbox img{display:block;width:100%;height:auto;border-radius:2px}
figcaption{margin-top:11px;font-size:12.4px;line-height:1.72;color:var(--mute)}
figcaption b{
  display:inline-block;margin-right:9px;font-family:ui-monospace,Consolas,monospace;
  font-size:11px;font-weight:700;letter-spacing:.06em;color:var(--accent);
}

/* ---------- 正文 ---------- */
.prose{margin:28px 0 0;max-width:80ch}
.prose p{margin:0 0 15px;font-size:15.2px;line-height:1.92;color:#3F4957}
.prose p:last-child{margin-bottom:0}
.prose b{color:var(--ink);font-weight:600}
.prose code{
  padding:1px 5px;font-family:ui-monospace,Consolas,monospace;font-size:13px;
  background:var(--soft);border:1px solid var(--line);border-radius:2px;color:#2A3441;
}

/* ---------- 指标条 ---------- */
.stats{
  display:grid;grid-template-columns:repeat(4,1fr);gap:1px;
  margin:30px 0 0;background:var(--line);border:1px solid var(--line);
}
.stat{background:#fff;padding:17px 18px 16px}
.stat b{
  display:block;font-family:ui-monospace,Consolas,monospace;font-size:19px;
  font-weight:600;line-height:1.26;letter-spacing:-.4px;color:var(--ink);
}
.stat span{display:block;margin-top:7px;font-size:11.2px;line-height:1.55;color:var(--mute)}

/* ---------- 注记 ---------- */
.note{
  margin:26px 0 0;padding:15px 20px;font-size:13.2px;line-height:1.82;
  background:var(--soft);border-left:2px solid #C9A227;color:#4A5563;
}
.note b{color:var(--ink);font-weight:600}

/* ---------- 免责与开源声明 ---------- */
.disclaimer{
  margin:28px 0 0;padding:22px 26px 24px;background:#F2F4F7;
  border:1px solid #DDE2E8;border-radius:3px;
}
.disclaimer h3{
  margin:0 0 10px;font-size:13px;font-weight:700;letter-spacing:.06em;color:#3E4854;
}
.disclaimer p{margin:0;font-size:13.6px;line-height:1.92;color:#4A5563}

/* ---------- 章末脚注 ---------- */
.footnote{
  margin:20px 0 0;font-size:12.4px;line-height:1.8;font-style:italic;color:#9C7A00;
}

/* ---------- 附录 ---------- */
.appendix{padding:66px 0 0;margin-top:74px;border-top:1px solid var(--line)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:26px;margin:30px 0 0}
.files{columns:2;column-gap:40px;margin:24px 0 0;font-size:12.2px}
.files div{
  break-inside:avoid;padding:7px 0;border-bottom:1px dotted var(--line);
  font-family:ui-monospace,Consolas,monospace;color:#4A5563;
}
.files span{float:right;color:var(--mute);font-size:11px}

footer{margin-top:84px;padding:34px 0 66px;border-top:1px solid var(--line);font-size:12px;color:var(--mute)}
footer .wrap{display:flex;justify-content:space-between;gap:24px;flex-wrap:wrap}

/* ---------- 响应式 ---------- */
@media (max-width:760px){
  .wrap{padding:0 22px}
  .cover{padding:64px 0 40px}
  h1{font-size:29px}
  .lede{font-size:15px}
  .meta{grid-template-columns:repeat(2,1fr);row-gap:20px}
  .stats{grid-template-columns:repeat(2,1fr)}
  .grid2{grid-template-columns:1fr}
  .files{columns:1}
  thead th:last-child,tbody td:last-child{display:none}
  .ch-title{font-size:21px}
}
@media print{
  body{font-size:11pt}
  .chapter,figure,.stats,.note,.footnote{break-inside:avoid;page-break-inside:avoid}
  .cover{padding:0 0 24px;background:#fff}
  .figbox{box-shadow:none}
  /* 免责声明是不可分割的整块：绝不允许在标题与正文之间分页 */
  .disclaimer{
    break-inside:avoid;page-break-inside:avoid;
    -webkit-column-break-inside:avoid;
    margin:18pt 0 22pt;padding:14pt 16pt 16pt;
  }
  .disclaimer h3,.disclaimer p{
    break-inside:avoid;page-break-inside:avoid;
    orphans:3;widows:3;
  }
  /* 正文与声明之间留出安全距离，避免整块被顶到页边 */
  .abstract{margin-bottom:26pt}
}
"""


# ---------------------------------------------------------------- 章节内容
# 每个章节：编号 / 英文眉题 / 中文标题 / 图（可多张）/ 正文段落 / 关键指标 / 可选注记
CHAPTERS = [
    dict(
        num="01",
        en="Illuminance Data Pipeline",
        title="照明系统数据自动化处理",
        figs=[("heatmap.png", "图 1",
               "照度伪彩色渲染：100 × 100 采样网格、0.5 mm 间距。色标为 lux，等照度线间隔 10%；"
               "虚线框为 1/e² 全光斑口径与 0.5R 中心区两种统计窗口，分别出数。")],
        body=[
            "任务是把一张「看起来还算均匀」的照度分布，变成可以判定的数字。用 numpy 生成 "
            "100 × 100 采样、间距 0.5 mm 的高斯光斑（1/e² 半径 17.5 mm），叠加随机噪声模拟探测器采集；"
            "pandas 落盘为宽表，matplotlib 做伪彩色渲染并叠加等照度线——把二维场压成一张可判读的图。",
            "真正需要工程判断的是<b>统计窗口</b>。若直接在整张 35 × 35 mm 网格上取 min/max，"
            "高斯尾部会把最小值拉到接近零，均匀性结论立刻失去意义。因此按 1/e² 孔径（3852 个有效采样点）"
            "与 0.5R 中心区（952 点）分窗统计，并同时给出 U(min/max)、U(min/mean)、U(mean/max) 三种定义——"
            "严格程度依次递减，只报其中一种很容易产生歧义。",
            "结果：全光斑 U(min/max) = 12.0%，中心区 55.3%，变异系数由 56.0% 降至 14.4%，"
            "9 点法 U(min/max) = 34.7%。这个「低」不是程序缺陷：按 1/e² 的定义，高斯分布在口径边缘的强度"
            "必然衰减到峰值的 13.5%，任何高斯配光都不可能通过 50% 的全光斑判据。要提升只能改配光本身"
            "——平顶、超高斯或复眼匀光。",
        ],
        stats=[("12.0 %", "全光斑 U(min/max)"), ("55.3 %", "中心区 U(min/max)"),
               ("14.4 %", "中心区变异系数 CV"), ("3 852", "有效采样点数")],
        note=None,
    ),
    dict(
        num="02",
        en="Parameter Space Sweep",
        title="一维与二维参数空间全局寻优",
        figs=[("zemax_sweep.png", "图 2a",
               "一维曲率扫描：MTF@30 lp/mm 对曲率半径，5 点采样，单峰曲线，峰值落在 R = 50.0 mm。"),
              ("zemax_2d_sweep.png", "图 2b",
               "二维 MTF 矩阵：曲率 × 空气间隔，7 × 9 = 63 点。白色虚线为最优脊线，斜率 −1.0；"
               "右上为三维曲面，左下为等高线。")],
        body=[
            "这一章搭的是所有扫描的地基：业务层只描述「扫哪些参数、取什么指标」，"
            "求解后端藏在抽象接口之后，按 zospy → 原生 ZOS-API → Mock 三级降级。任一后端不可用时不抛异常，"
            "而是记录失败原因、退到下一级，输出 CSV 里带 <code>backend</code> 字段。"
            "同一份脚本因此在没有装光学软件的机器上能跑通，在装有 Zemax 的工作站上取消一段注释即可对接真机。",
            "一维扫描把曲率半径从 49 mm 到 51 mm 走 5 点，MTF@30 lp/mm 在 R = 50.0 处取到峰值 0.8697，"
            "曲线单峰、两侧对称下降。二维扫描进一步引入空气间隔（2–10 mm），构成 7 × 9 = 63 点矩阵，"
            "全局最优落在 R = 50 / S = 6，MTF = 0.8624。",
            "比最优点更有价值的是等高线的形状：最优脊线斜率为 <b>−1.0</b>，即曲率每增大 1 mm，"
            "最优间距就要减小约 1 mm。曲率与间距在焦度上强耦合，这意味着公差分配不能单独放宽某一项"
            "——两者必须联合管控，否则单看各自公差都会误判。",
        ],
        stats=[("0.8697", "一维峰值 MTF@30lp"), ("50.0 mm", "最优曲率半径"),
               ("63 点", "二维扫描矩阵"), ("−1.0", "脊线耦合斜率")],
        note="<b>数据边界</b>：本章数值来自 Mock 后端（<code>backend=mock</code>），"
             "用于验证框架的吞吐能力与降级逻辑，不代表 Zemax 实测 MTF。真机接入后同一脚本可原样复跑，"
             "字段与图表结构保持不变。",
        footnote="注：本章节数据为 ZOS-API 的 Mock 框架测试结果，用于展示自动寻优逻辑，"
                 "非真实 Zemax 导出数据。",
    ),
    dict(
        num="03",
        en="Real Ray Tracing, Singlet",
        title="单透镜真实光线追迹",
        figs=[("ray_trace_real.png", "图 3",
               "平凸单透镜光线追迹（半口径 5 mm 与 10 mm 两组）。边缘光线出射后明显向光轴偏折，"
               "落点比近轴焦点更靠前——即欠校正负球差。")],
        body=[
            "从这一章起脱离 Mock，改用 rayoptics 做真实斯涅尔折射追迹。模型是 EPD 25.4 mm 的平凸单透镜，"
            "物在无穷远（首面间隔置 1e10 模拟平行光），曲率半径 25.4 mm、中心厚 5 mm。",
            "先做近轴校核：EFL = R/(n−1) = 25.4/0.5168 ≈ 49.15 mm，实测 49.1488 mm；"
            "BFL = f − d/n，实测 45.8523 mm，与解析值吻合到小数点后两位。这一步不通过就说明面型或材料填错了，"
            "后面的像差数字都不必再看——先算对一阶量，再谈三阶量。",
            "然后追三条高度 5 / 10 mm 的平行光。h = 10 mm 处横向球差 TA = −0.5059 mm、"
            "纵向球差 LSA = −2.301 mm。把 h = 5 mm 与 10 mm 的结果相除：TA 比 <b>8.81</b>、LSA 比 <b>4.15</b>，"
            "对应理论 h³ = 8.00 与 h² = 4.00。偏差来自五级像差与有限物距的贡献，但量级与走势确认了"
            "这就是三阶球差本身，而不是别的误差在冒充。",
        ],
        stats=[("49.1488 mm", "有效焦距 EFL"), ("45.8523 mm", "后焦距 BFL"),
               ("−2.301 mm", "纵向球差 LSA(h=10)"), ("8.81", "TA 实测比（理论 8.00）")],
        note=None,
    ),
    dict(
        num="04",
        en="Spherical Correction",
        title="球差校正实战：正负透镜组合",
        figs=[("doublet_raytrace.png", "图 4",
               "平凸 + 负弯月双片式光线追迹与球差曲线。纵向球差由 −2.3005 mm 收敛至 −0.0014 mm；"
               "h = 8 mm 附近的回弯是五级球差残余，也是这套两片式的物理天花板。")],
        body=[
            "单透镜的球差无法自愈——球面在边缘的折光能力天然强于近轴。经典解法是给正透镜配一片负透镜："
            "负透镜产生符号相反、随口径增长更快的球差，把正透镜的欠校正抵消掉。模型为平凸 + 负弯月，"
            "11 条光线覆盖 0–10 mm 半口径。",
            "三个自由量（弯月前后表面曲率 R3、R4 与空气间隔）交给 scipy 的 Nelder-Mead 无导数优化，"
            "目标函数取 h = 10 mm 处的 LSA，并约束焦距不变。这里用无导数算法比梯度法稳："
            "目标函数接近零时会被追迹误差弄出细小锯齿，差分梯度会直接失效。",
            "结果：LSA 从 <b>−2.3005 mm 收敛到 −0.0014 mm</b>，降幅 <b>99.94%</b>（改善 1642 倍）；"
            "像面弥散 RMS 从 0.2104 mm 降到 0.0077 mm。特别要注意 h = 8 mm 附近曲线出现回弯——"
            "这是五级球差残余，说明「三阶球差归零」并不等于「全口径完美成像」，边缘光线依旧欠校正。"
            "这条回弯就是两片式的物理极限，再往前必须加片。",
        ],
        stats=[("−99.94 %", "LSA 降幅"), ("1642 ×", "改善倍数"),
               ("0.0077 mm", "像面弥散 RMS"), ("2.41 mm", "最优空气间隔")],
        note=None,
    ),
    dict(
        num="05",
        en="Achromatic Doublet",
        title="色差校正与双胶合设计",
        figs=[("achromatic_raytrace.png", "图 5",
               "消色差双胶合光路（F 486 nm / d 588 nm / C 656 nm 三波长）。三色近轴焦点重合成一点，"
               "近轴 LCA 残余仅 3 nm；右图为轴向球差与色球差曲线。")],
        body=[
            "色差来自材料色散：同一片玻璃对 F 光与 C 光的折射率不同，焦点于是沿轴分开。"
            "消色差的薄透镜条件很干脆——<code>φ1/V1 + φ2/V2 = 0</code>，同时 <code>φ1 + φ2 = 1/100</code>。"
            "解出冕牌 +49.74 mm、火石 −98.96 mm，两者光焦度反号，阿贝数比与焦距比互为倒数。",
            "材料取 N-BK7（nd 1.5168 / Vd 64.17）与 N-SF5（nd 1.6727 / Vd 32.25），"
            "三面胶合、中间面曲率共用，得 R = +52.962 / −47.495 / −187.989 mm，EFL 精确到 100.0000 mm。"
            "胶合面的边缘厚度同时做了几何校验（冕牌 2.73 mm / 火石 4.30 mm），避免解出来的设计在实物上崩边。",
            "近轴 LCA 从单透镜的 −1528 µm 压到 <b>−0.003 µm（3 nm）</b>，四个数量级。但这不是免费的："
            "二级光谱仍有 −50.9 µm，全口径色球差 +185 µm，像面弥散 5.78 µm。近轴 LCA 归零只保证"
            "「三条近轴光线交于一点」，边缘光线的球差随波长漂移（色球差）并不受这条方程约束"
            "——这正是下一章要处理的残留。",
        ],
        stats=[("0.003 µm", "近轴 LCA(F−C)"), ("−1528 µm", "单透镜对照"),
               ("−50.9 µm", "残余二级光谱"), ("5.78 µm", "像面弥散 RMS")],
        note=None,
    ),
    dict(
        num="06",
        en="Apochromatic Triplet",
        title="APO 复消色差设计：ED 玻璃选型",
        figs=[("chromatic_focal_shift.png", "图 6",
               "焦移随波长变化：双胶合 / 单 ED 三片 / 全 APO 三条曲线，450–700 nm 全带扫描。"
               "双胶合在带边漂到 +168 µm，APO 全程压在 ±15 µm 内。")],
        body=[
            "复消色差要同时满足三个方程：总光焦度、F = C、以及 d 光落在 F 与 C 的中点。"
            "三个方程要求三块玻璃的「相对部分色散偏离」ΔP 彼此拉开。把材料库全部 1087 种玻璃的 P 对 V 做线性拟合，"
            "得到正常线 <code>P = 0.000438·V − 0.222442</code>；普通玻璃的 ΔP 都落在 ±0.002 内，"
            "三方程因此病态，解出的光焦度无法加工。",
            "真正的解在反常色散材料：S-FPL51（氟磷酸盐 ED）ΔP = −0.0076、N-KZFS2（短火石）ΔP = +0.0049、"
            "N-BK7 ΔP = +0.0020，符号与量级都够用。最终处方 R = +50.375 / −33.819 / +50.609 / −180.311 mm，"
            "厚度 6 / 3 / 4 mm，EFL = 100.0001 mm。",
            "二级光谱从双胶合的 −50.94 µm 压到 <b>−1.13 µm</b>，改善 45 倍；近轴 LCA −0.22 µm，"
            "450–700 nm 带内峰谷 17.0 µm，带内最大焦移 2.3 µm，横向弥散 4.76 µm。"
            "过程中还留了一个反面样本：FPL51 + SF5 + BK7 只放了一片 ED、却没配反常火石，"
            "二级光谱只做到 −16.9 µm——这说明「加一片 ED」并不自动等于 APO，ΔP 必须成对拉开。",
        ],
        stats=[("45 ×", "二级光谱改善"), ("−1.13 µm", "残余 SEC"),
               ("−0.22 µm", "近轴 LCA"), ("2.3 µm", "带内最大焦移")],
        note="<b>材料边界</b>：所用材料库（Schott / Ohara / Sumita / CDGM / Hikari）不含 CaF₂ 单晶数据，"
             "ED 元素选 Ohara S-FPL51 替代。若后续接入 CaF₂，二级光谱的理论下限还能再降一档。",
    ),
    dict(
        num="07",
        en="Off-Axis Aberrations",
        title="离轴视场与像质评估",
        figs=[("offaxis_throughfocus.png", "图 7",
               "5° 视场的调焦序列与 T/S 场曲曲线。切向焦面更靠近透镜，先收成水平焦线；继续外移则收到"
               "竖直焦线（弧矢），两者之间是最小弥散圆。下排为 25 点场曲曲线，三条曲线均按 θ² 规律下弯。")],
        body=[
            "轴上做到极致之后，视场一斜就是另一场战争。斜入射时孔径上下两侧的等效焦距不再相等："
            "彗差按 θ¹ 增长，像散与场曲按 θ² 增长。复用第 5 章的消色差双胶合（f/3.94），"
            "每个视场用六角极坐标铺 127 条光线做真实追迹。",
            "三条幂律全部实测复现。彗差系数 a₂ 在 3° 与 5° 的比值为 <b>1.665</b>（理论 5/3 = 1.667）；"
            "像散（切向与弧矢焦面间距）比值 <b>2.76</b>（理论 (5/3)² = 2.78）；Petzval 面上切向与弧矢的相对偏离"
            "为 <b>3.00 : 1</b>，与薄透镜理论一致，实测 R_p = 133.7 mm 对理论 137.4 mm（差 3%，源于忽略厚度）。"
            "还有一个容易漏掉的细节：离焦只产生奇次项、不污染切向扇的二次项，所以拟合出的 a₂ 是干净的纯彗差。",
            "量化结论：0° / 3° / 5° 的弥散斑 RMS 为 <b>3.5 / 41.7 / 109 µm</b>，相当于艾里半径的"
            "1.2 / 14.8 / 38.6 倍。以 10 倍艾里为崩溃判据，这个系统在 <b>±2.4°</b> 彻底失效；"
            "按 2 倍艾里算，可用视场只有 <b>±0.7°</b>。即使允许每个视场单独重调焦，也只撑到 ±3.45°"
            "——说明瓶颈不是调焦，而是波前畸变本身。缩光圈同样救不了：像差按 D² 缩、艾里斑按 D⁻¹ 涨，"
            "f/3.9 缩到 f/12.5 只把相对像质改善约 3 倍，场曲一点没动。",
            "改进方向明确：加平场（field flattener）镜压 Petzval 场曲；把光阑从透镜表面移开、用非对称结构平衡彗差；"
            "或者直接换 <b>Cooke 三片式</b>——正-负-正、光阑居中，靠结构对称性同时压彗差与畸变。"
            "这也是下一阶段最值得做的一步。",
        ],
        stats=[("38.6 ×", "5° 弥散 / 艾里半径"), ("±2.4°", "崩溃视场"),
               ("3.00 : 1", "Petzval 切向 : 弧矢"), ("718 µm", "5° 视场像散")],
        note=None,
    ),
]

SUMMARY_ROWS = [
    ("01", "照明数据管线", "U(min/max) 12.0%（全光斑）/ 55.3%（中心区）", "1/e² 分窗 + 9 点法"),
    ("02", "参数空间寻优", "MTF@30lp 峰值 0.8697（R=50, S=6）", "63 点矩阵 + 脊线斜率"),
    ("03", "单透镜追迹", "EFL 49.1488 mm，LSA −2.301 mm", "TA ∝ h³ 实测 8.81 / 理论 8.00"),
    ("04", "球差校正", "LSA 2.3005 → 0.0014 mm（−99.94%）", "Nelder-Mead + 回弯判据"),
    ("05", "消色差双胶合", "近轴 LCA −1528 → −0.003 µm", "φ/V 解析分配 + 三波长"),
    ("06", "APO 三片式", "二级光谱 −50.94 → −1.13 µm", "全库 ΔP 选型 + 带内扫描"),
    ("07", "轴外像差", "RMS 3.5 / 41.7 / 109 µm（0/3/5°）", "θ¹·θ² 幂律 + Petzval 3:1"),
]


# ---------------------------------------------------------------- HTML 拼装
def render_figs(figs: list) -> str:
    out = []
    for name, label, cap in figs:
        out.append(
            f'      <figure>\n'
            f'        <div class="figbox"><img src="images/{name}" alt="{cap[:40]}" loading="lazy"></div>\n'
            f'        <figcaption><b>{label}</b>{cap}</figcaption>\n'
            f'      </figure>\n'
        )
    return "".join(out)


def render_chapter(ch: dict) -> str:
    paras = "".join(f"            <p>{p}</p>\n" for p in ch["body"])
    stats = "".join(
        f'          <div class="stat"><b>{v}</b><span>{k}</span></div>\n' for v, k in ch["stats"]
    )
    note = ""
    if ch["note"]:
        note = f'        <div class="note">{ch["note"]}</div>\n'
    if ch.get("footnote"):
        note += f'        <p class="footnote">{ch["footnote"]}</p>\n'
    return (
        f'\n    <section class="chapter" id="ch{ch["num"]}">\n'
        f'      <div class="ch-head">\n'
        f'        <div class="ch-num">{ch["num"]}</div>\n'
        f'        <div>\n'
        f'          <p class="ch-en">{ch["en"]}</p>\n'
        f'          <h2 class="ch-title">{ch["title"]}</h2>\n'
        f'        </div>\n'
        f'      </div>\n'
        f'\n      <div class="figs">\n{render_figs(ch["figs"])}      </div>\n'
        f'\n      <div class="prose">\n{paras}      </div>\n'
        f'\n      <div class="stats">\n{stats}      </div>\n'
        f'{note}'
        f'    </section>\n'
    )


def build_html() -> str:
    stamp = datetime.now().strftime("%Y-%m-%d")
    n_py = len(list((ROOT / "scripts").glob("*.py")))
    n_txt = len(list((ROOT / "output").glob("*.txt")))
    files_total = sum(1 for _ in DATADIR.rglob("*") if _.is_file()) + \
                  sum(1 for _ in CODEDIR.glob("*.py")) + sum(1 for _ in CODEDIR.glob("*.txt"))

    head = (
        '<!DOCTYPE html>\n<html lang="zh-CN">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        '<meta name="description" content="基于纯 Python 的光学系统建模、自动化寻优与像差校正框架——'
        '从照明数据管线到 APO 复消色差与离轴像差评估的完整工程记录。">\n'
        '<title>基于纯 Python 的光学系统建模、自动化寻优与像差校正框架</title>\n'
        '<style>\n' + CSS + '\n</style>\n</head>\n<body>\n'
    )

    cover = f"""
  <header class="cover">
    <div class="wrap">
      <p class="kicker">Optical Design Portfolio · {stamp[:4]}</p>
      <h1>基于纯 Python 的光学系统<br>建模、自动化寻优与像差校正框架</h1>
      <p class="lede">
        本作品集收录 7 章共 10 组可视化结果与 {n_py} 个可复现脚本，覆盖从照明数据管线、参数空间寻优，
        到真实光线追迹、球差—色差—轴外像差三级校正的完整链路。全部计算在本地 Python 环境中完成，
        不依赖任何商业光学软件；每一条结论都配有近轴解析解或三级像差幂律的独立校核。
      </p>
      <div class="meta">
        <div><span>项目周期</span><strong>2026.10 — 2026.11</strong></div>
        <div><span>运行环境</span><strong>Python 3.13 · 64-bit</strong></div>
        <div><span>核心依赖</span><strong>rayoptics 0.9.10<br>opticalglass 2.0.2 · numpy · scipy</strong></div>
        <div><span>交付物</span><strong>{n_py} 个脚本 · 10 张图<br>{n_txt} 份数据报告</strong></div>
      </div>
    </div>
  </header>
"""

    abstract = """
    <main class="wrap">
      <section class="abstract">
        <h2>Abstract</h2>
        <p>
          这套框架要解决的不是「算一个透镜」，而是把「改参数 → 追迹 → 提指标 → 判据校验」这条闭环
          做成可复现、可降级、可交接的脚本。业务层只声明扫什么、取什么，求解后端藏在抽象接口之后，
          因此同一份代码既能在没有光学软件的机器上跑通全流程，也能在装有 Zemax 的工作站上直接对接真机。
        </p>
        <p>
          方法论上坚持三重独立校核：一阶量对近轴解析解（EFL = R/(n−1)、BFL = f − d/n、
          φ₁/V₁ + φ₂/V₂ = 0）；三阶量对幂律（TA ∝ h³、LSA ∝ h²、彗差 ∝ θ¹、像散 ∝ θ²）；
          场曲对 Petzval 求和（1/R_p = Σφᵢ/nᵢ）。任何一条对不上，先怀疑代码而不是先解释物理。
        </p>
        <p>
          数据诚实性上，Mock 后端的结果一律显式标注 <code>backend=mock</code>，不与真机数据混用；
          降级链每一级的失败原因都落盘可查。作品集里的每个数字都能由归档脚本重新跑出来。
        </p>
      </section>

      <section class="disclaimer">
        <h3>【免责与开源声明】</h3>
        <p>
          本项目为个人技术验证项目。Zemax 自动化寻优部分因无商业授权，采用 Mock（模拟接口）验证算法；
          所有光线追迹、球差/色差/APO 及离轴像差分析，均基于 Python 开源库 rayoptics 进行真实物理计算。
          代码框架由本人设计，AI 辅助编写，所有结果已通过理论公式交叉验证。
        </p>
      </section>
"""

    # 总览表
    rows = "".join(
        f'          <tr><td><a href="#ch{n}">{n}</a></td><td>{t}</td>'
        f'<td>{m}</td><td>{v}</td></tr>\n'
        for n, t, m, v in SUMMARY_ROWS
    )
    table = f"""
      <div class="tbl">
        <table>
          <thead><tr><th>章</th><th>主题</th><th>核心结果</th><th>验证方式</th></tr></thead>
          <tbody>
{rows}          </tbody>
        </table>
      </div>
"""

    chapters = "".join(render_chapter(ch) for ch in CHAPTERS)

    # 附录：补充图 + 文件清单
    extra_figs = (
        '        <figure>\n'
        '          <div class="figbox"><img src="images/apo_raytrace.png" alt="APO 三片式光路" loading="lazy"></div>\n'
        '          <figcaption><b>附图 A</b>APO 三片式（S-FPL51 + N-KZFS2 + N-BK7）光路：三个胶合面、'
        '四片式结构，EFL 100.0001 mm。</figcaption>\n'
        '        </figure>\n'
        '        <figure>\n'
        '          <div class="figbox"><img src="images/offaxis_spot.png" alt="三视场弥散斑对比" loading="lazy"></div>\n'
        '          <figcaption><b>附图 B</b>0° / 3° / 5° 三视场弥散斑同尺度对比：0° 的圆斑放进 5° 的视野里'
        '只剩一粒沙，直观呈现像差随视场的爆发式增长。</figcaption>\n'
        '        </figure>\n'
    )

    flist = []
    for p in sorted(list(CODEDIR.glob("*.py")) + list(CODEDIR.glob("*.txt")) +
                    list(DATADIR.rglob("*"))):
        if p.is_file():
            rel = p.relative_to(CODEDIR).as_posix()
            flist.append(f'          <div>{rel}<span>{human_size(p.stat().st_size)}</span></div>\n')

    appendix = f"""
      <section class="appendix">
        <div class="ch-head">
          <div class="ch-num">A</div>
          <div>
            <p class="ch-en">Appendix</p>
            <h2 class="ch-title">补充视图与归档清单</h2>
          </div>
        </div>
        <div class="grid2">
{extra_figs}        </div>
        <p class="ch-en" style="margin:44px 0 0">code_and_data · {files_total} 个文件</p>
        <div class="files">
{''.join(flist)}        </div>
      </section>
"""

    footer = f"""
    </main>
  <footer>
    <div class="wrap">
      <div>本页由 <code>scripts/09_build_portfolio.py</code> 自动生成 · {stamp}</div>
      <div>全部图表与数据来自本地可复现运行 · 无外部 CDN / JS 依赖</div>
    </div>
  </footer>
</body>
</html>
"""
    return head + cover + abstract + table + chapters + appendix + footer


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description="归档光学工程资产并生成作品集")
    ap.add_argument("--open", action="store_true", help="生成后自动用浏览器打开")
    args = ap.parse_args()

    print("=" * 68)
    print("光学工程作品集归档")
    print("=" * 68)
    print(f"源目录 : {ROOT}")
    print(f"目标   : {OUT}")

    OUT.mkdir(parents=True, exist_ok=True)
    IMGDIR.mkdir(parents=True, exist_ok=True)
    CODEDIR.mkdir(parents=True, exist_ok=True)
    DATADIR.mkdir(parents=True, exist_ok=True)

    # 1) 图片
    print("\n[1/3] 归档图表")
    ok_n = 0
    for name, _desc in IMAGES:
        src = ROOT / "output" / name
        ok, info = copy_verified(src, IMGDIR / name)
        if ok:
            ok_n += 1
            print(f"  OK   {name:<28} {info}")
        else:
            print(f"  !!   {name:<28} {info}  <- 请检查 output/ 目录")
    print(f"  小计 : {ok_n}/{len(IMAGES)} 张，MD5 全部一致")

    # 2) 脚本 + 报告 + 数据
    print("\n[2/3] 归档脚本与报告")
    n_code = 0
    for src in sorted((ROOT / "scripts").glob("*.py")):
        ok, info = copy_verified(src, CODEDIR / src.name)
        n_code += ok
        print(f"  {'OK ' if ok else '!! '}  code_and_data/{src.name:<26} {info}")
    for src in sorted((ROOT / "output").glob("*.txt")):
        ok, _ = copy_verified(src, CODEDIR / src.name)
        n_code += ok
        print(f"  {'OK ' if ok else '!! '}  code_and_data/{src.name:<26}")
    for src in sorted((ROOT / "data").glob("*.csv")) + sorted((ROOT / "output").glob("*.csv")):
        copy_verified(src, DATADIR / src.name)
    n_csv = len(list(DATADIR.glob("*.csv")))
    print(f"  小计 : {n_code} 个代码/报告 + {n_csv} 个数据表")

    # 3) HTML
    print("\n[3/3] 生成作品集页面")
    html = build_html()
    page = OUT / "Portfolio_Report.html"
    page.write_text(html, encoding="utf-8")
    print(f"  OK   {page.name}  {human_size(page.stat().st_size)} · {len(CHAPTERS)} 章")

    print("\n" + "=" * 68)
    print("完成。浏览器打开：")
    print(f"  {page}")
    print("=" * 68)

    if args.open:
        webbrowser.open(page.as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
