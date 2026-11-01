# Python-Optics-Portfolio

[English](#english) | [中文版](#中文版)

## English
A pure Python optical system modeling, automated optimization, and aberration correction framework (without relying on commercial optical software like Zemax).

**Core Achievements:**
- Built a ZOS-API automation framework with mock backend, achieving 2D parameter space global optimization and tolerance contour evaluation.
- Designed a doublet structure, correcting Longitudinal Spherical Aberration (LSA) by 99.94% (from 2.3mm to 0.0014mm).
- Designed an APO triplet, compressing secondary spectrum from 50.9 μm to 1.1 μm (45x compression).
- Reproduced off-axis higher-order aberrations (Coma/Astigmatism) following h³ and h² laws, accurately identifying field-of-view limits.

*(Note: The Zemax optimization module uses mock data due to the lack of a commercial license. All ray-tracing and aberration analyses are based on real physical calculations using the `rayoptics` Python library.)*

📄 **[Download the Full Project Report (PDF)](./你的PDF文件名.pdf)** *(请把这里的"你的PDF文件名.pdf"换成你实际上传的文件名)*

---

## 中文版
基于纯 Python 的光学系统建模、自动化寻优与像差校正框架。

**核心成果：**
- 构建了带 Mock 降级的 ZOS-API 自动化框架，实现二维参数空间（曲率x间距）的全局寻优与公差等高线评估。
- 设计了双透镜结构，将纵向球差（LSA）从 2.3mm 消除至 0.0014mm（优化率 99.94%）。
- 设计三片式 APO（复消色差）结构，将系统二级光谱从 50.9μm 压制至 1.1μm（压缩 45 倍）。
- 复现了高阶离轴像差（彗差/像散）随视场增大的物理规律（h³ 与 h² 定律），精准定位系统视场极限。

*(注：Zemax 自动化寻优部分因无商业授权采用 Mock 数据；所有光线追迹、球差/色差/APO及离轴像差分析，均基于 Python 开源库 `rayoptics` 进行真实物理计算。)*

📄 **[点击下载完整项目报告 (PDF)](./你的PDF文件名.pdf)**
