# -*- coding: utf-8 -*-
"""
光学设计与仿真环境体检脚本
------------------------------------------------
用途：每次开始新任务前跑一次，10 秒内确认：
  1) Python 与第三方依赖是否就绪
  2) pywin32 / COM 是否可用（CodeV、LightTools 必依赖）
  3) pythonnet 与 .NET 运行时是否可用（Zemax ZOS-API 必依赖）
  4) 本机是否安装了 Zemax / CodeV / LightTools / Lumerical，以及 ZOS-API 目录

设计原则：只读探测，不写注册表、不改任何软件配置、不动任何设计文件。
用法： python scripts/00_env_check.py
"""

import os
import sys
import glob
import platform
import importlib

SEP = "-" * 62


def h(title):
    """小节标题"""
    print(SEP)
    print(title)
    print(SEP)


# ============================================================
# 1. Python 解释器
# ============================================================
h("[1] Python 解释器")
IS_64 = sys.maxsize > 2**32          # 最可靠的位数判据，不依赖 platform.architecture 的大小写
print(f"版本      : {sys.version.split()[0]}")
print(f"架构      : {'64bit' if IS_64 else '32bit'}  (ZOS-API / LightTools 必须 64 位)")
print(f"可执行文件: {sys.executable}")

if not IS_64:
    print("!! 警告：32 位解释器无法加载 Zemax 64 位 ZOS-API 与 LightTools COM")

# ============================================================
# 2. 关键第三方依赖
# ============================================================
h("[2] 第三方依赖")
DEPS = {
    "win32com": "pywin32 — CodeV / LightTools COM 自动化",
    "pythoncom": "pywin32 — COM 单元初始化",
    "numpy": "数值计算",
    "scipy": "插值 / 拟合 / 优化",
    "pandas": "CSV / TXT 数据表处理",
    "matplotlib": "出图（MTF、点列图、照度均匀性）",
    "openpyxl": "Excel 报表输出",
    "clr": "pythonnet — Zemax ZOS-API（.NET 桥）",
}

missing = []
for mod, desc in DEPS.items():
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        print(f"  [OK]   {mod:<12} {str(ver):<10} {desc}")
    except Exception:
        print(f"  [缺失] {mod:<12} {'':<10} {desc}")
        missing.append(mod)

if missing:
    pip_names = {"clr": "pythonnet", "win32com": "pywin32", "pythoncom": "pywin32"}
    need = sorted({pip_names.get(m, m) for m in missing})
    print(f"\n  修复命令: python -m pip install {' '.join(need)}")

# ============================================================
# 3. COM 通道（CodeV / LightTools 依赖）
# ============================================================
h("[3] COM 通道自检")
try:
    import pythoncom
    import win32com.client

    pythoncom.CoInitialize()
    # 用一个系统自带的、纯数据型 COM 对象做最小可用性验证
    d = win32com.client.Dispatch("Scripting.Dictionary")
    d.Add("probe", 1)
    assert d.Item("probe") == 1
    print("  [OK] COM 派发正常（pywin32 可用）")
except Exception as e:
    print(f"  [失败] COM 不可用: {type(e).__name__}: {e}")
    print("  修复: python -m pywin32_postinstall -install  (需管理员权限终端)")

# ============================================================
# 4. .NET 运行时（Zemax ZOS-API 依赖）
# ============================================================
h("[4] .NET 运行时 / pythonnet")
try:
    import clr  # noqa: F401

    net_ver = "未知"
    try:
        from System import Environment

        net_ver = Environment.Version.ToString()
    except Exception:
        pass
    print(f"  [OK] pythonnet 可用，已加载 CLR（CLR 版本: {net_ver}）")

    # CLR 版本号本身看不出 .NET Framework 具体版本，读注册表 Release 键才准
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full") as k:
            rel = winreg.QueryValueEx(k, "Release")[0]
        # 微软官方 Release 键 → 版本对照（用阈值判断，兼容后续小版本号）
        if rel >= 533320:
            fw = "4.8.1 或更高"
        elif rel >= 528040:
            fw = "4.8"
        elif rel >= 461808:
            fw = "4.7.2"
        else:
            fw = f"低于 4.7.2 (Release={rel})"
        flag = "[OK]  " if rel >= 528040 else "[警告]"
        print(f"  {flag} .NET Framework {fw}  (Release={rel}) —— ZOS-API 要求 4.8 及以上")
    except Exception:
        print("  [跳过] 未能读取 .NET Framework 版本")
except Exception as e:
    print(f"  [失败] pythonnet 不可用: {type(e).__name__}: {e}")
    print("  修复: python -m pip install pythonnet")

# ============================================================
# 5. 光学软件安装探测（注册表 + COM ProgID + 常见路径）
# ============================================================
h("[5] 光学软件安装探测")

try:
    import winreg

    def reg_read(root, path, name=""):
        try:
            with winreg.OpenKey(root, path) as k:
                return winreg.QueryValueEx(k, name)[0]
        except Exception:
            return None

    # 5a. 卸载表 —— 最可靠的"是否安装"判据
    UNINSTALL = [
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
        (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
    ]
    PATTERN = ("zemax", "opticstudio", "code v", "codev", "lighttools",
               "lumerical", "synopsys", "ansys", "fred", "virtuallab", "rsoft")
    found_apps = []
    for root, base in UNINSTALL:
        try:
            with winreg.OpenKey(root, base) as k:
                for i in range(winreg.QueryInfoKey(k)[0]):
                    try:
                        sub = winreg.EnumKey(k, i)
                        dn = reg_read(root, f"{base}\\{sub}", "DisplayName")
                        loc = reg_read(root, f"{base}\\{sub}", "InstallLocation")
                        ver = reg_read(root, f"{base}\\{sub}", "DisplayVersion")
                        if dn and any(p in dn.lower() for p in PATTERN):
                            found_apps.append((dn, ver, loc))
                    except Exception:
                        continue
        except Exception:
            continue

    if found_apps:
        for dn, ver, loc in found_apps:
            print(f"  [已装] {dn}  v{ver}")
            if loc:
                print(f"         安装位置: {loc}")
    else:
        print("  [未装] 注册表卸载表中未发现 Zemax / CodeV / LightTools / Lumerical")

    # 5b. COM ProgID —— 直接决定 COM 自动化能不能连上
    PROGIDS = {
        "Zemax.Application": "Zemax OpticStudio (旧版 COM)",
        "ZOSAPI.ZOSAPI_Connection": "Zemax ZOS-API (.NET)",
        "CodeV.Application": "Code V",
        "LightTools.Application": "LightTools",
        "Lumerical.Application": "Lumerical",
    }
    print("\n  COM ProgID 注册状态:")
    for pid, label in PROGIDS.items():
        ok = reg_read(winreg.HKEY_CLASSES_ROOT, pid + r"\CLSID")
        print(f"    {'[OK]  ' if ok else '[缺失]'} {pid:<28} {label}")

    # 5c. 常见安装路径 + ZOS-API 目录定位
    CANDIDATES = [
        r"C:\Program Files\Zemax",
        r"C:\Program Files\Zemax OpticStudio",
        r"C:\Program Files (x86)\Zemax",
        r"C:\Program Files\Optical Research Associates",
        r"C:\Program Files\Synopsys",
        r"C:\Program Files\ANSYS",
        r"C:\Program Files\Lumerical",
        r"D:\Program Files\Zemax",
    ]
    hits = [p for p in CANDIDATES if os.path.isdir(p)]
    if hits:
        print("\n  安装路径命中:")
        for p in hits:
            print(f"    - {p}")
    else:
        print("\n  常见安装路径：全部未命中")

    # 5d. ZOS-API 关键 DLL —— 有它才能 import ZOSAPI
    print("\n  ZOS-API 关键文件搜索:")
    dll_hits = []
    for root_dir in [r"C:\Program Files", r"C:\Program Files (x86)", r"D:\Program Files", r"D:\\"]:
        if os.path.isdir(root_dir):
            for pat in ("ZOSAPI_NetHelper.dll", "ZOSAPI.dll"):
                dll_hits += glob.glob(os.path.join(root_dir, "**", pat), recursive=True)[:3]
    dll_hits = sorted(set(dll_hits))
    if dll_hits:
        for d in dll_hits:
            print(f"    [OK] {d}")
        api_dir = os.path.dirname(dll_hits[0])
        print(f"\n  建议 ZOSAPI_DIR = r\"{api_dir}\"")
    else:
        print("    [缺失] 未找到 ZOSAPI.dll / ZOSAPI_NetHelper.dll")
        print("    -> 当前无法进行 Zemax 自动化，需安装 OpticStudio（专业版及以上含 ZOS-API）")

except ImportError:
    print("  [跳过] 非 Windows 平台或 winreg 不可用")

# ============================================================
# 6. 结论
# ============================================================
h("[6] 体检结论")
if not missing and not found_apps:
    print("Python 侧：全部就绪 —— 可立即处理 CSV/TXT 数据、出图、写报表。")
    print("软件侧：未检测到光学仿真软件。")
    print("        -> 现在能做的：数据后处理、算法验证、脚本框架搭建（用 Mock 数据跑通）")
    print("        -> 暂不能做的：真实连 Zemax/CodeV/LightTools 读写设计文件")
else:
    print("详见上方各小节的 [缺失] / [未装] 项。")

print(SEP)
