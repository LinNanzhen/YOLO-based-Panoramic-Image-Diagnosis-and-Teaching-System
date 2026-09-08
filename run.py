import os
import re
import sys
import subprocess

# 切到脚本所在目录再干任何事。
# 原因：各页面用 "runs/winter"、"dataset/..." 这类相对路径找权重和数据，
# 从别的目录启动（常见于 systemd、云平台的启动命令、双击快捷方式）时
# 这些路径全部落空，表现为"下拉框里没有可选模型"而不是明确报错，极难排查。
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# 3.10 是硬下限：page_winter.py / scripts/finetune_winter.py 的函数签名里
# 直接写了 PEP 604 的 `str | None`，且没有 from __future__ import annotations，
# 导入时即求值，3.9 及以下会抛 TypeError。
MIN_PYTHON = (3, 10)
REQUIREMENTS_FILE = os.path.join(REPO_DIR, "requirements.txt")
PIP_MIRROR = "https://mirrors.aliyun.com/pypi/simple/"


def _force_utf8_stdout():
    """把 stdout/stderr 切到 UTF-8。

    Windows 控制台默认代码页是 GBK，而本脚本大量使用 ✓ ❌ ⚠ 🦷 等字符，
    一旦 stdout 不是 UTF-8（例如被重定向到管道、或由服务/计划任务拉起），
    print 就会抛 UnicodeEncodeError 当场中止启动。errors="replace" 保证
    即便终端字体不支持 emoji，也只是显示成方块而不会崩。

    顺带打开行缓冲：stdout 被重定向到文件时 Python 默认是块缓冲，
    `python run.py > log` 会在进程退出前一行都看不到（云服务器/机房部署
    正是这么起的），依赖检查跑着却像卡死了。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except (ValueError, OSError):
            pass

def _check_python():
    """解释器版本守卫。

    本机/服务器上 `python` 可能指向 Python 2（例如 MGLTools 自带的 2.7），
    而本项目的 f-string 与 pathlib 用法在 Python 2 下是 SyntaxError —— 直接
    崩在最外层，用户看不到任何有用信息。这里提前给一句能照做的提示。
    """
    if sys.version_info >= MIN_PYTHON:
        return
    want = ".".join(str(x) for x in MIN_PYTHON)
    print(f"❌ 当前解释器是 Python {sys.version.split()[0]}（{sys.executable}）")
    print(f"   本项目需要 Python {want}+。")
    print("   请用正确的解释器重新启动，例如：")
    print("     D:\\Anaconda\\python.exe run.py")
    print("   或先激活对应 conda 环境再运行 python run.py")
    sys.exit(1)


def load_requirements(path=REQUIREMENTS_FILE):
    """解析 requirements.txt -> [(发行包名, 版本约束或 None), ...]"""
    reqs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            m = re.match(r"^([A-Za-z0-9_.\-]+)\s*(.*)$", line)
            if not m:
                continue
            reqs.append((m.group(1), m.group(2).strip() or None))
    return reqs


def _installed_version(pkg):
    try:
        from importlib import metadata
    except ImportError:
        import importlib_metadata as metadata
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def _satisfies(installed, spec):
    """已装版本是否满足约束。优先用 packaging 走标准 PEP 440 语义
    （这样 torch==2.7.1 能正确匹配已装的 2.7.1+cu126），
    packaging 不可用时退化为比较去掉 local 标签的主版本。"""
    if not spec:
        return True
    try:
        from packaging.specifiers import SpecifierSet
        return installed in SpecifierSet(spec)
    except ImportError:
        pass
    m = re.match(r"^[=<>!~]+\s*(.+)$", spec)
    want = (m.group(1) if m else spec).strip()
    return installed.split("+")[0] == want.split("+")[0]


def check_and_install_requirements():
    """按 requirements.txt 核对依赖：缺的装（带固定版本），版本不符的只警告。

    与旧版的区别：旧版写死一份不带版本号的清单，缺失时 `pip install torch`
    会拉最新版。本项目锁的是 ultralytics 8.3.233 + streamlit 1.30.0，
    静默升级可能让页面读指标静默拿到 None（字段名变更），且很难回退。
    镜像用阿里云源（默认 pypi.org 与清华源在当前网络下经常不可达）。
    """
    print("检查依赖环境...")

    if not os.path.exists(REQUIREMENTS_FILE):
        print(f"⚠ 找不到 {REQUIREMENTS_FILE}，跳过依赖检查")
        return

    reqs = load_requirements()
    to_install = []
    mismatched = []

    for pkg, spec in reqs:
        installed = _installed_version(pkg)
        label = pkg if not spec else f"{pkg}{spec}"
        if installed is None:
            print(f"❌ {label} 未安装")
            to_install.append(label)
            continue
        if not _satisfies(installed, spec):
            print(f"⚠ {pkg} 版本不符：已装 {installed}，建议 {spec}")
            mismatched.append((pkg, installed, spec))
        else:
            print(f"✓ {label} 已安装（{installed}）")

    if mismatched:
        print("-" * 50)
        print(f"⚠ {len(mismatched)} 个包版本与 requirements.txt 不一致：")
        for pkg, installed, spec in mismatched:
            print(f"    {pkg}: 已装 {installed} / 建议 {spec}")
        print("  不自动改动，以免破坏当前可用环境。确需对齐请手动执行：")
        print(f"    pip install -r requirements.txt -i {PIP_MIRROR}")
        print("-" * 50)

    if not to_install:
        if not mismatched:
            print("\n✅ 环境完美，所有依赖已就绪！")
        print("=" * 60)
        return

    print(f"\n检测到 {len(to_install)} 个缺失的库: {', '.join(to_install)}")
    choice = ""
    if sys.stdin.isatty():
        choice = input("是否从阿里云镜像安装? (y/n, 默认: y): ").strip().lower()
    if choice and choice != "y":
        print("已跳过安装。")
        return

    print("-" * 50)
    cmd = [sys.executable, "-m", "pip", "install"] + to_install + ["-i", PIP_MIRROR]
    try:
        print("正在执行安装命令...")
        subprocess.check_call(cmd)
        print("\n✅ 所有依赖安装完成！")
    except subprocess.CalledProcessError:
        print("\n❌ 安装失败。建议手动运行以下命令安装:")
        print(f"pip install {' '.join(to_install)} -i {PIP_MIRROR}")
        sys.exit(1)

    cuda_pkgs = [p for p, _ in reqs if p.lower() in ("torch", "torchvision")]
    if cuda_pkgs:
        print("提示：默认源装的是 CPU 版 torch。需要 CUDA 请改装官方轮子：")
        print("  pip install torch==2.7.1 torchvision==0.22.1 "
              "--index-url https://download.pytorch.org/whl/cu126")
    print("=" * 60)


def run_app():
    """启动 Streamlit 应用"""
    print("\n" + "=" * 60)
    print("🦷 牙科 AI 教学平台正在启动...")
    print("=" * 60)

    app_path = os.path.join(REPO_DIR, "web_ui.py")
    if not os.path.exists(app_path):
        print(f"❌ 错误: 找不到 {app_path}")
        print("请确保 web_ui.py 与 run.py 在同一目录下")
        return

    port = os.environ.get("PORT", "8501")

    print("🚀 服务即将启动！")
    print(f"🌍 请在浏览器访问提供的 Network URL (通常是 http://<云服务器IP>:{port})")
    print("=" * 60)

    cmd = [sys.executable, "-m", "streamlit", "run", app_path,
           "--server.port", port, "--server.address", "0.0.0.0"]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n🛑 服务已停止")


if __name__ == "__main__":
    _force_utf8_stdout()
    _check_python()
    try:
        check_and_install_requirements()
        run_app()
    except KeyboardInterrupt:
        print("\n程序已退出")
