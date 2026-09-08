# -*- coding: utf-8 -*-
"""Windows GBK 控制台/管道下的输出编码验证。

这是本机实测时连续撞到的两个启动级崩溃：
  - run.py 打印 ✓ 时 UnicodeEncodeError，直接起不来；
  - 更隐蔽的一次是页面 3 微调跑到一半，scripts/prepare_winter_dataset.py
    print 一个 ⚠ 就崩在训练线程里，页面只显示一个和训练无关的编码错误。

修复点：run.py 自带 _force_utf8_stdout()；dental_common 导入时调 force_utf8_stdio()，
覆盖文档里的另一种启动方式 `streamlit run web_ui.py`（不经过 run.py）。

所有用例都在子进程里跑并显式设 PYTHONIOENCODING=gbk，这样无论宿主机 locale 是什么，
都能稳定复现"敌意控制台"，也不会因为父进程已经是 UTF-8 而变成空过。
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _gbk_codec_available() -> bool:
    """gbk 编解码器是否存在。非中文 locale 的系统上可能没有，此时
    PYTHONIOENCODING=gbk 会让解释器启动即 LookupError，复现不出目标缺陷。"""
    try:
        "✓".encode("gbk")
    except UnicodeEncodeError:
        return True          # 编不了这个字符，但 codec 本身在 —— 正是我们要的条件
    except LookupError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _gbk_codec_available(), reason="本机没有 gbk 编解码器，无法复现 GBK 控制台")


def _run(code: str, cwd=None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "gbk"      # 强制复现 GBK 控制台
    env.pop("PYTHONUTF8", None)          # 否则 UTF-8 模式会让测试空过
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    return subprocess.run([sys.executable, "-c", code], cwd=str(cwd or REPO_ROOT),
                          capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env,
                          stdin=subprocess.DEVNULL)   # 避免依赖检查卡在交互提示上


def test_gbk_console_baseline_really_crashes():
    """基线对照：不导入 dental_common 时，GBK 管道下 print('✓') 必然崩。

    保留这条是为了证明其余用例不是空过 —— 如果哪天它不再崩，说明宿主机环境
    变了（例如 Python 默认启用 UTF-8 模式），本文件的断言就得重新评估。
    """
    proc = _run("print('\\u2713')")
    assert proc.returncode != 0, "基线竟然没崩，测试环境已不是 GBK，其余断言不可信"
    assert "UnicodeEncodeError" in proc.stderr


def test_dental_common_switches_stdio_to_utf8():
    repo = str(REPO_ROOT)
    proc = _run(
        "import sys;"
        f"sys.path.insert(0, r'{repo}');"
        "import dental_common;"
        "print(sys.stdout.encoding, sys.stderr.encoding);"
        "print('\\u2713 \\u26a0 \\U0001f9b7 \\u274c \\u2705')"
    )
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.strip().splitlines()
    assert lines[0].split() == ["utf-8", "utf-8"], f"stdout/stderr 未切到 UTF-8: {lines[0]}"
    assert "✓" in lines[1] and "⚠" in lines[1] and "🦷" in lines[1]


def test_force_utf8_stdio_tolerates_streams_without_reconfigure():
    """被 pytest/Streamlit 替换过的流没有 reconfigure 方法，必须静默跳过而不是抛错。"""
    import dental_common as dc

    class NoReconfigure:
        pass

    orig = (sys.stdout, sys.stderr)
    try:
        sys.stdout, sys.stderr = NoReconfigure(), NoReconfigure()
        dc.force_utf8_stdio()          # 不应抛异常
    finally:
        sys.stdout, sys.stderr = orig


def test_documented_cli_scripts_survive_gbk_console():
    """README 里点名的两个命令行入口，导入后打印 ✓ 不应崩。

    这两个脚本原本不 import dental_common，是实测崩溃后才补上的引导。
    """
    repo, scripts = str(REPO_ROOT), str(REPO_ROOT / "scripts")
    for mod in ("prepare_winter_dataset", "train_pretrain"):
        proc = _run(
            "import sys;"
            f"sys.path.insert(0, r'{scripts}');"
            f"import {mod};"
            "print('\\u2713 " + mod + " ok')"
        )
        assert proc.returncode == 0, f"{mod} 在 GBK 控制台下崩溃:\n{proc.stderr}"
        assert f"✓ {mod} ok" in proc.stdout


def test_run_py_launcher_survives_gbk_console():
    """run.py 自己的 _force_utf8_stdout 必须生效（它是用户敲的第一条命令）。"""
    run_py = str(REPO_ROOT / "run.py")
    proc = _run(
        "import importlib.util;"
        f"spec = importlib.util.spec_from_file_location('runmod', r'{run_py}');"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m);"
        "m._force_utf8_stdout();"
        "print('\\u2713 \\U0001f9b7 \\u7259\\u79d1 AI')"
    )
    assert proc.returncode == 0, proc.stderr
    assert "✓ 🦷 牙科 AI" in proc.stdout


def test_run_py_dependency_check_prints_under_gbk():
    """端到端：真的跑一次依赖检查，确认 14 个 ✓ 都能打出来且不触发安装。"""
    run_py = str(REPO_ROOT / "run.py")
    proc = _run(
        "import importlib.util, sys;"
        f"spec = importlib.util.spec_from_file_location('runmod', r'{run_py}');"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m);"
        "m._force_utf8_stdout();"
        "m.check_and_install_requirements()"
    )
    assert proc.returncode == 0, proc.stderr
    assert "环境完美" in proc.stdout, f"依赖检查未通过:\n{proc.stdout}"
    assert proc.stdout.count("✓") >= 12


RUN_PY = str(REPO_ROOT / "run.py")


@pytest.mark.parametrize("bootstrap, flushed", [
    ("pass", False),                                     # 基线：什么都不装，就是块缓冲
    ("\n".join([
        "import importlib.util",
        f"_s = importlib.util.spec_from_file_location('runmod', r'{RUN_PY}')",
        "_m = importlib.util.module_from_spec(_s)",
        "_s.loader.exec_module(_m)",
        "_m._force_utf8_stdout()",
     ]), True),
    (f"sys.path.insert(0, r'{REPO_ROOT}')\nimport dental_common", True),
], ids=["baseline-block-buffered", "run.py", "dental_common"])
def test_redirected_output_is_line_buffered(tmp_path, bootstrap, flushed):
    """重定向到文件时必须逐行落盘，而不是等进程退出才一次性写出。

    云服务器/机房部署的文档写法是 `python run.py > log`，此时 stdout 不是 tty，
    Python 默认块缓冲：依赖检查和训练进度全都攒在缓冲区里，教师盯着一个空日志
    以为卡死了（实测过，日志里只有 streamlit 自己的几行）。

    baseline 那条用例故意不装任何修复，用来证明"没落盘"确实是块缓冲造成的，
    而不是子进程起不来/写不进去 —— 否则另外两条可能是空过。
    标记用纯 ASCII，避免把编码问题混进缓冲断言里。
    """
    log = tmp_path / "out.log"
    code = "\n".join([
        "import sys, time",
        bootstrap,
        "print('MARKER-FLUSH')",
        "time.sleep(30)",
    ])
    with open(log, "w", encoding="utf-8") as fh:
        child = subprocess.Popen([sys.executable, "-c", code], cwd=str(REPO_ROOT),
                                 stdout=fh, stderr=subprocess.DEVNULL,
                                 stdin=subprocess.DEVNULL)
    try:
        def _seen() -> bool:
            return "MARKER-FLUSH" in log.read_text(encoding="utf-8", errors="replace")

        if not flushed:
            time.sleep(3)
            assert child.poll() is None, "基线子进程提前退出了，无法证明块缓冲"
            assert not _seen(), "基线竟然已落盘，说明本机默认就是行缓冲，另两条断言不可信"
            return
        deadline = time.time() + 15
        while time.time() < deadline:
            if _seen():
                return                       # 进程还活着，内容已经落盘
            time.sleep(0.2)
        pytest.fail(f"子进程仍在运行但日志为空 —— 输出被块缓冲了:\n{log.read_text(errors='replace')!r}")
    finally:
        child.kill()
        child.wait(timeout=10)
