# -*- coding: utf-8 -*-
"""依赖清单与 run.py 版本校验的验证。

对应修复项③。旧 run.py 写死一份不带版本号的清单，缺失时 `pip install torch`
会拉最新版；本项目锁的是 ultralytics 8.3.233 + streamlit 1.30.0，静默升级可能
让页面读指标拿到 None。这里断言：清单本身全部锁版本、本机能全部满足、
版本不符时只警告不动手、缺失时按锁定版本安装。
"""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_PY = REPO_ROOT / "run.py"


@pytest.fixture(scope="module")
def runmod():
    """按文件路径加载 run.py（它不在包里，且模块级会 os.chdir）。

    chdir 到仓库根对本测试无害，但仍保存/恢复，免得影响同批次的其他用例。
    """
    spec = importlib.util.spec_from_file_location("_run_under_test", RUN_PY)
    mod = importlib.util.module_from_spec(spec)
    old_cwd = Path.cwd()
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        import os
        os.chdir(str(old_cwd))


def test_requirements_file_exists_and_is_fully_pinned(runmod):
    reqs = runmod.load_requirements()
    assert len(reqs) >= 12, "清单条目异常偏少，可能解析逻辑坏了"
    unpinned = [p for p, spec in reqs if not spec]
    assert not unpinned, f"这些依赖没锁版本，装的时候会拉最新: {unpinned}"
    for pkg, spec in reqs:
        assert spec.startswith("=="), f"{pkg} 应使用 == 精确锁定，实际是 {spec}"
    names = {p.lower() for p, _ in reqs}
    # 四个核心依赖必须在清单里，少一个就说明清单和实际 import 脱节了
    for core in ("torch", "ultralytics", "streamlit", "opencv-python"):
        assert core in names, f"核心依赖 {core} 不在 requirements.txt 里"


def test_every_pinned_requirement_is_installed_and_satisfied(runmod):
    """本机真实环境校验 —— 这就是"确保完全能用"的那一条。"""
    problems = []
    for pkg, spec in runmod.load_requirements():
        installed = runmod._installed_version(pkg)
        if installed is None:
            problems.append(f"{pkg}{spec} 未安装")
        elif not runmod._satisfies(installed, spec):
            problems.append(f"{pkg} 已装 {installed}，不满足 {spec}")
    assert not problems, "依赖未就绪：\n" + "\n".join(problems)


def test_satisfies_accepts_pep440_local_version(runmod):
    """torch 装的是 2.7.1+cu126，`==2.7.1` 必须判为满足，否则每次启动都误报。"""
    assert runmod._satisfies("2.7.1+cu126", "==2.7.1") is True
    assert runmod._satisfies("2.7.1", "==2.7.1") is True
    assert runmod._satisfies("2.8.0", "==2.7.1") is False
    assert runmod._satisfies("2.7.1+cu126", None) is True


def test_satisfies_falls_back_without_packaging(runmod, monkeypatch):
    """packaging 不可用时退化比较，仍要正确处理 +cu126 这类 local 标签。"""
    monkeypatch.setitem(sys.modules, "packaging.specifiers", None)
    assert runmod._satisfies("2.7.1+cu126", "==2.7.1") is True
    assert runmod._satisfies("9.9.9", "==2.7.1") is False


def test_min_python_matches_actual_pep604_usage(runmod):
    """MIN_PYTHON 不能低于代码真实需要：page_winter / finetune_winter 在签名里
    直接写了运行时求值的 `str | None`，3.9 及以下导入即 TypeError。"""
    assert runmod.MIN_PYTHON == (3, 10)
    assert sys.version_info >= runmod.MIN_PYTHON
    for path in (REPO_ROOT / "page_winter.py",
                 REPO_ROOT / "scripts" / "finetune_winter.py"):
        text = path.read_text(encoding="utf-8")
        assert "| None" in text, \
            f"{path.name} 已不再使用 PEP 604 联合类型，MIN_PYTHON 或许可以放宽"
        assert "from __future__ import annotations" not in text, \
            f"{path.name} 加了 __future__ annotations，MIN_PYTHON 的下限依据已变化"


def test_force_utf8_stdout_survives_gbk_console(runmod):
    """Windows 控制台/重定向管道下 print('✓') 会抛 UnicodeEncodeError 直接中止启动。

    这里只查调用形状；"重定向后真的逐行落盘"由 tests/test_console_encoding.py
    的 test_redirected_output_is_line_buffered 用子进程实测。
    """
    class FakeStream:
        def __init__(self):
            self.kwargs = None

        def reconfigure(self, **kw):
            self.kwargs = kw

    class NoReconfigure:
        pass

    out, err = FakeStream(), FakeStream()
    monkey_streams = (out, err)
    orig = (sys.stdout, sys.stderr)
    try:
        sys.stdout, sys.stderr = monkey_streams
        runmod._force_utf8_stdout()
    finally:
        sys.stdout, sys.stderr = orig
    for s in monkey_streams:
        assert s.kwargs == {"encoding": "utf-8", "errors": "replace",
                            "line_buffering": True}

    # 没有 reconfigure 的流（如被 capsys 替换）必须被静默跳过而不是抛错
    sys_out_orig = sys.stdout
    try:
        sys.stdout = NoReconfigure()
        runmod._force_utf8_stdout()
    finally:
        sys.stdout = sys_out_orig


def test_version_mismatch_warns_but_never_touches_env(runmod, monkeypatch, capsys):
    """版本不符只警告：自动升级可能把当前可用环境搞坏，且很难回退。"""
    calls = []
    monkeypatch.setattr(runmod.subprocess, "check_call",
                        lambda cmd, *a, **k: calls.append(cmd))
    monkeypatch.setattr(runmod, "_installed_version", lambda pkg: "0.0.1")
    runmod.check_and_install_requirements()
    out = capsys.readouterr().out
    assert calls == [], "版本不符时不应执行任何 pip 安装"
    assert "版本不符" in out and "不自动改动" in out
    assert "环境完美" not in out


def test_missing_package_is_installed_at_pinned_version(runmod, monkeypatch, capsys):
    """缺失时安装的必须是锁定版本，而不是裸包名（裸包名 = 拉最新）。"""
    calls = []
    monkeypatch.setattr(runmod.subprocess, "check_call",
                        lambda cmd, *a, **k: calls.append(cmd) or 0)
    monkeypatch.setattr(runmod, "_installed_version",
                        lambda pkg: None if pkg == "plotly" else "0.0.1")
    monkeypatch.setattr(runmod, "_satisfies", lambda installed, spec: True)
    monkeypatch.setattr(runmod.sys.stdin, "isatty", lambda: False, raising=False)

    runmod.check_and_install_requirements()
    out = capsys.readouterr().out

    assert len(calls) == 1, f"应恰好触发一次安装，实际 {len(calls)} 次"
    cmd = calls[0]
    assert cmd[0] == sys.executable and "pip" in cmd and "install" in cmd
    assert "plotly==5.9.0" in cmd, f"安装命令未带锁定版本: {cmd}"
    assert "plotly" not in cmd, f"不应出现裸包名（会拉最新版）: {cmd}"
    assert runmod.PIP_MIRROR in cmd
    assert "plotly==5.9.0" in out


def test_run_py_has_no_leftover_unpinned_logic(runmod):
    """回归守卫：旧版的死代码与写死清单不应再出现。"""
    text = RUN_PY.read_text(encoding="utf-8")
    assert "_IMPORT_NAME" not in text, "旧版的 import 名映射表是死代码，应已删除"
    assert "def check_package(" not in text, "旧版的 check_package 已被 load_requirements 取代"
    assert 'os.chdir(REPO_DIR)' in text, "run.py 必须切到仓库根目录，否则相对路径全部落空"
    assert "requirements.txt" in text
