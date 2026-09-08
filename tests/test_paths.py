# -*- coding: utf-8 -*-
"""路径基准与仓库自带资源的验证。

对应修复项①（CWD 路径依赖）与⑤（演示权重入库）。
核心断言：把进程 CWD 换到仓库之外，所有仓库内资源仍能被找到。
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_repo_root_points_at_repository():
    import dental_common as dc
    assert dc.REPO_ROOT == REPO_ROOT
    # 这几个标志文件同时存在，才能确认 REPO_ROOT 真的指到了仓库根而不是别处
    for marker in ("web_ui.py", "run.py", "requirements.txt", "dataset"):
        assert (dc.REPO_ROOT / marker).exists(), f"缺少标志文件 {marker}"


def test_derived_constants_are_absolute_and_inside_repo():
    import dental_common as dc
    for name in ("RUNS_DIR", "WINTER_RUNS_DIR", "DEMO_WEIGHTS_DIR"):
        p = getattr(dc, name)
        assert p.is_absolute(), f"{name} 必须是绝对路径，否则又回到依赖 CWD 的老问题"
        assert str(p).startswith(str(dc.REPO_ROOT)), f"{name} 应位于仓库内"


def test_demo_weights_ship_with_repository():
    """新克隆（runs/ 为空）时必须仍有可用的演示权重，否则诊断页下拉是空的。"""
    import dental_common as dc
    det = dc.DEMO_WEIGHTS_DIR / "expB-detect" / "weights" / "best.pt"
    cls = dc.DEMO_WEIGHTS_DIR / "expB-cls-angulation" / "weights" / "best.pt"
    metrics = dc.DEMO_WEIGHTS_DIR / "expB" / "metrics.json"
    for p in (det, cls, metrics):
        assert p.is_file(), f"内置演示权重缺失: {p}"
    assert det.stat().st_size > 1_000_000, "检测权重体积异常，可能不是完整的 .pt"


def test_demo_weights_discoverable_from_foreign_cwd(tmp_path):
    """只扫 weights/demo，模拟新克隆环境（runs/winter 不存在）。"""
    import ui_common as uc
    exps = uc.list_winter_experiments([uc.DEMO_WEIGHTS_DIR])
    assert [e["exp"] for e in exps] == ["expB"]
    e = exps[0]
    assert e["is_demo"] is True
    assert e["det_path"] and Path(e["det_path"]).is_file()
    assert e["cls_path"] and Path(e["cls_path"]).is_file()
    # metrics.json 里的指标要能带出来，下拉框才有 mAP/top1 可显示
    assert e["mAP50"] == pytest.approx(0.9943, abs=1e-3)
    assert e["top1"] == pytest.approx(0.9483, abs=1e-3)


def test_local_runs_take_priority_over_bundled_demo():
    """同名实验应优先用本机训练产物，而不是被内置演示覆盖。"""
    import ui_common as uc
    if not (uc.WINTER_RUNS_DIR / "expB-detect").is_dir():
        pytest.skip("本机没有 runs/winter/expB-detect，无法验证优先级")
    exps = {e["exp"]: e for e in uc.list_winter_experiments()}
    assert exps["expB"]["is_demo"] is False
    assert str(uc.WINTER_RUNS_DIR) in exps["expB"]["det_path"]


def test_all_paths_resolve_when_cwd_is_outside_repo(tmp_path):
    """在仓库外的 CWD 起一个子进程，验证路径解析与 CWD 无关。

    这是修复项①的端到端证明：以前从别的目录启动 streamlit，
    "runs/winter" 之类相对路径会全部落空且不报错。
    """
    code = (
        "import sys, os;"
        f"sys.path.insert(0, r'{REPO_ROOT}');"
        "import dental_common as dc, ui_common as uc;"
        "print(os.getcwd());"
        "print(dc.DEMO_WEIGHTS_DIR.is_dir());"
        "print(len(uc.list_winter_experiments([uc.DEMO_WEIGHTS_DIR])));"
        "print(uc.DEFAULT_DETECT_RUNS_DIR.is_absolute())"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(tmp_path), capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.strip().splitlines()
    assert Path(lines[0]) == tmp_path.resolve(), "子进程 CWD 应确实在仓库之外"
    assert lines[1] == "True", "仓库外 CWD 下仍应找到 weights/demo"
    assert lines[2] == "1", "仓库外 CWD 下仍应扫到内置 expB"
    assert lines[3] == "True"


def test_run_py_chdirs_to_repo_root(tmp_path):
    """run.py 一被加载就应把 CWD 切到仓库根目录。"""
    run_py = str(REPO_ROOT / "run.py")
    code = (
        "import os, sys, importlib.util;"
        f"spec = importlib.util.spec_from_file_location('runmod', r'{run_py}');"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m);"
        "print(os.getcwd())"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(tmp_path), capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert Path(proc.stdout.strip()) == REPO_ROOT


def test_page_modules_import_cleanly():
    """五个页面模块 + 入口都应能导入（web_ui 会调 set_page_config，单独跳过）。"""
    import importlib
    for mod in ("ui_common", "page_diagnose", "page_monitor",
                "page_train", "page_winter", "page_winter_train"):
        importlib.import_module(mod)


def test_finetune_winter_repo_path_helper():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import finetune_winter as fw
    finally:
        sys.path.pop(0)
    assert fw.REPO_ROOT == REPO_ROOT
    # 相对路径应被解析到仓库内，绝对路径应原样保留
    assert fw.repo_path("yolov8n.pt") == REPO_ROOT / "yolov8n.pt"
    assert fw.repo_path(str(REPO_ROOT / "x.pt")) == REPO_ROOT / "x.pt"
    # 仓库自带的 COCO 初始权重必须在（实验 A 基线依赖它）
    assert (REPO_ROOT / "yolov8n.pt").is_file()
    assert (REPO_ROOT / "yolov8n-cls.pt").is_file()


def test_no_cwd_relative_weight_literals_left():
    """回归守卫：已修过的几处不应再出现裸的相对路径字面量。"""
    checks = {
        "page_winter.py": ['list_winter_experiments("runs/winter")'],
        "page_winter_train.py": ['Path("runs/winter")'],
        "page_diagnose.py": ['"./runs/detect/results"'],
        "page_monitor.py": ['"./runs/detect/results"'],
        "dental_yolo_train.py": ["val: trainset"],
    }
    for fname, needles in checks.items():
        text = (REPO_ROOT / fname).read_text(encoding="utf-8")
        for needle in needles:
            assert needle not in text, f"{fname} 里仍存在应被移除的相对路径/错误配置: {needle}"
