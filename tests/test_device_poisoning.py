# -*- coding: utf-8 -*-
"""device=cpu 污染 CUDA_VISIBLE_DEVICES 的回归验证。

这是写测试时**新发现**的缺陷，不在原报告的 5 项里，但正好卡在课堂文档路径上：
学生在页面 3 用 CPU 微调（免费云 GPU 不可用时的推荐做法）→ 点"跳诊断页"→
页面 4/5 的推理在同一进程里以 `AssertionError: Invalid device id` 全部失败，
且只能重启 Streamlit 才能恢复。

根因：ultralytics 的 select_device 在 device="cpu" 时写
`os.environ["CUDA_VISIBLE_DEVICES"] = ""`，注释说是为了 force is_available()=False，
但 torch 2.7.1 上 is_available() 走 CUDA Runtime API（恒 True）、device_count() 走
NVML（返回 0），两者矛盾 → select_device 进 CUDA 分支 → get_device_properties(0) 越界。
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
KEY = "CUDA_VISIBLE_DEVICES"


def _cuda_present():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def test_guard_restores_env_var(monkeypatch):
    import dental_common as dc

    # 原本未设置：退出后必须仍然未设置，而不是留下一个 ""
    monkeypatch.delenv(KEY, raising=False)
    with dc.preserve_cuda_visible_devices():
        os.environ[KEY] = ""
    assert KEY not in os.environ

    # 原本已设置：退出后必须原样还原（多卡机器上用户可能自己指定了 "1"）
    monkeypatch.setenv(KEY, "1")
    with dc.preserve_cuda_visible_devices():
        os.environ[KEY] = ""
    assert os.environ[KEY] == "1"


def test_guard_restores_even_when_body_raises(monkeypatch):
    """训练中途报错也不能把污染留在进程里，否则错误会被第二个更难懂的错误掩盖。"""
    import dental_common as dc
    monkeypatch.setenv(KEY, "0")
    with pytest.raises(RuntimeError):
        with dc.preserve_cuda_visible_devices():
            os.environ[KEY] = ""
            raise RuntimeError("训练失败")
    assert os.environ[KEY] == "0"


def test_poisoned_env_makes_torch_self_inconsistent():
    """把根因固化下来：env="" 时 is_available() 与 device_count() 互相矛盾。

    必须在**子进程**里测：torch 的 device_count() 会把结果缓存进 `_cached_device_count`，
    但按源码注释只在 CUDA 初始化之后才缓存（"Do not cache the device count prior to
    CUDA initialization"）。所以在本进程里能否观察到 0，取决于前面哪些用例已经把
    CUDA 跑起来了 —— 顺序相关。新进程的 CUDA 状态是干净的，测量才可复现。

    这条一旦失败，说明 torch 换了行为（例如 is_available 改走 NVML），
    那么 select_device 的 cpu 分支就不再是隐患，本文件的其余断言需要重新评估。
    """
    if not _cuda_present():
        pytest.skip("本机无可用 CUDA，无法验证 is_available/device_count 的矛盾状态")

    repo = str(REPO_ROOT)
    code = (
        "import os, sys;"
        f"sys.path.insert(0, r'{repo}');"
        "import dental_common;"                  # 只为设置 KMP_DUPLICATE_LIB_OK
        "import torch;"
        "print(torch.cuda.is_available(), torch.cuda.device_count());"
        "os.environ['CUDA_VISIBLE_DEVICES'] = '';"
        "print(torch.cuda.is_available(), torch.cuda.device_count());"
        "os.environ.pop('CUDA_VISIBLE_DEVICES', None);"
        "print(torch.cuda.is_available(), torch.cuda.device_count())"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=repo,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    clean, poisoned, restored = (
        ln.split() for ln in proc.stdout.strip().splitlines()[-3:])

    assert clean == ["True", "1"], f"干净状态就不对: {clean}"
    # 矛盾点：Runtime API 说"有"，NVML 说"没有"。ultralytics 信前者，接着按下标 0
    # 取设备属性，而 torch 用后者做边界检查 → AssertionError: Invalid device id
    assert poisoned == ["True", "0"], f"env='' 时应出现 is_available/count 矛盾: {poisoned}"
    assert restored == ["True", "1"], f"还原环境变量后应立即恢复: {restored}"


@pytest.mark.slow
def test_cpu_inference_does_not_break_later_default_inference(demo_weights_pair):
    """端到端回归：先按 CPU 推理一次，再用默认设备推理，两次都必须成功。

    修复前第二次会以 AssertionError: Invalid device id 失败。
    """
    if not _cuda_present():
        pytest.skip("本机无可用 CUDA，无法复现该缺陷")
    import cv2
    import dental_common as dc
    from ultralytics import YOLO

    det_path, cls_path, img_path = demo_weights_pair
    img = cv2.imread(str(img_path))

    before = os.environ.get(KEY, None)
    det = YOLO(str(det_path))
    clf = YOLO(str(cls_path))
    first = dc.detect_and_classify(det, clf, img, crop_img=img, device="cpu")
    assert first, "CPU 推理本身应成功"
    assert os.environ.get(KEY, None) == before, \
        f"detect_and_classify 泄漏了 {KEY}: {before!r} -> {os.environ.get(KEY)!r}"

    det2 = YOLO(str(det_path))
    clf2 = YOLO(str(cls_path))
    second = dc.detect_and_classify(det2, clf2, img, crop_img=img)   # device=None
    assert second, "CPU 推理之后，默认设备推理不应再失败"
    assert all(f["angle_en"] is not None for f in second)


@pytest.mark.slow
def test_cpu_training_then_diagnose_page_survives(demo_weights_pair):
    """模拟课堂路径：页面 3 用 CPU 训练 → 页面 5 诊断。

    不真跑 epoch（太慢），而是走 finetune_winter 里同样的
    preserve_cuda_visible_devices + device='cpu' 组合，验证守卫确实生效。
    """
    if not _cuda_present():
        pytest.skip("本机无可用 CUDA，无法复现该缺陷")
    import dental_common as dc
    from ultralytics import YOLO

    det_path, _, img_path = demo_weights_pair
    before = os.environ.get(KEY, None)
    model = YOLO(str(det_path))
    with dc.preserve_cuda_visible_devices():
        model.predict(str(img_path), device="cpu", verbose=False)
    assert os.environ.get(KEY, None) == before

    import cv2
    img = cv2.imread(str(img_path))
    assert dc.detect_and_classify(YOLO(str(det_path)), None, img, crop_img=img), \
        "CPU 训练步骤之后，诊断页推理应仍然可用"


def test_all_inprocess_ultralytics_callsites_are_guarded():
    """回归守卫：所有会把 device 交给 ultralytics 的**进程内**调用都必须套上守卫。

    用 AST 而不是文本匹配：判断"在 with 块内"要看语法结构，看行距会误判。
    receiver 限定为 YOLO 对象（model/self.model/det/clf），这样就不会把
    `pipeline.train(...)` 这种调用我们自己已加守卫的封装方法也算进来。

    scripts/train_pretrain.py 与 scripts/auto_annotate.py 同样直接传 device，
    但它们是一次性 CLI 进程，污染随进程结束消失，故不在断言范围内。
    """
    import ast

    yolo_receivers = {"model", "self.model", "det", "clf"}
    methods = {"train", "val", "predict"}
    targets = ["dental_common.py", "dental_yolo_train.py", "scripts/finetune_winter.py"]

    for rel in targets:
        tree = ast.parse((REPO_ROOT / rel).read_text(encoding="utf-8"))

        guarded = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.With):
                continue
            is_guard = any(
                isinstance(item.context_expr, ast.Call)
                and getattr(item.context_expr.func, "id", "") == "preserve_cuda_visible_devices"
                for item in node.items)
            if is_guard:
                guarded.update(id(sub) for sub in ast.walk(node))

        found, unguarded = 0, []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in methods:
                continue
            if ast.unparse(node.func.value) not in yolo_receivers:
                continue
            found += 1
            if id(node) not in guarded:
                unguarded.append(f"{rel}:{node.lineno} {ast.unparse(node.func)}(...)")

        assert found, f"{rel} 里一个 ultralytics 调用都没扫到，receiver/方法名是否改了？"
        assert not unguarded, "以下调用未被 preserve_cuda_visible_devices 包住:\n  " + \
            "\n  ".join(unguarded)


@pytest.fixture(scope="module")
def demo_weights_pair():
    import dental_common as dc
    det = dc.DEMO_WEIGHTS_DIR / "expB-detect" / "weights" / "best.pt"
    cls = dc.DEMO_WEIGHTS_DIR / "expB-cls-angulation" / "weights" / "best.pt"
    img_dir = REPO_ROOT / "dataset" / "images" / "trainset"
    imgs = sorted(img_dir.glob("*.jpg")) if img_dir.is_dir() else []
    if not det.is_file() or not cls.is_file() or not imgs:
        pytest.skip("缺少内置演示权重或 dataset 图片")
    return det, cls, imgs[0]
