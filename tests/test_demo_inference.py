# -*- coding: utf-8 -*-
"""内置演示权重的端到端推理验证。

对应修复项⑤（weights/demo 入库）+ ①（CWD 无关）的联合验收：
新克隆的仓库不跑任何训练，也必须能在诊断页得出可用的 Winter 结论。
这条测试真的加载 .pt 并推理，所以慢（数十秒），用 -m slow 可跳过。
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = REPO_ROOT / "dataset" / "images" / "trainset"
LAB_DIR = REPO_ROOT / "dataset" / "labels" / "trainset"

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def demo_weights():
    import dental_common as dc
    det = dc.DEMO_WEIGHTS_DIR / "expB-detect" / "weights" / "best.pt"
    cls = dc.DEMO_WEIGHTS_DIR / "expB-cls-angulation" / "weights" / "best.pt"
    if not det.is_file() or not cls.is_file():
        pytest.skip(f"内置演示权重缺失: {det}")
    if not IMG_DIR.is_dir() or not any(IMG_DIR.glob("*.jpg")):
        pytest.skip("dataset/images/trainset 不在本机，无法做推理验证")
    return det, cls


def _models(det_path, cls_path):
    from ultralytics import YOLO
    return YOLO(str(det_path)), YOLO(str(cls_path))


def _single_tooth_images(limit):
    """挑出只标了 1 颗智齿的图：GT 唯一，便于逐图核对角度。"""
    out = []
    for img in sorted(IMG_DIR.glob("*.jpg")):
        lab = LAB_DIR / f"{img.stem}.txt"
        if not lab.is_file():
            continue
        ids = [ln.split()[0] for ln in lab.read_text(encoding="utf-8").split("\n")
               if ln.strip()]
        if len(ids) == 1:
            out.append((img, int(ids[0])))
        if len(out) >= limit:
            break
    return out


def test_demo_weights_load_and_class_names_match_curriculum(demo_weights):
    """权重要能加载，且分类头的 5 个类名与课堂口径（0..4 顺序）一致。

    类名对不上时页面不会报错，只会把"近中"显示成"垂直"——静默错判，最危险。
    """
    from winter_train_lib import ANGLE_CLASS_ORDER
    det, clf = _models(*demo_weights)
    assert det.names, "检测权重没有类别名，可能不是有效的 YOLO 权重"
    assert sorted(clf.names.values()) == sorted(ANGLE_CLASS_ORDER), \
        f"分类头类名 {clf.names} 与课堂口径 {ANGLE_CLASS_ORDER} 不一致"


def test_end_to_end_diagnose_produces_valid_findings(demo_weights):
    """跑通 检测 → 裁切 → 角度分类 全链路，并校验输出结构合法。"""
    import cv2
    import dental_common as dc

    det, clf = _models(*demo_weights)
    img_path, _ = _single_tooth_images(1)[0]
    img = cv2.imread(str(img_path))
    assert img is not None, f"图片读不出来（可能损坏）: {img_path}"
    h, w = img.shape[:2]

    findings = dc.detect_and_classify(det, clf, img, crop_img=img,
                                      conf=0.25, margin=0.2, device="cpu")
    assert findings, f"{img_path.name} 未检出任何智齿，演示权重可能无效"
    for f in findings:
        x1, y1, x2, y2 = f["box"]
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h, f"框越界: {f['box']} vs {(w, h)}"
        assert 0.0 < f["det_conf"] <= 1.0
        assert f["angle_en"] in dc.WINTER_ANGLE_ZH, f"未知角度类别: {f['angle_en']}"
        assert 0.0 < f["angle_conf"] <= 1.0
        assert f["angle_en"] in dc.WINTER_ANGLE_COLORS, "缺颜色映射会导致画框回退成白色"


def test_demo_weights_agree_with_human_labels(demo_weights):
    """逐图核对人工标注：准确率必须显著高于瞎猜（5 类随机 = 20%）。

    阈值取 0.7 而非文档里的 0.948：文档指标来自 33 张 val 集，这里抽的是
    train 侧样本且张数少，留足余量避免抖动；但只要权重被换成了没训练过的
    初始化，这条一定挂。
    """
    from winter_train_lib import ANGLE_CLASS_ORDER
    import cv2
    import dental_common as dc

    det, clf = _models(*demo_weights)
    samples = _single_tooth_images(15)
    assert len(samples) >= 8, "单智齿样本太少，无法做准确率校验"

    hit = judged = 0
    mismatches = []
    for img_path, gt_id in samples:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        findings = dc.detect_and_classify(det, clf, img, crop_img=img,
                                          conf=0.25, margin=0.2, device="cpu")
        if len(findings) != 1:
            mismatches.append(f"{img_path.name}: GT 1 颗 / 检出 {len(findings)} 颗")
            continue
        judged += 1
        pred = findings[0]["angle_en"]
        if pred == ANGLE_CLASS_ORDER[gt_id]:
            hit += 1
        else:
            mismatches.append(f"{img_path.name}: GT {ANGLE_CLASS_ORDER[gt_id]} / 判 {pred}")

    assert judged >= 6, f"有效样本太少（检出数不等于 1 的太多）：{mismatches}"
    acc = hit / judged
    assert acc >= 0.7, f"演示权重准确率 {acc:.0%} 过低，可能不是训练好的权重：{mismatches}"


def test_detect_only_degrades_gracefully(demo_weights):
    """没有分类权重时只画检测框，不应崩，也不应编造角度。"""
    import cv2
    import dental_common as dc

    det, _ = _models(*demo_weights)
    img_path, _ = _single_tooth_images(1)[0]
    img = cv2.imread(str(img_path))
    findings = dc.detect_and_classify(det, None, img, crop_img=img, device="cpu")
    assert findings
    for f in findings:
        assert f["angle_en"] is None and f["angle_conf"] is None


def test_page_winter_entrypoint_works_with_demo_weights(demo_weights):
    """走页面真实入口 run_winter_diagnose，确认标注图与中文结论都能产出。"""
    import cv2
    import dental_common as dc
    from page_winter import run_winter_diagnose

    det_path, cls_path = demo_weights
    img_path, _ = _single_tooth_images(1)[0]
    img = cv2.imread(str(img_path))

    ok, annotated, teeth, err = run_winter_diagnose(img, str(det_path), str(cls_path))
    assert ok, f"诊断失败: {err}"
    assert err is None
    assert annotated is not None and annotated.shape[:2] == img.shape[:2]
    assert teeth and len(teeth) >= 1
    t = teeth[0]
    assert t["angle_zh"] in dc.WINTER_ANGLE_ZH.values()
    assert t["side"] in ("图中左侧", "图中右侧")


def test_metrics_json_matches_documented_numbers(demo_weights):
    """内置 metrics.json 的指标要和文档/下拉框显示的一致。"""
    import dental_common as dc
    m = json.loads((dc.DEMO_WEIGHTS_DIR / "expB" / "metrics.json")
                   .read_text(encoding="utf-8"))
    assert m["mAP50"] == pytest.approx(0.9943, abs=1e-3)
    assert m["angulation"]["top1"] == pytest.approx(0.9483, abs=1e-3)
    cm = m["angulation"]["confusion"]
    assert len(cm) == 6 and all(len(r) == 6 for r in cm), "混淆矩阵应为 5 类 + 背景"


def test_predict_demo_resolves_weights_from_any_cwd(tmp_path):
    """scripts/predict_demo.py 的默认权重路径必须与 CWD 无关。

    在仓库外的目录起子进程：以前 --det 默认 "runs/winter/expB-detect/..."
    会解析成 <cwd>/runs/... 而找不到文件。
    """
    repo = str(REPO_ROOT)
    scripts = str(REPO_ROOT / "scripts")
    code = (
        "import sys;"
        f"sys.path.insert(0, r'{repo}');"
        f"sys.path.insert(0, r'{scripts}');"
        "import predict_demo as pd;"
        "from pathlib import Path;"
        "print(pd._demo_weight('detect'));"
        "print(pd._demo_weight('cls-angulation'));"
        "print(Path(pd._demo_weight('detect')).is_file())"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(tmp_path),
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.strip().splitlines()
    assert lines[2] == "True", f"仓库外 CWD 下没能解析到可用的演示权重: {lines[:2]}"
    for p in lines[:2]:
        assert Path(p).is_absolute(), f"默认权重路径必须是绝对路径: {p}"
