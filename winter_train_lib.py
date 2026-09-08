# -*- coding: utf-8 -*-
"""
winter_train_lib.py — Web/脚本共享的 Winter 微调编排库

把"整理数据集 → 检测微调 → 角度分类微调 → 指标汇总"串成可被 Web 页面直接调用的
函数，逻辑沿用 scripts/prepare_winter_dataset.py 与 scripts/finetune_winter.py，
不重复实现训练本身。

典型用法（页面侧）:
    from winter_train_lib import run_winter_finetune
    metrics = run_winter_finetune(
        images_dir=..., labels_dir=..., exp="exp_20210001",
        epochs=60, batch=16, cls_epochs=60, device="0")

产出（沿用 scripts 的目录约定）:
    runs/winter/<exp>-detect/weights/best.pt          智齿检测权重
    runs/winter/<exp>-cls-angulation/weights/best.pt  角度 5 类分类权重
    runs/winter/<exp>/metrics.json                    检测 mAP + 角度 top1/混淆矩阵
    runs/winter/<exp>_data/                           整理后的数据集（detect + cls/angulation）

类别口径（与 dataset/combos_angulation.csv 一致）: 0..4 = Mesioangular, Vertical,
Inverted, Horizontal, Buccolingual —— 只标"角度"一维。
"""
from __future__ import annotations

import json
import os
import re
import sys
from argparse import Namespace
from pathlib import Path

_LIB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_LIB_DIR))  # 保证以任意 CWD 都能 import scripts.*

# 默认输出/数据/组合映射（基于本文件位置，不依赖启动目录）
DEFAULT_OUT_ROOT = _LIB_DIR / "runs" / "winter"
DEFAULT_COMBOS_CSV = _LIB_DIR / "dataset" / "combos_angulation.csv"
PRETRAIN_WEIGHTS = _LIB_DIR / "runs" / "pretrain" / "weights" / "best.pt"

# 角度 5 类固定顺序（同 dataset/class_list_annotation.txt）——顺序即标签 ID，不能乱
ANGLE_CLASS_ORDER = ["Mesioangular", "Vertical", "Inverted", "Horizontal", "Buccolingual"]
FALLBACK_ANGULATION_COMBOS = {
    i: (name, "-", "-", name) for i, name in enumerate(ANGLE_CLASS_ORDER)
}

# ultralytics 训练/推理的模型尺寸字母
_SIZE_LETTER = {"nano": "n", "small": "s", "medium": "m"}


def safe_exp(user_id: str, prefix: str = "exp_") -> str:
    """把学号/ID 转成安全实验名，如 20210001 -> exp_20210001（已带/不带 exp_ 均可）。"""
    s = str(user_id or "").strip()
    if s.startswith(prefix):
        s = s[len(prefix):]
    s = re.sub(r"[^0-9A-Za-z_.-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._-")
    if not s:
        s = "student"
    return prefix + s


def load_angulation_combos(combos_csv: str | Path | None = None) -> dict[int, tuple]:
    """加载 5 类角度组合映射。CSV 缺失/损坏时回退到代码内固定顺序常量（并提示）。"""
    path = Path(combos_csv) if combos_csv else DEFAULT_COMBOS_CSV
    if path.is_file():
        try:
            from scripts.prepare_winter_dataset import load_custom_combos
            combos = load_custom_combos(path)
            if len(combos) == 5 and all(
                    combos[i][3] == ANGLE_CLASS_ORDER[i] for i in range(5)):
                return combos
            print(f"[winter_train_lib] 注意：{path.name} 与角度 5 类顺序不一致，改用内置顺序")
        except Exception as e:
            print(f"[winter_train_lib] 读取 {path} 失败（{e}），改用内置 5 类顺序")
    else:
        print(f"[winter_train_lib] 未找到 {path}，改用内置 5 类顺序")
    return dict(FALLBACK_ANGULATION_COMBOS)


def inspect_dataset(images_dir: str | Path, labels_dir: str | Path,
                    combos: dict | None = None) -> dict:
    """轻量预检：图片-标签配对 + 每类框数 + 格式问题，供上传后预览/报错。

    返回: {ok, n_images, n_boxes, by_class_id: {0..4: 数量}, missing: [图无标签],
          orphan: [标签无图], errors: [格式错误消息]}。不抛异常。
    """
    from scripts.prepare_winter_dataset import IMG_SUFFIXES, parse_label

    combos = combos if combos is not None else load_angulation_combos()
    img_dir, lb_dir = Path(images_dir), Path(labels_dir)
    result = {"ok": False, "n_images": 0, "n_boxes": 0,
              "by_class_id": {}, "missing": [], "orphan": [], "errors": []}
    if not img_dir.is_dir() or not lb_dir.is_dir():
        result["errors"].append(f"目录不存在: 图={img_dir} 标签={lb_dir}")
        return result

    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_SUFFIXES)
    lbs = {p.stem: p for p in lb_dir.iterdir() if p.suffix.lower() == ".txt"}
    result["n_images"] = len(imgs)
    for p in imgs:
        if p.stem not in lbs:
            result["missing"].append(p.name)
    for stem, p in lbs.items():
        if stem not in {i.stem for i in imgs}:
            result["orphan"].append(p.name)

    counts = {}
    n_boxes = 0
    for p in imgs:
        lb = lbs.get(p.stem)
        if lb is None:
            continue
        try:
            for cid, *_ in parse_label(lb, combos):
                counts[cid] = counts.get(cid, 0) + 1
                n_boxes += 1
        except Exception as e:
            result["errors"].append(f"{lb.name}: {e}")
    result["by_class_id"] = {i: counts.get(i, 0) for i in range(5)}
    result["n_boxes"] = n_boxes
    result["ok"] = (not result["errors"]) and (not result["missing"]) \
        and (not result["orphan"]) and len(imgs) > 0
    return result


def _resolve_init(init: str, model_size: str,
                  pretrain_weights: str | Path | None) -> tuple[str, str | None]:
    """返回 (init_mode, weights_path)。auto = 有域内预训练权重则用之，否则 COCO。"""
    det_pt = _LIB_DIR / f"yolov8{_SIZE_LETTER.get(model_size, 'n')}.pt"
    if init == "auto":
        init = "pretrain" if Path(pretrain_weights or PRETRAIN_WEIGHTS).is_file() else "coco"
        print(f"[winter_train_lib] 初始权重: auto → {init}")
    if init == "pretrain":
        w = Path(pretrain_weights) if pretrain_weights else PRETRAIN_WEIGHTS
        if not w.is_file():
            print(f"[winter_train_lib] 未找到域内预训练权重 {w}，回退 COCO 权重")
            init, w = "coco", det_pt
        return init, str(w)
    # coco（或未知值兜底）
    if not det_pt.is_file():
        raise FileNotFoundError(f"缺少初始权重 {det_pt}，请确认仓库内 yolov8*.pt 存在")
    return "coco", str(det_pt)


def run_winter_finetune(
    images_dir: str | Path,
    labels_dir: str | Path,
    exp: str = "exp_student",
    out_root: str | Path = DEFAULT_OUT_ROOT,
    combos_csv: str | Path | None = None,
    init: str = "auto",               # auto | coco | pretrain
    pretrain_weights: str | Path | None = None,
    model_size: str = "n",            # n/s/m
    epochs: int = 60,
    batch: int = 16,
    imgsz: int = 640,
    lr0: float = 0.001,
    patience: int = 30,
    cls_epochs: int = 60,
    cls_batch: int = 32,
    cls_imgsz: int = 224,
    cls_lr0: float = 0.0005,
    cls_patience: int = 20,
    device: str = "0",
    val_ratio: float = 0.15,
    seed: int = 42,
    margin: float = 0.2,
    data_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    """整理标注并微调 检测 + 角度分类 两个模型，写 metrics.json，返回指标 dict。"""
    from scripts.prepare_winter_dataset import build_winter_dataset

    exp = safe_exp(exp)
    out_root = Path(out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir).resolve() if data_dir else out_root / f"{exp}_data"
    combos = load_angulation_combos(combos_csv)

    # 1) 整理数据集（只生成 angulation 维度）
    if verbose:
        print(f"\n===== [1/3] 整理数据集 -> {data_dir} =====")
    report = build_winter_dataset(
        images_dir=images_dir, labels_dir=labels_dir, out_dir=data_dir,
        combos=combos, dims=("angulation",),
        val_ratio=val_ratio, seed=seed, margin=margin)

    # 角度分类至少要有 2 个不同类别，否则 ultralytics 会给出费解的报错
    by_ang = report.get("by_angulation") or {}
    n_distinct = sum(1 for c in by_ang.values() if c and c > 0)
    if n_distinct < 2:
        raise ValueError(
            f"标注数据里只出现 {n_distinct} 种角度类别，无法训练角度分类器。"
            f"请至少标注 2 种不同 Winter 角度（当前分布: {by_ang}），或先检查类别顺序是否标串。")

    # 2) 检测微调
    init_mode, weights = _resolve_init(init, model_size, pretrain_weights)
    det_yaml = data_dir / "detect" / "data.yaml"
    if not det_yaml.is_file():
        raise FileNotFoundError(f"检测数据集未生成: {det_yaml}")

    from scripts import finetune_winter as fw

    det_args = Namespace(
        data=str(det_yaml), init=init_mode, weights=weights,
        model_size=model_size, epochs=epochs, batch=batch, imgsz=imgsz,
        lr0=lr0, patience=patience, device=device,
        project=str(out_root), exp=exp)
    if verbose:
        print(f"\n===== [2/3] 检测微调（初始: {init_mode}）=====")
    det_metrics = fw.run_detect(det_args)

    # 3) 角度分类微调（数据集中只有 angulation 维度，其余维度自动跳过）
    cls_root = data_dir / "cls"
    if not (cls_root / "angulation" / "train").is_dir():
        raise FileNotFoundError(f"角度分类数据集未生成: {cls_root / 'angulation'}")
    cls_args = Namespace(
        cls_root=str(cls_root), init=init_mode, weights=weights,
        model_size=model_size, device=device,
        project=str(out_root), exp=exp,
        cls_epochs=cls_epochs, cls_batch=cls_batch, cls_imgsz=cls_imgsz,
        cls_lr0=cls_lr0, cls_patience=cls_patience)
    if verbose:
        print("\n===== [3/3] 角度分类微调 =====")
    cls_metrics = fw.run_cls(cls_args)

    # 4) 汇总指标
    metrics = {
        "exp": exp, "init": init_mode, "data_dir": str(data_dir),
        "created_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        **det_metrics, **cls_metrics,
    }
    metrics_dir = out_root / exp
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    if verbose:
        m = metrics
        top1 = m.get("angulation", {}).get("top1")
        print("\n" + "=" * 60)
        print(f"✓ 微调完成: {exp}")
        print(f"  检测 mAP50={m.get('mAP50'):.4f}  mAP50-95={m.get('mAP50-95'):.4f}")
        print(f"  角度分类 top1={top1:.4f}" if top1 is not None else "  角度分类: 无指标")
        print(f"  权重: {out_root / (exp + '-detect') / 'weights' / 'best.pt'}")
        print(f"        {out_root / (exp + '-cls-angulation') / 'weights' / 'best.pt'}")
        print(f"  指标: {metrics_dir / 'metrics.json'}")
    return metrics


def plot_angulation_confusion(cm_matrix, class_names=None):
    """把角度分类混淆矩阵画成 matplotlib 图（类别用英文，避免云服务器缺中文字体乱码）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = class_names or ANGLE_CLASS_ORDER
    cm = np.asarray(cm_matrix, dtype=float)
    if cm.ndim != 2 or cm.size == 0:
        return None
    fig, ax = plt.subplots(figsize=(max(5, len(names) * 1.1), max(4, len(names) * 0.9)))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(cm.shape[1]))
    ax.set_yticks(range(cm.shape[0]))
    ax.set_xticklabels(names[:cm.shape[1]], rotation=45, ha="right")
    ax.set_yticklabels(names[:cm.shape[0]])
    thresh = cm.max() / 2 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    ax.set_xlabel("预测")
    ax.set_ylabel("真实")
    ax.set_title("Winter 角度分类混淆矩阵（val 集）")
    fig.tight_layout()
    return fig
