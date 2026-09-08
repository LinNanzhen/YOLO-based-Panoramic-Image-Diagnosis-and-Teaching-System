# -*- coding: utf-8 -*-
"""
finetune_winter.py — Winter 微调实验驱动（实验 A/B 对比）

实验 A（基线）: 用官方 COCO 权重 yolov8n.pt 直接微调 Winter 数据
实验 B（核心）: 用域内预训练权重 runs/pretrain/weights/best.pt 微调
两者唯一区别是初始权重，其余超参数一致，从而量化预训练的增益。

检测（单类智齿框）:
    # 实验 A
    python scripts/finetune_winter.py --mode detect --init coco --exp expA
    # 实验 B
    python scripts/finetune_winter.py --mode detect --init pretrain \
        --weights runs/pretrain/weights/best.pt --exp expB

分类（三维度独立，裁切智齿区域，YOLO-cls）:
    python scripts/finetune_winter.py --mode cls --exp expB_cls

一键跑全部并对比:
    python scripts/finetune_winter.py --mode all --init pretrain \
        --weights runs/pretrain/weights/best.pt --exp expB \
        --compare-with runs/winter/expA/metrics.json

指标输出: runs/winter/<exp>/metrics.json（检测 mAP50/mAP50-95 + 三维度 top1/混淆矩阵）
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Anaconda 的 MKL 与 torch 各带一份 libiomp5md.dll，冲突会直接中止训练。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# 由脚本位置推导，与 CWD 无关：scripts/ 的上一级即仓库根目录
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from ultralytics import YOLO
except ImportError as e:
    sys.exit(f"[错误] 缺少 ultralytics: {e}")

# device=cpu 时 ultralytics 会把 CUDA_VISIBLE_DEVICES 写成 ""，不还原的话
# 同一进程（Web 页面）后续的 GPU 推理会全部报 Invalid device id
from dental_common import preserve_cuda_visible_devices

DIMS = ["relation", "position", "angulation"]


def repo_path(p) -> Path:
    """仓库相对路径 -> 绝对路径；已是绝对路径则原样返回。"""
    q = Path(p)
    return q if q.is_absolute() else REPO_ROOT / q


def run_detect(args) -> dict:
    data_yaml = Path(args.data)
    if not data_yaml.is_file():
        sys.exit(f"[错误] 未找到检测数据集: {data_yaml}（先运行 prepare_winter_dataset.py）")
    if args.init == "pretrain":
        weights = args.weights
        if not Path(weights).is_file():
            sys.exit(f"[错误] 预训练权重不存在: {weights}（先运行 train_pretrain.py）")
    else:
        weights = str(repo_path(f"yolov8{args.model_size[0]}.pt"))
    print(f"[检测] 初始权重: {weights}")
    model = YOLO(weights)
    with preserve_cuda_visible_devices():
        results = model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            lr0=args.lr0,
            patience=args.patience,
            device=args.device,
            project=str(args.project),
            name=f"{args.exp}-detect",
            exist_ok=True,
        )
        # 用 val 集（或 test 集，若 data.yaml 配置了 test）做评估
        val = model.val(data=str(data_yaml),
                        split="test" if data_yaml_test(data_yaml) else "val",
                        device=args.device, verbose=False)
    metrics = {
        "init": args.init,
        "mAP50": float(val.box.map50),
        "mAP50-95": float(val.box.map),
        "precision": float(val.box.mp),
        "recall": float(val.box.mr),
    }
    print(f"[检测] mAP50={metrics['mAP50']:.4f}  mAP50-95={metrics['mAP50-95']:.4f}  "
          f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}")
    return metrics


def data_yaml_test(data_yaml: Path) -> bool:
    return "test:" in data_yaml.read_text(encoding="utf-8")


def build_cls_model(args):
    """构建 YOLO-cls 模型：优先域内预训练骨干，其次本地 COCO 分类权重，最后随机初始化（离线兜底）。"""
    size = args.model_size[0]
    if args.init == "pretrain" and Path(args.weights).is_file():
        print(f"[分类] 用域内预训练骨干初始化: {args.weights}")
        model = YOLO(f"yolov8{size}-cls.yaml")
        model.load(str(args.weights))  # strict=False，只迁移骨干/颈部共同层
        return model
    w = repo_path(f"yolov8{size}-cls.pt")
    if w.is_file():
        print(f"[分类] 初始权重（COCO）: {w}")
        return YOLO(str(w))
    print("!" * 60)
    print(f"[分类] 警告：未找到 COCO 分类权重 {w}")
    print("       将改为随机初始化从零训练 —— 小数据下 top1 通常会明显偏低，")
    print("       这不代表你的标注或数据有问题。")
    print("       要恢复正常效果，联网执行一次即可（会下载到当前目录）：")
    print(f"         python -c \"from ultralytics import YOLO; YOLO('yolov8{size}-cls.pt')\"")
    print("!" * 60)
    return YOLO(f"yolov8{size}-cls.yaml")


def run_cls(args) -> dict:
    cls_root = Path(args.cls_root)
    if not cls_root.is_dir():
        sys.exit(f"[错误] 未找到分类数据集: {cls_root}（先运行 prepare_winter_dataset.py）")
    metrics = {}
    for dim in DIMS:
        dim_dir = cls_root / dim
        if not (dim_dir / "train").is_dir():
            print(f"[分类] 跳过 {dim}: 无 train 目录")
            continue
        print(f"[分类] {dim}: 训练 YOLO-cls（{args.model_size}）...")
        model = build_cls_model(args)
        with preserve_cuda_visible_devices():
            model.train(
                data=str(dim_dir),
                epochs=args.cls_epochs,
                batch=args.cls_batch,
                imgsz=args.cls_imgsz,
                lr0=args.cls_lr0,
                patience=args.cls_patience,
                device=args.device,
                project=str(args.project),
                name=f"{args.exp}-cls-{dim}",
                exist_ok=True,
            )
            val = model.val(device=args.device, verbose=False)
        top1 = float(val.top1)
        top5 = float(val.top5)
        cm = val.confusion_matrix.matrix if getattr(val, "confusion_matrix", None) else None
        metrics[dim] = {"top1": top1, "top5": top5,
                        "confusion": cm.tolist() if cm is not None else None}
        print(f"[分类] {dim}: top1={top1:.4f} top5={top5:.4f}")
    return metrics


def print_comparison(exp: str, metrics: dict, other_path: str | None) -> None:
    if not other_path:
        return
    other = Path(other_path)
    if not other.is_file():
        print(f"[对比] 未找到 {other}，跳过对比")
        return
    try:
        b = json.loads(other.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"[对比] 读取 {other} 失败: {e}")
        return
    print("\n" + "=" * 60)
    print(f"实验对比: {other.stem} (基线)  vs  {exp}")
    if "mAP50" in metrics and "mAP50" in b:
        print(f"  检测 mAP50    : {b['mAP50']:.4f} -> {metrics['mAP50']:.4f}  "
              f"(Δ {metrics['mAP50'] - b['mAP50']:+.4f})")
        print(f"  检测 mAP50-95 : {b['mAP50-95']:.4f} -> {metrics['mAP50-95']:.4f}  "
              f"(Δ {metrics['mAP50-95'] - b['mAP50-95']:+.4f})")
    for dim in DIMS:
        if dim in metrics and dim in b:
            d = b[dim].get("top1", 0)
            print(f"  分类 {dim:<10} top1: {d:.4f} -> {metrics[dim]['top1']:.4f}  "
                  f"(Δ {metrics[dim]['top1'] - d:+.4f})")


def main() -> int:
    ap = argparse.ArgumentParser(description="Winter 微调实验（检测 + 三维度分类）")
    ap.add_argument("--mode", choices=["detect", "cls", "all"], default="all")
    ap.add_argument("--data", default="winter_dataset/detect/data.yaml")
    ap.add_argument("--cls-root", default="winter_dataset/cls")
    ap.add_argument("--init", choices=["coco", "pretrain"], default="pretrain",
                    help="初始权重来源：coco=官方权重基线，pretrain=域内预训练")
    ap.add_argument("--weights", default="runs/pretrain/weights/best.pt",
                    help="pretrain 初始化时使用的权重")
    ap.add_argument("--model-size", default="n", help="n/s/m")
    ap.add_argument("--epochs", type=int, default=150, help="检测训练轮数")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--lr0", type=float, default=0.001)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--cls-epochs", type=int, default=100)
    ap.add_argument("--cls-batch", type=int, default=32)
    ap.add_argument("--cls-imgsz", type=int, default=224)
    ap.add_argument("--cls-lr0", type=float, default=0.0005)
    ap.add_argument("--cls-patience", type=int, default=20)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="runs/winter")
    ap.add_argument("--exp", default=None, help="实验名（默认 exp-<init>）")
    ap.add_argument("--compare-with", default=None, help="对比实验的 metrics.json 路径")
    args = ap.parse_args()
    args.exp = args.exp or f"exp-{args.init}"

    # 默认值都是仓库相对路径，统一解析成绝对路径，避免受运行时 CWD 影响
    for attr in ("data", "cls_root", "weights", "project"):
        setattr(args, attr, str(repo_path(getattr(args, attr))))

    metrics = {}
    if args.mode in ("detect", "all"):
        metrics.update(run_detect(args))
    if args.mode in ("cls", "all"):
        metrics.update(run_cls(args))

    out = Path(args.project) / args.exp
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ 指标已保存: {metrics_path}")

    if args.compare_with:
        print_comparison(args.exp, metrics, args.compare_with)
    return 0


if __name__ == "__main__":
    sys.exit(main())
