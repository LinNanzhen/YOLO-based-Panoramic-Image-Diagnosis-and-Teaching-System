# -*- coding: utf-8 -*-
"""
train_pretrain.py — 在云端 GPU 上用 pretrain/ 数据集对 YOLOv8 做域内预训练

为什么需要这一步：官方 yolov8n.pt 是在 COCO 自然图像上训练的，与牙科全景 X 光
域差距极大。先用 8423 张牙科全景片（含 16362 个 impacted tooth 实例）预训练，
让骨干网络学会牙齿/牙根/牙槽结构，再微调你的 Winter 小数据集（约 100 张）。

用法（云端 GPU）:
    # 1. 冒烟测试（约 3-5 分钟，验证数据能被正确读取）
    python scripts/train_pretrain.py --smoke

    # 2. 正式预训练（约 1-2 小时，视 GPU 而定）
    python scripts/train_pretrain.py --epochs 60 --batch 32

    # 3. 产出: runs/pretrain/weights/best.pt，后续所有实验以此为起点

参数:
    --data       pretrain/data.yaml 路径（默认自动查找）
    --model      初始权重（默认 yolov8n.pt，可换 yolov8s.pt）
    --epochs     训练轮数（smoke 模式固定 2）
    --batch      批次大小（smoke 模式固定 8）
    --imgsz      输入尺寸（smoke 模式固定 320）
    --device     GPU ID（默认 0）
    --project    输出目录（默认 runs/pretrain）
    --name       实验名（默认 pretrain）
    --no-amp     关闭 AMP（部分云 GPU/驱动需要）
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

# Anaconda 的 MKL 与 torch 各带一份 libiomp5md.dll，冲突会直接中止训练；
# 允许重复加载同一 OpenMP 运行时（二者同源，功能一致）。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from ultralytics import YOLO
except ImportError as e:
    sys.exit(f"[错误] 缺少 ultralytics: {e}\n请先执行: pip install ultralytics")


def find_data_yaml(arg: str | None, script_dir: Path) -> Path:
    candidates = []
    if arg:
        candidates.append(Path(arg))
    candidates += [
        Path("pretrain/data.yaml"),
        script_dir.parent / "pretrain" / "data.yaml",
    ]
    for c in candidates:
        if c.is_file():
            return c
    sys.exit(f"[错误] 未找到 pretrain/data.yaml，请确认 pretrain/ 目录已随仓库上传: {candidates}")


def read_names(data_yaml: Path) -> list[str]:
    """从 data.yaml 读取类别名（支持跨行 names 列表）。"""
    text = data_yaml.read_text(encoding="utf-8")
    start = text.find("names:")
    if start < 0:
        sys.exit("[错误] data.yaml 中没有 names: 字段")
    chunk = text[start + len("names:"):]
    lines, depth = [], 0
    for line in chunk.splitlines():
        s = line.strip()
        if not s:
            continue
        lines.append(s)
        depth += s.count("[") - s.count("]")
        if depth <= 0 and "]" in s:
            break
    joined = " ".join(lines).strip()
    if "[" in joined:
        joined = joined.split("[", 1)[1]
    if "]" in joined:
        joined = joined.split("]", 1)[0]
    names = [x.strip().strip("'\" ") for x in joined.split(",") if x.strip()]
    if not names:
        sys.exit("[错误] 无法解析 data.yaml 的 names 列表")
    return names


def make_smoke_yaml(data_yaml: Path, subset_root: Path,
                    n_train: int = 48, n_val: int = 16) -> Path:
    """构建冒烟测试用的最小数据集（复制少量图片+标签）与 data.yaml。"""
    src_root = data_yaml.parent
    names = read_names(data_yaml)
    for split, n in (("train", n_train), ("val", n_val)):
        src_img = src_root / split / "images"
        dst_img = subset_root / split / "images"
        dst_lbl = subset_root / split / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        picked = sorted(src_img.iterdir())[:n]
        for img in picked:
            shutil.copy2(img, dst_img / img.name)
            lb = src_root / split / "labels" / (img.stem + ".txt")
            if lb.exists():
                shutil.copy2(lb, dst_lbl / lb.name)
    smoke_yaml = subset_root / "data_smoke.yaml"
    # 注意：ultralytics 总是用 path/ 拼接 train/val，因此 path 必须写绝对路径、
    # train/val 写相对 path 的子目录，否则路径会翻倍。
    smoke_yaml.write_text(
        f"path: {subset_root.resolve().as_posix()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"nc: {len(names)}\n"
        f"names: {names!r}\n",
        encoding="utf-8",
    )
    return smoke_yaml


def main() -> int:
    ap = argparse.ArgumentParser(description="YOLOv8 牙科全景片域内预训练")
    ap.add_argument("--data", default=None, help="pretrain/data.yaml 路径")
    ap.add_argument("--model", default="yolov8n.pt", help="初始权重")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="runs", help="输出目录（默认 runs，配合 --name 生成 runs/pretrain/weights）")
    ap.add_argument("--name", default="pretrain", help="实验名（默认 pretrain，产出 runs/pretrain/weights/best.pt）")
    ap.add_argument("--smoke", action="store_true", help="冒烟测试：小数据集跑 2 epoch 验证流程")
    ap.add_argument("--no-amp", action="store_true", help="关闭 AMP")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_yaml = find_data_yaml(args.data, script_dir)
    names = read_names(data_yaml)
    print(f"[信息] 使用数据: {data_yaml}")
    print(f"[信息] 类别 ({len(names)}): {names}")

    if args.smoke:
        subset_root = Path("runs") / "pretrain_smoke"
        if subset_root.exists():
            shutil.rmtree(subset_root)
        data_yaml = make_smoke_yaml(data_yaml, subset_root)
        args.epochs, args.batch, args.imgsz = 2, 8, 320
        args.project, args.name = str(subset_root / "exp"), "smoke"
        print("[信息] 冒烟模式: 小数据集 2 epoch（验证标签解析与训练流程）")

    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        amp=not args.no_amp,
    )

    if args.smoke:
        print("\n" + "=" * 60)
        print("✓ 冒烟测试通过：pretrain 多边形标签可被正常读取并转换训练")
        print("  接下来运行正式预训练: python scripts/train_pretrain.py")
        return 0

    best = Path(args.project) / args.name / "weights" / "best.pt"
    last = Path(args.project) / args.name / "weights" / "last.pt"
    print("\n" + "=" * 60)
    print("✓ 预训练完成")
    print(f"  最佳权重: {best}")
    print(f"  最后权重: {last}")
    print("  后续微调示例:")
    print("  python scripts/finetune_winter.py --mode all --init pretrain \\")
    print(f"        --weights {best} ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
