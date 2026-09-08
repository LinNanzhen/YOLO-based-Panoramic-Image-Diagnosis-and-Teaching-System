# -*- coding: utf-8 -*-
"""
auto_annotate.py — 用预训练模型自动框出智齿，生成预标注（标注提速）

流程：预训练完成得到 runs/pretrain/weights/best.pt 后，对用户的全景片跑推理，
只保留 "impacted tooth"（类 12）的框，导出 YOLO 格式预标注。标注者在
X-AnyLabeling 中导入这些预标注，只需校正框的位置并把每颗智齿的 Winter
组合类别选好，工作量大幅下降。

用法:
    python scripts/auto_annotate.py --images dataset/images/trainset \
        --weights runs/pretrain/weights/best.pt --out annotated_presets

    python scripts/auto_annotate.py --images <更多未标注片目录> \
        --weights runs/pretrain/weights/best.pt --out annotated_presets2 --preview preview

参数:
    --images    未标注图片目录
    --weights   预训练权重（runs/pretrain/weights/best.pt）
    --out       输出目录（out/images + out/labels，YOLO 格式，可直接导入标注工具）
    --classes   保留的类别 ID（默认 12=impacted tooth，逗号分隔可多选）
    --conf      置信度阈值（默认 0.25）
    --iou       NMS IoU 阈值（默认 0.5）
    --imgsz     推理尺寸（默认 640）
    --min-area  过滤过小框（归一化面积，默认 0.001）
    --device    GPU ID（默认 0，-1 用 CPU）
    --preview   可选：把画好框的预览图输出到该目录，便于快速质检
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dental_common as dc  # noqa: E402  （须先于 ultralytics 导入，见其模块注释）
from PIL import Image, ImageDraw  # noqa: E402

try:
    from ultralytics import YOLO
except ImportError as e:
    sys.exit(f"[错误] 缺少 ultralytics: {e}")

IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main() -> int:
    ap = argparse.ArgumentParser(description="预标注：用预训练模型自动框出智齿")
    ap.add_argument("--images", required=True, help="未标注图片目录")
    ap.add_argument("--weights", required=True, help="预训练权重路径")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--classes", default="12", help="保留的类别 ID，逗号分隔（默认 12=impacted tooth）")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--min-area", type=float, default=0.001, help="过滤归一化面积小于该值的框")
    ap.add_argument("--device", default="0")
    ap.add_argument("--preview", default=None, help="可选：预览图输出目录")
    args = ap.parse_args()

    img_dir = Path(args.images)
    out_dir = Path(args.out)
    if not img_dir.is_dir():
        sys.exit(f"[错误] 图片目录不存在: {img_dir}")
    keep_classes = {int(c) for c in args.classes.split(",") if c.strip()}
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    preview_dir = Path(args.preview) if args.preview else None
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)

    print(f"[信息] 加载模型: {args.weights}")
    model = YOLO(args.weights)

    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_SUFFIXES)
    if not images:
        sys.exit(f"[错误] {img_dir} 中没有图片")
    print(f"[信息] 待处理图片: {len(images)} 张，保留类别 {sorted(keep_classes)}")

    total_boxes = 0
    no_detect = []
    skipped = []
    summary = {}
    for i, img_path in enumerate(images, 1):
        try:
            results = model.predict(
                str(img_path),
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )
        except Exception as e:
            # 个别损坏/无法解码的图片不应中断整批预标注
            skipped.append(img_path.name)
            print(f"  [警告] 图片读取失败，已跳过 {img_path.name}: {e}")
            continue
        r = results[0]
        boxes, confs, clss = r.boxes.xywhn.cpu().numpy(), r.boxes.conf.cpu().numpy(), \
            r.boxes.cls.cpu().numpy().astype(int)

        lines = []
        keep = []
        for (cx, cy, w, h), conf, cls_id in zip(boxes, confs, clss):
            if cls_id not in keep_classes:
                continue
            if w * h < args.min_area:
                continue
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            keep.append((cx, cy, w, h, float(conf)))

        stem = img_path.stem
        (out_dir / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        shutil.copy2(img_path, out_dir / "images" / img_path.name)

        if keep:
            total_boxes += len(keep)
        else:
            no_detect.append(img_path.name)

        if preview_dir:
            draw_preview(img_path, keep, preview_dir / f"{stem}_preview.jpg")

        if i % 20 == 0 or i == len(images):
            print(f"  [{i}/{len(images)}] 累计框数: {total_boxes}")
        summary[img_path.name] = {"boxes": len(keep)}

    summary["_total_boxes"] = total_boxes
    summary["_images_without_detection"] = no_detect
    summary["_images_skipped_unreadable"] = skipped
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"✓ 预标注完成: 共 {total_boxes} 个智齿框")
    print(f"  输出目录: {out_dir} (images/ + labels/，YOLO 格式)")
    if skipped:
        print(f"  ⚠ {len(skipped)} 张图无法读取已跳过（文件可能损坏，建议检查/替换）: {skipped[:5]}")
    if no_detect:
        print(f"  ⚠ {len(no_detect)} 张图未检出智齿（模型可能漏检，需人工补标）: {no_detect[:5]}")
    print("  下一步: 在 X-AnyLabeling 中打开 images/，导入 labels/ 预标注，")
    print("          校正框并给每颗智齿选择 Winter 组合类别，然后运行 prepare_winter_dataset.py")
    return 0


def draw_preview(img_path: Path, boxes: list, out_path: Path) -> None:
    """在图片上画框输出预览，便于快速质检。"""
    try:
        img = Image.open(img_path).convert("RGB")
        dr = ImageDraw.Draw(img)
        w, h = img.size
        font, _ = dc.load_draw_font(max(24, w // 60), prefer_cjk=False)
        for cx, cy, bw, bh, conf in boxes:
            x0, y0 = (cx - bw / 2) * w, (cy - bh / 2) * h
            x1, y1 = (cx + bw / 2) * w, (cy + bh / 2) * h
            dc.annotate_box(dr, (x0, y0, x1, y1), f"{conf:.2f}", (255, 0, 0), 3, font)
        img.save(out_path, quality=90)
    except Exception as e:  # 预览失败不影响主流程
        print(f"  [警告] 预览生成失败 {img_path.name}: {e}")


if __name__ == "__main__":
    sys.exit(main())
