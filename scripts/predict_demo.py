# -*- coding: utf-8 -*-
"""
predict_demo.py — 用微调后的 检测 + 角度分类 权重对全景片推理，输出带框可视化

用法:
    python scripts/predict_demo.py --images <图片目录或单张图> \
        --det runs/winter/expB-detect/weights/best.pt \
        --cls runs/winter/expB-cls-angulation/weights/best.pt \
        --out runs/demo_predictions \
        --labels winter_dataset/detect/labels/val   # 可选：叠加真实标注框做对比

输出: out/<图名>_pred.jpg（红框=预测 类别+置信度），有 --labels 时另存 _gt.jpg（绿框=真实）
同时终端打印每颗智齿的预测 vs 真实对照表。

字体/画框/裁切/推理流水线来自根目录 dental_common.py（与 Web 页面共用）。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dental_common as dc  # noqa: E402  （须先于 ultralytics 导入，见其模块注释）
from PIL import Image, ImageDraw  # noqa: E402
from ultralytics import YOLO  # noqa: E402

IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main() -> int:
    ap = argparse.ArgumentParser(description="智齿检测+角度分类推理可视化")
    ap.add_argument("--images", required=True, help="图片目录或单张图片路径")
    ap.add_argument("--det", default="runs/winter/expB-detect/weights/best.pt", help="检测权重")
    ap.add_argument("--cls", default="runs/winter/expB-cls-angulation/weights/best.pt",
                    help="角度分类权重（可省略只画检测框）")
    ap.add_argument("--labels", default=None, help="可选：YOLO 标签目录，画真实标注框做对比")
    ap.add_argument("--out", default="runs/demo_predictions")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    img_paths = sorted(Path(args.images).iterdir()) if Path(args.images).is_dir() \
        else [Path(args.images)]
    img_paths = [p for p in img_paths if p.suffix.lower() in IMG_SUFFIXES]
    if not img_paths:
        sys.exit(f"[错误] 没有找到图片: {args.images}")

    det = YOLO(args.det)
    clf = YOLO(args.cls) if args.cls and Path(args.cls).is_file() else None
    if not clf:
        print("[信息] 未提供分类权重，只画检测框")
    lbl_dir = Path(args.labels) if args.labels else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for img_path in img_paths:
        img = Image.open(img_path).convert("RGB")
        findings = dc.detect_and_classify(
            det, clf, str(img_path), crop_img=img,
            conf=args.conf, margin=args.margin, device=args.device)
        rows = []
        dr = ImageDraw.Draw(img)
        # 字号与线宽随图片尺寸缩放：约 2000px 宽的全景片用 ~33px 字
        font, _ = dc.load_draw_font(max(24, img.width // 60), prefer_cjk=False)
        line_w = max(4, img.width // 400)
        for f in findings:
            x1, y1, x2, y2 = f["box"]
            if f["angle_en"] is not None:
                label = f"{f['angle_en']} {f['angle_conf']:.2f}"
            else:
                label = f"{f['det_conf']:.2f}"
            dc.annotate_box(dr, f["box"], label, (255, 40, 40), line_w, font)
            rows.append((x1, y1, x2, y2, f["det_conf"], f["angle_en"] or "", f["angle_conf"]))

        pred_path = out / f"{img_path.stem}_pred.jpg"
        img.save(pred_path, quality=92)
        print(f"\n=== {img_path.name} → {pred_path}（{len(rows)} 颗智齿）===")
        for x1, y1, x2, y2, conf, cls_, pconf in rows:
            print(f"  框 ({x1},{y1})-({x2},{y2})  检测{conf:.2f}"
                  + (f"  → {cls_} {pconf:.2f}" if cls_ else ""))

        if lbl_dir:
            lb = lbl_dir / f"{img_path.stem}.txt"
            if lb.exists():
                gt_img = img.copy()
                gdr = ImageDraw.Draw(gt_img)
                w, h = gt_img.size
                for ln in lb.read_text().splitlines():
                    ln = ln.strip()
                    if not ln:
                        continue
                    vals = ln.split()
                    if len(vals) != 5:
                        continue
                    cx, cy, bw, bh = map(float, vals[1:])
                    gx0, gy0 = (cx - bw / 2) * w, (cy - bh / 2) * h
                    gx1, gy1 = (cx + bw / 2) * w, (cy + bh / 2) * h
                    gdr.rectangle([gx0, gy0, gx1, gy1], outline=(60, 220, 60), width=line_w)
                gt_path = out / f"{img_path.stem}_gt.jpg"
                gt_img.save(gt_path, quality=92)
                print(f"  真实标注对比: {gt_path}")
    print("\n✓ 完成，输出目录:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
