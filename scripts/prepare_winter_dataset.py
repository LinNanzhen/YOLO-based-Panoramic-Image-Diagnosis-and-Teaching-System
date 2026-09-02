# -*- coding: utf-8 -*-
"""
prepare_winter_dataset.py — 把 Winter 标注整理为可训练的数据集

标注约定（见 docs/winter_annotation_guide.md）：
- 每张全景片用检测框框出每一颗第三磨牙（智齿）
- 每颗智齿的 YOLO 标签类别 ID = Winter 组合 ID（如 "II-B-近中"）
- 默认组合空间 = 关系(I/II/III) × 位置(A/B/C) × 角度(近中/远中/垂直/水平/倒置) = 45 个 ID
- 也可以用 --combos 提供自定义 CSV（列: combo_id,relation,position,angulation）

输出（--out 目录）：
    detect/                    单类智齿检测数据集（nc=1），用于检测微调
        data.yaml  images/{train,val}  labels/{train,val}
    cls/                       三维度独立分类数据集（裁切智齿区域），用于 YOLO-cls 训练
        relation/   {train,val}/{I,II,III}/
        position/   {train,val}/{A,B,C}/
        angulation/ {train,val}/{近中,远中,垂直,水平,倒置}/
    crops_metadata.csv         每个裁切块的来源与标签，便于追溯
    report.json                各类别分布统计 + 样本量警告
    class_list_for_xanylabeling.txt   标注工具用的类别清单（按 ID 顺序）

用法:
    python scripts/prepare_winter_dataset.py \
        --images annotated_presets/images \
        --labels annotated_presets/labels \
        --out winter_dataset

参数:
    --images      标注完成的图片目录
    --labels      对应的 YOLO 标签目录（类别 ID = Winter 组合 ID）
    --out         输出目录（默认 winter_dataset）
    --combos      可选：自定义组合映射 CSV（combo_id,relation,position,angulation）
    --val-ratio   验证集比例（按图片划分，默认 0.15）
    --seed        随机种子（默认 42）
    --margin      裁切外扩比例（默认 0.2，即每边扩 20%）
    --test-images / --test-labels   可选：独立测试集（不参与训练/验证）
"""
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

DEFAULT_RELATIONS = ["I", "II", "III"]
DEFAULT_POSITIONS = ["A", "B", "C"]
DEFAULT_ANGLES = ["近中", "远中", "垂直", "水平", "倒置"]


def build_default_combos() -> dict[int, tuple[str, str, str, str]]:
    """默认组合空间：关系×位置×角度，ID 按固定顺序（关系→位置→角度）编号，保证稳定。"""
    combos = {}
    idx = 0
    for rel in DEFAULT_RELATIONS:
        for pos in DEFAULT_POSITIONS:
            for ang in DEFAULT_ANGLES:
                combos[idx] = (f"{rel}-{pos}-{ang}", rel, pos, ang)
                idx += 1
    return combos


def load_custom_combos(path: Path) -> dict[int, tuple[str, str, str, str]]:
    combos = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("combo_id"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            sys.exit(f"[错误] {path} 每行需 4 列 (combo_id,relation,position,angulation): {line}")
        cid = int(parts[0])
        rel, pos, ang = parts[1], parts[2], parts[3]
        # relation/position 为 "-" 表示该维度未标注（如只标角度），组合名直接取角度
        name = ang if rel == "-" and pos == "-" else f"{rel}-{pos}-{ang}"
        combos[cid] = (name, rel, pos, ang)
    if not combos:
        sys.exit(f"[错误] 未从 {path} 解析到任何组合")
    return combos


def parse_label(lb_path: Path, combos: dict) -> list[tuple[int, float, float, float, float]]:
    """解析 YOLO 标签，返回 [(combo_id, cx, cy, w, h), ...]。"""
    out = []
    for ln in lb_path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        vals = ln.split()
        if len(vals) != 5:
            sys.exit(f"[错误] {lb_path}: 每行需 5 列 (class cx cy w h)，得到 {len(vals)} 列: {ln}")
        cid = int(float(vals[0]))
        if cid not in combos:
            sys.exit(f"[错误] {lb_path}: 类别 ID {cid} 不在组合映射中（有效 ID: {sorted(combos)}）")
        out.append((cid, *map(float, vals[1:])))
    return out


def make_crop(img: Image.Image, box: tuple, margin: float) -> Image.Image:
    """按归一化框裁切，外扩 margin，返回裁剪图。"""
    w, h = img.size
    cx, cy, bw, bh = box
    x0 = int((cx - bw / 2 - bw * margin) * w)
    y0 = int((cy - bh / 2 - bh * margin) * h)
    x1 = int((cx + bw / 2 + bw * margin) * w)
    y1 = int((cy + bh / 2 + bh * margin) * h)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 - x0 < 8 or y1 - y0 < 8:  # 过小的裁切直接跳过并预警
        raise ValueError(f"裁切过小 ({x1-x0}x{y1-y0})")
    return img.crop((x0, y0, x1, y1))


def safe_stem(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name)


def main() -> int:
    ap = argparse.ArgumentParser(description="整理 Winter 标注为训练数据集")
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="winter_dataset")
    ap.add_argument("--combos", default=None, help="自定义组合 CSV")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--test-images", default=None)
    ap.add_argument("--test-labels", default=None)
    ap.add_argument("--dims", default="relation,position,angulation",
                    help="要生成的分类维度，逗号分隔（默认三个维度；只标了角度可传 angulation）")
    args = ap.parse_args()
    dims = [d.strip() for d in args.dims.split(",") if d.strip()]
    invalid = set(dims) - {"relation", "position", "angulation"}
    if invalid:
        sys.exit(f"[错误] 无效维度: {sorted(invalid)}（可选: relation,position,angulation）")

    img_dir, lb_dir = Path(args.images), Path(args.labels)
    if not img_dir.is_dir() or not lb_dir.is_dir():
        sys.exit(f"[错误] 图片/标签目录不存在: {img_dir}, {lb_dir}")
    combos = load_custom_combos(Path(args.combos)) if args.combos else build_default_combos()
    print(f"[信息] 组合空间: {len(combos)} 个组合（ID 0-{len(combos)-1}）")

    # 1. 读取全部标注
    entries = []  # (img_path, lb_path)
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_SUFFIXES)
    lbs = {p.stem: p for p in lb_dir.iterdir() if p.suffix.lower() == ".txt"}
    missing, orphan = [], []
    for p in imgs:
        if p.stem in lbs:
            entries.append((p, lbs[p.stem]))
        else:
            missing.append(p.name)
    orphan = [p for s, p in lbs.items() if s not in {i.stem for i in imgs}]
    if missing or orphan:
        sys.exit(f"[错误] 图片-标签不匹配: 缺标签 {len(missing)} 张 {missing[:5]}，孤儿标签 {len(orphan)} 个 {orphan[:5]}")

    # 2. 解析并切分（按图片划分，防裁切泄漏）
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(entries))
    n_val = max(1, round(len(entries) * args.val_ratio))
    val_idx = set(perm[:n_val].tolist())
    splits = {"train": [], "val": []}
    for i, (img, lb) in enumerate(entries):
        splits["val" if i in val_idx else "train"].append((img, lb))

    # 3. 输出目录
    out = Path(args.out)
    for split in ("train", "val"):
        (out / "detect" / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "detect" / "labels" / split).mkdir(parents=True, exist_ok=True)
    for dim in dims:
        for split in ("train", "val"):
            (out / "cls" / dim / split).mkdir(parents=True, exist_ok=True)

    by_combo = Counter()
    by_rel = Counter()
    by_pos = Counter()
    by_ang = Counter()
    warnings = []
    meta_rows = []
    total_boxes = 0

    def process(split: str, items: list, test_prefix: str = ""):
        nonlocal total_boxes
        for img_path, lb_path in items:
            boxes = parse_label(lb_path, combos)
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                warnings.append(f"图片无法打开 {img_path.name}: {e}")
                continue
            shutil.copy2(img_path, out / "detect" / "images" / split / img_path.name)
            det_lines = []
            for box_idx, (cid, cx, cy, bw, bh) in enumerate(boxes):
                name, rel, pos, ang = combos[cid]
                det_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                total_boxes += 1
                by_combo[name] += 1
                by_rel[rel] += 1
                by_pos[pos] += 1
                by_ang[ang] += 1
                stem = safe_stem(f"{test_prefix}{img_path.stem}__{box_idx}")
                try:
                    crop = make_crop(img, (cx, cy, bw, bh), args.margin)
                except ValueError as e:
                    warnings.append(f"{img_path.name} 框{box_idx} {e}，已跳过该裁切")
                else:
                    for dim, val in (("relation", rel), ("position", pos), ("angulation", ang)):
                        if dim not in dims:
                            continue
                        crop_path = out / "cls" / dim / split / val / f"{stem}.jpg"
                        crop_path.parent.mkdir(parents=True, exist_ok=True)
                        crop.save(crop_path, "JPEG", quality=95)
                meta_rows.append({
                    "image": img_path.name, "box_index": box_idx, "combo_id": cid,
                    "combo_name": name, "relation": rel, "position": pos,
                    "angulation": ang, "cx": round(cx, 4), "cy": round(cy, 4),
                    "w": round(bw, 4), "h": round(bh, 4), "split": split,
                })
            (out / "detect" / "labels" / split / (img_path.stem + ".txt")).write_text(
                "\n".join(det_lines) + ("\n" if det_lines else ""), encoding="utf-8")

    process("train", splits["train"])
    process("val", splits["val"])

    test_n = 0
    if args.test_images and args.test_labels:
        t_img, t_lb = Path(args.test_images), Path(args.test_labels)
        test_items = []
        for p in sorted(t_img.iterdir()):
            if p.suffix.lower() in IMG_SUFFIXES and (t_lb / (p.stem + ".txt")).exists():
                test_items.append((p, t_lb / (p.stem + ".txt")))
        if not test_items:
            sys.exit("[错误] 测试集目录中没有找到 图片+标签 配对")
        (out / "detect" / "images" / "test").mkdir(parents=True, exist_ok=True)
        (out / "detect" / "labels" / "test").mkdir(parents=True, exist_ok=True)
        for dim in dims:
            (out / "cls" / dim / "test").mkdir(parents=True, exist_ok=True)
        process("test", test_items, test_prefix="t_")
        test_n = len(test_items)

    # 4. data.yaml（detect）
    detect_yaml = out / "detect" / "data.yaml"
    yaml_lines = [
        f"path: {(out / 'detect').as_posix()}",
        "train: images/train",
        "val: images/val",
    ]
    if test_n:
        yaml_lines.append("test: images/test")
    yaml_lines += ["nc: 1", "names: ['wisdom']"]
    detect_yaml.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    # 5. 标注工具类别清单（按 ID 顺序）
    (out / "class_list_for_xanylabeling.txt").write_text(
        "\n".join(f"{cid}: {name}" for cid, (name, *_rest) in sorted(combos.items())), encoding="utf-8")

    # 6. 元数据与报告
    import csv
    with open(out / "crops_metadata.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()) if meta_rows else ["image"])
        writer.writeheader()
        writer.writerows(meta_rows)

    for name, cnt in by_combo.items():
        if cnt < 10:
            warnings.append(f"组合 {name} 仅 {cnt} 个样本，建议补标或从训练中剔除（评估时按类汇总）")

    report = {
        "total_images": len(entries), "total_boxes": total_boxes,
        "train_images": len(splits["train"]), "val_images": len(splits["val"]),
        "test_images": test_n,
        "by_combo": dict(sorted(by_combo.items())),
        "by_relation": dict(by_rel) if "relation" in dims else None,
        "by_position": dict(by_pos) if "position" in dims else None,
        "by_angulation": dict(by_ang) if "angulation" in dims else None,
        "warnings": warnings,
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 7. 打印摘要
    print("\n" + "=" * 60)
    print(f"图片: {len(entries)}（train {len(splits['train'])} / val {len(splits['val'])}"
          + (f" / test {test_n}" if test_n else "") + f"）  智齿框: {total_boxes}")
    if "relation" in dims:
        print(f"\n关系 (Winter Class): {dict(by_rel)}")
    if "position" in dims:
        print(f"位置 (A/B/C):        {dict(by_pos)}")
    if "angulation" in dims:
        print(f"角度:                {dict(by_ang)}")
    print(f"\n组合分布: {dict(sorted(by_combo.items(), key=lambda x: -x[1]))}")
    if warnings:
        print(f"\n⚠ 警告 {len(warnings)} 条:")
        for w in warnings:
            print(f"  - {w}")
    print(f"\n✓ 数据集已生成: {out}")
    print("  检测: winter_dataset/detect/data.yaml")
    print("  分类: winter_dataset/cls/{relation,position,angulation}")
    print("  下一步: python scripts/finetune_winter.py --mode all --init pretrain "
          "--weights runs/pretrain/weights/best.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
