# -*- coding: utf-8 -*-
"""
validate_pretrain.py — 校验 pretrain/ 数据集完整性与标签格式（无需 torch/GPU）

检查项：
1. train/val/test 三份的 图片-标签 一一对应（无孤儿标签/缺标签）
2. 每个标签文件逐行解析：
   - 类 ID 在 data.yaml 的 nc 范围内
   - 多边形（>6 值）坐标点数为偶数且 >= 3 个点，坐标归一化在 [0, 1]
   - 框格式（==5 值）合法
   - 同一文件内 框行 与 多边形行 混用（ultralytics 会视为损坏）-> 警告
3. 每类实例数、多边形点数、框面积统计
4. 图片可打开（PIL verify）
5. 与用户自有图片（默认 images/trainset + images/testset）做 MD5 查重

用法: python scripts/validate_pretrain.py [--pretrain-dir pretrain] [--user-dir images]
输出: 校验报告（发现问题返回退出码 1）
"""
import argparse
import hashlib
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_yaml_names(yaml_path: Path) -> list[str]:
    """极简 yaml 解析：支持 'names: [...]' 单行或跨多行列表。"""
    text = yaml_path.read_text(encoding="utf-8")
    # 合并 'names:' 之后的换行，直到方括号闭合
    start = text.find("names:")
    if start < 0:
        return []
    chunk = text[start + len("names:"):]
    lines = []
    depth = 0
    for line in chunk.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lines.append(stripped)
        depth += stripped.count("[") - stripped.count("]")
        if depth <= 0 and "]" in stripped:
            break
    joined = " ".join(lines).strip()
    if "[" in joined:
        joined = joined.split("[", 1)[1]
    if "]" in joined:
        joined = joined.split("]", 1)[0]
    return [x.strip().strip("'\" ") for x in joined.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="校验 pretrain 数据集")
    ap.add_argument("--pretrain-dir", default="pretrain", help="pretrain 数据集根目录")
    ap.add_argument("--user-dir", default="images", help="用户自有图片目录（查重用），可不存在")
    args = ap.parse_args()

    root = Path(args.pretrain_dir).resolve()
    yaml_path = root / "data.yaml"
    if not yaml_path.exists():
        print(f"[错误] 未找到 {yaml_path}")
        return 1

    names = parse_yaml_names(yaml_path)
    nc = len(names)
    print(f"数据集: {root}")
    print(f"类别数: {nc} -> {names}")
    print()

    issues = 0
    warnings = 0
    cls_counter = Counter()
    poly_points = Counter()
    areas = []
    per_split = {}

    for split in ("train", "val", "test"):
        img_dir = root / split / "images"
        lb_dir = root / split / "labels"
        if not img_dir.is_dir() or not lb_dir.is_dir():
            print(f"[错误] {split}: 缺少 images/labels 目录")
            issues += 1
            continue

        imgs = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMG_SUFFIXES}
        lbs = {p.stem: p for p in lb_dir.iterdir() if p.suffix.lower() == ".txt"}
        n_img, n_lb = len(imgs), len(lbs)

        orphan = sorted(set(lbs) - set(imgs))
        missing = sorted(set(imgs) - set(lbs))
        if orphan:
            issues += 1
            print(f"[错误] {split}: {len(orphan)} 个标签没有对应图片: {orphan[:5]}...")
        if missing:
            issues += 1
            print(f"[错误] {split}: {len(missing)} 张图片没有对应标签: {missing[:5]}...")

        split_cls = Counter()
        bad_images = []
        for stem, lb_path in sorted(lbs.items()):
            img_path = imgs[stem]
            # 图片可打开
            try:
                with Image.open(img_path) as im:
                    im.verify()
            except (UnidentifiedImageError, OSError) as e:
                bad_images.append((stem, str(e)))
                continue
            # 标签解析
            try:
                lines = [ln.split() for ln in lb_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            except OSError as e:
                issues += 1
                print(f"[错误] {lb_path}: 读取失败 {e}")
                continue
            has_poly = any(len(x) > 6 for x in lines)
            has_box = any(len(x) == 5 for x in lines)
            for ln in lines:
                try:
                    vals = np.array(ln, dtype=np.float32)
                except ValueError:
                    issues += 1
                    print(f"[错误] {lb_path}: 非数值行 {ln[:8]}")
                    continue
                cls_id = int(vals[0])
                if not (0 <= cls_id < nc):
                    issues += 1
                    print(f"[错误] {lb_path}: 类 ID {cls_id} 超出范围 0-{nc-1}")
                    continue
                split_cls[cls_id] += 1
                if len(vals) == 5:
                    if has_poly:
                        warnings += 1
                        print(f"[警告] {lb_path}: 框(5值)与多边形混用，ultralytics 会丢弃或报错")
                    cx, cy, w, h = vals[1:]
                    if w <= 0 or h <= 0:
                        issues += 1
                        print(f"[错误] {lb_path}: 非正框宽高 {w},{h}")
                    if vals[1:].max() > 1.01 or vals[1:].min() < -0.01:
                        issues += 1
                        print(f"[错误] {lb_path}: 坐标越界 {vals[1:]}")
                    areas.append(w * h)
                elif len(vals) > 6:
                    pts = vals[1:]
                    if len(pts) % 2 != 0 or len(pts) < 6:
                        issues += 1
                        print(f"[错误] {lb_path}: 多边形坐标数 {len(pts)} 非法（需偶数且 >=6）")
                        continue
                    if pts.max() > 1.01 or pts.min() < -0.01:
                        issues += 1
                        print(f"[错误] {lb_path}: 多边形坐标越界")
                        continue
                    poly_points[len(pts) // 2] += 1
                    xy = pts.reshape(-1, 2)
                    w = xy[:, 0].max() - xy[:, 0].min()
                    h = xy[:, 1].max() - xy[:, 1].min()
                    if w <= 0 or h <= 0:
                        issues += 1
                        print(f"[错误] {lb_path}: 多边形退化为线/点")
                        continue
                    areas.append(w * h)
                else:
                    issues += 1
                    print(f"[错误] {lb_path}: 行值数 {len(vals)} 非法（应为 5 或 >6）")
            if has_poly and has_box:
                warnings += 1

        per_split[split] = (n_img, n_lb, sum(split_cls.values()), len(bad_images))
        cls_counter.update(split_cls)
        if bad_images:
            issues += 1
            print(f"[错误] {split}: {len(bad_images)} 张图片无法打开: {bad_images[:3]}")

    print()
    print("=" * 64)
    print("分片统计 (图片 / 标签 / 实例):")
    for split, (n_img, n_lb, n_inst, n_bad) in per_split.items():
        print(f"  {split:<6}: 图片 {n_img:>5}  标签 {n_lb:>5}  实例 {n_inst:>6}")
    print()
    print("各类实例数:")
    for i, name in enumerate(names):
        print(f"  {i:<3} {name:<22} {cls_counter.get(i, 0):>6}")
    if poly_points:
        pts = sorted(poly_points.items())
        print(f"\n多边形点数分布 (点数 -> 多边形数):")
        print("  " + ", ".join(f"{k}点:{v}" for k, v in pts))
    if areas:
        a = np.array(areas)
        print(f"\n框面积统计 (归一化): 均值 {a.mean():.4f}  中位 {np.median(a):.4f}  "
              f"min {a.min():.4f}  max {a.max():.4f}")
        print(f"  小框(<0.005) 占比: {(a < 0.005).mean() * 100:.2f}%   "
              f"大框(>0.25) 占比: {(a > 0.25).mean() * 100:.2f}%")

    # 查重
    if args.user_dir and Path(args.user_dir).is_dir():
        user_imgs = [p for p in Path(args.user_dir).rglob("*") if p.suffix.lower() in IMG_SUFFIXES]
        if user_imgs:
            print(f"\n查重: 对 {len(user_imgs)} 张用户图片与 pretrain 图片做 MD5 比对...")
            print("  (计算 pretrain 图片 MD5，约需 1-2 分钟)", flush=True)
            pretrain_hashes = set()
            dup_found = []
            for p in root.rglob("*"):
                if p.suffix.lower() in IMG_SUFFIXES:
                    pretrain_hashes.add(md5(p))
            for up in user_imgs:
                if md5(up) in pretrain_hashes:
                    dup_found.append(str(up))
            if dup_found:
                warnings += 1
                print(f"[警告] 发现 {len(dup_found)} 张与 pretrain 重复的用户图片: {dup_found[:5]}")
            else:
                print("  ✓ 无重复图片")
        else:
            print(f"\n[跳过] {args.user_dir} 中没有图片")
    else:
        print(f"\n[跳过] 未找到用户图片目录 {args.user_dir}")

    print()
    print("=" * 64)
    print(f"校验完成: {issues} 个错误, {warnings} 个警告")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
