# 微调对比实验报告（Winter 智齿分类）

日期：2026-09-01 ｜ 设备：RTX 4060 Laptop 8GB ｜ 框架：Ultralytics 8.3.233 / torch 2.7.1+cu126

## 1. 数据

- 标注来源：`dataset/`（219 张全景片，其中 54 张无智齿；165 张含智齿，共 **384 颗智齿**）。
- 标注维度：**角度**（Winter 分类），5 类：Mesioangular(近中) 102、Vertical(垂直) 122、
  Horizontal(水平) 65、Inverted(倒置) 60、Buccolingual(颊舌向) 35。
- 切分（按图片，防裁切泄漏）：train 186 张 / val 33 张（角度裁切 326 / 58）。
- 组合映射：`dataset/combos_angulation.csv`（ID 0-4 按 `dataset/class_list_annotation.txt` 顺序）。

## 2. 实验设置

| 实验 | 检测初始化 | 分类初始化 |
|---|---|---|
| A（基线） | 官方 COCO 权重 yolov8n.pt | COCO 权重 yolov8n-cls.pt |
| B（核心） | 域内预训练 `runs/pretrain/weights/best.pt` | 域内预训练骨干（yolov8n-cls 架构 + 迁移 best.pt 骨干层） |

超参一致：检测 150 epochs / lr0 0.001 / batch 16 / imgsz 640 / patience 30；
分类 100 epochs / lr0 0.0005 / batch 32 / imgsz 224 / patience 20。模型均为 yolov8n。

## 3. 结果

### 检测（单类智齿框）

| 指标 | 实验 A (COCO) | 实验 B (预训练) | Δ |
|---|---|---|---|
| mAP50 | 0.9845 | **0.9943** | +0.0098 |
| mAP50-95 | 0.7288 | **0.7532** | +0.0244 |
| Precision | 0.9644 | — | — |
| Recall | 0.9655 | — | — |

→ **域内预训练对检测有明确增益**（mAP50-95 提升 2.4 个百分点）。

### 角度分类（YOLO-cls，top1）

| 实验 | top1 | top5 |
|---|---|---|
| A（COCO 初始化） | **0.9655** | 1.0000 |
| B（预训练骨干） | 0.9483 | — |

两实验的混淆矩阵几乎相同，唯一易混对是 **Vertical ↔ Mesioangular**（A 错 2 例、B 错 3 例），
其余四类全部正确。B 略低 1.7 个百分点（58 张验证裁切中约 1 例的差距），原因分析：
1. COCO 分类权重连分类头一起预训练，而 B 的分类头是随机初始化；
2. 骨干迁移后分类头训练轮数/学习率未单独调优，可能欠训练；
3. 两者均接近饱和，差异在噪声范围内。

## 4. 结论与建议

- **检测**：域内预训练收益明确，课堂演示用 `runs/winter/expB/expB-detect/weights/best.pt`。
- **分类**：当前数据量下 COCO 与预训练骨干效果相当（均 ~95%+）；若想让预训练骨干
  在分类上也占优，可把分类 lr0 提到 0.001-0.002 或延长训练（`--cls-lr0` / `--cls-epochs`）。
- **后续提升空间**：补标稀有类（Buccolingual 仅 35 例、Inverted 60 例）；对易混对
  Vertical/Mesioangular 可加旋转增强；扩充到 200-300 张后复测。

## 5. 复现命令

```bash
python scripts/prepare_winter_dataset.py --images dataset/images/trainset \
    --labels dataset/labels/trainset --combos dataset/combos_angulation.csv \
    --out winter_dataset --dims angulation

python scripts/finetune_winter.py --mode all --init coco --exp expA
python scripts/finetune_winter.py --mode all --init pretrain \
    --weights runs/pretrain/weights/best.pt --exp expB \
    --compare-with runs/winter/expA/metrics.json
```
