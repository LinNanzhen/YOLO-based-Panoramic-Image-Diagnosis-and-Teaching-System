# 云端执行指南：域内预训练 + Winter 分类微调

目标：用 `pretrain/`（8423 张牙科全景片、14 类）先对 YOLOv8 做**域内预训练**，
再把预训练权重作为起点，微调你标注的约 100 张 Winter 分类数据。
预训练不依赖你的标注，可以先跑。

## 0. 准备

1. 把整个仓库上传到趣算云（含 `pretrain/` 目录，约 430MB）。
2. 安装依赖（如未装过）：
   ```bash
   pip install ultralytics torch torchvision opencv-python pillow numpy
   ```
   （`run.py` 也可自动安装 web 端依赖。）

## 1. 冒烟测试（约 3-5 分钟）

验证 pretrain 多边形标签能被正确读取、训练流程通畅，再跑正式预训练：

```bash
python scripts/train_pretrain.py --smoke
```

看到 `✓ 冒烟测试通过` 即通过。若此处报标签解析错误，看文末「故障排查」。

## 2. 正式预训练（约 1-2 小时）

```bash
python scripts/train_pretrain.py --epochs 60 --batch 32
```

- 产出：`runs/pretrain/weights/best.pt`（后续所有实验的起点）。
- GPU 显存不够就把 `--batch` 调小（16 或 8）。
- 某些云 GPU 报 AMP 错误时加 `--no-amp`。

## 3. 标注辅助（数据到手后）

用预训练模型自动框出智齿，生成预标注：

```bash
python scripts/auto_annotate.py \
    --images <你的全景片目录> \
    --weights runs/pretrain/weights/best.pt \
    --out annotated_presets \
    --preview preview
```

- 自动只保留 "impacted tooth"（类 12）的框，导出 YOLO 格式到 `annotated_presets/`。
- `preview/` 里的预览图用于快速检查漏检/误检。
- 标注规范见 `docs/winter_annotation_guide.md`（给标注者）。

## 4. 整理数据集

标注完成后（每颗智齿框的类别 ID = Winter 组合 ID，如 "II-B-近中"）：

```bash
python scripts/prepare_winter_dataset.py \
    --images annotated_presets/images \
    --labels annotated_presets/labels \
    --out winter_dataset
```

产出：单类智齿检测集 `winter_dataset/detect/` + 三维度裁切分类集
`winter_dataset/cls/{relation,position,angulation}` + 类别分布报告
`winter_dataset/report.json`。**先看报告**：若某组合 <10 个样本，建议补标或接受其精度低。

## 5. 微调实验 A/B 对比

```bash
# 实验 A（基线）：官方 COCO 权重直接微调
python scripts/finetune_winter.py --mode all --init coco --exp expA

# 实验 B（核心）：域内预训练权重微调
python scripts/finetune_winter.py --mode all --init pretrain \
    --weights runs/pretrain/weights/best.pt --exp expB \
    --compare-with runs/winter/expA/metrics.json
```

- 对比表直接打印在终端：检测 mAP50/mAP50-95、三维度分类 top1 及 Δ。
- 指标 JSON 保存在 `runs/winter/<exp>/metrics.json`。
- 单跑检测或分类：`--mode detect` / `--mode cls`。
- 小模型抗过拟合用 nano（默认）；显存够可试 `--model-size s`。
- 微调超参默认已针对小数据调低学习率并开早停，可自行调整。

## 6. Web UI 集成与课堂演示

- 训练页「预训练权重来源」选择「域内预训练 (pretrain 数据集)」或填自定义路径，
  即可在 web 界面用预训练权重微调。
- 智能诊断页加载 `runs/winter/expB*/weights/best.pt` 即可课堂演示。
- 完整 Winter 结论（如 "II-B 近中阻生"）由检测框 + 三个分类器的输出组合得到，
  课堂演示可用脚本把三者合并显示（见下方示例）。

```python
# 课堂演示合并输出示例
from ultralytics import YOLO

det = YOLO("runs/winter/expB-detect/weights/best.pt")   # 检测智齿
cls_relation = YOLO("runs/winter/expB-cls-relation/weights/best.pt")
cls_position = YOLO("runs/winter/expB-cls-position/weights/best.pt")
cls_angulation = YOLO("runs/winter/expB-cls-angulation/weights/best.pt")

for r in det("全景片.jpg"):
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = r.orig_img[y1:y2, x1:x2]  # 按检测框裁切
        rel = cls_relation(crop)[0].probs.top1
        pos = cls_position(crop)[0].probs.top1
        ang = cls_angulation(crop)[0].probs.top1
        print(f"智齿 ({x1},{y1})-({x2},{y2}): {rel}-{pos}-{ang}")
```

## 7. 可选：进一步提性能

- **扩充标注**：用预训练模型对更多未标注全景片预标注 → 人工快速修正
  （约 100 张扩到 200-300 张，效果通常最明显）。
- **k-fold 交叉验证**：换 `--seed` 多次跑 `prepare_winter_dataset.py` +
  `finetune_winter.py`，指标取均值，结论更可靠。
- **类别不平衡**：稀有组合样本少时，训练时对罕见角度/位置做旋转、镜像增强。

## 故障排查

| 现象 | 处理 |
|---|---|
| 冒烟测试标签解析报错 | 运行 `python scripts/validate_pretrain.py` 看具体文件；若个别文件损坏，删除对应 图片+标签 后重跑 |
| 显存不足 (OOM) | `--batch` 减半；预训练用 `--batch 16` |
| AMP 报错（RTX 50 系等新卡） | 加 `--no-amp` |
| 预训练 mAP 很低 | 正常现象——预训练只求特征表达，不追求最终 mAP；评估只看微调后的结果 |
| web 页加载权重报错 | 确认路径存在且为 `best.pt`（YOLO 权重），重启 streamlit 生效 |
| Windows 报 `OMP: Error #15` | Anaconda 的 MKL 与 torch 各带一份 libiomp5md.dll 冲突；scripts/ 已内置 `KMP_DUPLICATE_LIB_OK=TRUE`，如仍复现可在运行命令前手动 `set KMP_DUPLICATE_LIB_OK=TRUE` |
| `auto_annotate.py` 报图片读取失败 | 个别图片文件损坏（截断 JPEG 等）会被自动跳过并列入 summary.json，无需中断；损坏图建议替换后重跑 |
