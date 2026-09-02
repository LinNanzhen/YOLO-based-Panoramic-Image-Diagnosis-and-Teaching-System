# YOLO-based Panoramic Image Diagnosis and Teaching System

## 项目简介

这是一个基于 YOLOv8 的牙科全景片 AI 诊断教学平台（Streamlit Web 应用）。当前主线：
**Winter 阻生智齿角度分类课堂演示** —— 域内预训练 → 微调 → 网页教学诊断 → 部署给学生用。

## 主要功能

- **Winter 教学诊断**：上传全景片，AI 自动框出智齿并按 Winter 分类给出中文角度结论
  （近中 / 垂直 / 水平 / 倒置 / 颊舌向阻生），五类彩色框 + 图例 + 读片说明
- **智能诊断（旧 3 类）**：龋齿 / 充填体 / 阻生牙检测
- **模型训练与监控**：网页端配置数据集与超参训练；实时训练曲线、混淆矩阵展示
- **命令行训练管线**：域内预训练、自动预标注、数据集整理、A/B 微调实验、推理可视化

## 仓库结构

```
impacted/
├── run.py                  # 启动器：检查/安装依赖后启动 Web
├── web_ui.py               # Web 入口：侧边栏导航 + 页面路由
├── ui_common.py            # 各页面共享的样式与 UI 部件
├── page_train.py           # 页面1 模型训练
├── page_monitor.py         # 页面2 训练监控
├── page_diagnose.py        # 页面3 智能诊断（旧 3 类）
├── page_winter.py          # 页面4 Winter 教学诊断
├── dental_common.py        # Web 与脚本共享：字体/画框/裁切/Winter 常量/推理流水线
├── dental_yolo_train.py    # DentalYOLOPipeline 训练核心
├── dental_yolo_visualization.py  # 可视化模块
├── scripts/                # 命令行训练管线（预训练/预标注/数据整理/微调/推理）
├── dataset/                # 用户 Winter 标注（219 张全景片 / 384 颗智齿，5 类）
├── pretrain/               # 域内预训练数据（8423 张，14 类，不入 git）
├── winter_dataset/         # 整理后的训练集（生成物，可由 scripts 重建）
├── runs/                   # 训练产物与权重（不入 git；expB best.pt 为演示权重）
└── docs/                   # 操作手册、标注规范、实验报告、学生指南、交接记录
```

## 使用方法

```bash
python run.py        # 检查依赖并启动 Web（http://localhost:8501）
```

要求 Python 3.11 + torch/ultralytics/streamlit 等，缺失时 run.py 会从阿里云镜像安装。

## Winter 分类训练流程（口腔外科课堂）

用 `pretrain/`（8423 张牙科全景片）对 YOLOv8 做域内预训练，再微调自己标注的
Winter 分类小数据集（219 张）以提升性能：

1. 冒烟测试 + 域内预训练：`python scripts/train_pretrain.py --smoke` → `python scripts/train_pretrain.py`
2. 自动预标注辅助：`python scripts/auto_annotate.py --images <图片目录> --weights runs/pretrain/weights/best.pt --out annotated_presets`
3. 整理数据集：`python scripts/prepare_winter_dataset.py --images ... --labels ...`
4. 微调对比实验 A/B：`python scripts/finetune_winter.py --mode all --init coco --exp expA` / `--init pretrain --exp expB`
5. 推理可视化：`python scripts/predict_demo.py --images <图或目录> --det <检测权重> --cls <分类权重>`

完整步骤见 `docs/cloud_training_guide.md`，标注规范见 `docs/winter_annotation_guide.md`，
实验结果见 `docs/experiment_report.md`，学生使用说明见 `docs/student_guide.md`。

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

## 致谢

- 感谢 YOLOv8（Ultralytics）的开发团队
- 感谢开源社区的支持

---

**最后更新**：2026-09-03
