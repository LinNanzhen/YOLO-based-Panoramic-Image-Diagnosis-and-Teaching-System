# YOLO-based Panoramic Image Diagnosis and Teaching System

## 项目简介

这是一个基于 YOLOv8 的牙科全景片 AI 诊断教学平台（Streamlit Web 应用）。当前主线：
**Winter 阻生智齿角度分类课堂演示** —— 域内预训练 → 微调 → 网页教学诊断 → 部署给学生用。

## 主要功能

- **Winter 教学诊断**：上传全景片，AI 自动框出智齿并按 Winter 分类给出中文角度结论
  （近中 / 垂直 / 水平 / 倒置 / 颊舌向阻生），五类彩色框 + 图例 + 读片说明
- **Winter 微调实验（课堂版）**：学生上传自己标注的数据，页面上点按钮即可在本机 GPU 微调
  出"属于自己的"智齿检测 + Winter 角度分类模型，页内看 mAP/top1/混淆矩阵，再跳到诊断页
  用自己的模型上传新全景片验证
- **智能诊断（旧 3 类）**：龋齿 / 充填体 / 阻生牙检测
- **模型训练与监控**：网页端配置数据集与超参训练；实时训练曲线、混淆矩阵展示
- **命令行训练管线**：域内预训练、自动预标注、数据集整理、A/B 微调实验、推理可视化

## 仓库结构

```
impacted/
├── run.py                  # 启动器：校验依赖版本后启动 Web
├── requirements.txt        # 锁定版本的依赖清单（复现实验报告指标的环境）
├── web_ui.py               # Web 入口：侧边栏导航 + 页面路由
├── ui_common.py            # 各页面共享的样式与 UI 部件
├── page_train.py           # 页面1 模型训练
├── page_monitor.py         # 页面2 训练监控
├── page_winter_train.py    # 页面3 Winter 微调（学生上传数据→GPU 微调→可视化）
├── page_diagnose.py        # 页面4 智能诊断（旧 3 类）
├── page_winter.py          # 页面5 Winter 教学诊断
├── dental_common.py        # Web 与脚本共享：字体/画框/裁切/Winter 常量/推理流水线
├── winter_train_lib.py     # Winter 微调编排库（整理数据集+检测/分类微调+混淆矩阵绘图）
├── dental_yolo_train.py    # DentalYOLOPipeline 训练核心
├── scripts/                # 命令行训练管线（预训练/预标注/数据整理/微调/推理）
├── dataset/                # 用户 Winter 标注（219 张全景片 / 384 颗智齿，5 类）
├── weights/demo/           # ★ 随仓库提交的课堂演示权重 expB（8.8MB，保证开箱即用）
├── pretrain/               # 域内预训练数据（8423 张，14 类，不入 git）
├── runs/                   # 训练产物与权重（不入 git，新克隆时为空）
├── tests/                  # 回归测试（pytest，80 条：路径/切分/依赖/ZIP/演示权重/设备/编码/API 兼容）
└── docs/                   # 操作手册、标注规范、实验报告、学生指南、交接记录
```

> 生成物目录（`winter_dataset/`、`annotated_presets/`、`runs/demo_predictions/` 等）不随仓库保留，
> 需要时用 `scripts/` 里对应命令重建，见 `docs/handoff.md` §6。

## 使用方法

```bash
python run.py        # 校验依赖版本并启动 Web（http://localhost:8501）
```

要求 **Python 3.10+**（实测通过环境 3.11.7）。`run.py` 按 `requirements.txt` 逐项核对：
缺失的才从阿里云镜像安装（带固定版本号），版本不符只打印警告、不擅自升级 ——
本项目锁定 ultralytics 8.3.233 + streamlit 1.30.0，静默升级可能让页面读指标拿到 None。

> ⚠️ Windows 上裸 `python` 可能指向别的 Python（例如某些软件自带的 2.7），
> 会直接 SyntaxError。请用正确解释器启动，如 `D:\Anaconda\python.exe run.py`，
> 或先激活对应 conda 环境。`run.py` 检测到解释器版本过低会给出明确提示。

### 回归测试

pytest 是开发依赖，`requirements.txt` 里故意注释掉了（学生端不需要），先装一次：
`python -m pip install pytest==8.4.2`

```bash
python -m pytest tests/ -q                 # 全部 80 条（含真实权重推理，约 45 秒）
python -m pytest tests/ -q -m "not slow"   # 跳过加载 .pt 的用例，十几秒
```

覆盖：仓库外 CWD 也能解析全部资源、train/val 按图片真正不相交且标签能被 ultralytics 定位、
requirements 的 pin 与实际安装一致、畸形 ZIP 被显式拒绝、内置演示权重端到端推理且与人工标注一致、
`device=cpu` 训练不会毒化同进程后续的 GPU 推理、GBK 控制台/重定向日志下不崩不丢输出。

> ⚠️ 升级 streamlit 后必须重跑：`tests/test_streamlit_compat.py` 是拿**当前安装的** streamlit
> 签名逐个校验 `st.*` 调用的关键字参数，代码超前于安装版本（例如 `st.image(use_container_width=...)`
> 在 1.42 之前不存在）会被点名到行号。

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

### Winter 微调课堂流程（网页版，学生每人一模型）

课堂版不需要命令行：每个学生在自己租的免费云 GPU 上启动 Web（`python run.py`），侧边栏进
「🧪 Winter 微调」页，**上传自己标注的数据 → 点按钮微调 → 页内看 mAP/混淆矩阵 → 跳转
「🦷 Winter 教学诊断」用自己 `exp_<学号>` 的模型上传新全景片看诊断**。训练按实验名隔离，
互不排队。

```text
学生：标注老师发的同一批全景片(5类角度,顺序即ID)
   → 上传数据ZIP/标签ZIP → 页面微调(检测+角度分类)
   → 上传新全景片 → 看到自己模型的 Winter 诊断结果
```

详细学生操作手册见 `docs/winter_web_finetune.md`（含标注顺序、云 GPU 启动、常见问题）。

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

## 致谢

- 感谢 YOLOv8（Ultralytics）的开发团队
- 感谢开源社区的支持

---

**最后更新**：2026-09-03
