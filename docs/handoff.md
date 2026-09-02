# 项目交接记录（更新于 2026-09-03）

> 给新对话的完整上下文。**新会话第一步：让 AI 阅读本文件**，即可无缝继续。
> 若本文件与仓库实际状态不一致，以仓库为准并更新本文件。

## 1. 项目一句话

YOLOv8 牙科全景片 AI 诊断教学平台（Streamlit Web 应用），当前主线任务：
**Winter 阻生智齿角度分类课堂演示**——域内预训练 → 微调 → Web 教学诊断页 → 部署给学生用。

## 2. 环境与运行须知（重要，踩过的坑）

| 项目 | 事实 |
|---|---|
| 操作系统 | Windows 10/11，Shell = Git Bash（命令路径用 `C:/...` 或 `/c/...`） |
| Python | `D:\Anaconda\python.exe`（3.11.7）；`python --version` 默认是 Python 2.7，**必须用全路径或激活 conda 环境** |
| 关键库 | torch 2.7.1+cu126（GPU 可用）、ultralytics 8.3.233、streamlit 1.30.0、cv2 4.11、numpy 1.26.4（MKL） |
| GPU | RTX 4060 Laptop 8GB，本地即可训练（预训练 60 epochs 约 65 分钟） |
| **OMP 冲突** | Windows 下报 `OMP: Error #15` → 训练前设 `KMP_DUPLICATE_LIB_OK=TRUE`。**web_ui.py 已内置**（dental_common.py 导入时自动设置）；scripts/*.py 也各自内置 |
| **网络受限** | pypi.org / github.com / 清华源经常不可达（间歇性恢复）。可用：阿里云 `https://mirrors.aliyun.com/pypi/simple/`（PyPI 包）与 `https://mirrors.aliyun.com/pytorch-wheels/cu126/`（CUDA 轮子，需 curl 直下 wheel 再 pip install，不能当 pip index 用）。pip 全局配置指向清华源（不可达），装包必须 `-i` 覆盖；**run.py 已改为阿里云镜像** |
| 路径约定 | 各脚本/Web 以**仓库根目录为 CWD** 运行；data.yaml 的 `path` 字段被 ultralytics 用 `path/` 拼接 train/val，**path 必须写绝对路径或确保 CWD 正确**（冒烟测试曾因相对路径翻倍出 bug） |
| 中文字体 | Web 标注图中文依赖系统字体：Windows 有 msyhbd.ttc；Linux 服务器若无 Noto CJK/WenQuanYi 会自动降级为英文标签（逻辑在 dental_common.load_draw_font） |
| 数据细节 | `dataset/labels/trainset/*.txt` 是 **CR（\r）换行符**——shell 的 wc/awk 会数错，Python `splitlines()` 正常；git 可能显示为单行。`.gitattributes` 已对 dataset/labels 设 `-text` 防 git 改写 |
| **改代码后生效** | 页面逻辑在 page_*.py / ui_common.py，Streamlit 对已导入模块热重载不可靠——**改完重启 run.py** |

## 3. 仓库结构（2026-09-03 重构后）

```
impacted/
├── run.py                        # 启动器（依赖检查 + 阿里云镜像安装 + streamlit 启动）
├── web_ui.py                     # Web 入口：页面配置 + 侧边栏导航 + 路由（~120 行）
├── ui_common.py                  # 各页面共享：科技风 CSS、权重扫描、上传+预览+置信度组件
├── page_train.py                 # 页面1 模型训练（含 ZIP 数据集解析 + DentalYOLOPipeline）
├── page_monitor.py               # 页面2 训练监控（plotly 曲线，读 runs/ 下 results.csv）
├── page_diagnose.py              # 页面3 智能诊断（旧 3 类：龋齿/充填体/阻生牙）
├── page_winter.py                # 页面4 Winter 教学诊断（含 run_winter_diagnose()）
├── dental_common.py              # ★ Web+脚本共享视觉模块：中文字体回退、画框/标签、
│                                 #   margin 裁切、Winter 常量、detect_and_classify() 流水线、
│                                 #   KMP_DUPLICATE_LIB_OK 自动设置
├── dental_yolo_train.py          # DentalYOLOPipeline 训练核心（勿动）
├── dental_yolo_visualization.py  # 可视化模块（旧，已无人引用，保留备用）
├── pretrain/                     # 8423 张牙科全景片（14 类 polygon 标注，Roboflow 数据集，不入 git）
├── dataset/                      # 用户 Winter 标注（219 张全景片，已入 git）
│   ├── images/trainset/          #   219 张图（testset 空）
│   ├── labels/trainset/          #   219 个 .txt（CR 换行；54 空=无智齿；共 384 颗智齿）
│   ├── combos_angulation.csv     #   类别映射：0=Mesioangular 1=Vertical 2=Inverted 3=Horizontal 4=Buccolingual
│   └── class_list_annotation.txt #   原标注类别清单
├── images/                       # 早期数据集（74 张 trainset，与 dataset/ 不同；77.jpg 损坏；不入 git）
├── winter_dataset/               # 整理后训练集：detect/（单类智齿）+ cls/angulation/（5 类裁切）+ report.json
├── runs/
│   ├── pretrain/weights/best.pt  # ★ 域内预训练权重（后续实验起点）
│   ├── winter/expA-detect|cls-angulation/   # 实验 A（COCO 基线）权重+曲线+混淆矩阵
│   ├── winter/expB-detect|cls-angulation/   # ★ 实验 B（预训练）权重（课堂演示用）
│   ├── winter/expA|expB/metrics.json        # 指标文件
│   ├── demo_predictions/         # 66 张预测/真实对照图 + winter_web_demo.jpg
│   └── _baseline/                # 2026-09-03 重构前的备份 + predict_demo 像素基线 + 验证脚本
├── annotated_presets/            # auto_annotate 产物（160 框预标注 + summary.json）
├── preview/                      # 73 张预标注质检预览图（不入 git）
├── scripts/
│   ├── train_pretrain.py         # 域内预训练（--smoke 冒烟）
│   ├── validate_pretrain.py      # 预训练数据校验
│   ├── auto_annotate.py          # 自动预标注（预览图用 dental_common 画框）
│   ├── prepare_winter_dataset.py # 标注→数据集（支持 --dims 单维度、--combos CSV）
│   ├── finetune_winter.py        # 微调实验 A/B 驱动
│   └── predict_demo.py           # 推理可视化（推理/画框走 dental_common，与 Web 完全一致）
└── docs/
    ├── handoff.md                # 本文件
    ├── cloud_training_guide.md   # 云端训练操作手册（含故障排查表）
    ├── winter_annotation_guide.md# 标注规范
    ├── experiment_report.md      # 实验 A/B 对比报告
    └── student_guide.md          # 学生使用指南（打印/投屏用）
```

## 4. 已完成的核心工作与结果

### 4.1 域内预训练（阶段 0-1）
- 冒烟测试修复 2 个 bug：`train_pretrain.py` 冒烟 data.yaml 路径翻倍（path 必须绝对/正确）；OMP 冲突（KMP_DUPLICATE_LIB_OK）。
- 正式预训练：60 epochs、imgsz 640、batch 16、本地 GPU 65 分钟。
- 验证集（2071 图）mAP50=0.453；**impacted tooth 类 mAP50=0.926 / Recall=0.954**。
- 产出：`runs/pretrain/weights/best.pt`。

### 4.2 Winter 数据集（阶段 2）
- 用户标注 = **仅角度一维**（非计划中的三维组合）：219 图 / 384 颗智齿，5 类：
  Mesioangular 102、Vertical 122、Horizontal 65、Inverted 60、Buccolingual 35。
- 切分（按图，防泄漏）：train 186 / val 33（裁切 326 / 58）。
- ⚠️ **类别 ID 映射为推断**：按用户 `labels.txt` 顺序 0=近中 1=垂直 2=倒置 3=水平 4=颊舌向，
  **未经用户确认**——若标注工具内顺序不同需改 `dataset/combos_angulation.csv` 重跑。

### 4.3 微调实验 A/B（阶段 3，nano）
| 实验 | 检测 mAP50 / mAP50-95 | 角度分类 top1 | 初始化 |
|---|---|---|---|
| A | 0.9845 / 0.7288 | **0.9655** | COCO（yolov8n.pt / yolov8n-cls.pt） |
| B | **0.9943 / 0.7532** | 0.9483 | 域内预训练 best.pt（分类=骨干迁移） |

- 结论：预训练对**检测**增益明确（mAP50-95 +2.4%）；分类两者均 ~95%+，唯一易混对 **Vertical↔Mesioangular**（A 错 2、B 错 3，58 张验证裁切）。
- B 分类略低原因：分类头随机初始化且未单独调 lr（可试 `--cls-lr0 0.002` 复跑）。
- 课堂演示权重：`runs/winter/expB-detect/weights/best.pt` + `runs/winter/expB-cls-angulation/weights/best.pt`。

### 4.4 Web Winter 教学诊断页（阶段 4）
- 页面 4 **🦷 Winter 教学诊断**（page_winter.py）：上传全景片 → 检测智齿 → 裁切角度分类 →
  中文结论"智齿#1·图中左侧·近中阻生 81%"，五类彩色框 + 图例 + 读片说明。
- 页面默认权重路径已填 expB；分类权重缺失自动降级为纯检测框；无中文字体自动切英文。
- 验证：08.jpg 推理结果与人工标注一致（近中阻生 81%）；样例图 `runs/demo_predictions/winter_web_demo.jpg`。

### 4.5 架构重构（2026-09-03，本次）
- **web_ui.py 1764 行单文件 → 入口(~120) + ui_common + page_train/monitor/diagnose/winter 五个文件**，
  每页一个 `render()` 函数，消除页面顶层代码块导致的变量跨分支泄漏。
- **新建 dental_common.py**：字体回退/画框/裁切/Winter 常量/检测分类流水线，Web 与
  predict_demo、auto_annotate 共用；内置 KMP_DUPLICATE_LIB_OK 防护（修复 web_ui 未内置问题）。
- 删除死代码：find_latest_results_csv、DentalYOLOVisualizer 死导入、sys/glob/json 死导入、
  未使用的 CSS 类（chart-container/status-idle）。
- **run.py**：清华镜像 → 阿里云镜像；改为"先检查依赖，缺失时才询问安装"。
- 清理垃圾文件：preview.zip(33M)、yolo11n.pt（零引用）、runs/demo_predictions.zip、.DS_Store、__pycache__。
- .gitignore 增补 pretrain/、images/、preview/；新增 .gitattributes（dataset/labels 设 -text）。
- **验证全过**：py_compile 全部文件；AppTest 渲染 4 页无异常；predict_demo 重构前后输出
  **像素级一致**（3 张样例 0 差异像素）；run_winter_diagnose(08.jpg) 结果与基线一致
  （近中 0.81 + 水平 1.00）；auto_annotate 2 图冒烟 OK；headless 启动 HTTP 200。
- 重构前旧版本备份在 `runs/_baseline/`（web_ui.py、run.py、predict_demo.py、auto_annotate.py）。

## 5. 未完成 / 待用户决策

1. **类别映射确认**（见 4.2 ⚠️）——影响分类结论是否可靠。
2. **部署方式**：推荐云端/机房服务器（学生浏览器访问，零安装）；备选教师机局域网。教师端操作已写进 student_guide.md 末尾。
3. **可选增强**：① 自测模式（学生先判读再揭晓 AI 结论，侧边栏学号输入框已预留）；② 补标稀有类（颊舌向 35 例最少）；③ 扩充数据至 200-300 张；④ 分类调参让 B 反超（--cls-lr0 0.002）。
4. 旧管线文件按用户要求保留未动：dental_yolo_train.py（仅页面1在用）、dental_yolo_visualization.py（无人引用）、index.html（设计稿）、images/、annotated_presets/、preview/、yolov8s.pt。
5. 早期 `images/trainset/77.jpg` 损坏（截断 JPEG）——auto_annotate 会跳过；如要用该图需替换原文件。

## 6. 常用命令速查（仓库根目录 CWD）

```bash
PY=/d/Anaconda/python

# 冒烟测试 / 正式预训练（已跑完，重跑可选）
$PY scripts/train_pretrain.py --smoke
$PY scripts/train_pretrain.py --epochs 60 --batch 16

# 标注整理 + 微调（已跑完；数据增补后重跑）
$PY scripts/prepare_winter_dataset.py --images dataset/images/trainset \
    --labels dataset/labels/trainset --combos dataset/combos_angulation.csv \
    --out winter_dataset --dims angulation
$PY scripts/finetune_winter.py --mode all --init coco --exp expA
$PY scripts/finetune_winter.py --mode all --init pretrain \
    --weights runs/pretrain/weights/best.pt --exp expB \
    --compare-with runs/winter/expA/metrics.json

# 任意图推理可视化（大字号标注，与 Web 页共用 dental_common 流水线）
$PY scripts/predict_demo.py --images <图或目录> \
    --det runs/winter/expB-detect/weights/best.pt \
    --cls runs/winter/expB-cls-angulation/weights/best.pt --out runs/demo_predictions

# 自动预标注（新图先跑这个，再去 X-AnyLabeling 校正补类）
$PY scripts/auto_annotate.py --images <未标注图目录> \
    --weights runs/pretrain/weights/best.pt --out annotated_presets --preview preview

# 启动 Web（学生端）
$PY run.py        # 或 streamlit run web_ui.py --server.port 8501

# 装包（默认源不可达时）
$PY -m pip install -i https://mirrors.aliyun.com/pypi/simple/ <包名>
```

## 7. 给下一会话的提示

- 用户是口腔外科教师（Nanzhen Lin），中文交流；目的是课堂让学生用 AI 辅助 Winter 分类判读。
- 代码布局：web_ui.py 只是入口/路由，**改页面逻辑去 page_*.py，改共享样式去 ui_common.py，
  改推理/画框去 dental_common.py**；改完重启 run.py 才生效。
- 训练/推理进程放后台用 `nohup ... > /tmp/x.log 2>&1 &`，进度看日志；GPU 显存 8GB，detect batch ≤16。
- 2026-09-03 已完成架构重构并提交 git；重构前备份与验证脚本在 runs/_baseline/。
