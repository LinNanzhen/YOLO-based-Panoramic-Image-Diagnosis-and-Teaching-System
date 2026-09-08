# 项目交接记录（更新于 2026-09-06）

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
├── web_ui.py                        # Web 入口：页面配置 + 侧边栏导航 + 路由 + session 初始化
├── ui_common.py                     # 各页面共享：科技风 CSS、权重扫描、上传+预览+置信度组件
├── page_train.py                    # 页面1 模型训练（旧 3 类）
├── page_monitor.py                  # 页面2 训练监控（plotly 曲线，读 runs/ 下 results.csv）
├── page_winter_train.py             # 页面3 🧪 Winter 微调（学生上传标注→GPU 微调→可视化→跳诊断）
├── page_diagnose.py                 # 页面4 智能诊断（旧 3 类：龋齿/充填体/阻生牙）
├── page_winter.py                   # 页面5 Winter 教学诊断（含 run_winter_diagnose + 模型来源下拉）
├── winter_train_lib.py              # ★ Winter 微调编排库：整理数据集 + 检测/角度微调 + 混淆矩阵绘图
├── dental_common.py                 # ★ Web+脚本共享视觉模块：中文字体回退、画框/标签、
│                                    #   margin 裁切、Winter 常量、detect_and_classify() 流水线、
│                                    #   KMP_DUPLICATE_LIB_OK 自动设置
├── dental_yolo_train.py          # DentalYOLOPipeline 训练核心（勿动）
├── pretrain/                     # 8423 张牙科全景片（14 类 polygon 标注，Roboflow 数据集，不入 git）
├── dataset/                      # 用户 Winter 标注（219 张全景片，已入 git）
│   ├── images/trainset/          #   219 张图（testset 空）
│   ├── labels/trainset/          #   219 个 .txt（CR 换行；54 空=无智齿；共 384 颗智齿）
│   ├── combos_angulation.csv     #   类别映射：0=Mesioangular 1=Vertical 2=Inverted 3=Horizontal 4=Buccolingual
│   └── class_list_annotation.txt #   原标注类别清单
├── winter_dataset/               # 整理后训练集（**当前已删，可由 prepare_winter_dataset.py 重建**）：
│                                 #   detect/（单类智齿）+ cls/angulation/（5 类裁切）+ report.json
│                                 # 注：早期数据集 images/（74 张 trainset，77.jpg 损坏）已于 2026-09-06
│                                 #   经用户确认后删除，其中 37 张不在 dataset/images/ 里且未入 git，已不可恢复
├── weights/demo/                 # ★ 随仓库提交的课堂演示权重 expB（8.8MB，新克隆即可用）
│   ├── expB-detect/weights/best.pt
│   ├── expB-cls-angulation/weights/best.pt
│   └── expB/metrics.json
├── runs/
│   ├── pretrain/weights/best.pt  # ★ 域内预训练权重（后续实验起点）
│   ├── winter/expA-detect|cls-angulation/   # 实验 A（COCO 基线）权重+曲线+混淆矩阵
│   ├── winter/expB-detect|cls-angulation/   # ★ 实验 B（预训练）权重（课堂演示用）
│   ├── winter/exp_<学号>-detect|cls-angulation/  # 学生课堂微调产物（每人一实验，不入 git）
│   ├── winter/expA|expB|exp_<学号>/metrics.json  # 指标文件
│   └── _baseline/                # 2026-09-03 重构前的备份 + predict_demo 像素基线 + 验证脚本
│   # 2026-09-06 已删（均可由 scripts 重跑再生）：demo_predictions/、pretrain_smoke/、
│   #   detect/、classify/（这两个只剩 val 曲线、0 个 best.pt）、winter/exp_test*（测试实验，
│   #   非 expA/expB 正式结果，含 71M 物化数据集副本）
├── annotated_presets/            # auto_annotate 产物（**当前已删**，重跑 auto_annotate.py 再生）
├── preview/                      # 预标注质检预览图（**当前已删**，随上一条一起再生）
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
    ├── student_guide.md          # 学生使用指南（打印/投屏用）
    └── winter_web_finetune.md    # 学生网页微调课堂手册（标注→上传→微调→诊断新片）
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
- 课堂演示权重：`weights/demo/expB-*/weights/best.pt`（随仓库提交，新克隆即可用）；本机重训产物在 `runs/winter/expB-*/weights/best.pt`。诊断页两处都扫，同名时 `runs/winter` 优先。

### 4.4 Web Winter 教学诊断页（阶段 4）
- 页面 4 **🦷 Winter 教学诊断**（page_winter.py）：上传全景片 → 检测智齿 → 裁切角度分类 →
  中文结论"智齿#1·图中左侧·近中阻生 81%"，五类彩色框 + 图例 + 读片说明。
- 页面默认权重路径已填 expB；分类权重缺失自动降级为纯检测框；无中文字体自动切英文。
- 验证：08.jpg 推理结果与人工标注一致（近中阻生 81%）。样例图 `runs/demo_predictions/winter_web_demo.jpg`
  已在 2026-09-06 清理中删除，需要时用 §6 的 `scripts/predict_demo.py` 命令重新生成。

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

### 4.6 网页版 Winter 微调课堂功能（2026-09-05）
- **需求**：学生各自标注同一批全景片 → 在**自己的免费云 GPU** 上跑本 Web，页面上传数据、
  微调出 `exp_<学号>` 的模型、页内可视化，再去 Winter 诊断页选自己的模型上传新片看结果。
- **新增**：`winter_train_lib.py`（整理数据集 + 检测/角度微调 + 混淆矩阵绘图编排，复用
  scripts 逻辑）；`page_winter_train.py`（页面3 🧪 Winter 微调：三种数据来源 + 参数 + 结果
  可视化 + 跳诊断）；`docs/winter_web_finetune.md`（学生手册）。
- **改动**：`scripts/prepare_winter_dataset.py` 抽出可导入的 `build_winter_dataset()`
  （CLI 输出不变，已用 dataset 全量回归 report.json 一致）；`ui_common.py` 加
  `list_winter_experiments`/`load_metrics_json`；`page_winter.py` 加「使用模型」下拉
  （按 exp 配对 det/cls 权重、带 mAP/top1，优先当前学号）；`web_ui.py` 导航 radio 加 key
  支持程序化跳页 + 新增路由/session 键。
- **验证**：py_compile 全过；prepare CLI 回归 report.json 一致；1-epoch CPU 端到端冒烟
  （6 图 → detect+cls 权重+metrics.json）通过。学生子集若只有 1 种角度会在库里直接报中文提示。
- **本机 GPU 实测（2026-09-05）**：219 张全量、检测/分类各 15 轮、RTX 4060，约 10 分钟跑完
  （检测 mAP50=0.973 / 角度 top1=0.931）；exp_test 权重推理 08.jpg 与人工标注一致（近中+水平）；
  诊断页下拉正确列出 exp_test/expA/expB 及指标，跨页跳转选中正常。（exp_test 仅为当时验收用，
  已于 2026-09-06 清理中删除；下拉现在列 expA/expB 与学生自己的 `exp_<学号>`。）
- **⚠ Windows DataLoader spawn 坑**：多进程 DataLoader 要求主脚本有 `if __name__ == "__main__":`
  保护——**Web 路径安全**（`streamlit.web.cli` 自带 guard，已确认），但用命令行脚本直接调
  `winter_train_lib.run_winter_finetune` 时必须加 guard，否则报
  `_check_not_importing_main` RuntimeError。另外受限网络下 AMP 自检下载 yolo11n.pt 会重试
  数分钟（最终自动跳过，不影响训练）。
- **约定**：课堂口径仍为"只标角度 5 类"（0 Mesioangular→4 Buccolingual），顺序即标签 ID；
  makesense 需按该顺序建标签。诊断页 expA/expB 为教师演示，学生模型 `exp_<学号>*` 不入 git。

### 4.7 可用性/安全修复（2026-09-06）

针对项目报告的"立即做"五项，逐项修复并本机实测。

**① CWD 路径依赖（影响面最广）**
- 症状：`streamlit run web_ui.py` 从非仓库根目录启动时（student_guide.md 末尾就写了这种启动方式），
  `runs/winter`、`./results`、`yolov8n.pt` 等相对路径全部落空，表现为"下拉框里没有可选模型"
  而非明确报错。
- 改法：`dental_common.py` 新增 `REPO_ROOT`/`RUNS_DIR`/`WINTER_RUNS_DIR`/`DEMO_WEIGHTS_DIR`
  （由文件位置推导）；`run.py` 启动即 `os.chdir(REPO_DIR)`；`ui_common.py` 新增
  `DEFAULT_DETECT_RUNS_DIR`；`scripts/finetune_winter.py` 与 `dental_yolo_train.py` 各自持有
  模块级 `REPO_ROOT`（不 import dental_common，避免与 ultralytics 的导入顺序耦合）。
  替换点：page_diagnose/page_monitor 的扫描目录、page_train 的输出目录与三处预训练权重、
  page_winter_train 读 results.png、page_winter 的实验扫描、finetune_winter 的 COCO 权重与
  四个路径参数（在 main() 里一次性归一化）。
- 顺带修掉一个默认可用性断点：页面1 训练默认写 `./results`，而页面2/3 扫的是
  `runs/detect/results` —— "训练完切到监控页看曲线"这条文档路径原先是断的，现已统一到同一常量。

**② 页面1 训练读到零标签（比报告里写的更严重）**
- 报告原以为是 `val: trainset` 导致指标虚高；实测发现问题更大：
  `_create_dataset_yaml()` **从未引用 `self.label_root`**。ultralytics 定位标签的方式是把图片
  路径里最后一段 `/images/` 替换成 `/labels/`，而本项目图片在 `<data_root>/trainset/`、标签在
  另一个扁平目录，路径里没有 `/images/` 段，替换是空操作。
  实测：`img2label_paths(['<tmp>/data_root/trainset/1.png'])` → `<tmp>/data_root/trainset/1.txt`（不存在）。
  即训练时一张标签都读不到，却不报任何错，各类 AP 恒为 0。
- 改法：`_create_dataset_yaml()` → `_prepare_split()` + `_build_dataset_yaml()`。按 label_root
  配对图片与同名 .txt，固定种子（seed=42, val_ratio=0.15）划出**真正独立**的 val，再用硬链接
  （跨卷回退 copy2）物化出 `<run>/_dataset/{images,labels}/{train,val[,test]}`，data.yaml 写绝对 path。
  硬链接避免每次训练复制几十 MB 用户数据，且不改动用户源目录。
- `test:` 仅在真的提供了 `test_label_root` 时才写 —— 原先声明 `test: testset` 却无任何测试标签，
  是纯装饰；宁可省略也不要指向无标签目录去产出全 0 的假测试指标。
  `DentalYOLOPipeline.__init__` 相应新增 `test_label_root` / `val_ratio` / `seed` 参数，
  page_train 从 `label_zip_info['test_label_root']` 传入（仅"服务器图像+上传标签"模式有该字段）。
- 顺带修掉：预训练权重回退分支拼的是 `f"yolov8{self.model_size}.pt"`，而 model_size 是
  `nano/small/medium`，得到 `yolov8nano.pt` 这种非法权重名，回退必然失败。新增 `_size_letter()` 归一化。
- **实测**：用仓库真实 `dataset/`（219 图 + 219 标签）跑 1 epoch，ultralytics 扫描日志
  `train: 186 images, 0 backgrounds, 0 corrupt` / `val: 33 images`，186+33=219，
  **与 experiment_report.md 记录的 train 186 / val 33 完全一致**；train/val 交集为空；源目录未被改动。

**③ 依赖版本锁定**
- 新增 `requirements.txt`（14 项，取自实测能复现 experiment_report.md 全部指标的环境）。
- `run.py` 改为解析该文件：缺失的才装（带固定版本号），版本不符只警告不擅自升级。
  旧版写死一份无版本清单，缺失时 `pip install torch` 会拉最新版，可能让页面读指标静默拿到 None。
- 实测确认 PEP 440 行为：`torch==2.7.1` 能匹配本机 `2.7.1+cu126`，不会误报版本不符。

**④ ZIP 解压路径校验（⚠ 原报告对严重性的判断被实测推翻，如实记录）**
- `page_train.py` 新增 `_safe_extractall()`：逐成员校验 `resolve()` 后仍落在目标目录内，
  并拒绝 symlink 成员（`external_attr >> 16` 的 Unix mode == `0o120000`）。
  `extract_and_detect_dataset` 与 `extract_labels_zip` 两处裸 `zf.extractall` 均已替换。
- **实测纠正**：原报告称这是"Zip Slip 漏洞、共享云实例上等于任意文件覆盖"，这是错的。
  实测（Python 3.11.7）构造含 `../../escaped.txt`、`/abs/escaped2.txt`、S_IFLNK 成员的 ZIP
  交给 `zipfile.extractall`：`../` 组件被剥掉后落在目标目录内、绝对路径降级为目录内相对路径、
  symlink 成员被当普通文件写入（内容是链接目标字符串，不是真软链）。仓库外**没有**任何文件被写出。
  即 CPython 标准库本身已中和经典 Zip Slip，原代码不存在可利用的任意文件写入。
- 改动的真实价值是**失败方式**：stdlib 静默改写后照常解压，用户上传畸形 ZIP 会得到一份路径被
  悄悄改过的残缺数据集（如 `../../labels/` 下的标签被平铺到根），页面只报"图片与标签不配对"，
  根因不可见；现在显式拒绝并指名是哪个成员。symlink 检查属防御性冗余（防标准库行为变更）。
- 教训：涉及安全结论时必须先实测再下判断，不要凭模式匹配（"裸 extractall = Zip Slip"）定性。

**⑤ 演示权重入库**
- 新增 `weights/demo/`，镜像 runs/winter 布局提交 expB 的 detect/cls `best.pt` + `metrics.json`（8.8MB）。
  镜像布局是为了让 `list_winter_experiments` 的后缀配对与 metrics 查找逻辑无需改动即可复用。
- `ui_common.list_winter_experiments(roots=None)` 改为多根扫描，默认 `[WINTER_RUNS_DIR, DEMO_WEIGHTS_DIR]`，
  顺序即优先级（本地训练覆盖内置演示），结果新增 `root`/`is_demo` 字段；page_winter 下拉据此多一档
  "🏫 内置演示"标签。**效果：新克隆的仓库不跑任何训练就能课堂演示。**
- 只复制 `best.pt` 不复制 `last.pt`（两者字节数相同，省一半体积）；未纳入 expA —— 若要让课堂做
  A/B 对比，需再提交约 9MB，待用户决定。
- **`yolov8s.pt`（22MB）后续处置**：本轮修复时曾判定"不移出 git 索引"（它并非零引用、用户先前要求
  保留、且 `git rm --cached` 不减仓库体积）。2026-09-06 清理时用户改为授权删除，现状态是：
  **磁盘文件已删，git 索引与历史里的 blob 仍在**（`git checkout -- yolov8s.pt` 可原样取回，
  仓库体积也因此没有变化）。同时把 `scripts/train_pretrain.py:20` 的 `--model` 说明从
  "可换 yolov8s.pt"改为注明需要时由 ultralytics 自动下载，避免文档指向一个本地不存在的文件。

**⑥ 本轮实测中新发现并修掉的启动级问题**
- `run.py` 在 Windows GBK 控制台下 `print("✓ ...")` 直接抛 `UnicodeEncodeError` 中止启动
  （stdout 被重定向到管道或由服务拉起时必现）。新增 `_force_utf8_stdout()`，在 `__main__` 最先调用。
- 同一个编码坑还埋在训练库深处：页面3 微调跑到一半，`scripts/prepare_winter_dataset.py` print 一个
  `⚠` 就崩在训练线程里，页面只显示一个和训练毫无关系的编码错误。改为在 `dental_common` 导入时调
  `force_utf8_stdio()` —— 文档里的另一种启动方式 `streamlit run web_ui.py` 不经过 run.py，而 web_ui
  会导入各页面模块、页面模块都导入 dental_common，放这里能一次覆盖两种启动路径；
  `prepare_winter_dataset.py` / `train_pretrain.py` 这两个不 import dental_common 的独立 CLI 各自补引导。
- 重定向到文件时 stdout 是**块缓冲**：`python run.py > log`（云服务器/机房部署的文档写法）在进程退出前
  一行都不落盘，教师盯着空日志以为卡死。两处 reconfigure 都加上 `line_buffering=True`，实测依赖检查
  的 14 行 ✓ 现在是边跑边出。
- `dental_yolo_train.py` 直接 import ultralytics 却没设 `KMP_DUPLICATE_LIB_OK`，DataLoader 子进程
  抛 `OMP: Error #15`（本机实测复现）。Web 路径靠 page_diagnose→ui_common→dental_common 的导入顺序
  侥幸被覆盖，但任何直接 import 它的脚本/测试都会中招。已在 import ultralytics 之前补上守卫。
- 解释器下限实测是 **Python 3.10**（page_winter.py:126 与 scripts/finetune_winter.py:151 的签名里
  有运行时求值的 PEP 604 联合类型，且无 `from __future__ import annotations`），`run.py` 据此加了版本守卫。
  另注：本机裸 `python` 指向 MGLTools 的 Python 2.7.11，照文档敲 `python run.py` 会 SyntaxError，
  正确解释器是 `/d/Anaconda/python.exe`。

**⑦ device=cpu 训练会永久毒化本进程的 GPU 推理（不在原报告里，浏览器实测时撞到）**
- 症状：页面3 用 CPU 微调（免费云 GPU 不可用时的推荐做法）→ 点"跳诊断页"→ 页面4/5 的推理在同一进程里
  全部 `AssertionError: Invalid device id`，且只能重启 Streamlit 才能恢复。页面只显示
  "检测推理失败: Invalid device id"，完全看不出与上一步的 CPU 训练有关。
- 根因：ultralytics `select_device` 在 `device="cpu"` 时写 `os.environ["CUDA_VISIBLE_DEVICES"] = ""`
  （源码注释说是 force `is_available()=False`），但 torch 2.7.1 上这个 force 不成立：
  `is_available()` 走 CUDA Runtime API（初始化时读一次，此后恒 True），`device_count()` 走 NVML
  （CUDA 初始化前每次重读环境变量，此时返回 0）。两者互相矛盾 → select_device 判定"有 GPU"进 CUDA 分支
  → `get_device_properties(0)` 因 `0 >= device_count()` 越界。
  （`_cached_device_count` 只在 `_initialized` 后才写，所以还原环境变量即可恢复，无需重启进程 —— 实测 1→0→1。）
- 改法：`dental_common.preserve_cuda_visible_devices()` 上下文管理器，调用前保存、退出时还原（异常也还原）。
  套在**所有同进程 ultralytics 调用点**上：dental_common ×2、finetune_winter ×4（train+val 各两处）、
  dental_yolo_train ×2。`scripts/auto_annotate.py` / `scripts/train_pretrain.py` 的 `device=` 有意不套 ——
  它们是一次性 CLI 进程，污染随进程结束消失。
- 防回归：`tests/test_device_poisoning.py` 用 AST 遍历这三个文件，断言每个 `train/val/predict` 调用都
  落在该 with 块内（receiver 限定 model/self.model/det/clf），改名或漏套会当场失败。

**⑧ st.image 参数与 pin 的 streamlit 版本不匹配（浏览器实测时撞到）**
- 症状：两个诊断页一上传图片就 `TypeError: ImageMixin.image() got an unexpected keyword argument
  'use_container_width'`，诊断根本走不到。HEAD 里就是这样，不是本轮改出来的。
- 根因：`st.image` 的"撑满列宽"参数在 streamlit **1.42** 才从 `use_column_width` 改名成
  `use_container_width`，而 requirements.txt pin 的（本机装的）是 1.30.0 —— 代码是按新版 API 写的。
  同名参数在 `button`/`plotly_chart`/`dataframe`/`pyplot` 上 1.30 就已支持，所以**只有 st.image 这一处会炸**，
  也正因如此它躲过了肉眼审查（实测逐个查过签名，见 tests/test_streamlit_compat.py）。
- 改法：`ui_common.show_image(image, caption)` 包装，按 `inspect.signature(st.image)` 挑当前版本认的名字；
  6 处 `st.image(..., use_container_width=True)` 全部改走它。之所以不直接把名字换回 `use_column_width`：
  云上 code-server 环境可能装的是新版，旧名在那边会弹弃用横幅；pin 只保证 run.py 会警告、不强制降级。
- 防回归：不再逐点硬编码断言，而是 AST 遍历全部 `st.xxx(...)` 调用，逐个关键字参数去**当前安装的**
  streamlit 签名里查存在性（跳过收 `**kwargs` 的和无签名的）。以后再写出超前于 pin 的 API 会被点名到行号。

**验证**：新增 `tests/`（pytest，8 个文件 80 条用例，本机 Python 3.11.7 + streamlit 1.30.0 + torch
2.7.1+cu126 全绿，`python -m pytest tests/ -q`）：

| 文件 | 条数 | 覆盖 |
| --- | --- | --- |
| `test_paths.py` | 10 | ①⑤ 仓库外 CWD 起子进程仍能解析全部资源、demo 权重齐备、本地 runs 优先于内置演示 |
| `test_dataset_split.py` | 12 | ② 旧布局确实定位不到标签（根因固化）、新布局能定位、train/val 按图不相交、切分可复现、源目录不被污染、219 张真实数据复现 186/33 |
| `test_requirements.py` | 9 | ③ 全部 pin 均已安装且满足、PEP 440 local version、缺 packaging 时的回退、版本不符只警告不装 |
| `test_zip_safety.py` | 7 | ④ 畸形成员被显式拒绝且一个都不写、目录成员不误判、**stdlib 基线证明不会逃逸仓库** |
| `test_demo_inference.py` | 7 | ⑤ 内置 expB 端到端推理、与人工标注一致、metrics.json 与文档数字对齐（`-m slow`） |
| `test_device_poisoning.py` | 6 | ⑦ 守卫还原语义、毒化态下 torch 自相矛盾（子进程）、CPU 训练后 GPU 诊断仍可用、AST 全覆盖 |
| `test_console_encoding.py` | 9 | ⑥ GBK 管道下基线真的会崩、两种启动路径都切到 UTF-8、重定向逐行落盘 |
| `test_streamlit_compat.py` | 20 | ⑧ 全部 `st.*` 关键字参数在安装版本里存在、show_image 选对名字 |

（条数曾为 81/21：`test_streamlit_compat.py` 按仓库根目录的 `*.py` 逐文件参数化，2026-09-06 清理删掉
`dental_yolo_visualization.py` 后自然少一个用例，不是断言丢失。）

除单元测试外还实测了：219 张真实数据的 1-epoch 训练（ultralytics 扫描日志 186/33）、
"CPU 微调 → 同进程 GPU 诊断"的完整课堂路径、从 `/tmp` 启动 run.py 后 `healthz=200`，
以及在真实浏览器里走完页面5「上传 08.jpg → 诊断」并核对结论（近中阻生 81% + 水平阻生 100%，
与 `dataset/labels/trainset/08.txt` 的类别 0/3 一致），五个页面均无异常。

### 4.8 仓库瘦身清理（2026-09-06，经用户逐项授权）

仓库从 **1.2G 降到 689M**（释放约 510M）。删前逐个核过体积、git 跟踪状态、硬链接实况与代码引用。

| 删除项 | 体积 | 可否恢复 |
| --- | --- | --- |
| `images/`（早期数据集，74 张 trainset） | ~20M | ❌ **37 张不在 `dataset/images/` 且未入 git，已永久丢失**（用户在看到该警告后仍选择删除） |
| `winter_dataset/`、`annotated_presets/`、`preview/` | ~120M | ✅ 跑 §6 的 `prepare_winter_dataset.py` / `auto_annotate.py` 再生 |
| `runs/{demo_predictions,pretrain_smoke,detect,classify}` | ~15M | ✅ `predict_demo.py` / `train_pretrain.py --smoke` 再生；`detect`/`classify` 本就只剩 val 曲线、**0 个 best.pt** |
| `runs/winter/exp_test*`（含 `_data` 物化副本） | ~71M | ✅ 测试用实验，非 expA/expB 正式结果；页面3 重跑一次微调即得 |
| `dental_yolo_visualization.py`、`index.html`、`yolov8s.pt` | ~22M | ✅ 均为 git 跟踪文件，`git checkout -- <path>` 取回（当前是未提交的工作区删除） |
| 各处 `__pycache__/`、`.pytest_cache/` | 少量 | ✅ 自动生成 |

**刻意保留**：`pretrain/`（444M，未入 git 的原始输入，删了不可再生）、`runs/pretrain/`、
`runs/winter/expA*`+`expB*`（正式实验结果）、`runs/_baseline/`（重构前备份 + 像素基线 + 验证脚本）、
`dataset/`（440 个跟踪文件，核心标注）、`weights/demo/`（开箱即用演示权重）、
`yolov8n.pt`/`yolov8n-cls.pt`（页面1/3 的 COCO 回落起点）、`.zcode/plans/`（用户历史决策记录）。

删后复跑全套测试 **80 passed**，`py_compile` 全过，Web 五个页面渲染无异常 —— 确认删掉的都不是运行时依赖。
注：本轮推翻了 `.zcode/plans/` 里"`dental_yolo_visualization.py`、`index.html`、`yolov8s.pt` 一律不动"
的先前决定，以本文件为准。


## 5. 未完成 / 待用户决策

1. **类别映射确认**（见 4.2 ⚠️）——影响分类结论是否可靠。
2. **部署方式**：推荐云端/机房服务器（学生浏览器访问，零安装）；备选教师机局域网。教师端操作已写进 student_guide.md 末尾。
3. **可选增强**：① 自测模式（学生先判读再揭晓 AI 结论，侧边栏学号输入框已预留）；② 补标稀有类（颊舌向 35 例最少）；③ 扩充数据至 200-300 张；④ 分类调参让 B 反超（--cls-lr0 0.002）。
4. ~~旧管线文件保留未动~~ → **已于 2026-09-06 授权删除**，明细与重建命令见 4.8。
   仍然成立的是：`dental_yolo_train.py` 在 4.7 ② 中被改过（原 `_create_dataset_yaml` 导致零标签，属必修），
   但对外构造参数保持向后兼容，仅新增可选参数。
5. 早期 `images/trainset/77.jpg` 曾是损坏的截断 JPEG（auto_annotate 会跳过）。该图连同整个 `images/`
   已在 4.8 中删除，此条只在有人恢复那批旧数据时才需要重新留意。
6. **本轮遗留待决策**（见 4.7 ⑤）：
   - 是否把 expA 权重也提交进 `weights/demo/`（约 +9MB）——课堂要做 A/B 对比就需要，否则下拉里只有 expB；
   - `yolov8s.pt` 的磁盘副本已删（4.8），但 blob 仍在 git 历史里、仓库体积没变。
     真要瘦身须 `git filter-repo` 改写历史，属破坏性操作，需明确授权后再做。
7. **页面4「智能诊断」没有可用权重**（浏览器实测确认）：它扫的是 `runs/detect/results`，
   该目录在 4.8 清理中已整个删除（删前也只剩 val 曲线、0 个 best.pt），`runs/` 又在 .gitignore 里、
   expA 检测权重从未入库 —— 所以**本机与新克隆现在表现一致**：`find_all_runs` 对缺失目录返回 `[]`，
   页面显示"⚠️ 未找到训练好的模型权重。请先进行训练或检查目录。"，是明确提示不是崩溃。
   开箱即用只有页面3/5（Winter，靠 `weights/demo/`）。要让页面4 也可用，需先在页面1 训一次，
   或按第 6 条把 expA 检测权重一并提交。
8. **4.7/4.8 的全部改动与删除都还没提交**，目前只是工作区状态（含 3 个跟踪文件的 `D`）。
   三个被删的跟踪文件在提交前都能 `git checkout --` 取回；提交后只能从历史里找。何时暂存/提交由用户决定。

## 6. 常用命令速查（4.7 ① 之后不再要求仓库根目录 CWD）

```bash
PY=/d/Anaconda/python

# 冒烟测试 / 正式预训练（已跑完，重跑可选）
$PY scripts/train_pretrain.py --smoke
$PY scripts/train_pretrain.py --epochs 60 --batch 16

# 标注整理 + 微调（输出目录 winter_dataset/ 已在 4.8 删除，这条就是重建命令）
$PY scripts/prepare_winter_dataset.py --images dataset/images/trainset \
    --labels dataset/labels/trainset --combos dataset/combos_angulation.csv \
    --out winter_dataset --dims angulation
$PY scripts/finetune_winter.py --mode all --init coco --exp expA
$PY scripts/finetune_winter.py --mode all --init pretrain \
    --weights runs/pretrain/weights/best.pt --exp expB \
    --compare-with runs/winter/expA/metrics.json

# 任意图推理可视化（大字号标注，与 Web 页共用 dental_common 流水线）
# 输出 runs/demo_predictions/ 已删，重跑即再生
$PY scripts/predict_demo.py --images <图或目录> \
    --det runs/winter/expB-detect/weights/best.pt \
    --cls runs/winter/expB-cls-angulation/weights/best.pt --out runs/demo_predictions

# 自动预标注（新图先跑这个，再去 X-AnyLabeling 校正补类）
# 输出 annotated_presets/ 与 preview/ 已删，重跑即再生
$PY scripts/auto_annotate.py --images <未标注图目录> \
    --weights runs/pretrain/weights/best.pt --out annotated_presets --preview preview

# 回归测试（80 条；slow 标记的会真加载 .pt 做推理，数十秒）
$PY -m pytest tests/ -q
$PY -m pytest tests/ -q -m "not slow"

# 启动 Web（学生端）
$PY run.py        # 或 streamlit run web_ui.py --server.port 8501

# 装包（默认源不可达时）
$PY -m pip install -i https://mirrors.aliyun.com/pypi/simple/ <包名>
```

## 7. 给下一会话的提示

- 用户是口腔外科教师（Nanzhen Lin），中文交流；目的是课堂让学生用 AI 辅助 Winter 分类判读。
- 代码布局：web_ui.py 只是入口/路由，**改页面逻辑去 page_*.py，改共享样式去 ui_common.py，
  改推理/画框去 dental_common.py**；改完重启 run.py 才生效。
- 改完任何页面/训练库，先跑 `$PY -m pytest tests/ -q -m "not slow"`（十几秒）。
  `test_streamlit_compat.py` 是拿**当前安装的** streamlit 签名做校验的，所以升级 streamlit 后
  必须重跑一遍 —— 4.7 ⑧ 那个"一上传图片就 TypeError"正是代码超前于安装版本造成的。
- 训练/推理进程放后台用 `nohup ... > /tmp/x.log 2>&1 &`，进度看日志；GPU 显存 8GB，detect batch ≤16。
- 2026-09-03 已完成架构重构并提交 git；重构前备份与验证脚本在 runs/_baseline/。
