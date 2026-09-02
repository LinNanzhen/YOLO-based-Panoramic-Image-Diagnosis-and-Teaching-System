# -*- coding: utf-8 -*-
"""页面 1：模型训练 —— 数据集上传/路径配置 + DentalYOLOPipeline 训练。"""
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import streamlit as st

# 导入即做核心模块存在性检查（本模块在 web_ui.py 启动时被导入，相当于启动门禁）
try:
    from dental_yolo_train import DentalYOLOPipeline
except ImportError:
    st.error("❌ 未找到核心模块，请确保 'dental_yolo_train.py' 与页面文件在同一目录下。")
    st.stop()


def extract_and_detect_dataset(zip_file, extract_to: str) -> dict:
    """
    解压 ZIP 数据集并自动识别目录结构。
    支持以下常见格式：
      - images/trainset/ + labels/trainset/   (本平台标准)
      - images/train/   + labels/train/       (YOLO 标准)
      - trainset/ + labels/trainset/          (扁平结构)
    返回 dict: success, data_root, label_root, n_train, n_test, n_labels
    """
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    def count_images(d: Path) -> int:
        if not d.exists() or not d.is_dir():
            return 0
        return sum(1 for f in d.iterdir() if f.is_file() and f.suffix.lower() in image_exts)

    def count_labels(d: Path) -> int:
        if not d.exists() or not d.is_dir():
            return 0
        return sum(1 for f in d.iterdir() if f.is_file() and f.suffix == '.txt')

    extract_path = Path(extract_to)
    extract_path.mkdir(parents=True, exist_ok=True)

    # 解压
    with zipfile.ZipFile(zip_file) as zf:
        zf.extractall(str(extract_path))

    # 如果 zip 内部只有一个顶层文件夹，进入它
    top_items = [d for d in extract_path.iterdir() if d.is_dir()]
    root = top_items[0] if len(top_items) == 1 else extract_path

    # 可能的 images 根目录
    img_root_candidates = [root / "images", root]

    # 训练/测试子目录别名
    train_aliases = ["trainset", "train"]
    test_aliases  = ["testset", "test", "val"]

    # 标签目录候选（相对于 root）
    label_candidates = [
        root / "labels" / "trainset",
        root / "labels" / "train",
        root / "labels",
    ]

    detected_img_root = None
    detected_train_dir = None
    detected_test_dir  = None

    for img_root in img_root_candidates:
        if not img_root.exists():
            continue
        for t_alias in train_aliases:
            t_dir = img_root / t_alias
            if t_dir.exists() and count_images(t_dir) > 0:
                detected_img_root  = img_root
                detected_train_dir = t_dir
                # 找对应的测试目录
                for v_alias in test_aliases:
                    v_dir = img_root / v_alias
                    if v_dir.exists():
                        detected_test_dir = v_dir
                        break
                break
        if detected_train_dir:
            break

    if detected_train_dir is None:
        all_dirs = sorted(str(d.relative_to(root)) for d in root.rglob("*") if d.is_dir())
        return {'success': False,
                'error': '未能识别目录结构，请确认 ZIP 内包含 trainset/ 或 train/ 文件夹',
                'found_dirs': all_dirs[:20]}

    # 统一重命名为平台规范名称（trainset / testset）
    if detected_train_dir.name != "trainset":
        new_train = detected_img_root / "trainset"
        detected_train_dir.rename(new_train)
        detected_train_dir = new_train

    if detected_test_dir and detected_test_dir.name != "testset":
        new_test = detected_img_root / "testset"
        detected_test_dir.rename(new_test)
        detected_test_dir = new_test

    # 找标签目录，并统一重命名
    detected_label_dir = None
    for lc in label_candidates:
        if lc.exists() and count_labels(lc) > 0:
            detected_label_dir = lc
            break

    if detected_label_dir and detected_label_dir.name != "trainset" \
            and detected_label_dir.parent.name == "labels":
        new_label = detected_label_dir.parent / "trainset"
        detected_label_dir.rename(new_label)
        detected_label_dir = new_label

    # 若标签仍未找到，尝试从训练集同目录找 .txt
    if detected_label_dir is None:
        if count_labels(detected_train_dir) > 0:
            detected_label_dir = detected_train_dir  # 图像和标签在同一目录

    return {
        'success': True,
        'data_root': str(detected_img_root),
        'label_root': str(detected_label_dir) if detected_label_dir else str(detected_train_dir),
        'n_train':  count_images(detected_train_dir),
        'n_test':   count_images(detected_test_dir) if detected_test_dir else 0,
        'n_labels': count_labels(detected_label_dir) if detected_label_dir else 0,
    }


def extract_labels_zip(zip_file, extract_to: str) -> dict:
    """
    解压仅含标签的 ZIP 并识别标签目录。
    支持以下格式：
      - labels/trainset/*.txt (+ labels/testset/*.txt)   (本平台标准)
      - labels/train/*.txt                               (YOLO 标准)
      - labels/*.txt / trainset/*.txt / 顶层直接放 .txt  (扁平结构)
    返回 dict: success, label_root, test_label_root, n_labels, n_test_labels, error
    """
    def count_labels(d: Path) -> int:
        if not d.exists() or not d.is_dir():
            return 0
        return sum(1 for f in d.iterdir() if f.is_file() and f.suffix == '.txt')

    extract_path = Path(extract_to)
    extract_path.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_file) as zf:
        zf.extractall(str(extract_path))

    # 如果 zip 内部只有一个顶层文件夹，进入它
    top_items = [d for d in extract_path.iterdir() if d.is_dir()]
    root = top_items[0] if len(top_items) == 1 else extract_path

    # 标签目录候选（train 优先）
    train_candidates = [
        root / "labels" / "trainset",
        root / "labels" / "train",
        root / "labels",
        root / "trainset",
        root / "train",
        root,
    ]
    test_candidates = [
        root / "labels" / "testset",
        root / "labels" / "test",
        root / "labels" / "val",
        root / "testset",
        root / "test",
        root / "val",
    ]

    detected_label_dir = None
    for lc in train_candidates:
        if lc.exists() and count_labels(lc) > 0:
            detected_label_dir = lc
            break

    if detected_label_dir is None:
        all_dirs = sorted(str(d.relative_to(root)) for d in root.rglob("*") if d.is_dir())
        return {'success': False,
                'error': 'ZIP 内未找到 .txt 标签文件，请确认包含 labels/trainset/ 或直接放置 .txt 文件',
                'found_dirs': all_dirs[:20]}

    # 统一整理为平台规范名称（labels/trainset）
    if detected_label_dir == root:
        # .txt 直接在 zip 顶层散落：归拢到 labels/trainset/
        new_label = extract_path / "labels" / "trainset"
        new_label.mkdir(parents=True, exist_ok=True)
        for txt in detected_label_dir.glob("*.txt"):
            txt.rename(new_label / txt.name)
        detected_label_dir = new_label
    elif detected_label_dir.parent.name != "labels" or detected_label_dir.name != "trainset":
        if detected_label_dir.parent.name == "labels":
            # .txt 直接散落在 labels/ 下：归拢到 labels/trainset/
            new_label = detected_label_dir.parent / "trainset"
            new_label.mkdir(exist_ok=True)
            for txt in detected_label_dir.glob("*.txt"):
                txt.rename(new_label / txt.name)
            detected_label_dir = new_label
        else:
            new_label = extract_path / "labels" / "trainset"
            new_label.parent.mkdir(parents=True, exist_ok=True)
            detected_label_dir.rename(new_label)
            detected_label_dir = new_label

    # 顺带识别测试集标签（可选）
    detected_test_label_dir = None
    for tc in test_candidates:
        if tc.exists() and count_labels(tc) > 0:
            detected_test_label_dir = tc
            break
    if detected_test_label_dir and detected_test_label_dir.parent.name != "labels":
        new_test_label = extract_path / "labels" / "testset"
        detected_test_label_dir.rename(new_test_label)
        detected_test_label_dir = new_test_label

    return {
        'success': True,
        'label_root': str(detected_label_dir),
        'test_label_root': str(detected_test_label_dir) if detected_test_label_dir else None,
        'n_labels': count_labels(detected_label_dir),
        'n_test_labels': count_labels(detected_test_label_dir) if detected_test_label_dir else 0,
    }


def render():
    st.markdown("""
    <h1 style="margin-bottom: 5px;">🚀 模型训练中心</h1>
    <p style="color: #6a9bc3; font-size: 16px; margin-top: 0;">
        配置参数，训练你的专属牙科疾病检测模型
    </p>
    """, unsafe_allow_html=True)

    # ── 数据集配置（全宽）──────────────────────────────────────
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📦 数据集配置")

    data_mode = st.radio(
        "数据来源",
        ["🖼️ 图像已在服务器 + 仅上传标签 ZIP（推荐，大数据集）",
         "📤 上传完整 ZIP 数据集",
         "📂 使用服务器路径（图像与标签均在本机）"],
        index=0,
        help="网页上传受代理服务器大小限制（超限报 axios error 413）。"
             "图像较大时请直接放入服务器，仅通过网页上传体积很小的标签 ZIP。"
    )

    image_exts_ui = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    # ---------- 模式 1: 图像已在服务器 + 仅上传标签 ZIP ----------
    if data_mode.startswith("🖼️"):
        st.markdown("""
        **适用于大体积数据集**：图像（.jpg 等）直接放入服务器，网页只上传标签（.txt，体积很小，
        不会触发 413 限制）。

        **操作步骤：**
        1. 将图像文件夹（含 `trainset/` 和 `testset/`）通过 code-server 左侧资源管理器
           拖拽上传到服务器任意目录；
        2. 在下方填写该图像根目录路径；
        3. 将标签打包为 labels.zip 在网页上传。

        **labels.zip 支持的内部格式：**
        ```
        labels.zip
        └── labels/
            └── trainset/   ← YOLO 格式 .txt 标签（或 train/ / 直接散落 .txt）
        ```
        """)

        split_data_root = st.text_input(
            "📁 服务器上图像根目录（含 trainset/ 和 testset/）",
            value="./images",
            key="split_data_root",
            help="例如 /home/user/dental_dataset/images"
        )

        uploaded_label_zip = st.file_uploader(
            "拖拽或点击上传标签 ZIP（仅含 .txt）",
            type=["zip"],
            key="label_zip_uploader",
            help="标签是纯文本文件，体积很小，网页上传不会触发大小限制"
        )

        if uploaded_label_zip is not None:
            file_id = f"{uploaded_label_zip.name}_{uploaded_label_zip.size}"
            if st.session_state.label_zip_file_id != file_id:
                # 清理旧的解压目录
                if st.session_state.label_zip_dir and \
                        Path(st.session_state.label_zip_dir).exists():
                    shutil.rmtree(st.session_state.label_zip_dir, ignore_errors=True)

                with st.spinner(f"正在解压 `{uploaded_label_zip.name}`..."):
                    tmp_dir = tempfile.mkdtemp(prefix="dental_labels_")
                    result = extract_labels_zip(uploaded_label_zip, tmp_dir)

                if result['success']:
                    st.session_state.label_zip_file_id    = file_id
                    st.session_state.label_zip_dir        = tmp_dir
                    st.session_state.label_zip_label_root = result['label_root']
                    st.session_state.label_zip_info       = result
                else:
                    st.session_state.label_zip_file_id    = None
                    st.session_state.label_zip_dir        = None
                    st.session_state.label_zip_label_root = None
                    st.session_state.label_zip_info       = None
                    st.error(f"❌ {result['error']}")
                    if result.get('found_dirs'):
                        st.code("发现的目录：\n" + "\n".join(result['found_dirs']))

        # 图像路径与标签都就绪后：统计 + 预校验标签匹配
        if st.session_state.label_zip_info:
            info = st.session_state.label_zip_info
            img_root = Path(split_data_root)
            train_dir = img_root / "trainset"
            test_dir = img_root / "testset"

            n_train = sum(1 for f in train_dir.iterdir()
                          if f.is_file() and f.suffix.lower() in image_exts_ui) \
                      if train_dir.exists() else 0
            n_test = sum(1 for f in test_dir.iterdir()
                         if f.is_file() and f.suffix.lower() in image_exts_ui) \
                     if test_dir.exists() else 0

            c1, c2, c3 = st.columns(3)
            c1.metric("训练图像", f"{n_train} 张")
            c2.metric("测试图像", f"{n_test} 张")
            c3.metric("标签文件", f"{info['n_labels']} 个")
            st.success(f"✅ 标签已就绪  |  图像目录: `{split_data_root}`  |  标签目录: `{info['label_root']}`")

            label_root = Path(info['label_root'])

            # 预校验：训练图像必须有同名标签
            if not img_root.exists():
                st.error(
                    f"❌ 图像目录不存在：`{split_data_root}`"
                    f"（实际查找位置：`{img_root.resolve()}`）\n\n"
                    "相对路径相对于 Streamlit 的**启动目录**解析。请检查：\n"
                    "- 图像文件夹是否真的在启动目录下，且目录名为 `images`"
                    "（内含 `trainset/` 与 `testset/`）；\n"
                    "- 或直接在上方填写图像目录的**绝对路径**。"
                )
            elif n_train == 0:
                st.error(f"❌ 未在 `{train_dir}` 找到图像，请确认目录包含 trainset/ 子目录")
            else:
                missing = [img.name for img in sorted(train_dir.iterdir())
                           if img.is_file() and img.suffix.lower() in image_exts_ui
                           and not (label_root / f"{img.stem}.txt").exists()]
                if missing:
                    st.error(
                        f"❌ {len(missing)} 张训练图像缺少标签:\n" +
                        "\n".join(missing[:5]) +
                        (f"\n... 和 {len(missing)-5} 张其他" if len(missing) > 5 else "")
                    )
                else:
                    st.success("✅ 预校验通过：所有训练图像都有对应标签")

                # 测试集标签是可选的，仅提示覆盖情况
                if n_test > 0:
                    if info.get('test_label_root'):
                        test_label_dir = Path(info['test_label_root'])
                        n_test_labeled = sum(
                            1 for img in test_dir.iterdir()
                            if img.is_file() and img.suffix.lower() in image_exts_ui
                            and (test_label_dir / f"{img.stem}.txt").exists())
                        st.caption(f"ℹ️ 测试集标签为可选项，当前覆盖 {n_test_labeled}/{n_test} 张测试图像")
                    else:
                        st.caption("ℹ️ 测试集标签为可选项（当前 ZIP 中未包含，训练流程不强制要求）")

    # ---------- 模式 2: 上传完整 ZIP ----------
    elif data_mode.startswith("📤"):
        st.markdown("""
        将包含训练数据的 ZIP 文件拖入下方区域，平台将自动解压并识别目录结构。

        ⚠️ **注意**：网页上传受代理服务器大小限制，超过约 80MB 会报
        `axios error 413`。图像较多时请改用上面的"图像已在服务器 + 仅上传标签"方式。

        **ZIP 内部支持的目录格式：**
        ```
        dataset.zip
        ├── images/
        │   ├── trainset/   ← 训练图像（或 train/）
        │   └── testset/    ← 测试图像（或 test/ / val/）
        └── labels/
            └── trainset/   ← YOLO 格式 .txt 标签
        ```
        """)

        uploaded_zip = st.file_uploader(
            "拖拽或点击上传数据集 ZIP",
            type=["zip"],
            key="dataset_zip_uploader",
            help="仅适合小数据集；大数据集请改用'图像已在服务器'方式"
        )

        if uploaded_zip is not None:
            # 仅在新文件上传时重新解压（用文件名+大小作为唯一标识）
            file_id = f"{uploaded_zip.name}_{uploaded_zip.size}"
            if st.session_state.upload_file_id != file_id:
                # 清理旧的解压目录
                if st.session_state.upload_dataset_dir and \
                        Path(st.session_state.upload_dataset_dir).exists():
                    shutil.rmtree(st.session_state.upload_dataset_dir, ignore_errors=True)

                with st.spinner(f"正在解压 `{uploaded_zip.name}`..."):
                    tmp_dir = tempfile.mkdtemp(prefix="dental_dataset_")
                    result = extract_and_detect_dataset(uploaded_zip, tmp_dir)

                if result['success']:
                    st.session_state.upload_file_id      = file_id
                    st.session_state.upload_dataset_dir  = tmp_dir
                    st.session_state.upload_data_root    = result['data_root']
                    st.session_state.upload_label_root   = result['label_root']
                    st.session_state.upload_dataset_info = result
                else:
                    st.session_state.upload_file_id      = None
                    st.session_state.upload_dataset_dir  = None
                    st.session_state.upload_data_root    = None
                    st.session_state.upload_label_root   = None
                    st.session_state.upload_dataset_info = None
                    st.error(f"❌ {result['error']}")
                    if result.get('found_dirs'):
                        st.code("发现的目录：\n" + "\n".join(result['found_dirs']))

        # 显示当前已解压的数据集状态
        if st.session_state.upload_dataset_info:
            info = st.session_state.upload_dataset_info
            c1, c2, c3 = st.columns(3)
            c1.metric("训练图像", f"{info['n_train']} 张")
            c2.metric("测试图像", f"{info['n_test']} 张")
            c3.metric("标签文件", f"{info['n_labels']} 个")
            st.success(f"✅ 数据集已就绪  |  图像目录: `{info['data_root']}`  |  标签目录: `{info['label_root']}`")

    # ---------- 模式 3: 服务器路径 ----------
    else:
        st.markdown("适用于数据集（图像和标签）已存在于服务器上的情况。")
        local_data_root  = st.text_input("📁 图像根目录（含 trainset/ 和 testset/）",
                                         value="./images",
                                         key="local_data_root")
        local_label_root = st.text_input("🏷️ 标签目录（直接含 .txt 文件）",
                                         value="./labels/trainset",
                                         key="local_label_root")

    st.markdown('</div>', unsafe_allow_html=True)

    # ── 参数配置 + 训练状态（两列）────────────────────────────
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🛠️ 训练参数配置")

        output_dir = st.text_input(
            "💾 结果输出目录",
            value="./results"
        )

        st.markdown("---")

        # 模型超参数
        c1_inner, c2_inner = st.columns(2)
        with c1_inner:
            model_size = st.selectbox(
                "🔬 模型规模",
                ["nano", "small", "medium"],
                index=0,
                help="Nano最快 | Medium最准"
            )
            epochs = st.number_input(
                "🔄 训练轮数",
                min_value=1,
                max_value=300,
                value=50
            )
        with c2_inner:
            batch_size = st.number_input(
                "📦 批次大小",
                min_value=1,
                max_value=64,
                value=8
            )
            lr0 = st.number_input(
                "📈 学习率",
                value=0.001,
                format="%.4f"
            )

        patience = st.slider(
            "⏱️ 早停耐心值",
            0, 50, 10,
            help="多少轮没提升则停止训练"
        )
        device_id = st.text_input(
            "🖥️ GPU设备 ID",
            value="0",
            help="-1 表示使用 CPU"
        )

        use_pretrained = st.checkbox("✅ 使用预训练权重", value=True)
        pretrained_path = ""
        if use_pretrained:
            pretrain_source = st.selectbox(
                "🎯 预训练权重来源",
                ["官方 COCO 权重", "域内预训练 (pretrain 数据集)", "自定义路径"],
                index=0,
                help="域内预训练: 先用 pretrain/ 数据集（8423 张牙科全景片）预训练 YOLOv8，"
                     "再以此权重微调你的数据，小数据集效果通常优于直接使用 COCO 权重。"
                     "预训练方法见 scripts/train_pretrain.py"
            )
            if pretrain_source == "官方 COCO 权重":
                pretrained_path = f"yolov8{model_size[0]}.pt"
            elif pretrain_source == "域内预训练 (pretrain 数据集)":
                pretrained_path = "runs/pretrain/weights/best.pt"
                if not os.path.exists(pretrained_path):
                    st.warning("⚠️ 未找到预训练权重，请先在云端运行 scripts/train_pretrain.py 生成")
            else:
                pretrained_path = st.text_input(
                    "📁 权重文件路径",
                    value="runs/pretrain/weights/best.pt",
                    help="填写权重文件路径，如 runs/pretrain/weights/best.pt"
                )
                if pretrained_path and not os.path.exists(pretrained_path):
                    st.warning(f"⚠️ 路径不存在: `{pretrained_path}`")
            st.caption(f"📥 将加载: `{pretrained_path}`")

        st.markdown("---")

        class_names_input = st.text_area(
            "🏷️ 疾病类别（每行一个，顺序对应标签ID 0, 1, 2...）",
            value="\n".join(st.session_state.class_names),
            height=120,
            help="每行填写一个类别名称，ID从0开始自动编号。修改后将在本次训练和推理中生效。"
        )
        class_names_list = [c.strip() for c in class_names_input.splitlines() if c.strip()]
        if class_names_list:
            st.session_state.class_names = class_names_list
            st.caption(f"共 {len(class_names_list)} 个类别: " + ", ".join(f"`{c}`" for c in class_names_list))

        st.markdown("<br>", unsafe_allow_html=True)
        start_btn = st.button("🔥 启动训练", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📈 训练状态")

        if start_btn:
            # 确定数据集路径来源
            if data_mode.startswith("🖼️"):
                data_root  = st.session_state.get("split_data_root", "./images")
                label_root = st.session_state.label_zip_label_root
                src_tag    = "服务器图像 + 上传标签"
            elif data_mode.startswith("📤"):
                data_root  = st.session_state.upload_data_root
                label_root = st.session_state.upload_label_root
                src_tag    = "上传数据集"
            else:
                data_root  = st.session_state.get("local_data_root", "./images")
                label_root = st.session_state.get("local_label_root", "./labels/trainset")
                src_tag    = "服务器路径"

            try:
                if data_root is None or label_root is None:
                    st.error("❌ 数据集未就绪，请先在上方「数据集配置」中完成图像路径填写或 ZIP 上传")
                elif not os.path.exists(data_root) or not os.path.exists(label_root):
                    st.error(f"❌ 目录不存在，请检查路径！\n- 图像: `{data_root}`\n- 标签: `{label_root}`")
                else:
                    st.info(f"📂 数据来源: {src_tag}  |  图像: `{data_root}`")

                    with st.spinner(f"正在初始化 YOLOv8-{model_size} 模型..."):
                        pipeline = DentalYOLOPipeline(
                            data_root=data_root,
                            label_root=label_root,
                            output_dir=output_dir,
                            model_size=model_size,
                            pretrained_weights=pretrained_path if use_pretrained else None,
                            class_names=st.session_state.class_names
                        )

                    st.success("✅ 模型初始化完成！")

                    progress_bar = st.progress(0)
                    status_text  = st.empty()
                    status_text.info("🔄 训练进行中... 请切换至「训练监控」页面查看实时曲线")

                    pipeline.train(
                        epochs=epochs,
                        batch_size=batch_size,
                        patience=patience,
                        lr0=lr0,
                        device=int(device_id)
                    )

                    progress_bar.progress(100)
                    st.balloons()

                    st.markdown("### 📊 训练结果")
                    results_dir = pipeline.results_subdir

                    res_img = results_dir / "results.png"
                    if res_img.exists():
                        st.image(str(res_img), caption="训练指标曲线", use_container_width=True)

                    cm_img = results_dir / "confusion_matrix.png"
                    if cm_img.exists():
                        st.image(str(cm_img), caption="混淆矩阵", use_container_width=True)

                    st.success(f"✅ 训练完成！权重保存至: `{results_dir}`")

            except Exception as e:
                st.error(f"❌ 运行出错: {str(e)}")
                st.exception(e)
        else:
            st.markdown("""
            <div class="neon-card">
                <p style="color: #00d4ff; font-size: 14px; margin: 0;">
                    👆 上传数据集后，在左侧配置参数并点击「启动训练」
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 💡 快速入门指南")
            st.markdown("""
            - 🖼️ **大数据集**: 图像直接放入服务器，网页仅上传标签 ZIP（避免 413 限制）
            - 📦 **小数据集**: 将 images/ + labels/ 打包成 ZIP 上传
            - 🧪 **初次测试**: 建议使用 Nano 模型 + 10 Epochs
            - 🖼️ **图片格式**: 支持 JPG / PNG / BMP
            - 💾 **显存不足**: 请减小 Batch Size
            - 📊 **实时监控**: 训练开始后切换至「训练监控」页面
            """)

        st.markdown('</div>', unsafe_allow_html=True)
