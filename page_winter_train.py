# -*- coding: utf-8 -*-
"""页面 5：Winter 微调 —— 学生上传自己的标注 → GPU 微调 → 可视化 → 跳转 Winter 诊断。

沿用「模型训练」页的交互模式：三种数据集来源（上传完整 ZIP / 图像在服务器 + 上传标签
ZIP / 服务器路径）+ 参数配置 + 阻塞式训练。训练逻辑复用 winter_train_lib +
scripts/（prepare_winter_dataset / finetune_winter），产物写入 runs/winter/<exp>。
"""
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

import winter_train_lib as wtl
from page_train import extract_and_detect_dataset, extract_labels_zip
from ui_common import show_image

# 角度 5 类的中文名（用于预览分布表）
_ZH = {"Mesioangular": "近中阻生", "Vertical": "垂直阻生", "Inverted": "倒置阻生",
       "Horizontal": "水平阻生", "Buccolingual": "颊舌向阻生"}

_IMG_UI = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
_TXT = ".txt"

# 本页专用 session key 前缀，避免与「模型训练」页共用状态
_K = {
    "full_zip_id": "wt_full_zip_id",
    "full_zip_dir": "wt_full_zip_dir",
    "full_data_root": "wt_full_data_root",
    "full_label_root": "wt_full_label_root",
    "full_info": "wt_full_info",
    "label_zip_id": "wt_label_zip_id",
    "label_zip_dir": "wt_label_zip_dir",
    "label_zip_root": "wt_label_zip_root",
    "label_zip_info": "wt_label_zip_info",
}


def _flat_images_dir(root) -> Path:
    """目录平铺化：含 trainset/train 子目录时进入它；否则视为已平铺的图像目录。"""
    r = Path(root)
    for sub in ("trainset", "train"):
        cand = r / sub
        if cand.is_dir() and any(p.is_file() and p.suffix.lower() in _IMG_UI
                                 for p in cand.iterdir()):
            return cand
    return r


def _flat_labels_dir(root) -> Path:
    """标签目录平铺化：尝试 labels/trainset 等常见子目录；否则视为已平铺的 .txt 目录。"""
    base = Path(root)
    for cand in (base / "labels" / "trainset", base / "labels" / "train",
                 base / "labels", base / "trainset", base / "train"):
        if cand.is_dir() and any(p.is_file() and p.suffix == _TXT for p in cand.iterdir()):
            return cand
    return base


def _count_images(d: Path) -> int:
    return sum(1 for p in d.iterdir()
               if p.is_file() and p.suffix.lower() in _IMG_UI) if d.is_dir() else 0


def _dataset_summary(images_dir, labels_dir):
    """校验并返回 (ok, summary_info)。用于上传后即时预览每类框数与问题。"""
    return wtl.inspect_dataset(images_dir, labels_dir)


def render_dataset_preview(images_dir, labels_dir):
    """展示数据集概览：图/标签/每类智齿框分布；有问题则以 error 红字给出。"""
    info = _dataset_summary(images_dir, labels_dir)
    n_img = _count_images(Path(images_dir))
    c1, c2, c3 = st.columns(3)
    c1.metric("图像", f"{n_img} 张")
    c2.metric("标签文件", f"{max(0, n_img - len(info['missing']))} 个")
    c3.metric("智齿框", f"{info['n_boxes']} 颗")

    rows = []
    for i, en in enumerate(wtl.ANGLE_CLASS_ORDER):
        rows.append({"类别 ID": i, "类别(标注顺序)": en, "中文": _ZH.get(en, ""),
                     "框数": info["by_class_id"].get(i, 0)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True,
                 hide_index=True)
    st.caption("类别 **ID 即标注顺序**（0 近中 → 4 颊舌向），请与标注时使用的类别清单核对，"
               "顺序错了模型会把角度学反，且不会报错。")

    if info["errors"]:
        st.error("❌ 标签格式问题:\n" + "\n".join(" - " + e for e in info["errors"][:5]))
    if info["missing"]:
        st.error(f"❌ {len(info['missing'])} 张图像缺标签: "
                 + ", ".join(info["missing"][:5]))
    if info["orphan"]:
        st.warning(f"⚠️ {len(info['orphan'])} 个标签找不到对应图像（已忽略）: "
                   + ", ".join(info["orphan"][:5]))
    if not info["ok"]:
        st.warning("数据集尚未通过预检，修正问题后再开始微调。")
    else:
        st.success("✅ 预检通过：图片-标签全部配对，格式无误。")
    return info["ok"]


def render():
    for k in _K.values():
        st.session_state.setdefault(k, None)

    st.markdown("""
    <h1 style="margin-bottom: 5px;">🧪 Winter 微调实验</h1>
    <p style="color: #6a9bc3; font-size: 16px; margin-top: 0;">
        上传你自己标注的智齿数据 → 本机 GPU 微调「智齿检测 + Winter 角度分类」→ 可视化结果，
        再到「Winter 教学诊断」上传新全景片试你的模型
    </p>
    """, unsafe_allow_html=True)

    # ── 数据集配置 ────────────────────────────────────────────
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📦 数据集配置")

    data_mode = st.radio(
        "数据来源",
        ["📤 上传完整 ZIP（含 images + labels）",
         "🖼️ 图像已在服务器 + 仅上传标注 ZIP（推荐）",
         "📂 使用服务器上的图像与标签路径"],
        index=0,
        help="网页上传受代理大小限制（超限报 413）；图像较大请用后两种方式。"
    )

    # 解析后得到的 (images_dir, labels_dir)，任意一种模式就绪后更新
    resolved = None

    # ---------- 模式 A：上传完整 ZIP ----------
    if data_mode.startswith("📤"):
        st.markdown("""
        **ZIP 内部支持的目录格式（与「模型训练」页一致）：**
        ```
        dataset.zip
        ├── images/trainset/  (或 images/train/) ← 全景片
        └── labels/trainset/  (或 labels/train/) ← YOLO .txt 标签（5 类角度）
        ```
        """)
        uploaded_zip = st.file_uploader("拖拽或点击上传数据集 ZIP", type=["zip"],
                                        key="wt_dataset_zip")
        if uploaded_zip is not None:
            file_id = f"{uploaded_zip.name}_{uploaded_zip.size}"
            if st.session_state[_K["full_zip_id"]] != file_id:
                if st.session_state[_K["full_zip_dir"]] and \
                        Path(st.session_state[_K["full_zip_dir"]]).exists():
                    shutil.rmtree(st.session_state[_K["full_zip_dir"]], ignore_errors=True)
                with st.spinner(f"正在解压 `{uploaded_zip.name}`..."):
                    tmp = tempfile.mkdtemp(prefix="winter_dataset_")
                    res = extract_and_detect_dataset(uploaded_zip, tmp)
                if res["success"]:
                    st.session_state[_K["full_zip_id"]] = file_id
                    st.session_state[_K["full_zip_dir"]] = tmp
                    st.session_state[_K["full_data_root"]] = res["data_root"]
                    st.session_state[_K["full_label_root"]] = res["label_root"]
                    st.session_state[_K["full_info"]] = res
                else:
                    for k in (_K["full_zip_id"], _K["full_zip_dir"],
                              _K["full_data_root"], _K["full_label_root"], _K["full_info"]):
                        st.session_state[k] = None
                    st.error(f"❌ {res.get('error')}")
                    if res.get("found_dirs"):
                        st.code("发现的目录：\n" + "\n".join(res["found_dirs"]))

        if st.session_state[_K["full_info"]]:
            info = st.session_state[_K["full_info"]]
            c1, c2, c3 = st.columns(3)
            c1.metric("训练图像", f"{info['n_train']} 张")
            c2.metric("测试图像", f"{info['n_test']} 张")
            c3.metric("标签文件", f"{info['n_labels']} 个")
            st.success(f"✅ 数据集已就绪  |  图像: `{info['data_root']}`  |  标签: `{info['label_root']}`")
            st.markdown("#### 🔍 标注预检与类别分布")
            images_dir = _flat_images_dir(info["data_root"])
            labels_dir = Path(info["label_root"])
            render_dataset_preview(images_dir, labels_dir)
            resolved = (images_dir, labels_dir)

    # ---------- 模式 B：图像已在服务器 + 上传标签 ZIP ----------
    elif data_mode.startswith("🖼️"):
        st.markdown("""
        **适用于大体积图像**：全景片直接放入本机（老师分发的图集），网页只上传标注 ZIP。
        1. 图像目录填下方（可以是含 `trainset/` 的根目录，**或直接是放满 .jpg 的平铺目录**）；
        2. 上传 makesense / X-AnyLabeling 导出的标签 ZIP（内含 .txt，支持
           `labels/trainset/`、`labels/`、或顶层直接散落 .txt）。
        """)
        wt_images_root = st.text_input("📁 服务器上的图像目录", value="./student_images",
                                       key="wt_images_root")
        uploaded_label_zip = st.file_uploader("上传标注 ZIP（仅含 .txt）", type=["zip"],
                                              key="wt_label_zip")
        if uploaded_label_zip is not None:
            file_id = f"{uploaded_label_zip.name}_{uploaded_label_zip.size}"
            if st.session_state[_K["label_zip_id"]] != file_id:
                if st.session_state[_K["label_zip_dir"]] and \
                        Path(st.session_state[_K["label_zip_dir"]]).exists():
                    shutil.rmtree(st.session_state[_K["label_zip_dir"]], ignore_errors=True)
                with st.spinner(f"正在解压 `{uploaded_label_zip.name}`..."):
                    tmp = tempfile.mkdtemp(prefix="winter_labels_")
                    res = extract_labels_zip(uploaded_label_zip, tmp)
                if res["success"]:
                    st.session_state[_K["label_zip_id"]] = file_id
                    st.session_state[_K["label_zip_dir"]] = tmp
                    st.session_state[_K["label_zip_root"]] = res["label_root"]
                    st.session_state[_K["label_zip_info"]] = res
                else:
                    for k in (_K["label_zip_id"], _K["label_zip_dir"],
                              _K["label_zip_root"], _K["label_zip_info"]):
                        st.session_state[k] = None
                    st.error(f"❌ {res.get('error')}")
                    if res.get("found_dirs"):
                        st.code("发现的目录：\n" + "\n".join(res["found_dirs"]))

        if st.session_state[_K["label_zip_info"]]:
            images_dir = _flat_images_dir(wt_images_root)
            labels_dir = Path(st.session_state[_K["label_zip_root"]])
            if not images_dir.is_dir() or _count_images(images_dir) == 0:
                st.error(f"❌ 图像目录无效：`{images_dir}`（相对路径基于启动目录解析）")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("图像", f"{_count_images(images_dir)} 张")
                c2.metric("标签文件", f"{st.session_state[_K['label_zip_info']]['n_labels']} 个")
                c3.metric("智齿框", "-")
                st.success(f"✅ 标签已就绪  |  图像: `{images_dir}`  |  标签: `{labels_dir}`")
                st.markdown("#### 🔍 标注预检与类别分布")
                render_dataset_preview(images_dir, labels_dir)
                resolved = (images_dir, labels_dir)

    # ---------- 模式 C：服务器路径 ----------
    else:
        wt_local_images = st.text_input("📁 图像目录（平铺 .jpg 或含 trainset/）",
                                        value="./dataset/images/trainset",
                                        key="wt_local_images")
        wt_local_labels = st.text_input("🏷️ 标签目录（平铺 .txt）",
                                        value="./dataset/labels/trainset",
                                        key="wt_local_labels")
        images_dir = _flat_images_dir(wt_local_images)
        labels_dir = _flat_labels_dir(wt_local_labels)
        if images_dir.is_dir() and labels_dir.is_dir():
            c1, c2 = st.columns(2)
            c1.metric("图像", f"{_count_images(images_dir)} 张")
            c2.metric("智齿框", "-")
            st.markdown("#### 🔍 标注预检与类别分布")
            render_dataset_preview(images_dir, labels_dir)
            resolved = (images_dir, labels_dir)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── 参数配置 + 训练状态（两列）────────────────────────────
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🛠️ 微调参数")

        exp_name = st.text_input(
            "🧑‍🎓 实验名（学号）",
            value=wtl.safe_exp(st.session_state.get("student_id", "Student_01")),
            key="wt_exp",
            help="每个学生一个实验名，权重与指标保存在 runs/winter/<实验名>(-detect/-cls-angulation)")
        exp = wtl.safe_exp(exp_name)

        init_choice = st.selectbox(
            "🎯 初始权重",
            ["auto（有域内预训练则用，否则 COCO）", "官方 COCO 权重", "域内预训练 (pretrain)"],
            index=0,
            help="域内预训练 = runs/pretrain/weights/best.pt（需先在服务器训练过）；"
                 "COCO = 仓库内自带 yolov8n.pt，最省事。")
        init_map = {"auto（有域内预训练则用，否则 COCO）": "auto",
                    "官方 COCO 权重": "coco",
                    "域内预训练 (pretrain)": "pretrain"}

        model_size = st.selectbox("🔬 模型规模", ["nano", "small", "medium"], index=0)
        device_id = st.text_input("🖥️ GPU 设备", value="0",
                                  help="云 GPU 填 0；无 GPU 调试填 cpu（会很慢）")

        c1p, c2p = st.columns(2)
        with c1p:
            epochs = st.number_input("检测轮数", 1, 300, 60)
            batch = st.number_input("检测批次", 1, 64, 16,
                                    help="8G 显存建议 ≤16；显存不足就调小")
        with c2p:
            cls_epochs = st.number_input("分类轮数", 1, 300, 60)
            cls_batch = st.number_input("分类批次", 1, 128, 32)

        st.markdown("---")
        start_btn = st.button("🔥 开始 Winter 微调", use_container_width=True,
                              type="primary")
        st.caption(f"实验名将保存为 `{exp}`；训练日志输出在服务器终端。")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📈 微调状态")

        if start_btn:
            if resolved is None:
                st.error("❌ 数据集未就绪，请先在上方完成图像/标签上传或路径填写")
            else:
                images_dir, labels_dir = resolved
                st.info(f"📂 数据: 图像 `{images_dir}`  |  标签 `{labels_dir}`")
                try:
                    init_mode = init_map[init_choice]
                    status_text = st.empty()
                    status_text.info("🔄 微调进行中（数据集整理 → 检测 → 角度分类）...\n\n"
                                     "可另开标签页到「📊 训练监控」把目录填 `runs/winter` "
                                     "并勾选「自动刷新」，实时看训练曲线；完成后本页会刷新结果。")

                    metrics = wtl.run_winter_finetune(
                        images_dir=images_dir, labels_dir=labels_dir, exp=exp,
                        init=init_mode, model_size=model_size, device=device_id,
                        epochs=int(epochs), batch=int(batch),
                        cls_epochs=int(cls_epochs), cls_batch=int(cls_batch),
                    )

                    status_text.empty()
                    st.balloons()
                    st.markdown("### ✅ 微调完成")

                    det = metrics.get("mAP50")
                    ang = metrics.get("angulation") or {}
                    c1r, c2r, c3r = st.columns(3)
                    c1r.metric("检测 mAP50", f"{det:.3f}" if det is not None else "-")
                    c2r.metric("检测 mAP50-95",
                               f"{metrics.get('mAP50-95'):.3f}" if metrics.get("mAP50-95") is not None else "-")
                    c3r.metric("角度分类 top1", f"{ang.get('top1'):.3f}"
                               if ang.get("top1") is not None else "-")

                    # 角度混淆矩阵
                    cm = ang.get("confusion")
                    if cm:
                        fig = wtl.plot_angulation_confusion(cm)
                        if fig is not None:
                            st.pyplot(fig)

                    det_run = Path(wtl.DEFAULT_OUT_ROOT) / f"{exp}-detect"
                    res_png = det_run / "results.png"
                    if res_png.exists():
                        show_image(str(res_png), caption="检测训练曲线")

                    st.markdown("**产物：**")
                    st.code(
                        f"检测: runs/winter/{exp}-detect/weights/best.pt\n"
                        f"角度: runs/winter/{exp}-cls-angulation/weights/best.pt\n"
                        f"指标: runs/winter/{exp}/metrics.json")

                    if st.button("🦷 去 Winter 教学诊断，上传新全景片试我的模型",
                                 use_container_width=True, type="primary",
                                 key="goto_winter_diag"):
                        st.session_state["winter_model_selection"] = exp
                        st.session_state["nav_page"] = "🦷 Winter 教学诊断"
                        st.rerun()

                except Exception as e:
                    st.error(f"❌ 微调出错: {e}")
                    st.exception(e)
                    st.caption("显存不足(OOM)请把批次调小再试；缺初始权重请换「官方 COCO 权重」。")
        else:
            st.markdown("""
            <div class="neon-card">
                <p style="color: #00d4ff; font-size: 14px; margin: 0;">
                    👆 上传你的标注数据后，在左侧配置参数并点击「开始 Winter 微调」
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("#### 💡 操作流程")
            st.markdown("""
            1. 📸 **准备数据**：把老师分发的全景片放进本机目录（如 `student_images/`），
               在 makesense/X-AnyLabeling 里按 **5 类角度顺序**（0 近中 Mesioangular → 1 垂直
               Vertical → 2 倒置 Inverted → 3 水平 Horizontal → 4 颊舌向 Buccolingual）
               逐颗框出智齿，导出 YOLO 标签；
            2. 📤 **上传**：上方选择数据来源，上传完整 ZIP 或仅上传标签 ZIP；
            3. 🔥 **微调**：填好学号实验名，点「开始 Winter 微调」，GPU 依次训练 检测 + 角度分类；
            4. 📊 **可视化**：完成后本页显示 mAP / top1 与混淆矩阵，过程中可去「训练监控」看实时曲线；
            5. 🦷 **验证**：跳转 Winter 教学诊断，选自己的实验，上传**没训练过的新全景片**看诊断结果。
            """)

        st.markdown('</div>', unsafe_allow_html=True)
