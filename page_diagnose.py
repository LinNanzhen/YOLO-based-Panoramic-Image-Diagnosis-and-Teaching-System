# -*- coding: utf-8 -*-
"""页面 3：智能诊断（旧 3 类）—— 龋齿 / 充填体 / 阻生牙检测。"""
import os
import tempfile
from pathlib import Path

import cv2
import streamlit as st
from PIL import Image

from ui_common import get_best_weights, upload_with_preview


def render():
    st.markdown("""
    <h1 style="margin-bottom: 5px;">👁️ 智能辅助诊断</h1>
    <p style="color: #6a9bc3; font-size: 16px; margin-top: 0;">
        上传牙科 X 光片，AI 自动标注龋齿、充填体和阻生牙
    </p>
    """, unsafe_allow_html=True)

    # 模型选择区
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🎯 选择诊断模型")

    output_dir_scan = st.text_input(
        "📂 权重搜索目录",
        value="./runs/detect/results",
        key="infer_root"
    )

    available_weights = get_best_weights(output_dir_scan)

    if not available_weights:
        st.warning("⚠️ 未找到训练好的模型权重。请先进行训练或检查目录。")
        selected_model_path = None
    else:
        weight_options = [w[0] for w in available_weights]
        selected_option = st.selectbox("🔬 选择模型版本", weight_options)
        selected_model_path = next(w[1] for w in available_weights if w[0] == selected_option)
        st.caption(f"📍 加载路径: `{selected_model_path}`")

    st.markdown('</div>', unsafe_allow_html=True)

    # 图像上传与推理
    col_input, col_output = st.columns(2)

    with col_input:
        uploaded_file, conf_thres = upload_with_preview(
            title="📤 上传影像",
            uploader_label="拖拽或点击上传 X-Ray 图像",
            uploader_key="xray_upload",
            slider_label="🎚️ 置信度阈值",
            slider_min=0.1, slider_max=1.0, slider_value=0.45,
            slider_help="调低可检测更多目标，但可能增加误检",
            preview_caption="📷 原始影像",
        )

    with col_output:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🔍 诊断结果")

        if uploaded_file is not None and selected_model_path:
            process_btn = st.button("⚡ 开始 AI 诊断", use_container_width=True)

            if process_btn:
                with st.spinner("🧠 AI 正在分析影像特征..."):
                    tfile = tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=Path(uploaded_file.name).suffix
                    )
                    tfile.write(uploaded_file.getvalue())
                    tfile.close()

                    try:
                        from ultralytics import YOLO

                        model = YOLO(selected_model_path)
                        results = model.predict(source=tfile.name, conf=conf_thres)
                        result = results[0]

                        img_cv = cv2.imread(tfile.name)
                        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                        h, w = img_cv.shape[:2]

                        detections = []
                        _names = st.session_state.class_names
                        CLASS_NAMES = {i: name for i, name in enumerate(_names)}
                        # 根据类别数量自动生成 RGB 颜色（HSV 均匀采样）
                        import colorsys as _cs
                        CLASS_COLORS = {
                            i: tuple(int(c * 255) for c in _cs.hsv_to_rgb(i / max(len(_names), 1), 0.75, 0.95))
                            for i in range(len(_names))
                        }

                        for box, cls, conf in zip(result.boxes.xywh, result.boxes.cls, result.boxes.conf):
                            cx, cy, bw, bh = box[:4].cpu().numpy()
                            class_id = int(cls.cpu().numpy())
                            confidence = float(conf.cpu().numpy())

                            x1 = max(0, int(cx - bw/2))
                            y1 = max(0, int(cy - bh/2))
                            x2 = min(w, int(cx + bw/2))
                            y2 = min(h, int(cy + bh/2))

                            color = CLASS_COLORS.get(class_id, (255, 255, 255))

                            # 绘制半透明填充
                            overlay = img_cv.copy()
                            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                            img_cv = cv2.addWeighted(overlay, 0.2, img_cv, 0.8, 0)

                            # 绘制边框
                            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 3)

                            # 标签背景
                            label_txt = f"{CLASS_NAMES.get(class_id, str(class_id))}: {confidence:.2f}"
                            (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                            cv2.rectangle(img_cv, (x1, y1-th-10), (x1+tw+10, y1), color, -1)
                            cv2.putText(img_cv, label_txt, (x1+5, y1-5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                            detections.append({
                                "class": CLASS_NAMES.get(class_id),
                                "confidence": confidence
                            })

                        st.image(img_cv, caption="🎯 AI 标注结果", use_container_width=True)

                        if detections:
                            st.success(f"✅ 检测到 {len(detections)} 处异常")

                            # 按类别统计
                            st.markdown("#### 📋 检测详情")
                            for i, d in enumerate(detections, 1):
                                st.markdown(f"🔍 **{d['class']}** - 置信度: `{d['confidence']:.2%}`")
                        else:
                            st.info("✨ 未检测到明显异常")

                    except Exception as e:
                        st.error(f"❌ 推理失败: {e}")
                    finally:
                        os.unlink(tfile.name)

        elif not selected_model_path:
            st.info("👆 请先在上方选择诊断模型")
        else:
            st.info("👈 请在左侧上传影像")

        st.markdown('</div>', unsafe_allow_html=True)
