# -*- coding: utf-8 -*-
"""页面 4：Winter 教学诊断 —— 检测智齿 → 裁切角度分类 → 中文结论。"""
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

import dental_common as dc
from ui_common import upload_with_preview


def run_winter_diagnose(img_bgr, det_path, cls_path, conf=0.25, margin=0.2):
    """Winter 教学诊断：检测智齿 → 按框裁切 → 角度分类。

    参数:
        img_bgr    OpenCV BGR 图像 (ndarray)
        det_path   智齿检测权重 (best.pt)
        cls_path   角度分类权重 (best.pt)，文件不存在时只画检测框
        conf       检测置信度阈值
        margin     裁切外扩比例（与训练数据一致）
    返回:
        (ok, 标注图 RGB ndarray 或 None, 智齿列表, 错误信息或 None)
    """
    try:
        from ultralytics import YOLO
    except Exception as e:
        return False, None, [], f"缺少 ultralytics 库: {e}"
    if not Path(det_path).is_file():
        return False, None, [], f"检测权重不存在: `{det_path}`\n请先完成训练，或填写正确的权重路径。"

    clf = None
    if Path(cls_path).is_file():
        try:
            clf = YOLO(cls_path)
        except Exception:
            clf = None

    try:
        det = YOLO(det_path)
        findings = dc.detect_and_classify(
            det, clf, img_bgr, crop_img=img_bgr,
            conf=conf, margin=margin, sort_left_right=True)
    except Exception as e:
        return False, None, [], f"检测推理失败: {e}"

    h, w = img_bgr.shape[:2]
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(pil_img)
    font, has_cjk = dc.load_draw_font(max(24, w // 60))
    line_w = max(4, w // 400)

    teeth = []
    for n, f in enumerate(findings, 1):
        x1, y1, x2, y2 = f["box"]
        det_conf = f["det_conf"]
        side = "图中右侧" if (x1 + x2) / 2 >= w / 2 else "图中左侧"
        angle_en, angle_conf = f["angle_en"], f["angle_conf"]
        color = dc.WINTER_ANGLE_COLORS.get(angle_en, (255, 255, 255))

        angle_zh = dc.WINTER_ANGLE_ZH.get(angle_en, angle_en)
        if has_cjk:
            label = (f"智齿{n}·{side}·{angle_zh} {angle_conf:.0%}" if angle_conf is not None
                     else f"智齿 {det_conf:.0%}")
        else:  # 服务器无中文字体时退回英文，避免乱码
            label = (f"#{n} {angle_en} {angle_conf:.0%}" if angle_conf is not None
                     else f"tooth {det_conf:.0%}")
        dc.annotate_box(dr, (x1, y1, x2, y2), label, color, line_w, font)

        teeth.append({
            "n": n, "side": side, "angle_zh": angle_zh, "angle_en": angle_en,
            "det_conf": det_conf, "angle_conf": angle_conf, "color": color,
        })

    return True, np.array(pil_img), teeth, None


def render():
    st.markdown("""
    <h1 style="margin-bottom: 5px;">🦷 Winter 阻生智齿教学诊断</h1>
    <p style="color: #6a9bc3; font-size: 16px; margin-top: 0;">
        上传全景 X 光片，AI 自动框出智齿并按 Winter 分类给出角度诊断（近中 / 垂直 / 水平 / 倒置 / 颊舌向阻生）
    </p>
    """, unsafe_allow_html=True)

    # 模型配置
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🎯 模型配置")
    winter_det_path = st.text_input(
        "🦷 智齿检测权重（best.pt）",
        value="runs/winter/expB-detect/weights/best.pt",
        key="winter_det_path",
        help="课堂演示默认使用实验 B（域内预训练微调）权重；也可换成 expA 或其他训练产物"
    )
    winter_cls_path = st.text_input(
        "🏷️ 角度分类权重（best.pt）",
        value="runs/winter/expB-cls-angulation/weights/best.pt",
        key="winter_cls_path",
        help="可留空 = 只画智齿框不做角度分类（如路径不存在则自动降级）"
    )
    det_ok = Path(winter_det_path).is_file()
    cls_ok = Path(winter_cls_path).is_file()
    if not det_ok:
        st.warning("⚠️ 检测权重不存在，请先运行训练（scripts/finetune_winter.py）或检查路径")
    if not cls_ok:
        st.info("ℹ️ 角度分类权重不存在，将只显示智齿检测框（不影响演示）")
    st.markdown('</div>', unsafe_allow_html=True)

    col_in, col_out = st.columns(2)
    with col_in:
        winter_file, winter_conf = upload_with_preview(
            title="📤 上传全景片",
            uploader_label="拖拽或点击上传 X-Ray 全景片",
            uploader_key="winter_xray_upload",
            slider_label="🎚️ 检测置信度阈值",
            slider_min=0.10, slider_max=0.95, slider_value=0.30,
            slider_help="调低可检出更多智齿，但可能引入误检",
            preview_caption="📷 原始全景片",
        )

    with col_out:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🔍 Winter 诊断结果")
        if winter_file is not None and det_ok:
            if st.button("⚡ 开始 Winter 诊断", use_container_width=True, key="winter_run_btn"):
                with st.spinner("🧠 AI 正在检测智齿并分析阻生角度..."):
                    buf = np.frombuffer(winter_file.getvalue(), np.uint8)
                    img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    ok, annotated, teeth, err = run_winter_diagnose(
                        img_bgr, winter_det_path,
                        winter_cls_path if cls_ok else "",
                        conf=winter_conf)
                if not ok:
                    st.error(f"❌ {err}")
                else:
                    st.session_state.winter_result = (annotated, teeth)

            if st.session_state.get("winter_result"):
                annotated, teeth = st.session_state.winter_result
                st.image(annotated, caption="🎯 AI Winter 诊断结果", use_container_width=True)
                if teeth:
                    st.success(f"✅ 共检出 {len(teeth)} 颗智齿")
                    st.markdown("#### 📋 诊断明细")
                    for t in teeth:
                        hexc = "#%02x%02x%02x" % t["color"]
                        angle_txt = (f"**{t['angle_zh']}**" if t["angle_conf"] is not None
                                     else "*未分类*")
                        conf_txt = (f"`{t['angle_conf']:.0%}`" if t["angle_conf"] is not None
                                    else "")
                        st.markdown(
                            f"<span style='color:{hexc}; font-size:18px;'>●</span> "
                            f"智齿 #{t['n']}（{t['side']}） {angle_txt} {conf_txt}"
                            f"　<small>检测置信度 {t['det_conf']:.0%}</small>",
                            unsafe_allow_html=True)
                else:
                    st.info("✨ 未检测到智齿（可调低置信度阈值后重试）")
        elif not det_ok:
            st.info("👆 请先提供有效的检测权重")
        else:
            st.info("👈 请先在左侧上传全景片")
        st.markdown('</div>', unsafe_allow_html=True)

    # 教学图例与说明
    with st.expander("📖 Winter 角度分类图例与读片说明"):
        st.markdown("**五类阻生角度（Winter 分类）颜色图例：**")
        for en, zh in dc.WINTER_ANGLE_ZH.items():
            hexc = "#%02x%02x%02x" % dc.WINTER_ANGLE_COLORS[en]
            st.markdown(f"<span style='color:{hexc}; font-size:16px;'>●</span> "
                        f"**{zh}**（{en}）", unsafe_allow_html=True)
        st.markdown("""
        ---
        **读片提示：**
        - 智齿按影像从左到右编号，**图中左侧/右侧**为影像方位（全景片左右镜像约定随设备而异，判读时可与原片方位核对）；
        - 置信度是模型对结论的信心程度，教学使用时建议同时结合影像原始特征独立判读；
        - 本系统角度分类基于课程标注数据（219 张全景片 / 384 颗智齿），近中与垂直阻生为最常见易混对，可重点对照教学。
        """)
