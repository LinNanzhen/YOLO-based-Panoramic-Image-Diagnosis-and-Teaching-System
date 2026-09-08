# -*- coding: utf-8 -*-
"""页面 4：Winter 教学诊断 —— 检测智齿 → 裁切角度分类 → 中文结论。"""
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

import dental_common as dc
from ui_common import list_winter_experiments, show_image, upload_with_preview
from winter_train_lib import safe_exp


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

    exps = list_winter_experiments()
    ready = [e for e in exps if e.get("det_path")]  # 至少要有检测权重才能诊断

    def exp_label(e: dict) -> str:
        if e.get("is_demo"):
            tag = "🏫 内置演示"
        elif e["exp"].startswith("exp_"):
            tag = "🧪 我的实验"
        else:
            tag = "🏫 教师演示"
        parts = [f"{tag} {e['exp']}"]
        if e.get("mAP50") is not None:
            parts.append(f"检测mAP50={e['mAP50']:.3f}")
        if e.get("top1") is not None:
            parts.append(f"角度top1={e['top1']:.3f}")
        return "  |  ".join(parts)

    exp_by_name = {e["exp"]: e for e in ready}

    # ---- 权重路径与模型下拉的协调 ----
    det_cur = st.session_state.get("winter_det_path")
    cls_cur = st.session_state.get("winter_cls_path")

    def match_by_path():
        for e in ready:
            if e["det_path"] == det_cur and (e.get("cls_path") or "") == (cls_cur or ""):
                return e
        return None

    def apply_exp(e: dict):
        st.session_state["winter_det_path"] = e["det_path"]
        st.session_state["winter_cls_path"] = e["cls_path"] or ""
        return e["det_path"], e["cls_path"] or "", e

    def default_exp() -> str | None:
        for cand in (safe_exp(st.session_state.get("student_id", "Student_01")), "expB"):
            if cand in exp_by_name:
                return cand
        return ready[0]["exp"] if ready else None

    matched = match_by_path()
    manual_typed = bool(det_cur) and matched is None   # 高级区已手填路径

    # 1) 跨页跳转请求（Winter 微调页完成后一次性消费）：优先于下拉旧状态
    jump = st.session_state.get("winter_model_selection")
    jump_applied = False
    if jump in exp_by_name and not manual_typed:
        det_cur, cls_cur, matched = apply_exp(exp_by_name[jump])
        jump_applied = True
    st.session_state["winter_model_selection"] = None

    # 2) 用户刚在下拉切换（widget 值 != 当前路径对应实验）时跟随下拉
    widget_val = st.session_state.get("winter_model_select")
    widget_exp = next((e for e in ready if exp_label(e) == widget_val), None)
    if widget_exp is not None and not jump_applied and not manual_typed \
            and (matched is None or matched["exp"] != widget_exp["exp"]):
        det_cur, cls_cur, matched = apply_exp(widget_exp)

    # 3) 首次/无任何匹配时的默认实验（学号 > expB > 最新）
    if not manual_typed and matched is None:
        cand = default_exp()
        if cand is not None:
            det_cur, cls_cur, matched = apply_exp(exp_by_name[cand])

    # 4) 模型来源下拉（切换即换用该实验权重）
    pseudo = "✍️ 手动指定路径（高级区填写，暂不与实验联动）"
    labels = [exp_label(e) for e in ready]
    label_to_exp = {exp_label(e): e["exp"] for e in ready}
    if manual_typed:
        labels = [pseudo] + labels
    if labels:
        cur_label = pseudo if manual_typed else \
            (exp_label(matched) if matched is not None else labels[0])
        if st.session_state.get("winter_model_select") != cur_label:
            st.session_state["winter_model_select"] = cur_label  # 跨页/新增模型后强制默认
        sel_label = st.selectbox(
            "🔬 使用模型（切换即换用该实验的权重）", labels, key="winter_model_select",
            help="下拉里的实验来自两处：runs/winter（你自己在「🧪 Winter 微调」页训练的 exp_<学号>，"
                 "以及教师演示 expA/expB）与 weights/demo（随仓库提交的内置演示 expB，保证新环境开箱即用）。")
        sel_exp = label_to_exp.get(sel_label)
        if sel_exp is not None and not manual_typed \
                and (matched is None or matched["exp"] != sel_exp):
            det_cur, cls_cur, matched = apply_exp(exp_by_name[sel_exp])
        if manual_typed:
            st.caption("当前使用手动填写的权重路径（选择下拉实验可一键切换）。")
    else:
        st.info("🤔 还没有训练好的模型。请先到左侧「🧪 Winter 微调」上传标注数据完成微调，"
                "或在下方高级区手动填写权重路径（如教师已共享 expA/expB 权重）。")

    with st.expander("🛠️ 高级：手动指定权重路径"):
        st.session_state["winter_det_path"] = det_cur or ""
        st.session_state["winter_cls_path"] = cls_cur or ""
        winter_det_path = st.text_input(
            "🦷 智齿检测权重（best.pt）",
            key="winter_det_path",
            help="课堂演示默认使用教师 expB；也可换成自己的 exp_<学号> 或其他产物")
        winter_cls_path = st.text_input(
            "🏷️ 角度分类权重（best.pt）",
            key="winter_cls_path",
            help="可留空 = 只画智齿框不做角度分类（如路径不存在则自动降级）")

    det_ok = Path(winter_det_path).is_file()
    cls_ok = Path(winter_cls_path).is_file()
    if not det_ok and not labels:
        st.warning("⚠️ 尚无可用权重：请先在「🧪 Winter 微调」页训练自己的模型，或填写教师提供的权重路径")
    elif not det_ok:
        st.warning(f"⚠️ 当前模型没有可用的检测权重：`{winter_det_path}`（可换下拉中的其他实验，或高级区手动填写）")
    if det_ok and not cls_ok:
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
                show_image(annotated, caption="🎯 AI Winter 诊断结果")
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
