# -*- coding: utf-8 -*-
"""
ui_common.py — 各页面（page_*.py）共享的样式与 UI 部件

包含：科技风全局 CSS、权重扫描、训练指标读取，以及两个诊断页共用的
「上传图片 + 置信度滑条 + 原图预览」卡片。
"""
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

# ==================== 科技风 CSS ====================
_CSS = """
<style>
/* ===== 导入 Google Fonts ===== */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700&family=Rajdhani:wght@300;400;500;600;700&family=Source+Code+Pro:wght@400;500&display=swap');

/* ===== 全局背景与基础样式 ===== */
.stApp {
    background: linear-gradient(135deg, #0a1628 0%, #1a2a4a 50%, #0d2137 100%);
    background-attachment: fixed;
    color: #e8f4fc;
    font-family: 'Rajdhani', sans-serif;
}

/* 添加网格背景效果 */
.stApp::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image:
        linear-gradient(rgba(0, 212, 255, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 212, 255, 0.03) 1px, transparent 1px);
    background-size: 50px 50px;
    pointer-events: none;
    z-index: 0;
}

/* ===== 侧边栏样式 ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1f3c 0%, #152642 50%, #0a1628 100%);
    border-right: 1px solid rgba(0, 212, 255, 0.3);
    box-shadow: 4px 0 20px rgba(0, 212, 255, 0.1);
}

section[data-testid="stSidebar"] .stMarkdown {
    color: #b8d4e8;
}

/* ===== 标题样式 ===== */
h1 {
    font-family: 'Orbitron', monospace !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, #00d4ff, #00ff88, #00d4ff);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradientShift 3s ease infinite;
    text-shadow: 0 0 30px rgba(0, 212, 255, 0.5);
    letter-spacing: 2px;
}

h2, h3 {
    font-family: 'Orbitron', monospace !important;
    font-weight: 600 !important;
    color: #00d4ff !important;
    text-shadow: 0 0 15px rgba(0, 212, 255, 0.4);
}

@keyframes gradientShift {
    0%, 100% { background-position: 0% center; }
    50% { background-position: 200% center; }
}

/* ===== 毛玻璃卡片样式 ===== */
.glass-card {
    background: rgba(13, 31, 60, 0.7);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow:
        0 8px 32px rgba(0, 0, 0, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.05);
    position: relative;
    overflow: hidden;
}

.glass-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00d4ff, transparent);
    animation: scanLine 3s ease-in-out infinite;
}

@keyframes scanLine {
    0% { left: -100%; }
    100% { left: 100%; }
}

/* ===== 霓虹边框卡片 ===== */
.neon-card {
    background: rgba(10, 22, 40, 0.85);
    border: 2px solid transparent;
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
    position: relative;
    box-shadow:
        0 0 15px rgba(0, 212, 255, 0.2),
        inset 0 0 30px rgba(0, 212, 255, 0.05);
}

.neon-card::after {
    content: '';
    position: absolute;
    inset: -2px;
    border-radius: 14px;
    background: linear-gradient(45deg, #00d4ff, #00ff88, #ff00ff, #00d4ff);
    background-size: 400% 400%;
    animation: neonBorder 8s ease infinite;
    z-index: -1;
    opacity: 0.6;
}

@keyframes neonBorder {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

/* ===== 按钮样式 ===== */
div.stButton > button {
    font-family: 'Orbitron', monospace !important;
    font-weight: 600;
    background: linear-gradient(135deg, #00d4ff 0%, #0099cc 50%, #00ff88 100%);
    background-size: 200% 200%;
    color: #0a1628 !important;
    border: none;
    padding: 12px 28px;
    border-radius: 8px;
    font-size: 14px;
    letter-spacing: 1px;
    text-transform: uppercase;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    box-shadow:
        0 4px 15px rgba(0, 212, 255, 0.4),
        0 0 30px rgba(0, 212, 255, 0.2);
}

div.stButton > button:hover {
    background-position: 100% 0;
    transform: translateY(-3px) scale(1.02);
    box-shadow:
        0 8px 25px rgba(0, 212, 255, 0.5),
        0 0 50px rgba(0, 212, 255, 0.3);
}

div.stButton > button:active {
    transform: translateY(0) scale(0.98);
}

/* ===== 输入框样式 ===== */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div > div {
    background: rgba(13, 31, 60, 0.8) !important;
    border: 1px solid rgba(0, 212, 255, 0.3) !important;
    border-radius: 8px !important;
    color: #e8f4fc !important;
    font-family: 'Source Code Pro', monospace !important;
    transition: all 0.3s ease;
}

.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: #00d4ff !important;
    box-shadow: 0 0 15px rgba(0, 212, 255, 0.3) !important;
}

/* ===== Slider 样式 ===== */
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #00d4ff, #00ff88) !important;
}

/* ===== 进度条样式 ===== */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #00d4ff, #00ff88, #00d4ff);
    background-size: 200% 100%;
    animation: progressGlow 2s ease infinite;
}

@keyframes progressGlow {
    0%, 100% { background-position: 0% 0; }
    50% { background-position: 100% 0; }
}

/* ===== 警告框样式 ===== */
.stSuccess {
    background: rgba(0, 255, 136, 0.1) !important;
    border: 1px solid rgba(0, 255, 136, 0.4) !important;
    border-radius: 8px;
}

.stInfo {
    background: rgba(0, 212, 255, 0.1) !important;
    border: 1px solid rgba(0, 212, 255, 0.4) !important;
    border-radius: 8px;
}

.stWarning {
    background: rgba(255, 170, 0, 0.1) !important;
    border: 1px solid rgba(255, 170, 0, 0.4) !important;
    border-radius: 8px;
}

.stError {
    background: rgba(255, 68, 68, 0.1) !important;
    border: 1px solid rgba(255, 68, 68, 0.4) !important;
    border-radius: 8px;
}

/* ===== 标签文字样式 ===== */
.stMarkdown, p, span, label {
    color: #b8d4e8 !important;
}

/* ===== 状态指示器 ===== */
.status-indicator {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 20px;
    font-family: 'Source Code Pro', monospace;
    font-size: 12px;
    font-weight: 500;
}

.status-active {
    background: rgba(0, 255, 136, 0.15);
    border: 1px solid rgba(0, 255, 136, 0.5);
    color: #00ff88;
}

/* ===== 数据展示标签 ===== */
.metric-label {
    font-family: 'Orbitron', monospace;
    font-size: 11px;
    color: #6a9bc3;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
}

.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 28px;
    font-weight: 700;
    color: #00d4ff;
    text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
}

/* ===== Expander 样式 ===== */
.streamlit-expanderHeader {
    background: rgba(13, 31, 60, 0.6) !important;
    border: 1px solid rgba(0, 212, 255, 0.2) !important;
    border-radius: 8px !important;
    color: #00d4ff !important;
}

/* ===== 隐藏默认 Streamlit 元素 ===== */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* ===== 自定义滚动条 ===== */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #0a1628;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #00d4ff, #0099cc);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, #00ff88, #00d4ff);
}

/* ===== 响应式调整 ===== */
@media (max-width: 768px) {
    .glass-card {
        padding: 16px;
    }
    h1 {
        font-size: 1.8rem !important;
    }
}
</style>
"""


def render_css():
    """渲染全局科技风样式（入口文件在每个脚本运行开始时调用一次）。"""
    st.markdown(_CSS, unsafe_allow_html=True)


# ==================== 共享辅助函数 ====================
def get_best_weights(output_dir):
    """递归查找所有 best.pt 文件，返回 [(相对路径显示名, 绝对路径)]（新者优先）。"""
    weights = []
    out_path = Path(output_dir)
    if out_path.exists():
        for p in out_path.rglob("best.pt"):
            display_name = str(p.relative_to(out_path))
            weights.append((display_name, str(p)))
    return sorted(weights, key=lambda x: x[1], reverse=True)


def load_training_metrics(csv_path):
    """加载训练指标数据（YOLO 生成的 CSV 列名前可能有空格，统一清理）。"""
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"读取训练数据失败: {e}")
        return None


def find_all_runs(output_dir):
    """查找目录下所有包含 results.csv 的训练运行（按最近修改排序）。"""
    out_path = Path(output_dir)
    if not out_path.exists():
        return []

    runs = []
    for p in out_path.iterdir():
        if not p.is_dir():
            continue

        # YOLO 的 results.csv 可能位置：p/results.csv（标准结构）、p/weights/ 下，
        # 或更深的任意子目录 —— 递归兜底
        csv_candidates = [
            p / "results.csv",
            p / "weights" / "results.csv",
        ]
        csv_candidates.extend(p.rglob("results.csv"))

        for csv_path in csv_candidates:
            if csv_path.exists():
                runs.append({
                    "name": p.name,
                    "path": str(p),
                    "csv_path": str(csv_path),
                    "mtime": csv_path.stat().st_mtime
                })
                break  # 找到一个就停止

    return sorted(runs, key=lambda x: x["mtime"], reverse=True)


def upload_with_preview(title, uploader_label, uploader_key,
                        slider_label, slider_min, slider_max, slider_value,
                        slider_help, preview_caption):
    """诊断页共用的「上传图片 + 置信度滑条 + 原图预览」毛玻璃卡片。

    返回 (uploaded_file, conf)。两个诊断页的滑条范围/默认值不同，通过参数传入。
    """
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(f"### {title}")

    uploaded_file = st.file_uploader(
        uploader_label,
        type=['jpg', 'png', 'jpeg', 'bmp'],
        key=uploader_key
    )

    conf = st.slider(slider_label, slider_min, slider_max, slider_value,
                     help=slider_help)

    if uploaded_file is not None:
        st.image(Image.open(uploaded_file), caption=preview_caption,
                 use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)
    return uploaded_file, conf
