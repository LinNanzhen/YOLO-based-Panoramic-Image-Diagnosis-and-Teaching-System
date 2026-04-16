"""
Dental AI Studio - 牙科AI教学平台
科技风UI + 实时训练曲线监控
"""

import streamlit as st
import sys
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import tempfile
import time
import glob
import pandas as pd
import json
import zipfile
import shutil
import plotly.graph_objects as go

# 导入核心模块
try:
    from dental_yolo_train import DentalYOLOPipeline
    from dental_yolo_visualization import DentalYOLOVisualizer
except ImportError:
    st.error("❌ 未找到核心模块，请确保 'dental_yolo_train.py' 和 'dental_yolo_visualization.py' 与本文件在同一目录下。")
    st.stop()

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="Dental AI Studio",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 科技风 CSS 样式 ====================
st.markdown("""
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

/* ===== 图表容器 ===== */
.chart-container {
    background: rgba(10, 22, 40, 0.9);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 12px;
    padding: 15px;
    margin: 10px 0;
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

.status-idle {
    background: rgba(0, 212, 255, 0.15);
    border: 1px solid rgba(0, 212, 255, 0.5);
    color: #00d4ff;
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
""", unsafe_allow_html=True)

# ==================== 侧边栏导航 ====================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <span style="font-family: 'Orbitron', monospace; font-size: 24px; font-weight: 700;
                     background: linear-gradient(90deg, #00d4ff, #00ff88);
                     -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🦷 DENTAL AI
        </span>
        <br>
        <span style="font-family: 'Rajdhani', sans-serif; font-size: 12px; color: #6a9bc3; letter-spacing: 3px;">
            STUDIO v2.0
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    page = st.radio(
        "📍 功能导航",
        ["🚀 模型训练", "📊 训练监控", "👁️ 智能诊断"],
        index=0
    )
    
    st.markdown("---")
    
    # 系统状态指示器
    st.markdown("""
    <div class="status-indicator status-active">
        <span style="width: 8px; height: 8px; background: #00ff88; border-radius: 50%; 
                     box-shadow: 0 0 10px #00ff88; animation: pulse 2s infinite;"></span>
        系统在线
    </div>
    <style>
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("### 👨‍🎓 操作员信息")
    student_id = st.text_input("学号/ID", value="Student_01", key="student_id")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.info(f"📂 工作目录: `{os.getcwd()}`")
    
    st.markdown("---")
    st.caption("© 2024 Dental AI Education System")


# ==================== Session State 初始化 ====================
if 'upload_dataset_dir' not in st.session_state:
    st.session_state.upload_dataset_dir = None   # 解压后的临时目录
if 'upload_data_root' not in st.session_state:
    st.session_state.upload_data_root = None     # 检测到的图像根目录
if 'upload_label_root' not in st.session_state:
    st.session_state.upload_label_root = None    # 检测到的标签目录
if 'upload_dataset_info' not in st.session_state:
    st.session_state.upload_dataset_info = None  # 数据集统计信息
if 'upload_file_id' not in st.session_state:
    st.session_state.upload_file_id = None       # 用于检测新上传
if 'class_names' not in st.session_state:
    st.session_state.class_names = ['Caries', 'Restoration', 'Impacted tooth']  # 疾病类别


# ==================== 辅助函数 ====================
def get_best_weights(output_dir):
    """递归查找所有 best.pt 文件"""
    weights = []
    out_path = Path(output_dir)
    if out_path.exists():
        for p in out_path.rglob("best.pt"):
            display_name = str(p.relative_to(out_path))
            weights.append((display_name, str(p)))
    return sorted(weights, key=lambda x: x[1], reverse=True)


def find_latest_results_csv(output_dir):
    """查找最新的 results.csv 文件"""
    out_path = Path(output_dir)
    if not out_path.exists():
        return None
    
    csv_files = list(out_path.rglob("results.csv"))
    if not csv_files:
        return None
    
    # 按修改时间排序，返回最新的
    return max(csv_files, key=lambda p: p.stat().st_mtime)


def load_training_metrics(csv_path):
    """加载训练指标数据"""
    try:
        df = pd.read_csv(csv_path)
        # 清理列名（YOLO 生成的 CSV 列名前可能有空格）
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"读取训练数据失败: {e}")
        return None


def find_all_runs(output_dir):
    """查找所有训练运行目录"""
    out_path = Path(output_dir)
    if not out_path.exists():
        return []
    
    runs = []
    
    # 遍历 output_dir 下的所有目录
    for p in out_path.iterdir():
        if not p.is_dir():
            continue
        
        # YOLO 的 results.csv 可能在多个位置：
        # 1. ./results/run_xxx/weights/results.csv (你的代码结构)
        # 2. ./results/run_xxx/results.csv (标准结构)
        # 3. 递归查找所有 results.csv
        
        csv_candidates = [
            p / "results.csv",               # 标准结构（YOLO project/name 模式）
            p / "weights" / "results.csv",   # 旧版兼容
        ]
        
        # 也递归查找
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


# ==================== 页面 1: 模型训练 ====================
if page == "🚀 模型训练":
    st.markdown("""
    <h1 style="margin-bottom: 5px;">🚀 模型训练中心</h1>
    <p style="color: #6a9bc3; font-size: 16px; margin-top: 0;">
        配置参数，训练你的专属牙科疾病检测模型
    </p>
    """, unsafe_allow_html=True)

    # ── 数据集配置（全宽）──────────────────────────────────────
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📦 数据集配置")

    tab_upload, tab_local = st.tabs(["📤 上传 ZIP 数据集", "📂 使用服务器路径"])

    # ---------- Tab 1: 上传 ZIP ----------
    with tab_upload:
        st.markdown("""
        将包含训练数据的 ZIP 文件拖入下方区域，平台将自动解压并识别目录结构。

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
            help="建议将 images/ 和 labels/ 打包成一个 zip 文件上传"
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

    # ---------- Tab 2: 服务器路径 ----------
    with tab_local:
        st.markdown("适用于数据集已存在于服务器上的情况。")
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
            pretrained_path = f"yolov8{model_size[0]}.pt"
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
            using_upload = (st.session_state.upload_data_root is not None)
            if using_upload:
                data_root  = st.session_state.upload_data_root
                label_root = st.session_state.upload_label_root
            else:
                data_root  = local_data_root
                label_root = local_label_root

            try:
                if not os.path.exists(data_root) or not os.path.exists(label_root):
                    st.error(f"❌ 目录不存在，请检查路径！\n- 图像: `{data_root}`\n- 标签: `{label_root}`")
                else:
                    src_tag = "上传数据集" if using_upload else "服务器路径"
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
            - 📦 **上传数据集**: 将 images/ + labels/ 打包成 ZIP 上传
            - 🧪 **初次测试**: 建议使用 Nano 模型 + 10 Epochs
            - 🖼️ **图片格式**: 支持 JPG / PNG / BMP
            - 💾 **显存不足**: 请减小 Batch Size
            - 📊 **实时监控**: 训练开始后切换至「训练监控」页面
            """)

        st.markdown('</div>', unsafe_allow_html=True)


# ==================== 页面 2: 训练监控 ====================
elif page == "📊 训练监控":
    st.markdown("""
    <h1 style="margin-bottom: 5px;">📊 训练曲线监控</h1>
    <p style="color: #6a9bc3; font-size: 16px; margin-top: 0;">
        实时追踪训练进度，分析模型性能变化
    </p>
    """, unsafe_allow_html=True)
    
    # 配置区
    col_config, col_refresh = st.columns([3, 1])
    with col_config:
        results_dir = st.text_input(
            "📂 训练结果目录",
            value="./runs/detect/results",
            key="monitor_dir"
        )
    with col_refresh:
        st.markdown("<br>", unsafe_allow_html=True)
        auto_refresh = st.checkbox("🔄 自动刷新 (5s)", value=False)
    
    # 调试：显示目录是否存在
    results_path = Path(results_dir)
    if not results_path.exists():
        st.error(f"❌ 目录不存在: `{results_dir}`")
        st.info("请确认训练结果目录路径正确")
    else:
        # 查找所有训练运行
        all_runs = find_all_runs(results_dir)
        
        # 调试信息
        with st.expander("🔧 目录扫描调试信息", expanded=False):
            st.markdown(f"**扫描目录**: `{results_path.resolve()}`")
            
            subdirs = [d.name for d in results_path.iterdir() if d.is_dir()]
            st.markdown(f"**子目录列表** ({len(subdirs)} 个):")
            if subdirs:
                st.code("\n".join(subdirs[:20]))  # 最多显示20个
            else:
                st.warning("该目录下没有子目录")
            
            # 查找所有 results.csv
            all_csvs = list(results_path.rglob("results.csv"))
            st.markdown(f"**找到的 results.csv 文件** ({len(all_csvs)} 个):")
            if all_csvs:
                for csv in all_csvs[:10]:
                    st.code(str(csv))
            else:
                st.warning("未找到任何 results.csv 文件")
        
        if not all_runs:
            st.warning("⚠️ 未找到训练记录。请先进行模型训练。")
            st.markdown("""
            **可能的原因**：
            1. 还没有开始训练
            2. 训练还在进行中，尚未生成 `results.csv`
            3. 目录路径不正确
            
            **YOLO 训练输出结构**：
            ```
            ./runs/detect/results/
            └── run_20241201_120000/   ← YOLO save_dir
                ├── results.csv        ← 训练曲线数据
                └── weights/
                    ├── best.pt
                    └── last.pt
            ```
            """)
        else:
            # 选择训练运行
            run_names = [r["name"] for r in all_runs]
            selected_run = st.selectbox(
                "📁 选择训练记录",
                run_names,
                index=0
            )
            
            # 获取选中的运行信息
            selected_run_info = next(r for r in all_runs if r["name"] == selected_run)
            csv_path = selected_run_info["csv_path"]
            
            st.caption(f"📍 CSV 路径: `{csv_path}`")
            
            # 加载数据
            df = load_training_metrics(csv_path)
            
            if df is None:
                st.error("❌ 无法读取 CSV 文件")
            elif df.empty:
                st.warning("⚠️ CSV 文件为空，训练可能刚开始")
            else:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                
                # 显示关键指标
                st.markdown("### 📈 关键指标概览")
                
                # 获取最新一行数据
                latest = df.iloc[-1]
                total_epochs = len(df)
                
                # 指标卡片
                m1, m2, m3, m4 = st.columns(4)
                
                with m1:
                    st.markdown(f"""
                    <div class="neon-card" style="text-align: center;">
                        <div class="metric-label">当前轮次</div>
                        <div class="metric-value">{total_epochs}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with m2:
                    # 查找 box_loss 列
                    box_loss = None
                    for col in df.columns:
                        if 'box_loss' in col.lower():
                            box_loss = latest.get(col)
                            break
                    
                    if box_loss is not None and isinstance(box_loss, (int, float)):
                        box_loss_str = f"{box_loss:.4f}"
                    else:
                        box_loss_str = "N/A"
                    
                    st.markdown(f"""
                    <div class="neon-card" style="text-align: center;">
                        <div class="metric-label">Box Loss</div>
                        <div class="metric-value" style="color: #ff6b6b;">{box_loss_str}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with m3:
                    # 尝试多种可能的列名
                    map50 = None
                    for col in df.columns:
                        col_lower = col.lower()
                        if 'map50' in col_lower or 'map_0.5' in col_lower or 'map50(b)' in col_lower:
                            if '95' not in col_lower:  # 排除 mAP50-95
                                map50 = latest.get(col)
                                break
                    
                    if map50 is not None and isinstance(map50, (int, float)):
                        map50_str = f"{map50:.3f}"
                    else:
                        map50_str = "N/A"
                    
                    st.markdown(f"""
                    <div class="neon-card" style="text-align: center;">
                        <div class="metric-label">mAP@50</div>
                        <div class="metric-value" style="color: #00ff88;">{map50_str}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with m4:
                    map5095 = None
                    for col in df.columns:
                        col_lower = col.lower()
                        if 'map50-95' in col_lower or 'map_0.5:0.95' in col_lower:
                            map5095 = latest.get(col)
                            break
                    
                    if map5095 is not None and isinstance(map5095, (int, float)):
                        map5095_str = f"{map5095:.3f}"
                    else:
                        map5095_str = "N/A"
                    
                    st.markdown(f"""
                    <div class="neon-card" style="text-align: center;">
                        <div class="metric-label">mAP@50-95</div>
                        <div class="metric-value" style="color: #00d4ff;">{map5095_str}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Loss 曲线图
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("### 📉 Loss 曲线")
                
                # 查找 loss 相关列（更宽松的匹配）
                loss_cols = [col for col in df.columns if 'loss' in col.lower()]
                
                if loss_cols:
                    epochs = list(range(1, len(df) + 1))
                    fig_loss = go.Figure()
                    colors = ['#ff6b6b', '#ffd700', '#ff9f43', '#ee5a24', '#0652DD', '#1289A7']
                    for i, col in enumerate(loss_cols):
                        fig_loss.add_trace(go.Scatter(
                            x=epochs, y=df[col],
                            mode='lines', name=col.strip(),
                            line=dict(color=colors[i % len(colors)], width=2)
                        ))
                    fig_loss.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(15,22,40,0.8)',
                        font=dict(color='#c8d6e5'),
                        xaxis=dict(title='Epoch', gridcolor='rgba(0,212,255,0.1)', color='#c8d6e5'),
                        yaxis=dict(title='Loss', gridcolor='rgba(0,212,255,0.1)', color='#c8d6e5'),
                        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#c8d6e5')),
                        height=350, margin=dict(l=0, r=0, t=10, b=0)
                    )
                    st.plotly_chart(fig_loss, use_container_width=True)
                    st.caption(f"📊 显示的 Loss 列: {', '.join(loss_cols)}")
                else:
                    st.info("未找到 Loss 数据列")
                    st.caption(f"可用列: {df.columns.tolist()}")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # mAP 曲线图
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("### 📈 mAP / Precision / Recall 曲线")
                
                # 查找 metrics 相关列
                metric_cols = [col for col in df.columns 
                               if any(kw in col.lower() for kw in ['map', 'precision', 'recall'])]
                
                if metric_cols:
                    epochs = list(range(1, len(df) + 1))
                    fig_metric = go.Figure()
                    colors = ['#00ff88', '#00d4ff', '#a855f7', '#ffd700']
                    for i, col in enumerate(metric_cols):
                        fig_metric.add_trace(go.Scatter(
                            x=epochs, y=df[col],
                            mode='lines', name=col.strip(),
                            line=dict(color=colors[i % len(colors)], width=2)
                        ))
                    fig_metric.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(15,22,40,0.8)',
                        font=dict(color='#c8d6e5'),
                        xaxis=dict(title='Epoch', gridcolor='rgba(0,212,255,0.1)', color='#c8d6e5'),
                        yaxis=dict(title='Value', gridcolor='rgba(0,212,255,0.1)', color='#c8d6e5'),
                        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#c8d6e5')),
                        height=350, margin=dict(l=0, r=0, t=10, b=0)
                    )
                    st.plotly_chart(fig_metric, use_container_width=True)
                    st.caption(f"📊 显示的指标列: {', '.join(metric_cols)}")
                else:
                    st.info("未找到 mAP/Precision/Recall 数据列")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # 显示原始数据表格
                with st.expander("📋 查看原始训练数据"):
                    st.dataframe(df, use_container_width=True)
                
                # 显示可用的列名（调试用）
                with st.expander("🔧 调试信息 - 可用数据列"):
                    st.markdown("**所有列名**:")
                    st.code(str(df.columns.tolist()))
                    st.markdown(f"**数据行数**: {len(df)}")
                    st.markdown("**数据预览** (前3行):")
                    st.dataframe(df.head(3))
            
            # 自动刷新逻辑
            if auto_refresh:
                time.sleep(5)
                st.rerun()


# ==================== 页面 3: 智能诊断 ====================
elif page == "👁️ 智能诊断":
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
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📤 上传影像")
        
        uploaded_file = st.file_uploader(
            "拖拽或点击上传 X-Ray 图像",
            type=['jpg', 'png', 'jpeg', 'bmp'],
            key="xray_upload"
        )
        
        conf_thres = st.slider(
            "🎚️ 置信度阈值",
            0.1, 1.0, 0.45,
            help="调低可检测更多目标，但可能增加误检"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="📷 原始影像", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
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