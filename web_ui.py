"""
Dental AI Studio - 牙科AI教学平台（入口）

本文件只负责：页面配置 + 全局样式 + 侧边栏导航 + 页面路由。
四个功能页面分别在 page_train.py / page_monitor.py / page_diagnose.py /
page_winter.py 中实现；共享样式与部件在 ui_common.py；与 scripts 共享的
绘制/推理工具在 dental_common.py。

提示：修改 page_*.py / ui_common.py 后若页面上未生效，重启 run.py（Streamlit
对已导入模块的热重载不可靠）。
"""
import os

import streamlit as st

# set_page_config 必须是第一个 Streamlit 命令；页面模块只在其 render() 内发 UI 命令
st.set_page_config(
    page_title="Dental AI Studio",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

import page_diagnose   # noqa: E402
import page_monitor    # noqa: E402
import page_train      # noqa: E402  （导入时会检查核心模块 dental_yolo_train 是否存在）
import page_winter     # noqa: E402
import page_winter_train  # noqa: E402
import ui_common       # noqa: E402

# ==================== 全局样式 ====================
ui_common.render_css()

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

    # 供其他页面程序化跳转（如 Winter 微调完成后自动切到诊断页）
    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = "🚀 模型训练"

    page = st.radio(
        "📍 功能导航",
        ["🚀 模型训练", "📊 训练监控", "🧪 Winter 微调", "👁️ 智能诊断", "🦷 Winter 教学诊断"],
        key="nav_page",
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
    st.text_input("学号/ID", value="Student_01", key="student_id")

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
if 'label_zip_dir' not in st.session_state:
    st.session_state.label_zip_dir = None        # 标签 ZIP 解压的临时目录
if 'label_zip_label_root' not in st.session_state:
    st.session_state.label_zip_label_root = None # 检测到的标签目录
if 'label_zip_file_id' not in st.session_state:
    st.session_state.label_zip_file_id = None    # 用于检测新上传
if 'label_zip_info' not in st.session_state:
    st.session_state.label_zip_info = None       # 标签 ZIP 统计信息
if 'class_names' not in st.session_state:
    st.session_state.class_names = ['Caries', 'Restoration', 'Impacted tooth']  # 疾病类别
if 'winter_result' not in st.session_state:
    st.session_state.winter_result = None        # Winter 页的 (标注图, 智齿列表)
if 'winter_model_selection' not in st.session_state:
    st.session_state.winter_model_selection = None  # Winter 诊断页当前使用的实验名


# ==================== 页面路由 ====================
PAGES = {
    "🚀 模型训练": page_train.render,
    "📊 训练监控": page_monitor.render,
    "🧪 Winter 微调": page_winter_train.render,
    "👁️ 智能诊断": page_diagnose.render,
    "🦷 Winter 教学诊断": page_winter.render,
}
PAGES[page]()
