# -*- coding: utf-8 -*-
"""页面 2：训练监控 —— 读取 runs/ 下的 results.csv，用 plotly 展示训练曲线。"""
import time
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from ui_common import DEFAULT_DETECT_RUNS_DIR, find_all_runs, load_training_metrics


def render():
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
            value=str(DEFAULT_DETECT_RUNS_DIR),
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

            **YOLO 训练输出结构**（相对仓库根目录）：
            ```
            runs/detect/results/
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
