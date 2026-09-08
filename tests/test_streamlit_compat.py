# -*- coding: utf-8 -*-
"""应用代码与实际安装的 streamlit 版本之间的 API 兼容性验证。

写测试时在浏览器里实测撞到的：两个诊断页一上传图片就崩
`TypeError: ImageMixin.image() got an unexpected keyword argument 'use_container_width'`。
根因是 st.image 的"撑满列宽"参数在 streamlit 1.42 才从 use_column_width 改名成
use_container_width，而 requirements.txt pin 的（本机装的）是 1.30.0 —— 代码是按
新版 API 写的。同名的 use_container_width 在 button/plotly_chart/dataframe 上
1.30 就已经支持，所以只有 st.image 这一处会炸，也正因如此它躲过了肉眼审查。

这里不做逐个调用的硬编码断言，而是把整类问题一次扫掉：遍历所有 `st.xxx(...)`
调用，检查每个关键字参数在**当前安装的** streamlit 里确实存在。以后升级 streamlit
或再写出超前于 pin 的代码，这条会当场指出是哪个文件哪一行。
"""
import ast
import inspect
from pathlib import Path

import pytest
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]

# 只扫随应用一起跑的文件；runs/ 里是训练产物，tests/ 自己不算
APP_MODULES = sorted(
    p for p in REPO_ROOT.glob("*.py")
    if p.name not in {"run.py"}
) + sorted((REPO_ROOT / "scripts").glob("*.py"))


def _st_call_targets(tree):
    """产出 (行号, 函数名, [关键字参数名]) ，只针对形如 st.foo(...) 的调用。"""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "st"):
            continue
        # st.foo(**kwargs) 这种展开无法静态判断，跳过
        if any(kw.arg is None for kw in node.keywords):
            continue
        yield node.lineno, func.attr, [kw.arg for kw in node.keywords]


def test_app_files_are_actually_scanned():
    """守卫本身别空过：至少要扫到几个页面模块，且里面确实有 st.* 调用。"""
    names = {p.name for p in APP_MODULES}
    for expected in ("web_ui.py", "page_winter.py", "page_diagnose.py", "ui_common.py"):
        assert expected in names, f"{expected} 不在扫描列表里"
    total = 0
    for path in APP_MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        total += sum(1 for _ in _st_call_targets(tree))
    assert total > 50, f"只扫到 {total} 处 st.* 调用，AST 匹配规则大概失效了"


@pytest.mark.parametrize("path", APP_MODULES, ids=lambda p: p.name)
def test_every_streamlit_kwarg_exists_in_installed_version(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bad = []
    for lineno, attr, kwargs in _st_call_targets(tree):
        func = getattr(st, attr, None)
        if func is None or not callable(func):
            continue                      # 不是 streamlit 顶层 API（或版本里没有），交给运行时
        try:
            params = inspect.signature(func).parameters
        except (TypeError, ValueError):
            continue                      # C 实现/无签名，无法静态校验
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            continue                      # 收 **kwargs，任何名字都"合法"
        for kw in kwargs:
            if kw not in params:
                bad.append(f"{path.name}:{lineno} st.{attr}({kw}=...) 不存在，"
                           f"当前 streamlit {st.__version__} 只接受 {sorted(params)}")
    assert not bad, "\n".join(bad)


def test_show_image_picks_a_kwarg_this_streamlit_accepts():
    """show_image 选出来的参数名必须是当前 st.image 真能吃的。"""
    import ui_common as uc
    params = inspect.signature(st.image).parameters
    if uc._IMAGE_WIDTH_KWARG is None:
        pytest.skip("当前 streamlit 的 st.image 两个宽度参数都不支持，包装退化为不传")
    assert uc._IMAGE_WIDTH_KWARG in params
    # 1.42 之前只有旧名，之后应优先新名（旧名会弹弃用横幅）
    if "use_container_width" in params:
        assert uc._IMAGE_WIDTH_KWARG == "use_container_width"


def test_no_page_calls_st_image_with_container_width_directly():
    """回归守卫：所有 st.image 都该走 show_image，别再有人直接写新参数名。"""
    offenders = []
    for path in APP_MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for lineno, attr, kwargs in _st_call_targets(tree):
            if attr == "image" and "use_container_width" in kwargs:
                offenders.append(f"{path.name}:{lineno}")
    assert not offenders, f"这些 st.image 调用在 streamlit<1.42 上会 TypeError: {offenders}"


def test_show_image_runs_without_a_script_context():
    """裸跑一次（无 Streamlit 运行时），确认包装本身不会抛异常。"""
    import numpy as np
    import ui_common as uc
    uc.show_image(np.zeros((8, 8, 3), dtype="uint8"), caption="smoke")
