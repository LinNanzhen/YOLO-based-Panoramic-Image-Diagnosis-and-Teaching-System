# -*- coding: utf-8 -*-
"""pytest 共享配置。

把仓库根目录放进 sys.path，使测试能从任意 CWD 运行（这本身也是被测特性之一：
所有仓库内路径都应由文件位置推导，而非依赖进程工作目录）。
"""
import os
import sys
from pathlib import Path

# 必须早于任何 ultralytics/torch 导入，见 dental_common 模块注释
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: 会真实加载 .pt 权重做推理的用例（数十秒）。跳过：-m 'not slow'",
    )
