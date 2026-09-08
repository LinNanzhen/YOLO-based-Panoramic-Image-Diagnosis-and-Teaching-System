# -*- coding: utf-8 -*-
"""
dental_common.py — Web 页面与 scripts 脚本共享的视觉工具模块

包含：中文字体回退加载、检测框绘制（深色底白字标签）、按 margin 裁切、
Winter 五类角度的中文名/颜色常量，以及"检测 → 逐框裁切 → 角度分类"共享流水线。
web_ui 与 scripts/predict_demo.py、scripts/auto_annotate.py 都从这里导入，
调整绘制样式只需改这一处。

注意：导入本模块会做两件进程级修复 —— ① 设置 KMP_DUPLICATE_LIB_OK（Anaconda 的
MKL 与 torch 各带一份 libiomp5md.dll，不设会直接报 `OMP: Error #15` 中止进程）；
② 把 stdout/stderr 切到 UTF-8（Windows GBK 控制台下 print ✓/⚠ 会抛
UnicodeEncodeError）。依赖 torch 的入口应先 import 本模块，再 import ultralytics/torch。
"""
import contextlib
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def force_utf8_stdio():
    """把 stdout/stderr 切到 UTF-8，终端不支持时退化为占位符而不是崩。

    Windows 控制台与重定向管道默认用 GBK，而本项目从训练库深处就会 print
    ⚠ ✓ 🦷 之类字符（例如 scripts/prepare_winter_dataset.py 的数据集告警），
    一旦编码不了就抛 UnicodeEncodeError —— 在页面的训练线程里表现为一个和训练
    本身毫无关系的编码报错，极难联想到根因。

    run.py 已为自己处理过，但文档里的另一种启动方式 `streamlit run web_ui.py`
    不经过 run.py；web_ui 会导入各页面模块，页面模块都导入本模块，所以放这里
    能一次覆盖两种启动路径。

    同时打开行缓冲：训练进度是从这里 print 到服务进程的 stdout 的，
    `streamlit run web_ui.py > log` 这种起法下块缓冲会让日志一直空着。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:      # 被 pytest/Streamlit 替换过的流没有该方法
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except (ValueError, OSError):
            pass


force_utf8_stdio()

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------- 仓库路径基准 ----------
# 由本文件位置推导，与进程 CWD 无关。所有仓库内资源路径都应基于它拼接，
# 否则 streamlit 从别的目录启动时 "runs/winter" 之类的相对路径会全部落空。
REPO_ROOT = Path(__file__).resolve().parent
RUNS_DIR = REPO_ROOT / "runs"
WINTER_RUNS_DIR = RUNS_DIR / "winter"
DEMO_WEIGHTS_DIR = REPO_ROOT / "weights" / "demo"


@contextlib.contextmanager
def preserve_cuda_visible_devices():
    """把一次 ultralytics 调用对 CUDA_VISIBLE_DEVICES 的改动限制在这次调用内。

    ultralytics 的 select_device 在 device="cpu" 时会写
    `os.environ["CUDA_VISIBLE_DEVICES"] = ""`（源码注释：force is_available()=False）。
    但实测 torch 2.7.1 上这个 "force" 并不成立：`is_available()` 走 CUDA Runtime API
    （初始化时读一次，此后恒为 True），而 `device_count()` 走 NVML（CUDA 初始化前每次
    都重读环境变量，此时返回 0；初始化后才缓存进 `_cached_device_count`）。两者互相
    矛盾，于是 select_device 判定"有 GPU"进入 CUDA 分支，再调 `get_gpu_info(0)` →
    `get_device_properties(0)`，因 `0 >= device_count()` 抛
    `AssertionError: Invalid device id`。

    后果是同一进程内后续所有默认设备的推理全部失败，页面只显示"检测推理失败:
    Invalid device id"。课堂文档路径「无 GPU 时用 CPU 微调 → 跳诊断页看图」正好
    踩中：训练与推理在同一个 Streamlit 进程里。

    恢复环境变量即可让两者重新一致（已实测 device_count 回到 1，无需重启进程），
    所以在每次调用前后保存/恢复就能断开这条因果链。
    """
    key = "CUDA_VISIBLE_DEVICES"
    missing = object()
    saved = os.environ.get(key, missing)
    try:
        yield
    finally:
        if saved is missing:
            os.environ.pop(key, None)
        else:
            os.environ[key] = saved

# ---------- Winter 阻生角度常量 ----------
WINTER_ANGLE_ZH = {
    "Mesioangular": "近中阻生",
    "Vertical":     "垂直阻生",
    "Inverted":     "倒置阻生",
    "Horizontal":   "水平阻生",
    "Buccolingual": "颊舌向阻生",
}
WINTER_ANGLE_COLORS = {
    "Mesioangular": (0, 150, 255),   # 蓝
    "Vertical":     (0, 200, 90),    # 绿
    "Horizontal":   (255, 130, 0),   # 橙
    "Inverted":     (230, 60, 220),  # 品红
    "Buccolingual": (255, 70, 70),   # 红
}

# ---------- 字体回退链 ----------
_CJK_FONT_PATHS = [
    "C:/Windows/Fonts/msyhbd.ttc",                  # Windows 微软雅黑粗体
    "C:/Windows/Fonts/simhei.ttf",                  # Windows 黑体
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]
_LATIN_FONT_PATHS = [
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]


def load_draw_font(size: int, prefer_cjk: bool = True):
    """加载可用的绘图字体。返回 (font, 是否支持中文)；都找不到则退回 PIL 默认字体。

    prefer_cjk=True  优先中文字体（Web 教学诊断的中文标签）；
    prefer_cjk=False 优先西文字体（脚本输出的英文类别名）。
    """
    first, second = (_CJK_FONT_PATHS, _LATIN_FONT_PATHS) if prefer_cjk \
        else (_LATIN_FONT_PATHS, _CJK_FONT_PATHS)
    for path in first:
        try:
            return ImageFont.truetype(path, size), True
        except OSError:
            continue
    for path in second:
        try:
            return ImageFont.truetype(path, size), False
        except OSError:
            continue
    return ImageFont.load_default(), False


# ---------- 绘制 ----------
def draw_label(dr: ImageDraw.ImageDraw, xy, text: str, font,
               text_color=(255, 255, 255), bar_color=(20, 20, 20)):
    """在 (x, y) 上方画带深色底的白字标签，保证 X 光片上清晰可读。"""
    x, y = xy
    bbox = dr.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    dr.rectangle([x - 2, y - th - 6, x + tw + 4, y + 2], fill=bar_color)
    dr.text((x, y - th - 4), text, font=font, fill=text_color)


def annotate_box(dr: ImageDraw.ImageDraw, box, label: str, color,
                 line_w: int, font):
    """画一个检测框（实线边框）+ 框上方标签。box = (x1, y1, x2, y2)。"""
    x1, y1, x2, y2 = box
    dr.rectangle([x1, y1, x2, y2], outline=color, width=line_w)
    draw_label(dr, (x1, max(0, y1 - 6)), label, font)


# ---------- 裁切 ----------
def clamp_margin_box(box, w: int, h: int, margin: float = 0.2):
    """把检测框按 margin 比例外扩并裁剪到图像范围内。返回整数 (x0, y0, x1, y1)。

    与训练数据（prepare_winter_dataset.py）的裁切 margin 语义一致。
    """
    x0, y0, x1, y1 = map(int, box)
    mw, mh = (x1 - x0) * margin, (y1 - y0) * margin
    x0, y0 = max(0, int(x0 - mw)), max(0, int(y0 - mh))
    x1, y1 = min(w, int(x1 + mw)), min(h, int(y1 + mh))
    return x0, y0, x1, y1


# ---------- 共享推理流水线 ----------
def detect_and_classify(det, clf, source, crop_img=None, conf: float = 0.25,
                        margin: float = 0.2, device=None,
                        sort_left_right: bool = False) -> list:
    """检测 + 逐框裁切 + 角度分类的共享流水线。

    参数:
        det      已加载的 YOLO 检测模型
        clf      已加载的 YOLO 分类模型；None 或单框分类失败时 angle 字段为 None
        source   传给 det.predict 的输入（图片路径 / ndarray 均可）
        crop_img 用于裁切分类的图像（ndarray 或 PIL.Image，与 source 对应的原图）；
                 None 则跳过分类
        conf     检测置信度阈值
        margin   裁切外扩比例（与训练数据一致）
        device   推理设备（None = 模型默认）
        sort_left_right  True 时结果按框左边界从左到右排序
    返回:
        findings 列表，每项:
        {"box": (x1, y1, x2, y2) 整数元组, "det_conf": float,
         "angle_en": str|None, "angle_conf": float|None}
    """
    det_kw = {"conf": conf, "verbose": False}
    clf_kw = {"verbose": False}
    if device is not None:
        det_kw["device"] = device
        clf_kw["device"] = device

    with preserve_cuda_visible_devices():
        r = det.predict(source, **det_kw)[0]
    h, w = r.orig_shape
    boxes = r.boxes
    order = range(len(boxes))
    if sort_left_right:
        order = sorted(range(len(boxes)), key=lambda i: float(boxes.xyxy[i][0]))

    findings = []
    for i in order:
        box = tuple(int(v) for v in boxes.xyxy[i].tolist())
        det_conf = float(boxes.conf[i])
        angle_en = angle_conf = None
        if clf is not None and crop_img is not None:
            cx0, cy0, cx1, cy1 = clamp_margin_box(box, w, h, margin)
            if isinstance(crop_img, np.ndarray):
                crop = crop_img[cy0:cy1, cx0:cx1]
            else:
                crop = crop_img.crop((cx0, cy0, cx1, cy1))
            try:
                with preserve_cuda_visible_devices():
                    probs = clf.predict(crop, **clf_kw)[0].probs
                angle_en = clf.names[int(probs.top1)]
                angle_conf = float(probs.top1conf)
            except Exception:
                angle_en, angle_conf = None, None
        findings.append({"box": box, "det_conf": det_conf,
                         "angle_en": angle_en, "angle_conf": angle_conf})
    return findings
