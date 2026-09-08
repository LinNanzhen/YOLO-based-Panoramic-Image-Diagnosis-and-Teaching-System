# -*- coding: utf-8 -*-
"""ZIP 解压路径校验的验证。

对应修复项④。注意：这里断言的是"畸形 ZIP 被显式拒绝"，而不是"堵住了任意文件
写入"——实测 CPython 的 zipfile.extractall 本身就会剥掉 ../ 组件、把绝对路径
降级到目标目录内、把 symlink 成员当普通文件写，仓库外不会被写出任何东西。
改动的价值在于把静默改写换成当场报错。
"""
import io
import zipfile
from pathlib import Path

import pytest

from page_train import _safe_extractall


def _zip(members: dict) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    buf.seek(0)
    return buf


def test_legit_dataset_zip_extracts_normally(tmp_path):
    buf = _zip({
        "images/trainset/1.jpg": "img1",
        "images/trainset/2.jpg": "img2",
        "labels/trainset/1.txt": "0 0.5 0.5 0.2 0.2\n",
        "labels/trainset/2.txt": "1 0.4 0.4 0.3 0.3\n",
    })
    _safe_extractall(buf, tmp_path / "out")
    out = tmp_path / "out"
    assert (out / "images" / "trainset" / "1.jpg").read_text() == "img1"
    assert (out / "labels" / "trainset" / "2.txt").is_file()


def test_parent_traversal_member_is_rejected(tmp_path):
    buf = _zip({"../../escaped.txt": "pwned", "ok.txt": "fine"})
    with pytest.raises(ValueError, match="越界"):
        _safe_extractall(buf, tmp_path / "out")
    # 拒绝必须是"一个都不写"，而不是写了一半
    assert not (tmp_path / "out" / "ok.txt").exists()
    assert not (tmp_path.parent / "escaped.txt").exists()


def test_absolute_path_member_is_rejected(tmp_path):
    buf = _zip({"/etc/escaped.txt": "pwned"})
    with pytest.raises(ValueError, match="越界"):
        _safe_extractall(buf, tmp_path / "out")


def test_symlink_member_is_rejected(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zi = zipfile.ZipInfo("link")
        zi.external_attr = 0o120777 << 16      # S_IFLNK | 0777
        zf.writestr(zi, "/etc/passwd")
    buf.seek(0)
    with pytest.raises(ValueError, match="符号链接"):
        _safe_extractall(buf, tmp_path / "out")


def test_directory_entries_are_not_flagged(tmp_path):
    """目录成员（名字以 / 结尾）resolve 后等于其自身目录，不应被误判为越界。"""
    buf = _zip({"labels/": "", "labels/a.txt": "0 0.5 0.5 0.1 0.1\n"})
    _safe_extractall(buf, tmp_path / "out")
    assert (tmp_path / "out" / "labels" / "a.txt").is_file()


def test_stdlib_baseline_confirms_no_escape(tmp_path):
    """基线对照：证明未加守卫的 extractall 也不会写到目标目录之外。

    保留这条是为了让"为什么④不算安全漏洞"这个结论可被复验，
    避免下一个会话又把它当成高危漏洞复述一遍。
    """
    target = tmp_path / "out"
    target.mkdir()
    buf = _zip({"../../escaped.txt": "pwned", "/abs/escaped2.txt": "pwned2"})
    with zipfile.ZipFile(buf) as zf:
        zf.extractall(str(target))          # 故意用未加守卫的方式
    assert not (tmp_path.parent / "escaped.txt").exists()
    # ../ 被剥掉后落在目标目录内 —— 静默改写，不报错，这正是改动要消除的行为
    assert (target / "escaped.txt").exists()
    assert (target / "abs" / "escaped2.txt").exists()


def test_both_upload_entrypoints_use_the_guarded_helper():
    """回归守卫：两个上传入口都不应再出现裸 zf.extractall。"""
    src = Path(__file__).resolve().parents[1] / "page_train.py"
    text = src.read_text(encoding="utf-8")
    assert text.count("zf.extractall(") == 1, "只应剩 _safe_extractall 内部那一处"
    assert text.count("_safe_extractall(zip_file, extract_path)") == 2
