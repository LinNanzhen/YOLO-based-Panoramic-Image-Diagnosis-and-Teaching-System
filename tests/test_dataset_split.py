# -*- coding: utf-8 -*-
"""train/val 切分与标签物化的验证。

对应修复项②。旧代码 data.yaml 写的是 `val: trainset`，且图片/标签的目录布局
（<data_root>/trainset/*.jpg + 扁平 label_root/*.txt）里没有 `/images/` 段，
ultralytics 的 img2label_paths 做替换时是空操作 —— 训练读到零个标签且不报错。

这里同时断言两件事：
  (a) 旧布局确实定位不到标签（把根因固化成可复验的测试，而不是口头结论）；
  (b) 新布局能定位到，且 train/val 按图片不相交、切分可复现、源目录不被污染。
"""
import os
from pathlib import Path

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]

CLASS_NAMES = ["Caries", "Restoration", "Impacted tooth"]


def _make_dataset(root: Path, n_train: int, n_test: int = 0):
    """造一个最小的合成数据集：<root>/images/{trainset,testset} + <root>/labels/trainset"""
    img_root = root / "images"
    lab_root = root / "labels" / "trainset"
    (img_root / "trainset").mkdir(parents=True, exist_ok=True)
    (img_root / "testset").mkdir(parents=True, exist_ok=True)
    lab_root.mkdir(parents=True, exist_ok=True)

    for i in range(n_train):
        name = f"img{i:03d}"
        Image.new("RGB", (32, 32), (i * 3 % 256, 40, 90)).save(
            img_root / "trainset" / f"{name}.jpg")
        (lab_root / f"{name}.txt").write_text(
            f"{i % 3} 0.5 0.5 0.2 0.3\n", encoding="utf-8")

    test_lab_root = root / "labels" / "testset"
    test_lab_root.mkdir(parents=True, exist_ok=True)
    for i in range(n_test):
        name = f"tst{i:03d}"
        Image.new("RGB", (32, 32), (10, 200, 30)).save(
            img_root / "testset" / f"{name}.jpg")
        (test_lab_root / f"{name}.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")

    return img_root, lab_root, test_lab_root


def _pipeline(img_root, lab_root, out_dir, test_label_root=None,
              class_names_override=None, model_size="nano", **kw):
    from dental_yolo_train import DentalYOLOPipeline
    return DentalYOLOPipeline(
        data_root=img_root, label_root=lab_root, output_dir=out_dir,
        pretrained_weights="", model_size=model_size,
        class_names=class_names_override or CLASS_NAMES,
        test_label_root=test_label_root, **kw)


@pytest.fixture
def tiny(tmp_path):
    img_root, lab_root, test_lab_root = _make_dataset(tmp_path / "ds", n_train=20, n_test=4)
    out = tmp_path / "out"
    return img_root, lab_root, test_lab_root, out


def test_old_layout_cannot_locate_labels(tmp_path):
    """根因复现：扁平标签布局下 img2label_paths 指向的文件不存在。

    这条是修复项②的证据。删掉它，"为什么要物化目录树"就只剩注释里的一句话。
    """
    from ultralytics.data.utils import img2label_paths

    ds = tmp_path / "data_root"
    (ds / "trainset").mkdir(parents=True)
    img = ds / "trainset" / "1.jpg"
    Image.new("RGB", (16, 16)).save(img)
    flat_lab = tmp_path / "labels" / "trainset"
    flat_lab.mkdir(parents=True)
    (flat_lab / "1.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    derived = img2label_paths([str(img)])[0]
    assert not Path(derived).exists(), (
        "旧布局居然能定位到标签，说明对根因的判断有误，需要重新核实")
    # 替换是空操作：路径里根本没有 /images/ 段，标签名原样落到图片旁边
    assert derived == str(ds / "labels" / "trainset" / "1.txt") or \
        derived == str(ds / "trainset" / "1.txt")


def test_new_layout_labels_are_locatable(tiny):
    """物化后 img2label_paths 必须命中真实存在的标签文件。"""
    from ultralytics.data.utils import img2label_paths

    img_root, lab_root, _, out = tiny
    p = _pipeline(img_root, lab_root, out)
    dataset_root, stats = p._prepare_split()

    train_imgs = sorted(str(x) for x in (dataset_root / "images" / "train").iterdir())
    val_imgs = sorted(str(x) for x in (dataset_root / "images" / "val").iterdir())
    assert stats["n_train"] == len(train_imgs)
    assert stats["n_val"] == len(val_imgs)

    for split, imgs in (("train", train_imgs), ("val", val_imgs)):
        derived = img2label_paths(imgs)
        assert len(derived) == len(imgs)
        for d in derived:
            assert Path(d).is_file(), f"{split} 的标签定位失败: {d}"
            assert f"{os.sep}labels{os.sep}{split}{os.sep}" in d


def test_train_and_val_are_disjoint_by_image(tiny):
    """按图片划分：同一张图不能既在 train 又在 val（裁切级划分会泄漏）。"""
    img_root, lab_root, _, out = tiny
    p = _pipeline(img_root, lab_root, out, val_ratio=0.25)
    dataset_root, stats = p._prepare_split()

    train_names = {x.name for x in (dataset_root / "images" / "train").iterdir()}
    val_names = {x.name for x in (dataset_root / "images" / "val").iterdir()}
    assert train_names.isdisjoint(val_names), f"train/val 重叠: {train_names & val_names}"
    assert train_names | val_names == {f"img{i:03d}.jpg" for i in range(20)}
    assert stats["n_val"] == 5 and stats["n_train"] == 15

    # 标签必须跟图片同侧，否则 val 图片会配上 train 标签
    assert {x.name for x in (dataset_root / "labels" / "train").iterdir()} == \
        {n.replace(".jpg", ".txt") for n in train_names}
    assert {x.name for x in (dataset_root / "labels" / "val").iterdir()} == \
        {n.replace(".jpg", ".txt") for n in val_names}


def test_split_is_reproducible_and_actually_shuffled(tiny):
    img_root, lab_root, _, out = tiny
    a = _pipeline(img_root, lab_root, out / "a", seed=42)
    b = _pipeline(img_root, lab_root, out / "b", seed=42)
    ra, _ = a._prepare_split()
    rb, _ = b._prepare_split()
    va = {x.name for x in (ra / "images" / "val").iterdir()}
    vb = {x.name for x in (rb / "images" / "val").iterdir()}
    assert va == vb, "同种子必须给出同一切分，否则实验无法复现"

    # 换种子应得到不同的验证集（20 张里取 3 张，偶然相同的概率极低）
    c = _pipeline(img_root, lab_root, out / "c", seed=7)
    rc, _ = c._prepare_split()
    vc = {x.name for x in (rc / "images" / "val").iterdir()}
    assert vc != va, "换种子后切分没变，说明 seed 没有真正生效"

    # 且不是简单取前 N 个文件（那种"切分"等于按文件名排序，有系统偏差）
    all_names = sorted(f"img{i:03d}.jpg" for i in range(20))
    assert va != set(all_names[:len(va)])


def test_source_directories_are_not_modified(tiny):
    """硬链接物化不能污染用户的数据目录，也不能改动原文件内容。"""
    img_root, lab_root, _, out = tiny
    before_imgs = {x.name: x.read_bytes() for x in (img_root / "trainset").iterdir()}
    before_labs = {x.name: x.read_bytes() for x in lab_root.iterdir()}
    before_tree = sorted(str(x.relative_to(img_root.parent))
                         for x in img_root.parent.rglob("*") if x.is_file())

    p = _pipeline(img_root, lab_root, out)
    p._prepare_split()

    assert {x.name: x.read_bytes() for x in (img_root / "trainset").iterdir()} == before_imgs
    assert {x.name: x.read_bytes() for x in lab_root.iterdir()} == before_labs
    # 源目录里不应多出 images/ 或 labels/{train,val} 之类的切分产物
    assert sorted(str(x.relative_to(img_root.parent))
                  for x in img_root.parent.rglob("*") if x.is_file()) == before_tree


def test_link_or_copy_preserves_content(tiny):
    """_link_or_copy 优先硬链接、失败回退复制；两条路径的内容都必须一致。"""
    from dental_yolo_train import DentalYOLOPipeline
    img_root, lab_root, _, out = tiny
    src = img_root / "trainset" / "img000.jpg"
    dst = out / "probe" / "copy.jpg"
    DentalYOLOPipeline._link_or_copy(src, dst)
    assert dst.is_file() and dst.read_bytes() == src.read_bytes()
    try:
        assert os.stat(src).st_ino == os.stat(dst).st_ino, "同卷下应走硬链接（零额外磁盘占用）"
    except AssertionError:
        pytest.skip("当前文件系统不支持硬链接，回退复制属预期行为")
    # 幂等：目标已存在时不重复写
    DentalYOLOPipeline._link_or_copy(src, dst)
    assert dst.read_bytes() == src.read_bytes()


def test_missing_labels_are_reported_not_silently_dropped(tmp_path):
    """_prepare_split 要报告缺标签的图，而不是把它们悄悄算进 train 计数。

    注意必须在构造之后再删标签：构造函数里的 DataValidator 会先因缺标签抛错。
    """
    img_root, lab_root, _ = _make_dataset(tmp_path / "ds", n_train=6)[:3]
    p = _pipeline(img_root, lab_root, tmp_path / "out")
    (lab_root / "img002.txt").unlink()          # 人为制造一张缺标签的图
    _, stats = p._prepare_split()
    assert stats["missing_train_labels"] == ["img002.jpg"]
    assert stats["n_train"] + stats["n_val"] == 5, "缺标签的图不应被计入切分"


def test_no_trainable_pairs_raises(tmp_path):
    img_root, lab_root, _ = _make_dataset(tmp_path / "ds", n_train=3)[:3]
    p = _pipeline(img_root, lab_root, tmp_path / "out")
    for f in lab_root.glob("*.txt"):            # 全删：一张都配不上
        f.unlink()
    with pytest.raises(FileNotFoundError):
        p._prepare_split()


def test_test_split_only_when_labels_exist(tmp_path):
    """dataset/images/testset 是空的；指向无标签目录会产出全 0 的假测试指标。"""
    img_root, lab_root, test_lab_root = _make_dataset(tmp_path / "ds", n_train=6, n_test=3)

    p_no = _pipeline(img_root, lab_root, tmp_path / "out_no")
    _, s_no = p_no._prepare_split()
    assert s_no["n_test"] == 0
    assert not (p_no.results_subdir / "_dataset" / "images" / "test").exists()
    assert "test:" not in p_no._build_dataset_yaml(p_no.results_subdir / "_dataset", False)

    p_yes = _pipeline(img_root, lab_root, tmp_path / "out_yes",
                      test_label_root=test_lab_root)
    _, s_yes = p_yes._prepare_split()
    assert s_yes["n_test"] == 3
    test_dir = p_yes.results_subdir / "_dataset" / "images" / "test"
    assert len(list(test_dir.iterdir())) == 3
    assert "test: images/test" in p_yes._build_dataset_yaml(test_dir.parent.parent, True)


def test_dataset_yaml_uses_absolute_path_and_real_val(tiny):
    img_root, lab_root, _, out = tiny
    p = _pipeline(img_root, lab_root, out)
    dataset_root, stats = p._prepare_split()
    text = p._build_dataset_yaml(dataset_root, stats["n_test"] > 0)

    assert "val: trainset" not in text, "回归守卫：旧写法会让 val 与 train 完全重合"
    assert f"path: {dataset_root.resolve()}" in text, "path 必须是绝对路径，否则又依赖 CWD"
    assert "train: images/train" in text and "val: images/val" in text
    assert "nc: 3" in text
    for i, name in enumerate(CLASS_NAMES):
        assert f"  {i}: {name}" in text


def test_size_letter_maps_ultralytics_names(tiny):
    img_root, lab_root, _, out = tiny
    for given, want in [("nano", "n"), ("small", "s"), ("medium", "m"),
                        ("NANO", "n"), ("n", "n"), ("s", "s"), ("m", "m")]:
        p = _pipeline(img_root, lab_root, out / given, model_size=given)
        assert p._size_letter() == want, f"{given} 应映射为 {want}"
    # 旧代码直接拼 f"yolov8{model_size}.pt"，nano 会得到不存在的 yolov8nano.pt
    p = _pipeline(img_root, lab_root, out / "guard")
    assert f"yolov8{p.model_size}.pt" != f"yolov8{p._size_letter()}.pt"


def test_bundled_dataset_reproduces_documented_split(tmp_path):
    """用仓库里真实的 219 张标注跑一遍切分，核对 docs/experiment_report.md 的 186/33。

    这条把"文档里的数字"和"代码的行为"绑在一起：数字对不上就说明切分逻辑漂了。
    """
    img_root = REPO_ROOT / "dataset" / "images"
    lab_root = REPO_ROOT / "dataset" / "labels" / "trainset"
    if not img_root.is_dir() or not any(lab_root.glob("*.txt")):
        pytest.skip("dataset/ 不在本机，跳过真实数据校验")

    from winter_train_lib import ANGLE_CLASS_ORDER
    # Winter 标注的类别 ID 是 0-4，构造函数里就得给 5 个类名，
    # 否则 DataValidator 会以"类别ID 超出范围 [0,2]"直接拒掉
    names = list(ANGLE_CLASS_ORDER)
    p = _pipeline(img_root, lab_root, tmp_path / "out", class_names_override=names)
    _, stats = p._prepare_split()

    assert (stats["n_train"], stats["n_val"]) == (186, 33)
    assert stats["n_test"] == 0, "dataset/images/testset 是空的，不应生成 test"
    assert not stats["missing_train_labels"], "219 张图应全部配到同名标签"
    assert p.num_classes == 5
    assert "nc: 5" in p._build_dataset_yaml(tmp_path / "out" / "x", False)
