"""道路标线数据增强工具。

增强目标：
1. Brightness/Contrast：模拟夜间与曝光变化；
2. GaussianBlur：模拟摄像头模糊；
3. RandomRain：模拟雨天；
4. RandomShadow：模拟道路阴影。

支持 YOLO 标注格式：class x_center y_center width height，坐标归一化到 [0, 1]。
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

try:
    import albumentations as A
except Exception as exc:  # pragma: no cover - 依赖缺失时由调用方处理
    A = None
    _ALBUMENTATIONS_IMPORT_ERROR = exc
else:
    _ALBUMENTATIONS_IMPORT_ERROR = None


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _require_albumentations() -> None:
    if A is None:
        raise ImportError(
            "数据增强模块需要 albumentations，请执行 `pip install albumentations`。"
        ) from _ALBUMENTATIONS_IMPORT_ERROR


def _pad_if_needed(img_size: int):
    """兼容 Albumentations 1.x(value) 与 2.x(fill) 的 Padding 参数。"""

    try:
        return A.PadIfNeeded(
            min_height=img_size,
            min_width=img_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
    except TypeError:
        return A.PadIfNeeded(
            min_height=img_size,
            min_width=img_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=(114, 114, 114),
        )


def _bbox_params(label_fields: list[str], min_visibility: float = 0.0):
    """兼容不同 Albumentations 版本的 BboxParams。"""

    kwargs = {
        "format": "yolo",
        "label_fields": label_fields,
        "min_visibility": min_visibility,
    }
    try:
        return A.BboxParams(**kwargs, clip=True)
    except TypeError:
        return A.BboxParams(**kwargs)


def _random_rain():
    """兼容 RandomRain 新旧参数名。"""

    if not hasattr(A, "RandomRain"):
        return A.NoOp(p=1.0)
    try:
        return A.RandomRain(
            slant_range=(-10, 10),
            drop_length=18,
            drop_width=1,
            drop_color=(180, 180, 180),
            blur_value=3,
            brightness_coefficient=0.75,
            p=0.18,
        )
    except TypeError:
        return A.RandomRain(
            slant_lower=-10,
            slant_upper=10,
            drop_length=18,
            drop_width=1,
            drop_color=(180, 180, 180),
            blur_value=3,
            brightness_coefficient=0.75,
            p=0.18,
        )


def build_train_transform(img_size: int = 640):
    """构建训练增强流水线。"""

    _require_albumentations()
    rain = _random_rain()
    shadow = A.RandomShadow(p=0.25) if hasattr(A, "RandomShadow") else A.NoOp(p=1.0)

    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size, p=1.0),
            _pad_if_needed(img_size),
            A.RandomBrightnessContrast(brightness_limit=(-0.45, 0.15), contrast_limit=0.25, p=0.45),
            A.GaussianBlur(blur_limit=(3, 7), p=0.20),
            rain,
            shadow,
            A.HorizontalFlip(p=0.15),
        ],
        bbox_params=_bbox_params(["class_labels"], min_visibility=0.15),
    )


def build_val_transform(img_size: int = 640):
    """验证集仅做尺寸适配，不改变图像语义。"""

    _require_albumentations()
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size, p=1.0),
            _pad_if_needed(img_size),
        ],
        bbox_params=_bbox_params(["class_labels"]),
    )


def read_yolo_label(label_path: Path) -> tuple[list[list[float]], list[int]]:
    """读取 YOLO 标签文件。"""

    boxes: list[list[float]] = []
    class_ids: list[int] = []
    if not label_path.exists():
        return boxes, class_ids

    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"标签格式错误: {label_path} -> {line}")
        class_ids.append(int(float(parts[0])))
        boxes.append([float(v) for v in parts[1:]])
    return boxes, class_ids


def write_yolo_label(label_path: Path, boxes: Iterable[Iterable[float]], class_ids: Iterable[int]) -> None:
    """写入 YOLO 标签文件。"""

    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for class_id, box in zip(class_ids, boxes):
        x, y, w, h = [min(1.0, max(0.0, float(v))) for v in box]
        if w <= 0 or h <= 0:
            continue
        lines.append(f"{int(class_id)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def augment_yolo_sample(
    image: np.ndarray,
    boxes: list[list[float]],
    class_ids: list[int],
    img_size: int = 640,
) -> tuple[np.ndarray, list[list[float]], list[int]]:
    """对单张图像和 YOLO 框执行增强。"""

    transform = build_train_transform(img_size)
    augmented = transform(image=image, bboxes=boxes, class_labels=class_ids)
    return augmented["image"], list(augmented["bboxes"]), list(augmented["class_labels"])


def iter_images(image_dir: Path) -> list[Path]:
    """枚举图像文件。"""

    return sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def copy_split(source_root: Path, target_root: Path, split: str) -> None:
    """复制某个 split 的原始图片和标签。"""

    src_img_dir = source_root / "images" / split
    src_label_dir = source_root / "labels" / split
    dst_img_dir = target_root / "images" / split
    dst_label_dir = target_root / "labels" / split
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_label_dir.mkdir(parents=True, exist_ok=True)

    for image_path in iter_images(src_img_dir):
        rel = image_path.relative_to(src_img_dir)
        dst_image = dst_img_dir / rel
        dst_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, dst_image)

        label_path = src_label_dir / rel.with_suffix(".txt")
        if label_path.exists():
            dst_label = dst_label_dir / rel.with_suffix(".txt")
            dst_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(label_path, dst_label)


def create_augmented_yolo_dataset(
    source_root: str | Path = "dataset",
    target_root: str | Path = "dataset_aug",
    repeats: int = 1,
    img_size: int = 640,
    seed: int = 42,
) -> Path:
    """离线生成增强版 YOLO 数据集。

    该函数会：
    - 复制原始 train/val；
    - 对 train 额外生成 repeats 份增强样本；
    - 保持 labels 与 images 目录结构一致。
    """

    _require_albumentations()
    source_root = Path(source_root)
    target_root = Path(target_root)
    random.seed(seed)
    np.random.seed(seed)

    if target_root.exists():
        shutil.rmtree(target_root)
    copy_split(source_root, target_root, "train")
    copy_split(source_root, target_root, "val")

    transform = build_train_transform(img_size)
    src_img_dir = source_root / "images" / "train"
    src_label_dir = source_root / "labels" / "train"
    dst_img_dir = target_root / "images" / "train"
    dst_label_dir = target_root / "labels" / "train"

    for image_path in iter_images(src_img_dir):
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rel = image_path.relative_to(src_img_dir)
        label_path = src_label_dir / rel.with_suffix(".txt")
        boxes, class_ids = read_yolo_label(label_path)

        for index in range(repeats):
            augmented = transform(image=image, bboxes=boxes, class_labels=class_ids)
            aug_image = cv2.cvtColor(augmented["image"], cv2.COLOR_RGB2BGR)
            aug_boxes = list(augmented["bboxes"])
            aug_classes = list(augmented["class_labels"])

            stem = f"{rel.stem}_aug{index + 1}"
            out_image = dst_img_dir / rel.with_name(stem).with_suffix(".jpg")
            out_label = dst_label_dir / rel.with_name(stem).with_suffix(".txt")
            out_image.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_image), aug_image)
            write_yolo_label(out_label, aug_boxes, aug_classes)

    return target_root
