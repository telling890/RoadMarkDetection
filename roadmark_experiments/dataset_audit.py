"""Dataset audit utilities for YOLO road-mark data."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .roadmark_missing import load_semantic_metadata


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class DatasetConfig:
    yaml_path: Path
    root: Path
    train: Path
    val: Path
    names: list[str]


def load_dataset_config(data_yaml: str | Path) -> DatasetConfig:
    yaml_path = Path(data_yaml).resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"未找到数据集配置: {yaml_path}")
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    root_raw = Path(raw.get("path", "."))
    root = root_raw if root_raw.is_absolute() else (yaml_path.parent / root_raw).resolve()
    names_raw = raw.get("names", [])
    if isinstance(names_raw, dict):
        names = [str(names_raw[key]) for key in sorted(names_raw)]
    else:
        names = [str(name) for name in names_raw]
    return DatasetConfig(
        yaml_path=yaml_path,
        root=root,
        train=root / raw.get("train", "images/train"),
        val=root / raw.get("val", "images/val"),
        names=names,
    )


def label_dir_for_image_dir(image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        return Path(*parts)
    return image_dir.parent.parent / "labels" / image_dir.name


def iter_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        return []
    return sorted(path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def parse_label_file(label_path: Path, class_count: int) -> tuple[Counter[int], list[str]]:
    counts: Counter[int] = Counter()
    errors: list[str] = []
    if not label_path.exists():
        return counts, errors

    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"line {line_number}: expected 5 fields, got {len(parts)}")
            continue
        try:
            class_id = int(float(parts[0]))
            x, y, w, h = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"line {line_number}: non-numeric label values")
            continue
        if class_id < 0 or class_id >= class_count:
            errors.append(f"line {line_number}: class id {class_id} outside [0, {class_count - 1}]")
            continue
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            errors.append(f"line {line_number}: bbox values outside YOLO normalized range")
            continue
        counts[class_id] += 1
    return counts, errors


def audit_split(split: str, image_dir: Path, names: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]], list[str], list[dict[str, str]]]:
    label_dir = label_dir_for_image_dir(image_dir)
    images = iter_images(image_dir)
    class_counts: Counter[int] = Counter()
    missing_labels: list[str] = []
    invalid_rows: list[dict[str, str]] = []
    label_file_count = 0
    empty_label_count = 0

    for image_path in images:
        relative = image_path.relative_to(image_dir)
        label_path = label_dir / relative.with_suffix(".txt")
        if not label_path.exists():
            missing_labels.append(str(relative))
            continue
        label_file_count += 1
        counts, errors = parse_label_file(label_path, len(names))
        if not counts and not label_path.read_text(encoding="utf-8").strip():
            empty_label_count += 1
        class_counts.update(counts)
        for error in errors:
            invalid_rows.append({"split": split, "label": str(label_path), "error": error})

    summary = {
        "split": split,
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "image_count": len(images),
        "label_file_count": label_file_count,
        "missing_label_count": len(missing_labels),
        "empty_label_count": empty_label_count,
        "box_count": sum(class_counts.values()),
        "invalid_label_count": len(invalid_rows),
    }
    distribution = [
        {
            "split": split,
            "class_id": class_id,
            "class_name": names[class_id] if class_id < len(names) else f"class_{class_id}",
            "box_count": int(class_counts.get(class_id, 0)),
        }
        for class_id in range(len(names))
    ]
    return summary, distribution, missing_labels, invalid_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def audit_dataset(data_yaml: str | Path, output_dir: str | Path) -> dict[str, Path]:
    config = load_dataset_config(data_yaml)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    all_missing: dict[str, list[str]] = {}
    invalid_rows: list[dict[str, str]] = []

    for split, image_dir in (("train", config.train), ("val", config.val)):
        summary, distribution, missing, invalid = audit_split(split, image_dir, config.names)
        stats_rows.append(summary)
        distribution_rows.extend(distribution)
        all_missing[split] = missing
        invalid_rows.extend(invalid)

    stats_path = output_dir / "dataset_stats.csv"
    distribution_path = output_dir / "class_distribution.csv"
    invalid_path = output_dir / "invalid_labels.csv"
    missing_path = output_dir / "missing_labels.json"
    report_path = output_dir / "dataset_audit.md"

    write_csv(
        stats_path,
        stats_rows,
        [
            "split",
            "image_dir",
            "label_dir",
            "image_count",
            "label_file_count",
            "missing_label_count",
            "empty_label_count",
            "box_count",
            "invalid_label_count",
        ],
    )
    write_csv(distribution_path, distribution_rows, ["split", "class_id", "class_name", "box_count"])
    write_csv(invalid_path, invalid_rows, ["split", "label", "error"])
    missing_path.write_text(json.dumps(all_missing, ensure_ascii=False, indent=2), encoding="utf-8")
    write_dataset_report(
        report_path,
        config,
        stats_rows,
        distribution_rows,
        invalid_rows,
        all_missing,
        load_semantic_metadata(config.yaml_path),
    )

    return {
        "stats": stats_path,
        "distribution": distribution_path,
        "invalid": invalid_path,
        "missing": missing_path,
        "report": report_path,
    }


def write_dataset_report(
    path: Path,
    config: DatasetConfig,
    stats_rows: list[dict[str, Any]],
    distribution_rows: list[dict[str, Any]],
    invalid_rows: list[dict[str, str]],
    missing: dict[str, list[str]],
    semantic_metadata: dict[str, Any],
) -> None:
    lines = [
        "# EXP-00 路面标线缺失数据集与类别协议审计",
        "",
        f"- 数据配置: `{config.yaml_path}`",
        f"- 数据根目录: `{config.root}`",
        f"- 类别数: {len(config.names)}",
        "",
        "## Split 统计",
        "",
        "| split | images | labels | boxes | missing labels | empty labels | invalid lines |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in stats_rows:
        lines.append(
            f"| {row['split']} | {row['image_count']} | {row['label_file_count']} | {row['box_count']} | "
            f"{row['missing_label_count']} | {row['empty_label_count']} | {row['invalid_label_count']} |"
        )

    lines.extend(["", "## 类别分布", "", "| split | class id | class name | boxes |", "|---|---:|---|---:|"])
    for row in distribution_rows:
        lines.append(f"| {row['split']} | {row['class_id']} | {row['class_name']} | {row['box_count']} |")

    if semantic_metadata:
        verified = bool(semantic_metadata.get("verified_for_target_task", False))
        lines.extend(
            [
                "",
                "## 标签语义状态",
                "",
                f"- 源任务: `{semantic_metadata.get('source_task', 'unknown')}`",
                f"- 目标任务: `{semantic_metadata.get('target_task', 'unknown')}`",
                f"- 已人工验证目标语义: `{verified}`",
            ]
        )
        if not verified:
            lines.append("- 警告: 当前指标只能验证代码链路，不能作为路面标线缺失识别准确率。")

    lines.extend(
        [
            "",
            "## 完成标准",
            "",
            f"- 缺失标签文件: {sum(len(items) for items in missing.values())}",
            f"- 非法标签行: {len(invalid_rows)}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
