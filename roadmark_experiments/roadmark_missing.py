"""Class protocol for road-mark missing detection experiments."""

from __future__ import annotations

from pathlib import Path

import yaml


SOURCE_RDD_CLASSES = ["D00", "D10", "D20", "D30", "D40", "D50", "D60", "D70", "D80", "D90"]

ROADMARK_MISSING_CLASSES = [
    "lane_line_missing",
    "lane_line_break",
    "edge_line_missing",
    "stop_line_missing",
    "crosswalk_missing",
    "arrow_missing",
    "guide_line_missing",
    "worn_marking_missing",
    "occluded_marking_missing",
    "other_marking_missing",
]

SOURCE_TO_MISSING_NAME = dict(zip(SOURCE_RDD_CLASSES, ROADMARK_MISSING_CLASSES))


def source_to_missing_class_id(source_name: str) -> int | None:
    """Return the road-mark missing class id for an imported source class."""

    try:
        return SOURCE_RDD_CLASSES.index(source_name)
    except ValueError:
        return None


def missing_class_name(source_name: str) -> str:
    return SOURCE_TO_MISSING_NAME.get(source_name, source_name)


def load_semantic_metadata(data_yaml: str | Path) -> dict:
    data_yaml = Path(data_yaml).resolve()
    candidates = [
        data_yaml.with_name(f"{data_yaml.stem}_semantics.yaml"),
        data_yaml.with_name("label_semantics.yaml"),
    ]
    metadata_path = next((path for path in candidates if path.exists()), None)
    if metadata_path is None:
        return {}
    return yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}


def semantic_warning(data_yaml: str | Path) -> str | None:
    metadata = load_semantic_metadata(data_yaml)
    if not metadata or metadata.get("verified_for_target_task", False):
        return None
    source_task = metadata.get("source_task", "unknown")
    target_task = metadata.get("target_task", "unknown")
    return (
        f"标签语义尚未验证: source_task={source_task}, target_task={target_task}。"
        "当前指标只能验证代码链路，不能作为路面标线缺失识别准确率。"
    )
