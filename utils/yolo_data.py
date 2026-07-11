"""Helpers for YOLO dataset YAML files."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_data_yaml(data_yaml: str | Path) -> tuple[Path, dict]:
    path = Path(data_yaml).resolve()
    if not path.exists():
        raise FileNotFoundError(f"未找到数据集配置: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return path, yaml.safe_load(handle) or {}


def resolve_dataset_root(data_yaml: Path, data: dict) -> Path:
    root = Path(data.get("path", "."))
    return root if root.is_absolute() else (data_yaml.parent / root).resolve()


def write_resolved_data_yaml(data_yaml: str | Path, output_dir: str | Path) -> Path:
    """Write a runtime copy whose `path` is absolute for Ultralytics."""

    source_yaml, data = load_data_yaml(data_yaml)
    data = dict(data)
    data["path"] = str(resolve_dataset_root(source_yaml, data))

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{source_yaml.stem}_resolved.yaml"
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)
    return output_path
