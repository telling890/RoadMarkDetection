"""Raw data selection and YOLO dataset preparation utilities."""

from __future__ import annotations

import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import yaml

from .dataset_audit import IMAGE_SUFFIXES, load_dataset_config, parse_label_file


@dataclass(frozen=True)
class SelectionOptions:
    train_ratio: float = 0.8
    seed: int = 42
    max_images: int | None = None
    max_images_per_class: int | None = None
    min_width: int = 1
    min_height: int = 1
    include_empty_labels: bool = False
    force: bool = False


@dataclass(frozen=True)
class RawSample:
    image_path: Path
    label_path: Path
    relative_path: Path
    width: int
    height: int
    class_counts: Counter[int]

    @property
    def box_count(self) -> int:
        return int(sum(self.class_counts.values()))

    @property
    def primary_class(self) -> int:
        if not self.class_counts:
            return -1
        return sorted(self.class_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


@dataclass(frozen=True)
class PreparedDatasetResult:
    data_yaml: Path
    selected_csv: Path
    rejected_csv: Path
    summary_md: Path
    summary_json: Path
    selected_count: int
    rejected_count: int


def _image_root(raw_root: Path) -> Path:
    images = raw_root / "images"
    return images if images.exists() else raw_root


def _relative_image_path(raw_root: Path, image_path: Path) -> Path:
    base = _image_root(raw_root)
    try:
        return image_path.relative_to(base)
    except ValueError:
        return image_path.relative_to(raw_root)


def _find_label_path(raw_root: Path, image_path: Path) -> Path:
    relative = _relative_image_path(raw_root, image_path).with_suffix(".txt")
    candidates = [
        raw_root / "labels" / relative,
        image_path.with_suffix(".txt"),
        raw_root / relative,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def iter_raw_images(raw_root: str | Path) -> list[Path]:
    raw_root = Path(raw_root).resolve()
    base = _image_root(raw_root)
    if not base.exists():
        return []
    return sorted(path for path in base.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def _read_image_size(image_path: Path) -> tuple[int, int] | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    height, width = image.shape[:2]
    return int(width), int(height)


def _reject_row(image_path: Path, label_path: Path, reason: str, detail: str) -> dict[str, str]:
    return {"image": str(image_path), "label": str(label_path), "reason": reason, "detail": detail}


def collect_candidates(
    raw_root: str | Path,
    class_count: int,
    options: SelectionOptions,
) -> tuple[list[RawSample], list[dict[str, str]]]:
    raw_root = Path(raw_root).resolve()
    candidates: list[RawSample] = []
    rejected: list[dict[str, str]] = []

    for image_path in iter_raw_images(raw_root):
        label_path = _find_label_path(raw_root, image_path)
        relative_path = _relative_image_path(raw_root, image_path)
        image_size = _read_image_size(image_path)
        if image_size is None:
            rejected.append(_reject_row(image_path, label_path, "unreadable_image", "OpenCV cannot read this image"))
            continue
        width, height = image_size
        if width < options.min_width or height < options.min_height:
            rejected.append(
                _reject_row(
                    image_path,
                    label_path,
                    "image_too_small",
                    f"{width}x{height} < {options.min_width}x{options.min_height}",
                )
            )
            continue
        if not label_path.exists():
            rejected.append(_reject_row(image_path, label_path, "missing_label", "No matching YOLO label file"))
            continue

        counts, errors = parse_label_file(label_path, class_count)
        if errors:
            rejected.append(_reject_row(image_path, label_path, "invalid_label", "; ".join(errors)))
            continue
        if not counts and not options.include_empty_labels:
            rejected.append(_reject_row(image_path, label_path, "empty_label", "No valid boxes in label file"))
            continue
        candidates.append(RawSample(image_path, label_path, relative_path, width, height, counts))

    return candidates, rejected


def select_samples(candidates: list[RawSample], options: SelectionOptions) -> list[RawSample]:
    rng = random.Random(options.seed)
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    selected: list[RawSample] = []
    per_class_images: Counter[int] = Counter()

    for sample in shuffled:
        if options.max_images is not None and len(selected) >= options.max_images:
            break
        if options.max_images_per_class is not None and sample.class_counts:
            if any(per_class_images[class_id] >= options.max_images_per_class for class_id in sample.class_counts):
                continue
        selected.append(sample)
        for class_id in sample.class_counts:
            per_class_images[class_id] += 1
    return selected


def split_samples(samples: list[RawSample], train_ratio: float, seed: int) -> dict[str, list[RawSample]]:
    rng = random.Random(seed)
    grouped: dict[int, list[RawSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.primary_class].append(sample)

    train: list[RawSample] = []
    val: list[RawSample] = []
    for group in grouped.values():
        group = list(group)
        rng.shuffle(group)
        if len(group) <= 1:
            train.extend(group)
            continue
        train_count = max(1, min(len(group) - 1, round(len(group) * train_ratio)))
        train.extend(group[:train_count])
        val.extend(group[train_count:])

    train.sort(key=lambda sample: sample.relative_path.as_posix())
    val.sort(key=lambda sample: sample.relative_path.as_posix())
    return {"train": train, "val": val}


def prepare_dataset_from_raw(
    raw_root: str | Path,
    output_root: str | Path,
    source_data_yaml: str | Path,
    report_dir: str | Path,
    options: SelectionOptions,
) -> PreparedDatasetResult:
    raw_root = Path(raw_root).resolve()
    output_root = Path(output_root).resolve()
    report_dir = Path(report_dir).resolve()
    config = load_dataset_config(source_data_yaml)

    if raw_root == output_root:
        raise ValueError("--raw-data and --prepared-data cannot point to the same directory.")
    if not raw_root.exists():
        raise FileNotFoundError(f"原始数据目录不存在: {raw_root}")

    candidates, rejected = collect_candidates(raw_root, len(config.names), options)
    selected = select_samples(candidates, options)
    splits = split_samples(selected, options.train_ratio, options.seed)

    if selected:
        _prepare_output_root(output_root, options.force)
        for split, samples in splits.items():
            for sample in samples:
                _copy_sample(sample, output_root, split)

    report_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = report_dir / "prepared_road_mark.yaml"
    selected_csv = report_dir / "selected_samples.csv"
    rejected_csv = report_dir / "rejected_samples.csv"
    summary_md = report_dir / "selection_summary.md"
    summary_json = report_dir / "selection_summary.json"

    _write_data_yaml(data_yaml, output_root, config.names)
    _write_selected_csv(selected_csv, splits)
    _write_rejected_csv(rejected_csv, rejected)
    summary = _selection_summary(raw_root, output_root, config.names, candidates, selected, rejected, splits, options)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary_md(summary_md, summary)

    return PreparedDatasetResult(data_yaml, selected_csv, rejected_csv, summary_md, summary_json, len(selected), len(rejected))


def _prepare_output_root(output_root: Path, force: bool) -> None:
    image_dir = output_root / "images"
    label_dir = output_root / "labels"
    existing_data = [
        path
        for base in (image_dir, label_dir)
        if base.exists()
        for path in base.rglob("*")
        if path.is_file() and path.name != ".gitkeep"
    ]
    if existing_data and not force:
        raise FileExistsError(
            f"目标数据目录已有 {len(existing_data)} 个文件。若要重新生成，请添加 --force-selection: {output_root}"
        )
    if force:
        for child in (image_dir, label_dir):
            if child.exists():
                resolved = child.resolve()
                if output_root.resolve() not in resolved.parents:
                    raise RuntimeError(f"拒绝清理非目标目录内路径: {resolved}")
                shutil.rmtree(resolved)
    for split in ("train", "val"):
        (image_dir / split).mkdir(parents=True, exist_ok=True)
        (label_dir / split).mkdir(parents=True, exist_ok=True)


def _copy_sample(sample: RawSample, output_root: Path, split: str) -> None:
    image_target = output_root / "images" / split / sample.relative_path
    label_target = output_root / "labels" / split / sample.relative_path.with_suffix(".txt")
    image_target.parent.mkdir(parents=True, exist_ok=True)
    label_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sample.image_path, image_target)
    shutil.copy2(sample.label_path, label_target)


def _write_data_yaml(path: Path, output_root: Path, names: list[str]) -> None:
    payload = {"path": output_root.as_posix(), "train": "images/train", "val": "images/val", "nc": len(names), "names": names}
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _write_selected_csv(path: Path, splits: dict[str, list[RawSample]]) -> None:
    fields = ["split", "image", "label", "relative_path", "width", "height", "box_count", "class_ids"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split, samples in splits.items():
            for sample in samples:
                writer.writerow(
                    {
                        "split": split,
                        "image": str(sample.image_path),
                        "label": str(sample.label_path),
                        "relative_path": sample.relative_path.as_posix(),
                        "width": sample.width,
                        "height": sample.height,
                        "box_count": sample.box_count,
                        "class_ids": " ".join(str(class_id) for class_id in sorted(sample.class_counts)),
                    }
                )


def _write_rejected_csv(path: Path, rejected: list[dict[str, str]]) -> None:
    fields = ["image", "label", "reason", "detail"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rejected)


def _selection_summary(
    raw_root: Path,
    output_root: Path,
    names: list[str],
    candidates: list[RawSample],
    selected: list[RawSample],
    rejected: list[dict[str, str]],
    splits: dict[str, list[RawSample]],
    options: SelectionOptions,
) -> dict[str, Any]:
    class_counts: dict[str, dict[str, int]] = {}
    for split, samples in splits.items():
        counter: Counter[int] = Counter()
        for sample in samples:
            counter.update(sample.class_counts)
        class_counts[split] = {names[class_id]: int(counter.get(class_id, 0)) for class_id in range(len(names))}

    return {
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "rejected_count": len(rejected),
        "split_counts": {split: len(samples) for split, samples in splits.items()},
        "class_box_counts": class_counts,
        "rejected_reasons": dict(sorted(Counter(row["reason"] for row in rejected).items())),
        "options": {
            "train_ratio": options.train_ratio,
            "seed": options.seed,
            "max_images": options.max_images,
            "max_images_per_class": options.max_images_per_class,
            "min_width": options.min_width,
            "min_height": options.min_height,
            "include_empty_labels": options.include_empty_labels,
            "force": options.force,
        },
    }


def _write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# 数据挑选与处理报告",
        "",
        f"- 原始数据目录: `{summary['raw_root']}`",
        f"- 输出数据目录: `{summary['output_root']}`",
        f"- 候选样本数: {summary['candidate_count']}",
        f"- 选中样本数: {summary['selected_count']}",
        f"- 剔除样本数: {summary['rejected_count']}",
        "",
        "## 划分统计",
        "",
        "| split | images |",
        "|---|---:|",
    ]
    for split, count in summary["split_counts"].items():
        lines.append(f"| {split} | {count} |")

    lines.extend(["", "## 类别框数量"])
    for split, counts in summary["class_box_counts"].items():
        lines.extend(["", f"### {split}", "", "| class | boxes |", "|---|---:|"])
        for class_name, count in counts.items():
            lines.append(f"| {class_name} | {count} |")

    lines.extend(["", "## 剔除原因", "", "| reason | count |", "|---|---:|"])
    for reason, count in summary["rejected_reasons"].items():
        lines.append(f"| {reason} | {count} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
