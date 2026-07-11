"""Import Roboflow YOLO txt zip archives into the project dataset layout."""

from __future__ import annotations

import csv
import json
import re
import shutil
import struct
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .dataset_audit import IMAGE_SUFFIXES
from .roadmark_missing import ROADMARK_MISSING_CLASSES, SOURCE_RDD_CLASSES, source_to_missing_class_id


CANONICAL_RDD_CLASSES = SOURCE_RDD_CLASSES
ROADMARK_MISSING_CLASS_NAMES = ROADMARK_MISSING_CLASSES
LABEL_DIR_NAMES = ("labels", "labelTxt")


@dataclass(frozen=True)
class ZipImportOptions:
    include_empty_labels: bool = False
    force: bool = False


@dataclass(frozen=True)
class ZipImportResult:
    data_yaml: Path
    manifest_csv: Path
    summary_csv: Path
    rejected_csv: Path
    summary_json: Path
    imported_images: int
    rejected_images: int
    imported_boxes: int


@dataclass(frozen=True)
class ImageEntry:
    zip_path: Path
    entry_name: str
    source_split: str
    target_split: str
    region: str
    relative_image: Path


def discover_txt_zips(raw_root: str | Path) -> list[Path]:
    raw_root = Path(raw_root).resolve()
    if raw_root.is_file() and raw_root.name.lower().endswith("_txt.zip"):
        return [raw_root]
    if not raw_root.exists():
        return []
    return sorted(path for path in raw_root.rglob("*_txt.zip") if path.is_file())


def import_rdd_txt_zips(
    raw_root: str | Path,
    output_root: str | Path,
    data_yaml: str | Path,
    report_dir: str | Path,
    options: ZipImportOptions | None = None,
    class_names: list[str] | None = None,
) -> ZipImportResult:
    """Import all ``*_txt.zip`` archives under raw_root into YOLO train/val folders."""

    options = options or ZipImportOptions()
    class_names = class_names or ROADMARK_MISSING_CLASS_NAMES
    raw_root = Path(raw_root).resolve()
    output_root = Path(output_root).resolve()
    data_yaml = Path(data_yaml).resolve()
    report_dir = Path(report_dir).resolve()

    zip_paths = discover_txt_zips(raw_root)
    if not zip_paths:
        raise FileNotFoundError(f"未在 {raw_root} 下找到 *_txt.zip 数据包。")

    _prepare_output_root(output_root, options.force)
    report_dir.mkdir(parents=True, exist_ok=True)

    class_to_id = {name: index for index, name in enumerate(class_names)}
    manifest_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    total_boxes = 0
    total_images = 0

    for zip_path in zip_paths:
        region = _region_slug(zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            source_names = _read_source_names(archive)
            source_to_target = _source_class_map(source_names, class_to_id)
            entries = list(_iter_image_entries(archive, zip_path, region))
            zip_boxes = 0
            zip_imported = 0
            zip_rejected = 0
            split_counts: Counter[str] = Counter()
            class_counts: Counter[str] = Counter()

            lower_lookup = {name.lower(): name for name in archive.namelist()}
            for image in entries:
                label_name = _find_label_entry(image.entry_name, lower_lookup)
                if label_name is None:
                    rejected_rows.append(_reject_row(image, "", "missing_label", "No matching label file"))
                    zip_rejected += 1
                    continue

                image_size = _read_image_size_from_archive(archive, image.entry_name)
                if image_size is None:
                    rejected_rows.append(_reject_row(image, label_name, "unreadable_image", "Cannot determine image size"))
                    zip_rejected += 1
                    continue

                raw_label = archive.read(label_name).decode("utf-8", errors="replace")
                converted, counts, errors = _convert_label_lines(raw_label, source_to_target, class_names, image_size)
                if errors:
                    rejected_rows.append(_reject_row(image, label_name, "invalid_label", "; ".join(errors)))
                    zip_rejected += 1
                    continue
                if not converted and not options.include_empty_labels:
                    rejected_rows.append(_reject_row(image, label_name, "empty_label", "No valid boxes in label file"))
                    zip_rejected += 1
                    continue

                output_image, output_label = _output_paths(output_root, image)
                output_image.parent.mkdir(parents=True, exist_ok=True)
                output_label.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(image.entry_name) as source, output_image.open("wb") as target:
                    shutil.copyfileobj(source, target)
                output_label.write_text("\n".join(converted) + ("\n" if converted else ""), encoding="utf-8")

                image_boxes = int(sum(counts.values()))
                total_boxes += image_boxes
                zip_boxes += image_boxes
                total_images += 1
                zip_imported += 1
                split_counts[image.target_split] += 1
                for class_id, count in counts.items():
                    class_counts[class_names[class_id]] += count
                manifest_rows.append(
                    {
                        "split": image.target_split,
                        "region": image.region,
                        "zip": str(zip_path),
                        "source_image": image.entry_name,
                        "source_label": label_name,
                        "output_image": str(output_image),
                        "output_label": str(output_label),
                        "boxes": image_boxes,
                        "classes": " ".join(class_names[class_id] for class_id in sorted(counts)),
                    }
                )

            summary_rows.append(
                {
                    "region": region,
                    "zip": str(zip_path),
                    "images": zip_imported,
                    "train_images": split_counts.get("train", 0),
                    "val_images": split_counts.get("val", 0),
                    "boxes": zip_boxes,
                    "rejected": zip_rejected,
                    "source_names": " ".join(source_names),
                    "class_counts": json.dumps(dict(sorted(class_counts.items())), ensure_ascii=False),
                }
            )

    _write_data_yaml(data_yaml, output_root, class_names)
    manifest_csv = report_dir / "data_split_manifest.csv"
    summary_csv = report_dir / "zip_import_summary.csv"
    rejected_csv = report_dir / "zip_import_rejected.csv"
    summary_json = report_dir / "zip_import_summary.json"
    _write_csv(
        manifest_csv,
        manifest_rows,
        ["split", "region", "zip", "source_image", "source_label", "output_image", "output_label", "boxes", "classes"],
    )
    _write_csv(
        summary_csv,
        summary_rows,
        ["region", "zip", "images", "train_images", "val_images", "boxes", "rejected", "source_names", "class_counts"],
    )
    _write_csv(rejected_csv, rejected_rows, ["region", "zip", "source_image", "source_label", "reason", "detail"])
    summary_payload = _summary_payload(raw_root, output_root, data_yaml, class_names, manifest_rows, rejected_rows, summary_rows)
    summary_json.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return ZipImportResult(
        data_yaml=data_yaml,
        manifest_csv=manifest_csv,
        summary_csv=summary_csv,
        rejected_csv=rejected_csv,
        summary_json=summary_json,
        imported_images=total_images,
        rejected_images=len(rejected_rows),
        imported_boxes=total_boxes,
    )


def import_extracted_roboflow_dirs(
    raw_root: str | Path,
    output_root: str | Path,
    data_yaml: str | Path,
    report_dir: str | Path,
    options: ZipImportOptions | None = None,
    class_names: list[str] | None = None,
) -> ZipImportResult:
    """Import already extracted Roboflow YOLO txt folders into the project dataset layout."""

    options = options or ZipImportOptions()
    class_names = class_names or ROADMARK_MISSING_CLASS_NAMES
    raw_root = Path(raw_root).resolve()
    output_root = Path(output_root).resolve()
    data_yaml = Path(data_yaml).resolve()
    report_dir = Path(report_dir).resolve()

    roots = _discover_extracted_roots(raw_root)
    if not roots:
        raise FileNotFoundError(f"未在 {raw_root} 下找到含 data.yaml 的已解压 Roboflow 数据目录。")

    _prepare_output_root(output_root, options.force)
    report_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    total_boxes = 0
    total_images = 0

    for root in roots:
        region = _region_slug(root)
        source_names = _read_source_names_from_file(root / "data.yaml")
        source_to_target = _source_class_map(source_names, {name: index for index, name in enumerate(class_names)})
        zip_boxes = 0
        zip_imported = 0
        zip_rejected = 0
        split_counts: Counter[str] = Counter()
        class_counts: Counter[str] = Counter()

        for image_path in _iter_extracted_images(root):
            source_split = image_path.relative_to(root).parts[0]
            target_split = _normalize_split(source_split)
            if target_split is None:
                continue
            relative_image = image_path.relative_to(root / source_split / "images")
            label_path = _find_extracted_label_path(root, source_split, relative_image)
            image = ImageEntry(
                zip_path=root,
                entry_name=str(image_path),
                source_split=source_split,
                target_split=target_split,
                region=region,
                relative_image=relative_image,
            )
            if label_path is None:
                rejected_rows.append(_reject_row(image, "", "missing_label", "No matching label file"))
                zip_rejected += 1
                continue
            image_size = _read_image_size_from_file(image_path)
            if image_size is None:
                rejected_rows.append(_reject_row(image, str(label_path), "unreadable_image", "Cannot determine image size"))
                zip_rejected += 1
                continue
            raw_label = label_path.read_text(encoding="utf-8", errors="replace")
            converted, counts, errors = _convert_label_lines(raw_label, source_to_target, class_names, image_size)
            if errors:
                rejected_rows.append(_reject_row(image, str(label_path), "invalid_label", "; ".join(errors)))
                zip_rejected += 1
                continue
            if not converted and not options.include_empty_labels:
                rejected_rows.append(_reject_row(image, str(label_path), "empty_label", "No valid boxes in label file"))
                zip_rejected += 1
                continue

            output_image, output_label = _output_paths(output_root, image)
            output_image.parent.mkdir(parents=True, exist_ok=True)
            output_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, output_image)
            output_label.write_text("\n".join(converted) + ("\n" if converted else ""), encoding="utf-8")

            image_boxes = int(sum(counts.values()))
            total_boxes += image_boxes
            zip_boxes += image_boxes
            total_images += 1
            zip_imported += 1
            split_counts[target_split] += 1
            for class_id, count in counts.items():
                class_counts[class_names[class_id]] += count
            manifest_rows.append(
                {
                    "split": target_split,
                    "region": region,
                    "zip": str(root),
                    "source_image": str(image_path),
                    "source_label": str(label_path),
                    "output_image": str(output_image),
                    "output_label": str(output_label),
                    "boxes": image_boxes,
                    "classes": " ".join(class_names[class_id] for class_id in sorted(counts)),
                }
            )

        summary_rows.append(
            {
                "region": region,
                "zip": str(root),
                "images": zip_imported,
                "train_images": split_counts.get("train", 0),
                "val_images": split_counts.get("val", 0),
                "boxes": zip_boxes,
                "rejected": zip_rejected,
                "source_names": " ".join(source_names),
                "class_counts": json.dumps(dict(sorted(class_counts.items())), ensure_ascii=False),
            }
        )

    _write_data_yaml(data_yaml, output_root, class_names)
    manifest_csv = report_dir / "data_split_manifest.csv"
    summary_csv = report_dir / "zip_import_summary.csv"
    rejected_csv = report_dir / "zip_import_rejected.csv"
    summary_json = report_dir / "zip_import_summary.json"
    _write_csv(
        manifest_csv,
        manifest_rows,
        ["split", "region", "zip", "source_image", "source_label", "output_image", "output_label", "boxes", "classes"],
    )
    _write_csv(
        summary_csv,
        summary_rows,
        ["region", "zip", "images", "train_images", "val_images", "boxes", "rejected", "source_names", "class_counts"],
    )
    _write_csv(rejected_csv, rejected_rows, ["region", "zip", "source_image", "source_label", "reason", "detail"])
    summary_payload = _summary_payload(raw_root, output_root, data_yaml, class_names, manifest_rows, rejected_rows, summary_rows)
    summary_json.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return ZipImportResult(data_yaml, manifest_csv, summary_csv, rejected_csv, summary_json, total_images, len(rejected_rows), total_boxes)


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
        raise FileExistsError(f"目标数据目录已有 {len(existing_data)} 个文件。若要重新生成，请添加 --force-selection: {output_root}")
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


def _read_source_names(archive: zipfile.ZipFile) -> list[str]:
    data_yaml_name = next((name for name in archive.namelist() if Path(name).name in {"data.yaml", "data.yml"}), None)
    if data_yaml_name is None:
        raise FileNotFoundError("zip 中缺少 data.yaml")
    payload = yaml.safe_load(archive.read(data_yaml_name).decode("utf-8", errors="replace")) or {}
    raw_names = payload.get("names", [])
    if isinstance(raw_names, dict):
        return [str(raw_names[key]) for key in sorted(raw_names)]
    return [str(name) for name in raw_names]


def _read_source_names_from_file(path: Path) -> list[str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_names = payload.get("names", [])
    if isinstance(raw_names, dict):
        return [str(raw_names[key]) for key in sorted(raw_names)]
    return [str(name) for name in raw_names]


def _discover_extracted_roots(raw_root: Path) -> list[Path]:
    if (raw_root / "data.yaml").exists():
        return [raw_root]
    return sorted(path.parent for path in raw_root.rglob("data.yaml") if (path.parent / "train" / "images").exists())


def _iter_extracted_images(root: Path) -> list[Path]:
    images: list[Path] = []
    for split_dir in root.iterdir():
        if not split_dir.is_dir() or _normalize_split(split_dir.name) is None:
            continue
        image_dir = split_dir / "images"
        if not image_dir.exists():
            continue
        images.extend(path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    return sorted(images)


def _find_extracted_label_path(root: Path, source_split: str, relative_image: Path) -> Path | None:
    relative_label = relative_image.with_suffix(".txt")
    for label_dir in LABEL_DIR_NAMES:
        candidate = root / source_split / label_dir / relative_label
        if candidate.exists():
            return candidate
    same_dir = root / source_split / "images" / relative_label
    return same_dir if same_dir.exists() else None


def _source_class_map(source_names: list[str], class_to_id: dict[str, int]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for source_id, name in enumerate(source_names):
        if name in class_to_id:
            mapping[source_id] = class_to_id[name]
            continue
        missing_id = source_to_missing_class_id(name)
        if missing_id is not None:
            mapping[source_id] = missing_id
    return mapping


def _iter_image_entries(archive: zipfile.ZipFile, zip_path: Path, region: str) -> list[ImageEntry]:
    entries: list[ImageEntry] = []
    for name in archive.namelist():
        path = Path(name)
        if name.endswith("/") or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        parts = name.replace("\\", "/").split("/")
        if len(parts) < 3 or parts[1].lower() != "images":
            continue
        target_split = _normalize_split(parts[0])
        if target_split is None:
            continue
        relative_image = Path(*parts[2:])
        entries.append(
            ImageEntry(
                zip_path=zip_path,
                entry_name=name,
                source_split=parts[0],
                target_split=target_split,
                region=region,
                relative_image=relative_image,
            )
        )
    return sorted(entries, key=lambda item: (item.target_split, item.entry_name))


def _normalize_split(split: str) -> str | None:
    split = split.lower()
    if split == "train":
        return "train"
    if split in {"valid", "val", "test"}:
        return "val"
    return None


def _find_label_entry(image_name: str, lower_lookup: dict[str, str]) -> str | None:
    parts = image_name.replace("\\", "/").split("/")
    if len(parts) < 3:
        return None
    split = parts[0]
    relative_label = "/".join(parts[2:])
    relative_label = str(Path(relative_label).with_suffix(".txt")).replace("\\", "/")
    candidates = [f"{split}/{label_dir}/{relative_label}" for label_dir in LABEL_DIR_NAMES]
    candidates.append(str(Path(image_name).with_suffix(".txt")).replace("\\", "/"))
    for candidate in candidates:
        found = lower_lookup.get(candidate.lower())
        if found is not None:
            return found
    return None


def _convert_label_lines(
    raw_label: str,
    source_to_target: dict[int, int],
    class_names: list[str],
    image_size: tuple[int, int],
) -> tuple[list[str], Counter[int], list[str]]:
    converted: list[str] = []
    counts: Counter[int] = Counter()
    errors: list[str] = []
    image_width, image_height = image_size
    for line_number, line in enumerate(raw_label.splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) == 5:
            try:
                source_class = int(float(parts[0]))
                x, y, w, h = [float(value) for value in parts[1:]]
            except ValueError:
                errors.append(f"line {line_number}: non-numeric label values")
                continue
        elif len(parts) >= 9:
            parsed = _parse_quadrilateral_label(parts, image_width, image_height)
            if parsed is None:
                errors.append(f"line {line_number}: invalid quadrilateral label")
                continue
            class_name, x, y, w, h = parsed
            source_class = _source_index_for_class_name(class_name, source_to_target, class_names)
            if source_class is None:
                errors.append(f"line {line_number}: class name {class_name} is not in source names")
                continue
        else:
            errors.append(f"line {line_number}: expected YOLO 5 fields or quadrilateral 9+ fields, got {len(parts)}")
            continue
        if source_class not in source_to_target:
            errors.append(f"line {line_number}: source class {source_class} is not in canonical classes")
            continue
        target_class = source_to_target[source_class]
        if target_class < 0 or target_class >= len(class_names):
            errors.append(f"line {line_number}: target class {target_class} outside canonical range")
            continue
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            errors.append(f"line {line_number}: bbox values outside YOLO normalized range")
            continue
        converted.append(f"{target_class} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        counts[target_class] += 1
    return converted, counts, errors


def _parse_quadrilateral_label(parts: list[str], image_width: int, image_height: int) -> tuple[str, float, float, float, float] | None:
    try:
        values = [float(value) for value in parts[:8]]
    except ValueError:
        return None
    class_name = parts[8]
    xs = values[0::2]
    ys = values[1::2]
    x_min = max(0.0, min(xs))
    x_max = min(float(image_width), max(xs))
    y_min = max(0.0, min(ys))
    y_max = min(float(image_height), max(ys))
    box_width = x_max - x_min
    box_height = y_max - y_min
    if image_width <= 0 or image_height <= 0 or box_width <= 0 or box_height <= 0:
        return None
    x_center = ((x_min + x_max) / 2.0) / image_width
    y_center = ((y_min + y_max) / 2.0) / image_height
    width = box_width / image_width
    height = box_height / image_height
    return class_name, x_center, y_center, width, height


def _source_index_for_class_name(class_name: str, source_to_target: dict[int, int], class_names: list[str]) -> int | None:
    if class_name not in class_names:
        target_class = source_to_missing_class_id(class_name)
        if target_class is None:
            return None
    else:
        target_class = class_names.index(class_name)
    for source_class, mapped_target in source_to_target.items():
        if mapped_target == target_class:
            return source_class
    return None


def _read_image_size_from_archive(archive: zipfile.ZipFile, image_name: str) -> tuple[int, int] | None:
    data = archive.read(image_name)
    return _read_image_size_from_bytes(data)


def _read_image_size_from_file(path: Path) -> tuple[int, int] | None:
    return _read_image_size_from_bytes(path.read_bytes())


def _read_image_size_from_bytes(data: bytes) -> tuple[int, int] | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        if len(data) >= 24:
            width, height = struct.unpack(">II", data[16:24])
            return int(width), int(height)
        return None
    if data[:2] == b"\xff\xd8":
        return _jpeg_size(data)
    if data[:6] in {b"GIF87a", b"GIF89a"} and len(data) >= 10:
        width, height = struct.unpack("<HH", data[6:10])
        return int(width), int(height)
    if data[:2] == b"BM" and len(data) >= 26:
        width, height = struct.unpack("<ii", data[18:26])
        return abs(int(width)), abs(int(height))
    return None


def _jpeg_size(data: bytes) -> tuple[int, int] | None:
    index = 2
    while index < len(data):
        while index < len(data) and data[index] == 0xFF:
            index += 1
        if index >= len(data):
            return None
        marker = data[index]
        index += 1
        if marker in {0xD8, 0xD9, 0x01} or 0xD0 <= marker <= 0xD7:
            continue
        if index + 2 > len(data):
            return None
        segment_length = struct.unpack(">H", data[index : index + 2])[0]
        if segment_length < 2 or index + segment_length > len(data):
            return None
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            if index + 7 > len(data):
                return None
            height = struct.unpack(">H", data[index + 3 : index + 5])[0]
            width = struct.unpack(">H", data[index + 5 : index + 7])[0]
            return int(width), int(height)
        index += segment_length
    return None


def _output_paths(output_root: Path, image: ImageEntry) -> tuple[Path, Path]:
    relative_image = Path(image.region) / image.relative_image
    output_image = output_root / "images" / image.target_split / relative_image
    output_label = output_root / "labels" / image.target_split / relative_image.with_suffix(".txt")
    return output_image, output_label


def _region_slug(zip_path: Path) -> str:
    raw = zip_path.name if zip_path.is_dir() else (zip_path.parent.name or zip_path.stem)
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", raw.strip()).strip("_")
    return slug or "region"


def _reject_row(image: ImageEntry, label_name: str, reason: str, detail: str) -> dict[str, Any]:
    return {
        "region": image.region,
        "zip": str(image.zip_path),
        "source_image": image.entry_name,
        "source_label": label_name,
        "reason": reason,
        "detail": detail,
    }


def _write_data_yaml(path: Path, output_root: Path, names: list[str]) -> None:
    try:
        dataset_path = Path("../") / output_root.relative_to(path.parent.parent)
    except ValueError:
        dataset_path = output_root
    payload = {
        "path": dataset_path.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "nc": len(names),
        "names": names,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary_payload(
    raw_root: Path,
    output_root: Path,
    data_yaml: Path,
    class_names: list[str],
    manifest_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    split_counts: Counter[str] = Counter(row["split"] for row in manifest_rows)
    class_counts: Counter[str] = Counter()
    for row in manifest_rows:
        for class_name in str(row["classes"]).split():
            class_counts[class_name] += 1
    box_counts_by_class: defaultdict[str, int] = defaultdict(int)
    for summary in summary_rows:
        for class_name, count in json.loads(summary["class_counts"]).items():
            box_counts_by_class[class_name] += int(count)
    return {
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "data_yaml": str(data_yaml),
        "classes": class_names,
        "split_images": dict(sorted(split_counts.items())),
        "total_images": len(manifest_rows),
        "total_boxes": int(sum(int(row["boxes"]) for row in manifest_rows)),
        "rejected_images": len(rejected_rows),
        "image_class_presence": dict(sorted(class_counts.items())),
        "box_counts_by_class": {name: int(box_counts_by_class.get(name, 0)) for name in class_names},
        "zips": summary_rows,
    }
