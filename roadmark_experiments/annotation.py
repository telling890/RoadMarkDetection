"""Candidate selection, manual review, and YOLO export for road-mark gaps."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from .dataset_audit import IMAGE_SUFFIXES


MANIFEST_FIELDS = (
    "candidate_id",
    "source_image",
    "original_split",
    "logical_source",
    "content_hash",
    "width",
    "height",
    "marking_score",
    "blur_score",
    "brightness",
    "source_batch",
    "capture_group",
    "scene",
    "weather",
    "time_of_day",
    "status",
    "box_count",
    "notes",
)
REVIEWED_STATUSES = {"positive", "negative"}


@dataclass(frozen=True)
class SelectionResult:
    manifest: Path
    candidates: int
    high_score_candidates: int
    random_candidates: int


@dataclass(frozen=True)
class ExportResult:
    data_yaml: Path
    train_images: int
    val_images: int
    positive_images: int
    negative_images: int


@dataclass(frozen=True)
class IngestResult:
    manifest: Path
    imported_images: int
    duplicate_images: int
    rejected_images: int


@dataclass(frozen=True)
class LabelImgWorkspaceResult:
    image_dir: Path
    label_dir: Path
    class_file: Path
    images: int


@dataclass(frozen=True)
class LabelImgSyncResult:
    positive: int
    negative: int
    pending: int
    updated: int


def logical_source_name(path: Path) -> str:
    return re.sub(r"\.rf\.[0-9a-fA-F]{32}$", "", path.stem)


def iter_source_images(source_root: str | Path) -> list[Path]:
    source_root = Path(source_root).resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"未找到候选图片目录: {source_root}")
    return sorted(path for path in source_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def road_marking_score(image: np.ndarray) -> float:
    """Estimate visible white/yellow road-marking content in the road ROI."""

    height, width = image.shape[:2]
    roi = image[int(height * 0.42) :, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 145), (179, 90, 255))
    yellow = cv2.inRange(hsv, (8, 65, 90), (42, 255, 255))
    mask = cv2.bitwise_or(white, yellow)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    ratio = float(np.count_nonzero(mask)) / float(mask.size)
    edges = cv2.Canny(mask, 50, 150)
    min_length = max(20, int(width * 0.06))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25, minLineLength=min_length, maxLineGap=20)
    line_length = 0.0
    if lines is not None:
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            line_length += math.hypot(float(x2 - x1), float(y2 - y1))
    normalized_lines = min(line_length / max(float(width * 8), 1.0), 1.0)
    return round(min(ratio * 4.0, 1.0) * 0.55 + normalized_lines * 0.45, 6)


def _candidate_id(source_root: Path, image_path: Path) -> str:
    relative = image_path.relative_to(source_root).as_posix()
    digest = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:12]
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", logical_source_name(image_path))[:48]
    return f"{stem}_{digest}"


def _original_split(source_root: Path, image_path: Path) -> str:
    relative_parts = [part.lower() for part in image_path.relative_to(source_root).parts]
    for split in ("train", "val", "valid", "test"):
        if split in relative_parts:
            return "val" if split in {"val", "valid", "test"} else "train"
    return "unknown"


def _read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def select_annotation_candidates(
    source_root: str | Path,
    workspace: str | Path,
    max_images: int = 1000,
    random_fraction: float = 0.25,
    seed: int = 42,
    scan_limit: int | None = None,
) -> SelectionResult:
    if max_images <= 0:
        raise ValueError("max_images 必须大于 0")
    if not 0.0 <= random_fraction <= 1.0:
        raise ValueError("random_fraction 必须在 0 到 1 之间")

    source_root = Path(source_root).resolve()
    workspace = Path(workspace).resolve()
    manifest = workspace / "manifest.csv"
    previous = {row["candidate_id"]: row for row in _read_manifest(manifest)}

    unique_images: dict[str, Path] = {}
    for image_path in iter_source_images(source_root):
        unique_images.setdefault(logical_source_name(image_path), image_path)

    images_to_score = list(unique_images.values())
    if scan_limit is not None and scan_limit > 0 and len(images_to_score) > scan_limit:
        images_to_score = random.Random(seed).sample(images_to_score, scan_limit)

    scored: list[tuple[float, Path]] = []
    for index, image_path in enumerate(images_to_score, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        scored.append((road_marking_score(image), image_path))
        if index % 1000 == 0:
            print(f"候选筛选进度: {index}/{len(images_to_score)}")

    scored.sort(key=lambda item: (-item[0], str(item[1])))
    total = min(max_images, len(scored))
    random_count = min(int(round(total * random_fraction)), total)
    high_count = total - random_count
    high = scored[:high_count]
    remainder = scored[high_count:]
    rng = random.Random(seed)
    random_rows = rng.sample(remainder, min(random_count, len(remainder)))
    selected = high + random_rows
    selected.sort(key=lambda item: (-item[0], str(item[1])))

    rows: list[dict[str, str]] = []
    for score, image_path in selected:
        candidate_id = _candidate_id(source_root, image_path)
        existing = previous.get(candidate_id, {})
        rows.append(
            {
                "candidate_id": candidate_id,
                "source_image": str(image_path),
                "original_split": _original_split(source_root, image_path),
                "logical_source": logical_source_name(image_path),
                "content_hash": existing.get("content_hash", ""),
                "width": existing.get("width", ""),
                "height": existing.get("height", ""),
                "marking_score": f"{score:.6f}",
                "blur_score": existing.get("blur_score", ""),
                "brightness": existing.get("brightness", ""),
                "source_batch": existing.get("source_batch", "existing_dataset"),
                "capture_group": existing.get("capture_group", logical_source_name(image_path)),
                "scene": existing.get("scene", "unknown"),
                "weather": existing.get("weather", "unknown"),
                "time_of_day": existing.get("time_of_day", "unknown"),
                "status": existing.get("status", "pending"),
                "box_count": existing.get("box_count", "0"),
                "notes": existing.get("notes", ""),
            }
        )

    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "labels").mkdir(exist_ok=True)
    _write_manifest(manifest, rows)
    return SelectionResult(manifest, len(rows), high_count, len(random_rows))


def _label_state(path: Path) -> dict[str, int | str] | None:
    if not path.exists():
        return None
    return {"mtime_ns": path.stat().st_mtime_ns, "sha256": _sha256(path)}


def prepare_labelimg_workspace(workspace: str | Path) -> LabelImgWorkspaceResult:
    workspace = Path(workspace).resolve()
    rows = _read_manifest(workspace / "manifest.csv")
    if not rows:
        raise RuntimeError(f"标注清单为空，请先执行 select 或 ingest: {workspace / 'manifest.csv'}")

    labelimg_root = workspace / "labelimg"
    image_dir = labelimg_root / "images"
    label_dir = labelimg_root / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    class_file = labelimg_root / "predefined_classes.txt"
    class_file.write_text("road_mark_missing\n", encoding="utf-8")
    (label_dir / "classes.txt").write_text("road_mark_missing\n", encoding="utf-8")

    baseline: dict[str, dict[str, int | str] | None] = {}
    prepared = 0
    for row in rows:
        source = Path(row["source_image"])
        if not source.exists():
            row["notes"] = ";".join(filter(None, (row.get("notes", ""), "missing_source_image")))
            continue
        image_target = image_dir / f"{row['candidate_id']}{source.suffix.lower()}"
        if not image_target.exists():
            try:
                os.link(source, image_target)
            except OSError:
                shutil.copy2(source, image_target)

        source_label = workspace / "labels" / f"{row['candidate_id']}.txt"
        label_target = label_dir / f"{row['candidate_id']}.txt"
        if source_label.exists() and (not label_target.exists() or row.get("status") == "prelabel"):
            shutil.copy2(source_label, label_target)
        baseline[row["candidate_id"]] = _label_state(label_target)
        prepared += 1

    _write_manifest(workspace / "manifest.csv", rows)
    (labelimg_root / "baseline.json").write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return LabelImgWorkspaceResult(image_dir, label_dir, class_file, prepared)


def _validate_labelimg_yolo(path: Path) -> int:
    count = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path.name}:{line_number} 不是 YOLO 5 字段格式")
        class_id = int(float(parts[0]))
        values = [float(value) for value in parts[1:]]
        if class_id != 0:
            raise ValueError(f"{path.name}:{line_number} 类别必须为 0")
        x, y, width, height = values
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"{path.name}:{line_number} 坐标超出 YOLO 范围")
        count += 1
    return count


def sync_labelimg_annotations(
    workspace: str | Path, accept_unlabeled_negative: bool = False
) -> LabelImgSyncResult:
    workspace = Path(workspace).resolve()
    manifest = workspace / "manifest.csv"
    rows = _read_manifest(manifest)
    labelimg_root = workspace / "labelimg"
    label_dir = labelimg_root / "labels"
    baseline_path = labelimg_root / "baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else {}
    canonical_labels = workspace / "labels"
    canonical_labels.mkdir(parents=True, exist_ok=True)
    updated = 0

    for row in rows:
        candidate_id = row["candidate_id"]
        source_label = label_dir / f"{candidate_id}.txt"
        current_state = _label_state(source_label)
        previous_state = baseline.get(candidate_id)
        canonical_label = canonical_labels / f"{candidate_id}.txt"
        canonical_state = _label_state(canonical_label)
        changed = current_state is not None and (
            current_state != previous_state or current_state.get("sha256") != (canonical_state or {}).get("sha256")
        )
        already_reviewed = row.get("status") in REVIEWED_STATUSES
        if source_label.exists() and (changed or already_reviewed):
            box_count = _validate_labelimg_yolo(source_label)
            shutil.copy2(source_label, canonical_label)
            row["status"] = "positive" if box_count else "negative"
            row["box_count"] = str(box_count)
            updated += int(changed)
        elif accept_unlabeled_negative and not already_reviewed:
            negative_label = canonical_labels / f"{candidate_id}.txt"
            negative_label.write_text("", encoding="utf-8")
            row["status"] = "negative"
            row["box_count"] = "0"
            updated += 1

    _write_manifest(manifest, rows)
    status = annotation_status(workspace)
    statuses = status["statuses"]
    return LabelImgSyncResult(
        positive=int(statuses.get("positive", 0)),
        negative=int(statuses.get("negative", 0)),
        pending=int(status["total"]) - int(statuses.get("positive", 0)) - int(statuses.get("negative", 0)),
        updated=updated,
    )


def run_labelimg(workspace: str | Path) -> LabelImgSyncResult:
    workspace_result = prepare_labelimg_workspace(workspace)
    launcher = Path(__file__).resolve().parents[1] / "labelimg_app.py"
    command = [
        sys.executable,
        str(launcher),
        str(workspace_result.image_dir),
        str(workspace_result.class_file),
        str(workspace_result.label_dir),
    ]
    print("启动 LabelImg:", " ".join(command))
    subprocess.run(command, check=True)
    return sync_labelimg_annotations(workspace)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ingest_collected_images(
    source_root: str | Path,
    workspace: str | Path,
    source_batch: str | None = None,
    scene: str = "unknown",
    weather: str = "unknown",
    time_of_day: str = "unknown",
    min_width: int = 640,
    min_height: int = 480,
) -> IngestResult:
    """Copy a new collection batch into the durable annotation workspace."""

    source_root = Path(source_root).resolve()
    workspace = Path(workspace).resolve()
    manifest = workspace / "manifest.csv"
    rows = _read_manifest(manifest)
    workspace_images = workspace / "source_images"
    workspace_images.mkdir(parents=True, exist_ok=True)
    (workspace / "labels").mkdir(parents=True, exist_ok=True)

    existing_hashes: set[str] = set()
    for row in rows:
        content_hash = row.get("content_hash", "")
        source_image = Path(row.get("source_image", ""))
        if not content_hash and source_image.is_file():
            content_hash = _sha256(source_image)
            row["content_hash"] = content_hash
        if content_hash:
            existing_hashes.add(content_hash)

    imported = 0
    duplicates = 0
    rejected_rows: list[dict[str, str]] = []
    batch_name = source_batch or source_root.name
    for image_path in iter_source_images(source_root):
        content_hash = _sha256(image_path)
        if content_hash in existing_hashes:
            duplicates += 1
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            rejected_rows.append({"source_image": str(image_path), "reason": "unreadable_image", "detail": ""})
            continue
        height, width = image.shape[:2]
        if width < min_width or height < min_height:
            rejected_rows.append(
                {
                    "source_image": str(image_path),
                    "reason": "resolution_too_small",
                    "detail": f"{width}x{height} < {min_width}x{min_height}",
                }
            )
            continue

        relative = image_path.relative_to(source_root)
        candidate_id = f"{re.sub(r'[^A-Za-z0-9_-]+', '_', image_path.stem)[:48]}_{content_hash[:12]}"
        destination = workspace_images / f"{candidate_id}{image_path.suffix.lower()}"
        shutil.copy2(image_path, destination)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        quality_notes = []
        if blur_score < 30.0:
            quality_notes.append("soft_image")
        if brightness < 35.0:
            quality_notes.append("dark_scene")
        elif brightness > 225.0:
            quality_notes.append("overexposed_scene")
        relative_parent = relative.parent.as_posix()
        capture_group = f"{batch_name}/{relative_parent}" if relative_parent != "." else f"{batch_name}/{logical_source_name(image_path)}"
        rows.append(
            {
                "candidate_id": candidate_id,
                "source_image": str(destination),
                "original_split": "new",
                "logical_source": logical_source_name(image_path),
                "content_hash": content_hash,
                "width": str(width),
                "height": str(height),
                "marking_score": f"{road_marking_score(image):.6f}",
                "blur_score": f"{blur_score:.3f}",
                "brightness": f"{brightness:.3f}",
                "source_batch": batch_name,
                "capture_group": capture_group,
                "scene": scene,
                "weather": weather,
                "time_of_day": time_of_day,
                "status": "pending",
                "box_count": "0",
                "notes": ";".join(quality_notes),
            }
        )
        existing_hashes.add(content_hash)
        imported += 1

    _write_manifest(manifest, rows)
    rejected_path = workspace / "ingest_rejected.csv"
    with rejected_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("source_image", "reason", "detail"))
        writer.writeheader()
        writer.writerows(rejected_rows)
    return IngestResult(manifest, imported, duplicates, len(rejected_rows))


def annotation_status(workspace: str | Path) -> dict[str, object]:
    workspace = Path(workspace).resolve()
    rows = _read_manifest(workspace / "manifest.csv")
    status_counts: dict[str, int] = {}
    for row in rows:
        status = row.get("status", "pending") or "pending"
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "total": len(rows),
        "statuses": status_counts,
        "boxes": sum(int(row.get("box_count", "0") or 0) for row in rows if row.get("status") == "positive"),
        "batches": len({row.get("source_batch", "") for row in rows if row.get("source_batch", "")}),
        "capture_groups": len({row.get("capture_group", "") for row in rows if row.get("capture_group", "")}),
    }


def prelabel_candidates(
    workspace: str | Path,
    weights: str | Path,
    device: str = "0",
    confidence: float = 0.25,
    image_size: int = 768,
) -> int:
    """Generate model-assisted boxes. Every result still requires manual review."""

    from ultralytics import YOLO

    from models.register_ultralytics import register_custom_modules

    workspace = Path(workspace).resolve()
    weights = Path(weights).resolve()
    if not weights.exists():
        raise FileNotFoundError(f"未找到预标注权重: {weights}")
    manifest = workspace / "manifest.csv"
    rows = _read_manifest(manifest)
    pending_rows = [row for row in rows if row.get("status", "pending") == "pending"]
    if not pending_rows:
        return 0

    register_custom_modules()
    model = YOLO(str(weights))
    sources = [row["source_image"] for row in pending_rows]
    results = model.predict(sources, imgsz=image_size, conf=confidence, device=device, stream=True, verbose=False)
    updated = 0
    for row, result in zip(pending_rows, results):
        height, width = result.orig_shape
        boxes: list[tuple[int, int, int, int]] = []
        if result.boxes is not None:
            for xyxy, class_id in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                if int(class_id) == 0:
                    x1, y1, x2, y2 = map(int, xyxy.tolist())
                    boxes.append((x1, y1, x2, y2))
        _write_yolo_boxes(workspace / "labels" / f"{row['candidate_id']}.txt", boxes, width, height)
        row["status"] = "prelabel"
        row["box_count"] = str(len(boxes))
        updated += 1
        if updated % 50 == 0:
            _write_manifest(manifest, rows)
    _write_manifest(manifest, rows)
    return updated


def _grouped_train_val_split(
    rows: list[dict[str, str]], train_ratio: float, seed: int
) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        group = row.get("capture_group", "") or row.get("logical_source", "") or row["candidate_id"]
        groups.setdefault(group, []).append(row)
    if len(groups) < 2:
        raise RuntimeError("至少需要 2 个独立采集组才能划分 train/val；请增加不同道路或不同采集会话。")

    rng = random.Random(seed)
    group_names = list(groups)
    rng.shuffle(group_names)
    target_val = max(1, int(round(len(rows) * (1.0 - train_ratio))))
    val_groups: set[str] = set()
    val_count = 0
    for group in group_names:
        if val_count >= target_val or len(val_groups) >= len(group_names) - 1:
            break
        val_groups.add(group)
        val_count += len(groups[group])

    def status_in(split_groups: set[str], status: str) -> bool:
        return any(row.get("status") == status for group in split_groups for row in groups[group])

    all_groups = set(group_names)
    for status in ("positive", "negative"):
        status_groups = {group for group in group_names if any(row.get("status") == status for row in groups[group])}
        if len(status_groups) < 2:
            continue
        train_groups = all_groups - val_groups
        if not status_in(val_groups, status):
            candidate = min(status_groups & train_groups, key=lambda group: len(groups[group]))
            val_groups.add(candidate)
        train_groups = all_groups - val_groups
        if not status_in(train_groups, status):
            candidate = min(status_groups & val_groups, key=lambda group: len(groups[group]))
            val_groups.remove(candidate)

    train_rows = [row for group in group_names if group not in val_groups for row in groups[group]]
    val_rows = [row for group in group_names if group in val_groups for row in groups[group]]
    if not train_rows or not val_rows:
        raise RuntimeError("按采集组划分后 train 或 val 为空，请增加独立采集组。")
    return {"train": train_rows, "val": val_rows}


def _read_yolo_boxes(path: Path, width: int, height: int) -> list[tuple[int, int, int, int]]:
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, x, y, w, h = map(float, line.split())
        if int(class_id) != 0:
            continue
        boxes.append(
            (
                int((x - w / 2) * width),
                int((y - h / 2) * height),
                int((x + w / 2) * width),
                int((y + h / 2) * height),
            )
        )
    return boxes


def _write_yolo_boxes(path: Path, boxes: list[tuple[int, int, int, int]], width: int, height: int) -> None:
    lines = []
    for x1, y1, x2, y2 in boxes:
        x1, x2 = sorted((max(0, x1), min(width, x2)))
        y1, y2 = sorted((max(0, y1), min(height, y2)))
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue
        x = (x1 + x2) / 2.0 / width
        y = (y1 + y2) / 2.0 / height
        w = (x2 - x1) / width
        h = (y2 - y1) / height
        lines.append(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


class AnnotationSession:
    def __init__(self, workspace: str | Path, start: int = 0) -> None:
        self.workspace = Path(workspace).resolve()
        self.manifest_path = self.workspace / "manifest.csv"
        self.rows = _read_manifest(self.manifest_path)
        if not self.rows:
            raise RuntimeError(f"标注清单为空，请先执行 select: {self.manifest_path}")
        pending = next((index for index, row in enumerate(self.rows) if row["status"] not in REVIEWED_STATUSES), 0)
        self.index = min(max(start, 0), len(self.rows) - 1) if start else pending
        self.boxes: list[tuple[int, int, int, int]] = []
        self.image: np.ndarray | None = None
        self.display_scale = 1.0
        self.drag_start: tuple[int, int] | None = None
        self.drag_end: tuple[int, int] | None = None
        self.window = "RoadMark Missing Annotation"

    @property
    def row(self) -> dict[str, str]:
        return self.rows[self.index]

    @property
    def label_path(self) -> Path:
        return self.workspace / "labels" / f"{self.row['candidate_id']}.txt"

    def load(self) -> None:
        self.image = cv2.imread(self.row["source_image"])
        if self.image is None:
            raise RuntimeError(f"无法读取图片: {self.row['source_image']}")
        height, width = self.image.shape[:2]
        self.boxes = _read_yolo_boxes(self.label_path, width, height)
        self.drag_start = None
        self.drag_end = None

    def _mouse(self, event: int, x: int, y: int, _flags: int, _param) -> None:
        point = (int(x / self.display_scale), int(y / self.display_scale))
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = point
            self.drag_end = point
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            self.drag_end = point
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            self.drag_end = point
            x1, y1 = self.drag_start
            x2, y2 = self.drag_end
            if abs(x2 - x1) >= 3 and abs(y2 - y1) >= 3:
                self.boxes.append((x1, y1, x2, y2))
            self.drag_start = None
            self.drag_end = None

    def _render(self) -> np.ndarray:
        assert self.image is not None
        canvas = self.image.copy()
        for x1, y1, x2, y2 in self.boxes:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if self.drag_start and self.drag_end:
            cv2.rectangle(canvas, self.drag_start, self.drag_end, (0, 255, 255), 2)
        reviewed = sum(row["status"] in REVIEWED_STATUSES for row in self.rows)
        status = f"{self.index + 1}/{len(self.rows)} reviewed={reviewed} status={self.row['status']} boxes={len(self.boxes)}"
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(canvas, status, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        max_width, max_height = 1280, 900
        self.display_scale = min(max_width / canvas.shape[1], max_height / canvas.shape[0], 1.5)
        return cv2.resize(canvas, None, fx=self.display_scale, fy=self.display_scale, interpolation=cv2.INTER_AREA)

    def save(self, status: str) -> None:
        assert self.image is not None
        if status == "positive" and not self.boxes:
            print("正样本至少需要一个框；无缺失目标请按 N。")
            return
        if status == "negative":
            self.boxes = []
        height, width = self.image.shape[:2]
        _write_yolo_boxes(self.label_path, self.boxes, width, height)
        self.row["status"] = status
        self.row["box_count"] = str(len(self.boxes))
        _write_manifest(self.manifest_path, self.rows)
        self.index = min(self.index + 1, len(self.rows) - 1)
        self.load()

    def run(self) -> None:
        print("鼠标左键拖动框选缺失区域。S=保存正样本，N=确认负样本，Z=撤销，A/D=上一张/下一张，Q=退出。")
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self._mouse)
        self.load()
        while True:
            cv2.imshow(self.window, self._render())
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                self.save("positive")
            elif key == ord("n"):
                self.save("negative")
            elif key == ord("z") and self.boxes:
                self.boxes.pop()
            elif key == ord("a"):
                self.index = max(0, self.index - 1)
                self.load()
            elif key == ord("d"):
                self.index = min(len(self.rows) - 1, self.index + 1)
                self.load()
        cv2.destroyAllWindows()


def export_reviewed_dataset(
    workspace: str | Path,
    output_root: str | Path,
    data_yaml: str | Path,
    train_ratio: float = 0.8,
    seed: int = 42,
    force: bool = False,
) -> ExportResult:
    if not 0.5 <= train_ratio < 1.0:
        raise ValueError("train_ratio 必须在 [0.5, 1.0) 范围内")
    workspace = Path(workspace).resolve()
    output_root = Path(output_root).resolve()
    data_yaml = Path(data_yaml).resolve()
    rows = [row for row in _read_manifest(workspace / "manifest.csv") if row["status"] in REVIEWED_STATUSES]
    if len(rows) < 2:
        raise RuntimeError("至少需要 2 张已复核图片才能导出 train/val 数据集。")
    if not any(row["status"] == "positive" for row in rows):
        raise RuntimeError("没有包含缺失框的正样本，不能导出。")

    if output_root.exists() and any(output_root.iterdir()):
        if not force:
            raise FileExistsError(f"输出目录非空: {output_root}。确认后使用 --force。")
        shutil.rmtree(output_root)

    positives = [row for row in rows if row["status"] == "positive"]
    negatives = [row for row in rows if row["status"] == "negative"]
    split_map = _grouped_train_val_split(rows, train_ratio, seed)

    for split, split_rows_value in split_map.items():
        image_dir = output_root / "images" / split
        label_dir = output_root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for row in split_rows_value:
            source = Path(row["source_image"])
            destination = image_dir / f"{row['candidate_id']}{source.suffix.lower()}"
            shutil.copy2(source, destination)
            source_label = workspace / "labels" / f"{row['candidate_id']}.txt"
            destination_label = label_dir / f"{row['candidate_id']}.txt"
            if source_label.exists():
                shutil.copy2(source_label, destination_label)
            else:
                destination_label.write_text("", encoding="utf-8")

    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    relative_root = os.path.relpath(output_root, data_yaml.parent).replace("\\", "/")
    payload = {
        "path": relative_root,
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["road_mark_missing"],
    }
    data_yaml.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    semantics = {
        "target_task": "road_mark_missing_detection",
        "source_task": "manual_review_of_road_images",
        "verified_for_target_task": True,
        "annotation_method": "manual_bbox_review",
        "classes": ["road_mark_missing"],
    }
    semantics_path = data_yaml.with_name(f"{data_yaml.stem}_semantics.yaml")
    semantics_path.write_text(yaml.safe_dump(semantics, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return ExportResult(data_yaml, len(split_map["train"]), len(split_map["val"]), len(positives), len(negatives))
