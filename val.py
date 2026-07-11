"""YOLO26 路面标线缺失验证脚本。

输出 Precision、Recall、mAP50、mAP50-95、FPS，并生成 confusion_matrix.png 与 PR_curve.png。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ultralytics import YOLO

from models.register_ultralytics import register_custom_modules
from roadmark_experiments.roadmark_missing import load_semantic_metadata, semantic_warning
from utils.yolo_data import write_resolved_data_yaml


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证 YOLO26 路面标线缺失检测模型")
    parser.add_argument("--weights", default="runs/train/best.pt", help="模型权重")
    parser.add_argument("--data", default="data/road_mark.yaml")
    parser.add_argument("--img-size", "--imgsz", dest="img_size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--project", default="runs/val")
    parser.add_argument("--name", default="roadmark_val")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--tta", action="store_true", help="启用测试时增强；精度优先，速度会下降")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def select_device(device: str) -> str:
    if device != "auto":
        if device.lower() != "cpu" and not torch.cuda.is_available():
            raise RuntimeError(
                f"请求使用 GPU --device {device}，但当前 PyTorch 未检测到 CUDA。"
                "请安装 CUDA 版 PyTorch，或临时改用 --device cpu。"
            )
        return device
    return "0" if torch.cuda.is_available() else "cpu"


def to_float(value) -> float:
    arr = np.asarray(value, dtype=float)
    return float(arr.mean()) if arr.size else 0.0


def to_array(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=float)
    return np.asarray(value, dtype=float).reshape(-1)


def per_class_results(metrics) -> list[dict[str, float | int | str]]:
    names_raw = getattr(metrics, "names", {}) or {}
    names = {int(key): str(value) for key, value in names_raw.items()} if isinstance(names_raw, dict) else {
        index: str(value) for index, value in enumerate(names_raw)
    }
    precision = to_array(getattr(metrics.box, "p", None))
    recall = to_array(getattr(metrics.box, "r", None))
    map50 = to_array(getattr(metrics.box, "ap50", None))
    map50_95 = to_array(getattr(metrics.box, "maps", None))
    class_count = max(len(names), precision.size, recall.size, map50.size, map50_95.size)

    rows: list[dict[str, float | int | str]] = []
    for class_id in range(class_count):
        p = float(precision[class_id]) if class_id < precision.size else 0.0
        r = float(recall[class_id]) if class_id < recall.size else 0.0
        rows.append(
            {
                "class_id": class_id,
                "class_name": names.get(class_id, f"class_{class_id}"),
                "Precision": p,
                "Recall": r,
                "F1": 2.0 * p * r / (p + r) if p + r > 0 else 0.0,
                "mAP50": float(map50[class_id]) if class_id < map50.size else 0.0,
                "mAP50-95": float(map50_95[class_id]) if class_id < map50_95.size else 0.0,
            }
        )
    return rows


def main() -> dict[str, Any]:
    args = parse_args()
    register_custom_modules()
    weights = resolve_path(args.weights)
    source_data = resolve_path(args.data)
    semantic_metadata = load_semantic_metadata(source_data)
    warning = semantic_warning(source_data)
    if warning:
        print(f"警告: {warning}")
    data = write_resolved_data_yaml(source_data, ROOT / "runs" / "_runtime_data")
    if not weights.exists():
        raise FileNotFoundError(f"未找到权重文件: {weights}")

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data),
        imgsz=args.img_size,
        batch=args.batch,
        device=select_device(args.device),
        project=str(resolve_path(args.project)),
        name=args.name,
        exist_ok=True,
        plots=True,
        conf=args.conf,
        iou=args.iou,
        augment=args.tta,
    )

    speed = getattr(metrics, "speed", {}) or {}
    total_ms = float(speed.get("preprocess", 0.0)) + float(speed.get("inference", 0.0)) + float(
        speed.get("postprocess", 0.0)
    )
    fps = 1000.0 / total_ms if total_ms > 0 else 0.0

    precision = to_float(metrics.box.p)
    recall = to_float(metrics.box.r)
    result: dict[str, Any] = {
        "Precision": precision,
        "Recall": recall,
        "F1": 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0,
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "FPS": fps,
        "TTA": bool(args.tta),
        "LabelSemanticsVerified": bool(semantic_metadata.get("verified_for_target_task", False)),
        "SourceTask": semantic_metadata.get("source_task", "unknown"),
        "TargetTask": semantic_metadata.get("target_task", "unknown"),
        "PerClass": per_class_results(metrics),
    }

    save_dir = Path(metrics.save_dir)
    metrics_json = save_dir / "metrics.json"
    metrics_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("验证结果:")
    for key in ("Precision", "Recall", "F1", "mAP50", "mAP50-95", "FPS"):
        print(f"{key}: {result[key]:.4f}")
    print("逐类别结果:")
    for row in result["PerClass"]:
        print(
            f"- {row['class_name']}: P={row['Precision']:.4f}, R={row['Recall']:.4f}, "
            f"F1={row['F1']:.4f}, mAP50={row['mAP50']:.4f}, mAP50-95={row['mAP50-95']:.4f}"
        )
    print(f"图表输出目录: {save_dir}")
    print(f"混淆矩阵: {save_dir / 'confusion_matrix.png'}")
    print(f"PR 曲线: {save_dir / 'PR_curve.png'}")
    return result


if __name__ == "__main__":
    main()
