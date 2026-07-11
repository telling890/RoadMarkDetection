"""End-to-end integrity checks for the road-mark missing experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.register_ultralytics import patch_wise_iou_loss, register_custom_modules  # noqa: E402
from roadmark_experiments.dataset_audit import iter_images, load_dataset_config  # noqa: E402
from roadmark_experiments.roadmark_missing import ROADMARK_MISSING_CLASSES, semantic_warning  # noqa: E402
from roadmark_experiments.training_profiles import get_training_profile  # noqa: E402


MODEL_YAMLS = (
    ROOT / "models" / "yolo26_roadmark_ema.yaml",
    ROOT / "models" / "yolo26_roadmark_ema_bifpn.yaml",
    ROOT / "models" / "yolo26_roadmark_full.yaml",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查路面标线缺失实验的完整运行链路")
    parser.add_argument("--require-gpu", action="store_true", help="没有 CUDA 时直接失败")
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--data", default="data/road_mark.yaml", help="要检查的数据集 YAML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    data_path = data_path if data_path.is_absolute() else ROOT / data_path
    data = load_dataset_config(data_path)
    train_count = len(iter_images(data.train))
    val_count = len(iter_images(data.val))
    assert train_count > 0 and val_count > 0, "train/val 数据集不能为空"
    supported_protocols = (ROADMARK_MISSING_CLASSES, ["road_mark_missing"])
    assert data.names in supported_protocols, "数据 YAML 必须使用单类正式协议或旧版 10 类兼容协议"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.require_gpu and device == "cpu":
        raise RuntimeError("完整性测试要求 GPU，但当前 PyTorch 未检测到 CUDA。")

    register_custom_modules()
    model_rows: list[str] = []
    for model_yaml in MODEL_YAMLS:
        payload = yaml.safe_load(model_yaml.read_text(encoding="utf-8")) or {}
        assert int(payload.get("nc", -1)) == len(ROADMARK_MISSING_CLASSES), f"{model_yaml.name} 的基础类别数异常"
        yolo = YOLO(str(model_yaml))
        model = yolo.model.to(device).eval()
        sample = torch.zeros(1, 3, args.img_size, args.img_size, device=device)
        with torch.no_grad():
            output = model(sample)
        assert output is not None
        params = sum(parameter.numel() for parameter in model.parameters())
        model_rows.append(f"{model_yaml.name}: {params} params")
        del sample, output, model, yolo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    patch_wise_iou_loss()
    import ultralytics.utils.loss as yolo_loss

    assert getattr(yolo_loss.BboxLoss.forward, "_roadmark_wise_iou", False), "Wise-IoU patch 未生效"
    profile = get_training_profile("accuracy")
    assert profile.img_size >= 768 and profile.cos_lr and profile.multi_scale > 0

    print(f"device: {device}")
    print(f"dataset: {data.yaml_path} | train={train_count}, val={val_count}, classes={len(data.names)}")
    for row in model_rows:
        print(row)
    warning = semantic_warning(data.yaml_path)
    if warning:
        print(f"semantic warning: {warning}")
    print("Integrity test passed.")


if __name__ == "__main__":
    main()
