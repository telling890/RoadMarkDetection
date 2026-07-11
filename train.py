"""YOLO26 路面标线缺失训练脚本。

默认参数：
epochs=200, batch=16, img_size=640, optimizer=AdamW, lr=0.001

示例：
    python train.py --variant full --loss wise_iou
    python train.py --resume
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

from models.register_ultralytics import patch_wise_iou_loss, register_custom_modules
from roadmark_experiments.roadmark_missing import semantic_warning
from roadmark_experiments.training_profiles import TRAINING_PROFILES, get_training_profile
from utils.augmentation import create_augmented_yolo_dataset, iter_images
from utils.yolo_data import write_resolved_data_yaml


ROOT = Path(__file__).resolve().parent


VARIANT_MODELS = {
    "baseline": "yolo26n.pt",
    "ema": "models/yolo26_roadmark_ema.yaml",
    "ema_bifpn": "models/yolo26_roadmark_ema_bifpn.yaml",
    "full": "models/yolo26_roadmark_full.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 YOLO26 路面标线缺失检测模型")
    parser.add_argument("--variant", choices=VARIANT_MODELS.keys(), default="baseline", help="模型实验版本")
    parser.add_argument("--model", default=None, help="自定义模型权重或 YAML，优先级高于 --variant")
    parser.add_argument("--pretrained", default="yolo26n.pt", help="自定义 YAML 使用的预训练权重")
    parser.add_argument("--data", default="data/road_mark.yaml", help="数据集 YAML")
    parser.add_argument("--profile", choices=TRAINING_PROFILES, default="accuracy", help="训练超参数配置")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=None, help="覆盖 profile 的 batch")
    parser.add_argument("--img-size", "--imgsz", dest="img_size", type=int, default=None, help="覆盖 profile 的输入尺寸")
    parser.add_argument("--optimizer", default=None, help="覆盖 profile 的优化器")
    parser.add_argument("--lr", type=float, default=None, help="覆盖 profile 的初始学习率")
    parser.add_argument("--patience", type=int, default=None, help="覆盖 profile 的 early stopping patience")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto、cpu、0、0,1 等")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true", help="从最近一次训练恢复")
    parser.add_argument("--loss", choices=["ciou", "wise_iou"], default="ciou", help="边框回归损失")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default=None)
    parser.add_argument("--offline-aug-repeats", type=int, default=0, help="训练前离线增强 train 集的重复次数")
    parser.add_argument("--fraction", type=float, default=1.0, help="训练集使用比例，完整实验保持 1.0")
    parser.add_argument("--no-val", action="store_true", help="跳过训练期验证，仅用于快速完整性测试")
    parser.add_argument("--allow-empty", action="store_true", help="允许数据集为空，仅用于检查脚本参数")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def select_device(device: str) -> str:
    """自动选择 CUDA/CPU。"""

    if device != "auto":
        if device.lower() != "cpu" and not torch.cuda.is_available():
            raise RuntimeError(
                f"请求使用 GPU --device {device}，但当前 PyTorch 未检测到 CUDA。"
                "请安装 CUDA 版 PyTorch，或临时改用 --device cpu。"
            )
        return device
    return "0" if torch.cuda.is_available() else "cpu"


def ensure_dataset(data_yaml: Path, allow_empty: bool = False) -> bool:
    """检查 YOLO 数据集目录是否存在。"""

    if not data_yaml.exists():
        raise FileNotFoundError(f"未找到数据集配置: {data_yaml}")
    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    dataset_root = (data_yaml.parent / data.get("path", ".")).resolve()
    train_dir = dataset_root / data["train"]
    val_dir = dataset_root / data["val"]
    if not train_dir.exists() or not val_dir.exists():
        if allow_empty:
            print(f"数据集目录尚未准备完整: train={train_dir}, val={val_dir}")
            return False
        raise FileNotFoundError(f"数据集目录不存在: train={train_dir}, val={val_dir}")

    train_images = iter_images(train_dir)
    val_images = iter_images(val_dir)
    if not allow_empty and (not train_images or not val_images):
        raise RuntimeError(
            "数据集为空。请把图片放入 dataset/images/train 与 dataset/images/val，"
            "标签放入 dataset/labels/train 与 dataset/labels/val。"
        )
    if allow_empty and (not train_images or not val_images):
        print(
            "数据集目录已找到，但 train/val 图片为空；"
            "--allow-empty 模式只完成参数检查，不启动训练。"
        )
        return False
    return True


def build_augmented_data_yaml(source_yaml: Path, repeats: int, img_size: int) -> Path:
    """离线增强并生成新的 data yaml。"""

    with source_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    source_root = (source_yaml.parent / data.get("path", ".")).resolve()
    target_root = ROOT / "dataset_aug"
    create_augmented_yolo_dataset(source_root, target_root, repeats=repeats, img_size=img_size)

    aug_yaml = ROOT / "data" / "road_mark_aug.yaml"
    aug_data = dict(data)
    aug_data["path"] = "../dataset_aug"
    with aug_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(aug_data, f, allow_unicode=True, sort_keys=False)
    return aug_yaml


def load_yolo_model(model_path: str, pretrained: str | None = None) -> YOLO:
    """加载 YOLO26 或改进模型。"""

    register_custom_modules()
    model_resolved = resolve_path(model_path) if model_path.endswith((".yaml", ".yml")) else model_path
    model = YOLO(str(model_resolved))

    if str(model_resolved).endswith((".yaml", ".yml")) and pretrained:
        # 自定义结构使用官方 YOLO26 预训练权重进行可匹配层加载。
        model.load(pretrained)
    return model


def main() -> None:
    args = parse_args()
    profile = get_training_profile(args.profile)
    batch = args.batch if args.batch is not None else profile.batch
    img_size = args.img_size if args.img_size is not None else profile.img_size
    optimizer = args.optimizer or profile.optimizer
    lr = args.lr if args.lr is not None else profile.lr0
    patience = args.patience if args.patience is not None else profile.patience
    data_yaml = resolve_path(args.data)
    warning = semantic_warning(data_yaml)
    if warning:
        print(f"警告: {warning}")

    if args.offline_aug_repeats > 0:
        data_yaml = build_augmented_data_yaml(data_yaml, args.offline_aug_repeats, img_size)

    dataset_ready = ensure_dataset(data_yaml, allow_empty=args.allow_empty)
    if not dataset_ready:
        print("参数检查完成。放入真实数据集后，去掉 --allow-empty 即可训练。")
        return
    runtime_data_yaml = write_resolved_data_yaml(data_yaml, ROOT / "runs" / "_runtime_data")

    model_path = args.model or VARIANT_MODELS[args.variant]
    run_name = args.name or f"roadmark_missing_{args.variant}_{args.loss}"
    device = select_device(args.device)

    if args.loss == "wise_iou":
        patch_wise_iou_loss()

    model = load_yolo_model(model_path, args.pretrained)
    print(f"使用设备: {device}")
    print(f"模型: {model_path}")
    print(f"数据集: {runtime_data_yaml}")
    print(f"Loss: {args.loss}")
    print(f"训练配置: {args.profile} | imgsz={img_size} | batch={batch} | optimizer={optimizer} | lr0={lr}")

    results = model.train(
        data=str(runtime_data_yaml),
        epochs=args.epochs,
        batch=batch,
        imgsz=img_size,
        optimizer=optimizer,
        lr0=lr,
        lrf=profile.lrf,
        weight_decay=profile.weight_decay,
        patience=patience,
        cos_lr=profile.cos_lr,
        close_mosaic=profile.close_mosaic,
        multi_scale=profile.multi_scale,
        mosaic=profile.mosaic,
        mixup=profile.mixup,
        degrees=profile.degrees,
        translate=profile.translate,
        scale=profile.scale,
        shear=profile.shear,
        perspective=profile.perspective,
        hsv_h=profile.hsv_h,
        hsv_s=profile.hsv_s,
        hsv_v=profile.hsv_v,
        fliplr=profile.fliplr,
        warmup_epochs=profile.warmup_epochs,
        box=profile.box,
        cls=profile.cls,
        dfl=profile.dfl,
        seed=args.seed,
        deterministic=True,
        amp=True,
        fraction=args.fraction,
        val=not args.no_val,
        device=device,
        workers=args.workers,
        resume=args.resume,
        project=str(resolve_path(args.project)),
        name=run_name,
        exist_ok=True,
        plots=True,
    )

    save_dir = Path(getattr(results, "save_dir", resolve_path(args.project) / run_name))
    best = save_dir / "weights" / "best.pt"
    if best.exists():
        target = resolve_path(args.project) / "best.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, target)
        print(f"best.pt 已保存: {best}")
        print(f"同时复制到: {target}")
    else:
        print(f"未找到 best.pt，请检查训练输出目录: {save_dir}")


if __name__ == "__main__":
    main()
