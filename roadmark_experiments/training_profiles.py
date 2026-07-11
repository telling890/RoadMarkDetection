"""Reproducible training profiles for road-mark missing detection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingProfile:
    img_size: int
    batch: int
    optimizer: str
    lr0: float
    lrf: float
    weight_decay: float
    patience: int
    cos_lr: bool
    close_mosaic: int
    multi_scale: float
    mosaic: float
    mixup: float
    degrees: float
    translate: float
    scale: float
    shear: float
    perspective: float
    hsv_h: float
    hsv_s: float
    hsv_v: float
    fliplr: float
    warmup_epochs: float
    box: float
    cls: float
    dfl: float


TRAINING_PROFILES: dict[str, TrainingProfile] = {
    "standard": TrainingProfile(
        img_size=640,
        batch=16,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        patience=100,
        cos_lr=False,
        close_mosaic=10,
        multi_scale=0.0,
        mosaic=1.0,
        mixup=0.0,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        warmup_epochs=3.0,
        box=7.5,
        cls=0.5,
        dfl=1.5,
    ),
    "accuracy": TrainingProfile(
        img_size=768,
        batch=8,
        optimizer="AdamW",
        lr0=0.0008,
        lrf=0.01,
        weight_decay=0.0005,
        patience=60,
        cos_lr=True,
        close_mosaic=20,
        multi_scale=0.20,
        mosaic=0.80,
        mixup=0.05,
        degrees=2.0,
        translate=0.10,
        scale=0.40,
        shear=1.0,
        perspective=0.0005,
        hsv_h=0.01,
        hsv_s=0.55,
        hsv_v=0.35,
        fliplr=0.5,
        warmup_epochs=5.0,
        box=8.0,
        cls=0.7,
        dfl=1.5,
    ),
}


def get_training_profile(name: str) -> TrainingProfile:
    try:
        return TRAINING_PROFILES[name]
    except KeyError as exc:
        available = ", ".join(sorted(TRAINING_PROFILES))
        raise ValueError(f"未知训练配置: {name}。可选: {available}") from exc
