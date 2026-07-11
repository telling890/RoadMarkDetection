"""YOLO26 路面标线缺失推理脚本。

支持图片、视频、摄像头：
    python detect.py --source test.jpg
    python detect.py --source video.mp4
    python detect.py --source 0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from models.register_ultralytics import register_custom_modules


ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO26 路面标线缺失实时检测")
    parser.add_argument("--weights", default="runs/train/best.pt")
    parser.add_argument("--source", required=True, help="图片、视频路径或摄像头编号，如 0")
    parser.add_argument("--img-size", "--imgsz", dest="img_size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="roadmark_missing_detect")
    parser.add_argument("--show", action="store_true", help="实时显示检测窗口")
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


def load_model(weights: str) -> YOLO:
    register_custom_modules()
    weights_path = resolve_path(weights)
    if weights_path.exists():
        return YOLO(str(weights_path))
    print(f"未找到训练权重 {weights_path}，临时回退到官方 yolo26n.pt。")
    return YOLO("yolo26n.pt")


def print_detections(result, fps: float) -> None:
    names = result.names
    if result.boxes is None or len(result.boxes) == 0:
        print(f"FPS {fps:.2f} | 未检测到路面标线缺失")
        return
    items = []
    for box in result.boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        xyxy = [round(float(v), 2) for v in box.xyxy[0].tolist()]
        items.append(f"{names[cls_id]} {conf:.2f} {xyxy}")
    print(f"FPS {fps:.2f} | " + " | ".join(items))


def predict_frame(model: YOLO, frame, args: argparse.Namespace):
    start = time.perf_counter()
    result = model.predict(
        frame,
        imgsz=args.img_size,
        conf=args.conf,
        iou=args.iou,
        device=select_device(args.device),
        verbose=False,
    )[0]
    elapsed = time.perf_counter() - start
    fps = 1.0 / elapsed if elapsed > 0 else 0.0
    annotated = result.plot()
    return annotated, result, fps


def detect_image(model: YOLO, image_path: Path, args: argparse.Namespace, out_dir: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    annotated, result, fps = predict_frame(model, image, args)
    print_detections(result, fps)
    out_path = out_dir / image_path.name
    cv2.imwrite(str(out_path), annotated)
    print(f"检测结果已保存: {out_path}")


def detect_stream(model: YOLO, source: str, args: argparse.Namespace, out_dir: Path) -> None:
    is_camera = source.isdigit()
    cap = cv2.VideoCapture(int(source) if is_camera else source)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频/摄像头: {source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.img_size
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.img_size
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    fps_in = fps_in if fps_in and fps_in > 1 else 25
    out_path = out_dir / ("camera_result.mp4" if is_camera else f"{Path(source).stem}_result.mp4")
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (width, height))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            annotated, result, fps = predict_frame(model, frame, args)
            annotated = cv2.resize(annotated, (width, height))
            writer.write(annotated)
            print_detections(result, fps)
            if args.show:
                cv2.imshow("RoadMarkMissingDetection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()
    print(f"检测结果视频已保存: {out_path}")


def main() -> None:
    args = parse_args()
    model = load_model(args.weights)
    out_dir = resolve_path(args.project) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    source = args.source
    source_path = resolve_path(source) if not source.isdigit() else None
    if source.isdigit():
        detect_stream(model, source, args, out_dir)
    elif source_path and source_path.suffix.lower() in IMAGE_SUFFIXES:
        detect_image(model, source_path, args, out_dir)
    elif source_path and source_path.suffix.lower() in VIDEO_SUFFIXES:
        detect_stream(model, str(source_path), args, out_dir)
    else:
        raise ValueError(f"不支持的 source: {source}")


if __name__ == "__main__":
    main()
