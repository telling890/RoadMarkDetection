"""Launch LabelImg with the project-required YOLO single-class settings."""

from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="以 YOLO 单类模式启动 LabelImg")
    parser.add_argument("image_dir")
    parser.add_argument("class_file")
    parser.add_argument("save_dir")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from labelImg.labelImg import get_main_app
        from libs.constants import FORMAT_YOLO
    except ImportError as exc:
        raise RuntimeError("未安装 LabelImg。请执行: pip install labelImg==1.8.6 lxml") from exc

    labelimg_argv = [sys.argv[0], args.image_dir, args.class_file, args.save_dir]
    app, window = get_main_app(labelimg_argv)
    window.set_format(FORMAT_YOLO)
    window.auto_saving.setChecked(True)
    window.single_class_mode.setChecked(True)
    return int(app.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
