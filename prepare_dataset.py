"""Select raw road-mark data and prepare the YOLO train/val dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from roadmark_experiments.data_selection import SelectionOptions, prepare_dataset_from_raw
from roadmark_experiments.zip_import import (
    ZipImportOptions,
    discover_txt_zips,
    import_extracted_roboflow_dirs,
    import_rdd_txt_zips,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="挑选原始路面标线缺失数据，并生成 YOLO train/val 数据集")
    parser.add_argument("--raw-data", required=True, help="原始数据目录，支持 images/labels 结构或图片与标签同目录")
    parser.add_argument("--data", default="data/road_mark.yaml", help="类别协议 YAML")
    parser.add_argument("--prepared-data", default="dataset", help="筛选后 YOLO 数据集输出目录")
    parser.add_argument("--outputs", default="runs/experiments", help="挑选报告输出目录")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=None, help="最多挑选多少张有效图片")
    parser.add_argument("--max-images-per-class", type=int, default=None, help="每个类别最多挑选多少张包含该类的图片")
    parser.add_argument("--min-width", type=int, default=1)
    parser.add_argument("--min-height", type=int, default=1)
    parser.add_argument("--include-empty-labels", action="store_true", help="保留无目标空标签图片")
    parser.add_argument("--force-selection", action="store_true", help="允许清空 prepared-data/images 与 labels 后重新生成")
    parser.add_argument("--import-zips", action="store_true", help="从 raw-data 下的 Roboflow *_txt.zip 导入数据")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_data).resolve()
    output_root = Path(args.prepared_data).resolve()
    report_dir = Path(args.outputs).resolve()
    data_yaml = Path(args.data).resolve()
    if args.import_zips or discover_txt_zips(raw_root) or (raw_root / "data.yaml").exists() or any(raw_root.rglob("data.yaml")):
        importer = import_rdd_txt_zips if discover_txt_zips(raw_root) else import_extracted_roboflow_dirs
        result = importer(
            raw_root=raw_root,
            output_root=output_root,
            data_yaml=data_yaml,
            report_dir=report_dir,
            options=ZipImportOptions(include_empty_labels=args.include_empty_labels, force=args.force_selection),
        )
        print("zip 数据导入完成:")
        print(f"- imported images: {result.imported_images}")
        print(f"- imported boxes: {result.imported_boxes}")
        print(f"- rejected images: {result.rejected_images}")
        print(f"- data yaml: {result.data_yaml}")
        print(f"- manifest csv: {result.manifest_csv}")
        print(f"- summary csv: {result.summary_csv}")
        print(f"- rejected csv: {result.rejected_csv}")
        return

    options = SelectionOptions(
        train_ratio=args.train_ratio,
        seed=args.seed,
        max_images=args.max_images,
        max_images_per_class=args.max_images_per_class,
        min_width=args.min_width,
        min_height=args.min_height,
        include_empty_labels=args.include_empty_labels,
        force=args.force_selection,
    )
    result = prepare_dataset_from_raw(
        raw_root=raw_root,
        output_root=output_root,
        source_data_yaml=data_yaml,
        report_dir=report_dir,
        options=options,
    )
    print("数据挑选与处理完成:")
    print(f"- selected samples: {result.selected_count}")
    print(f"- rejected samples: {result.rejected_count}")
    print(f"- data yaml: {result.data_yaml}")
    print(f"- selected csv: {result.selected_csv}")
    print(f"- rejected csv: {result.rejected_csv}")
    print(f"- summary: {result.summary_md}")


if __name__ == "__main__":
    main()
