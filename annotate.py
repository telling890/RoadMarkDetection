"""Manual annotation workflow for a single-class road-mark-missing dataset."""

from __future__ import annotations

import argparse

from roadmark_experiments.annotation import (
    AnnotationSession,
    annotation_status,
    export_reviewed_dataset,
    ingest_collected_images,
    prelabel_candidates,
    select_annotation_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="筛选、标注并导出路面标线缺失 YOLO 数据集")
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("select", help="从现有道路图片中筛选标注候选")
    select.add_argument("--source", default="dataset/images")
    select.add_argument("--workspace", default="annotations/road_mark_missing")
    select.add_argument("--max-images", type=int, default=1000)
    select.add_argument("--random-fraction", type=float, default=0.25)
    select.add_argument("--seed", type=int, default=42)
    select.add_argument("--scan-limit", type=int, default=6000, help="最多评分多少张去重后的原图；0 表示全部")

    ingest = subparsers.add_parser("ingest", help="导入并整理新采集的道路图片")
    ingest.add_argument("--source", required=True)
    ingest.add_argument("--workspace", default="annotations/road_mark_missing")
    ingest.add_argument("--batch", default=None, help="采集批次名称")
    ingest.add_argument("--scene", default="unknown", help="如 urban、rural、highway")
    ingest.add_argument("--weather", default="unknown", help="如 sunny、rain、cloudy")
    ingest.add_argument("--time-of-day", default="unknown", help="如 day、night、dusk")
    ingest.add_argument("--min-width", type=int, default=640)
    ingest.add_argument("--min-height", type=int, default=480)

    review = subparsers.add_parser("review", help="打开鼠标框选工具")
    review.add_argument("--workspace", default="annotations/road_mark_missing")
    review.add_argument("--start", type=int, default=0)

    prelabel = subparsers.add_parser("prelabel", help="使用已有单类模型为待标图片生成预标注")
    prelabel.add_argument("--workspace", default="annotations/road_mark_missing")
    prelabel.add_argument("--weights", required=True)
    prelabel.add_argument("--device", default="0")
    prelabel.add_argument("--conf", type=float, default=0.25)
    prelabel.add_argument("--img-size", type=int, default=768)

    status = subparsers.add_parser("status", help="显示标注进度和采集批次统计")
    status.add_argument("--workspace", default="annotations/road_mark_missing")

    export = subparsers.add_parser("export", help="导出已人工复核的单类 YOLO 数据集")
    export.add_argument("--workspace", default="annotations/road_mark_missing")
    export.add_argument("--output", default="new data1")
    export.add_argument("--data", default="data/road_mark_missing.yaml")
    export.add_argument("--train-ratio", type=float, default=0.8)
    export.add_argument("--seed", type=int, default=42)
    export.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "select":
        result = select_annotation_candidates(
            source_root=args.source,
            workspace=args.workspace,
            max_images=args.max_images,
            random_fraction=args.random_fraction,
            seed=args.seed,
            scan_limit=args.scan_limit or None,
        )
        print(f"候选清单: {result.manifest}")
        print(f"候选图片: {result.candidates}，高标线得分: {result.high_score_candidates}，随机负样本池: {result.random_candidates}")
    elif args.command == "review":
        AnnotationSession(args.workspace, start=args.start).run()
    elif args.command == "ingest":
        result = ingest_collected_images(
            source_root=args.source,
            workspace=args.workspace,
            source_batch=args.batch,
            scene=args.scene,
            weather=args.weather,
            time_of_day=args.time_of_day,
            min_width=args.min_width,
            min_height=args.min_height,
        )
        print(f"标注清单: {result.manifest}")
        print(f"新增={result.imported_images}, 重复跳过={result.duplicate_images}, 质量拒绝={result.rejected_images}")
    elif args.command == "prelabel":
        count = prelabel_candidates(args.workspace, args.weights, args.device, args.conf, args.img_size)
        print(f"已生成预标注: {count} 张。必须运行 review 人工确认后才能导出。")
    elif args.command == "status":
        result = annotation_status(args.workspace)
        print(f"总图片: {result['total']}，已确认框: {result['boxes']}，采集批次: {result['batches']}，采集组: {result['capture_groups']}")
        for name, count in sorted(result["statuses"].items()):
            print(f"- {name}: {count}")
    elif args.command == "export":
        result = export_reviewed_dataset(
            workspace=args.workspace,
            output_root=args.output,
            data_yaml=args.data,
            train_ratio=args.train_ratio,
            seed=args.seed,
            force=args.force,
        )
        print(f"数据配置: {result.data_yaml}")
        print(f"train={result.train_images}, val={result.val_images}, positive={result.positive_images}, negative={result.negative_images}")


if __name__ == "__main__":
    main()
