"""路面标线缺失检测实验调度脚本。

该入口按照 README 中的 EXP-00 到 EXP-09 执行实验：
- EXP-00: 数据集与标签审计；
- EXP-01/03/04/05/06/07: 训练、验证、消融汇总；
- EXP-02: 复杂环境增强/场景验证结果表；
- EXP-08: 图片/视频/摄像头推理；
- EXP-09: PyQt5 Demo 检查清单或启动 GUI。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from ultralytics import YOLO

from models.register_ultralytics import register_custom_modules
from roadmark_experiments.data_selection import SelectionOptions, prepare_dataset_from_raw
from roadmark_experiments.dataset_audit import audit_dataset, iter_images, load_dataset_config
from roadmark_experiments.plan import EXPERIMENT_PLAN, ModelRun, all_experiment_ids, get_experiment, runs_for_experiment
from roadmark_experiments.training_profiles import TRAINING_PROFILES, get_training_profile
from roadmark_experiments.reports import (
    METRIC_FIELDS,
    load_metrics,
    model_result_row,
    write_ablation_plot,
    write_complex_template,
    write_manifest,
    write_realtime_template,
    write_results_markdown,
    write_rows_csv,
)
from roadmark_experiments.roadmark_missing import semantic_warning
from roadmark_experiments.zip_import import (
    ZipImportOptions,
    discover_txt_zips,
    import_extracted_roboflow_dirs,
    import_rdd_txt_zips,
)


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按 README 执行路面标线缺失检测实验")
    parser.add_argument("--exp", default="EXP-07", help="实验编号、逗号列表或 all。默认 EXP-07 完整模型主对比。")
    parser.add_argument("--data", default="data/road_mark.yaml")
    parser.add_argument("--raw-data", default=None, help="原始 YOLO 数据池目录；传入后 EXP-00 会先挑选/清洗数据。")
    parser.add_argument("--prepared-data", default="dataset", help="挑选后生成的 YOLO 数据集目录。")
    parser.add_argument("--force-selection", action="store_true", help="允许清空 prepared-data/images 与 labels 后重新生成。")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="数据挑选后的 train/val 划分比例。")
    parser.add_argument("--selection-seed", type=int, default=42, help="数据挑选和划分随机种子。")
    parser.add_argument("--max-images", type=int, default=None, help="最多挑选多少张有效图片。")
    parser.add_argument("--max-images-per-class", type=int, default=None, help="每个类别最多挑选多少张包含该类的图片。")
    parser.add_argument("--min-width", type=int, default=1, help="筛选图片的最小宽度。")
    parser.add_argument("--min-height", type=int, default=1, help="筛选图片的最小高度。")
    parser.add_argument("--include-empty-labels", action="store_true", help="保留无目标空标签图片。")
    parser.add_argument("--profile", choices=TRAINING_PROFILES, default="accuracy", help="训练超参数配置。")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=None, help="覆盖 profile 的 batch。")
    parser.add_argument("--img-size", "--imgsz", dest="img_size", type=int, default=None, help="覆盖 profile 的输入尺寸。")
    parser.add_argument("--lr", type=float, default=None, help="覆盖 profile 的初始学习率。")
    parser.add_argument("--patience", type=int, default=None, help="覆盖 profile 的 early stopping patience。")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0", help="默认使用第 1 张 GPU；无 CUDA 时手动传 cpu。")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--outputs", default="runs/experiments")
    parser.add_argument("--skip-train", action="store_true", help="跳过训练，仅验证/汇总已有权重")
    parser.add_argument("--skip-val", action="store_true", help="跳过验证，仅训练/汇总已有 metrics")
    parser.add_argument("--tta-val", action="store_true", help="验证时启用 TTA；精度优先但速度更慢")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要执行的命令")
    parser.add_argument("--offline-aug-repeats", type=int, default=1, help="EXP-02 使用的离线增强重复次数")
    parser.add_argument(
        "--scenario-data",
        action="append",
        default=[],
        metavar="NAME=YAML",
        help="EXP-02 复杂环境验证数据，如 night=data/night.yaml。可重复传入。",
    )
    parser.add_argument("--source", default=None, help="EXP-08 推理输入：图片、视频路径或摄像头编号")
    parser.add_argument("--weights", default="runs/train/best.pt", help="EXP-08/EXP-09 默认权重")
    parser.add_argument("--show", action="store_true", help="EXP-08 摄像头/视频推理时显示窗口")
    parser.add_argument("--launch-gui", action="store_true", help="EXP-09 启动 PyQt5 Demo")
    parser.add_argument("--list", action="store_true", help="打印实验拆分清单")
    return parser.parse_args()


def print_experiment_plan() -> None:
    print("| ID | Phase | 名称 | 目标 | 关键产物 |")
    print("|---|---|---|---|---|")
    for spec in EXPERIMENT_PLAN:
        print(f"| {spec.exp_id} | {spec.phase} | {spec.title} | {spec.objective} | {spec.artifact} |")


def parse_exp_ids(raw: str) -> list[str]:
    if raw == "all":
        return all_experiment_ids()
    exp_ids = [part.strip() for part in raw.split(",") if part.strip()]
    for exp_id in exp_ids:
        get_experiment(exp_id)
    return exp_ids


def run_command(cmd: list[str], dry_run: bool = False) -> None:
    print("执行:", " ".join(cmd))
    if not dry_run:
        try:
            subprocess.run(cmd, cwd=ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"命令执行失败，退出码 {exc.returncode}: {' '.join(cmd)}") from exc


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def count_params(weights: Path) -> int:
    if not weights.exists():
        return 0
    register_custom_modules()
    model = YOLO(str(weights))
    return sum(param.numel() for param in model.model.parameters())


def train_command(run: ModelRun, args: argparse.Namespace) -> list[str]:
    profile = get_training_profile(args.profile)
    batch = args.batch if args.batch is not None else profile.batch
    img_size = args.img_size if args.img_size is not None else profile.img_size
    command = [
        sys.executable,
        "train.py",
        "--variant",
        run.variant,
        "--loss",
        run.loss,
        "--data",
        args.data,
        "--profile",
        args.profile,
        "--epochs",
        str(args.epochs),
        "--batch",
        str(batch),
        "--img-size",
        str(img_size),
        "--device",
        args.device,
        "--workers",
        str(args.workers),
        "--seed",
        str(args.seed),
        "--name",
        run.run_name,
    ]
    if args.lr is not None:
        command.extend(["--lr", str(args.lr)])
    if args.patience is not None:
        command.extend(["--patience", str(args.patience)])
    return command


def val_command(run: ModelRun, data: str, args: argparse.Namespace, name: str | None = None) -> list[str]:
    profile = get_training_profile(args.profile)
    batch = args.batch if args.batch is not None else profile.batch
    img_size = args.img_size if args.img_size is not None else profile.img_size
    weights = ROOT / "runs" / "train" / run.run_name / "weights" / "best.pt"
    command = [
        sys.executable,
        "val.py",
        "--weights",
        str(weights),
        "--data",
        data,
        "--img-size",
        str(img_size),
        "--batch",
        str(batch),
        "--device",
        args.device,
        "--name",
        name or run.run_name,
    ]
    if args.tta_val:
        command.append("--tta")
    return command


def dataset_image_counts(data_yaml: str | Path) -> tuple[int, int]:
    config = load_dataset_config(resolve_path(data_yaml))
    return len(iter_images(config.train)), len(iter_images(config.val))


def ensure_dataset_ready_for_model_runs(args: argparse.Namespace, output_dir: Path) -> None:
    if args.dry_run or args.skip_train:
        return

    train_count, val_count = dataset_image_counts(args.data)
    if train_count > 0 and val_count > 0:
        return

    if args.raw_data:
        print("当前训练/验证集为空，先根据 --raw-data 执行 EXP-00 数据挑选与处理。")
        run_exp00(args, output_dir)
        train_count, val_count = dataset_image_counts(args.data)
        if train_count > 0 and val_count > 0:
            return

    raise SystemExit(
        "\n数据集为空，不能启动训练。\n"
        "请先把数据放入 dataset/images/train 与 dataset/images/val，或先执行数据挑选：\n\n"
        "  python experiment.py --exp EXP-00 --raw-data <原始数据目录> --prepared-data dataset --force-selection\n\n"
        "然后再运行：\n\n"
        "  python experiment.py --exp EXP-01 --device 0\n\n"
        "如果只想查看将要执行的命令，可以加 --dry-run。"
    )


def run_model_experiments(runs: list[ModelRun], args: argparse.Namespace, output_dir: Path) -> list[dict[str, object]]:
    ensure_dataset_ready_for_model_runs(args, output_dir)
    warning = semantic_warning(resolve_path(args.data))
    if warning and not args.dry_run:
        print(f"警告: {warning}")
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for run in runs:
        if run.run_name in seen:
            continue
        seen.add(run.run_name)
        if not args.skip_train:
            run_command(train_command(run, args), dry_run=args.dry_run)
        if not args.skip_val:
            run_command(val_command(run, args.data, args), dry_run=args.dry_run)
        if args.dry_run:
            continue
        params = count_params(ROOT / "runs" / "train" / run.run_name / "weights" / "best.pt") if not args.dry_run else 0
        rows.append(model_result_row(run, ROOT, params=params))

    if args.dry_run:
        print("dry-run: 未写入实验表、Markdown 或消融图。")
        return rows

    csv_path = output_dir / "experiment_results.csv"
    md_path = output_dir / "experiment_results.md"
    plot_path = output_dir / "figures" / "figure4_ablation_metrics.png"
    write_rows_csv(csv_path, rows, METRIC_FIELDS)
    write_results_markdown(md_path, rows)
    write_ablation_plot(plot_path, rows)
    print(f"实验表已保存: {csv_path}")
    print(f"Markdown 表已保存: {md_path}")
    print(f"消融图已保存: {plot_path}")
    return rows


def run_exp00(args: argparse.Namespace, output_dir: Path) -> None:
    data_for_audit = resolve_path(args.data)
    if args.dry_run:
        if args.raw_data:
            print(f"dry-run: 将从 {resolve_path(args.raw_data)} 挑选/导入数据到 {resolve_path(args.prepared_data)}。")
        print(f"dry-run: 将审计 {data_for_audit}，未写入审计报告。")
        return
    if args.raw_data:
        raw_root = resolve_path(args.raw_data)
        if discover_txt_zips(raw_root) or (raw_root / "data.yaml").exists() or any(raw_root.rglob("data.yaml")):
            importer = import_rdd_txt_zips if discover_txt_zips(raw_root) else import_extracted_roboflow_dirs
            imported = importer(
                raw_root=raw_root,
                output_root=resolve_path(args.prepared_data),
                data_yaml=resolve_path(args.data),
                report_dir=output_dir,
                options=ZipImportOptions(include_empty_labels=args.include_empty_labels, force=args.force_selection),
            )
            args.data = str(imported.data_yaml)
            data_for_audit = imported.data_yaml
            print(f"imported images: {imported.imported_images}")
            print(f"imported boxes: {imported.imported_boxes}")
            print(f"rejected images: {imported.rejected_images}")
            print(f"prepared data yaml: {imported.data_yaml}")
            print(f"manifest csv: {imported.manifest_csv}")
            print(f"zip summary csv: {imported.summary_csv}")
            print(f"zip rejected csv: {imported.rejected_csv}")
        else:
            options = SelectionOptions(
                train_ratio=args.train_ratio,
                seed=args.selection_seed,
                max_images=args.max_images,
                max_images_per_class=args.max_images_per_class,
                min_width=args.min_width,
                min_height=args.min_height,
                include_empty_labels=args.include_empty_labels,
                force=args.force_selection,
            )
            prepared = prepare_dataset_from_raw(
                raw_root=raw_root,
                output_root=resolve_path(args.prepared_data),
                source_data_yaml=resolve_path(args.data),
                report_dir=output_dir,
                options=options,
            )
            args.data = str(prepared.data_yaml)
            data_for_audit = prepared.data_yaml
            print(f"selected samples: {prepared.selected_count}")
            print(f"rejected samples: {prepared.rejected_count}")
            print(f"prepared data yaml: {prepared.data_yaml}")
            print(f"selection summary: {prepared.summary_md}")
            print(f"selected csv: {prepared.selected_csv}")
            print(f"rejected csv: {prepared.rejected_csv}")

    written = audit_dataset(data_for_audit, output_dir)
    for label, path in written.items():
        print(f"{label}: {path}")


def parse_scenario_data(items: list[str]) -> list[tuple[str, str]]:
    scenarios: list[tuple[str, str]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"--scenario-data 必须使用 NAME=YAML 格式: {item}")
        name, data = item.split("=", 1)
        scenarios.append((name.strip(), data.strip()))
    return scenarios


def run_exp02(args: argparse.Namespace, output_dir: Path) -> None:
    complex_csv = output_dir / "complex_env_results.csv"
    scenarios = parse_scenario_data(args.scenario_data)
    full_run = ModelRun(
        "YOLO26+C2f-DCN+EMA+BiFPN+WiseIoU",
        "full",
        "wise_iou",
        ("EXP-02",),
        "完整模型 + 复杂环境增强",
    )

    if not args.skip_train:
        cmd = train_command(full_run, args)
        cmd.extend(["--offline-aug-repeats", str(args.offline_aug_repeats)])
        run_command(cmd, dry_run=args.dry_run)

    if args.dry_run:
        if scenarios and not args.skip_val:
            for scenario, data_yaml in scenarios:
                name = f"{full_run.run_name}_{scenario}"
                run_command(val_command(full_run, data_yaml, args, name=name), dry_run=True)
        print("dry-run: 未写入复杂环境结果表。")
        return

    rows: list[dict[str, object]] = []
    if scenarios and not args.skip_val:
        for scenario, data_yaml in scenarios:
            name = f"{full_run.run_name}_{scenario}"
            run_command(val_command(full_run, data_yaml, args, name=name), dry_run=args.dry_run)
            metrics_path = ROOT / "runs" / "val" / name / "metrics.json"
            metrics = load_metrics(metrics_path)
            rows.append(
                {
                    "Scenario": scenario,
                    "Model": full_run.name,
                    "Data": data_yaml,
                    "Weights": str(ROOT / "runs" / "train" / full_run.run_name / "weights" / "best.pt"),
                    "Precision": round(metrics["Precision"], 4),
                    "Recall": round(metrics["Recall"], 4),
                    "mAP50": round(metrics["mAP50"], 4),
                    "mAP50-95": round(metrics["mAP50-95"], 4),
                    "FPS": round(metrics["FPS"], 2),
                    "Notes": "scenario validation",
                }
            )

    if rows:
        write_rows_csv(
            complex_csv,
            rows,
            ["Scenario", "Model", "Data", "Weights", "Precision", "Recall", "mAP50", "mAP50-95", "FPS", "Notes"],
        )
    else:
        write_complex_template(complex_csv)
    print(f"复杂环境结果表已保存: {complex_csv}")


def run_exp08(args: argparse.Namespace, output_dir: Path) -> None:
    realtime_csv = output_dir / "realtime_results.csv"
    if not args.source:
        if args.dry_run:
            print("dry-run: 未提供 --source，未写入实时推理结果模板。")
            return
        write_realtime_template(realtime_csv)
        print(f"未提供 --source，已生成实时推理结果模板: {realtime_csv}")
        return

    profile = get_training_profile(args.profile)
    img_size = args.img_size if args.img_size is not None else profile.img_size
    cmd = [
        sys.executable,
        "detect.py",
        "--weights",
        args.weights,
        "--source",
        args.source,
        "--img-size",
        str(img_size),
        "--device",
        args.device,
        "--project",
        "runs/detect",
        "--name",
        "roadmark_missing_detect",
    ]
    if args.show:
        cmd.append("--show")
    run_command(cmd, dry_run=args.dry_run)
    if args.dry_run:
        print("dry-run: 未写入实时推理结果表。")
        return
    write_rows_csv(
        realtime_csv,
        [
            {
                "Source": args.source,
                "InputType": "camera" if str(args.source).isdigit() else "file",
                "Weights": args.weights,
                "OutputDir": str(ROOT / "runs" / "detect" / "roadmark_missing_detect"),
                "FPS": "",
                "AvgLatencyMs": "",
                "Notes": "FPS is printed by detect.py during inference.",
            }
        ],
        ["Source", "InputType", "Weights", "OutputDir", "FPS", "AvgLatencyMs", "Notes"],
    )
    print(f"实时推理结果表已保存: {realtime_csv}")


def run_exp09(args: argparse.Namespace, output_dir: Path) -> None:
    checklist = output_dir / "demo_checklist.md"
    lines = [
        "# EXP-09 路面标线缺失检测 PyQt5 Demo 检查清单",
        "",
        "- [ ] `python app.py` 能正常启动。",
        "- [ ] 无训练权重时可回退到 `yolo26n.pt` 演示流程。",
        "- [ ] 有训练权重时优先加载 `runs/train/best.pt`。",
        "- [ ] 图片、视频、摄像头入口可打开。",
        "- [ ] 表格显示类别、置信度、xyxy 坐标和 FPS。",
        "",
        f"默认权重: `{args.weights}`",
    ]
    if args.dry_run:
        print(f"dry-run: 未写入 Demo 检查清单: {checklist}")
        if args.launch_gui:
            run_command([sys.executable, "app.py", "--weights", args.weights], dry_run=True)
        return
    checklist.parent.mkdir(parents=True, exist_ok=True)
    checklist.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Demo 检查清单已保存: {checklist}")
    if args.launch_gui:
        run_command([sys.executable, "app.py", "--weights", args.weights], dry_run=args.dry_run)


def run_experiment(exp_id: str, args: argparse.Namespace, output_dir: Path) -> None:
    print(f"\n== {exp_id} {get_experiment(exp_id).title} ==")
    if exp_id == "EXP-00":
        run_exp00(args, output_dir)
    elif exp_id == "EXP-02":
        run_exp02(args, output_dir)
    elif exp_id == "EXP-08":
        run_exp08(args, output_dir)
    elif exp_id == "EXP-09":
        run_exp09(args, output_dir)
    else:
        runs = runs_for_experiment(exp_id)
        if not runs:
            print(f"{exp_id} 暂无可执行模型组。")
            return
        run_model_experiments(runs, args, output_dir)


def main() -> None:
    args = parse_args()
    if args.list:
        print_experiment_plan()
        return

    output_dir = resolve_path(args.outputs)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_manifest(output_dir / "experiment_manifest.md")

    for exp_id in parse_exp_ids(args.exp):
        run_experiment(exp_id, args, output_dir)

    if args.dry_run:
        print("\ndry-run 完成，未写入任何实验产物。")
    else:
        print(f"\n实验输出目录: {output_dir}")


if __name__ == "__main__":
    main()


