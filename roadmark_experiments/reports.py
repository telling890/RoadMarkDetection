"""Result table and plotting helpers for road-mark experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .plan import EXPERIMENT_PLAN, ModelRun


METRIC_FIELDS = [
    "Model",
    "Variant",
    "Loss",
    "Description",
    "Precision",
    "Recall",
    "mAP50",
    "mAP50-95",
    "FPS",
    "Params",
    "Weights",
    "MetricsPath",
]


def load_metrics(metrics_path: Path) -> dict[str, float]:
    if not metrics_path.exists():
        return {"Precision": 0.0, "Recall": 0.0, "mAP50": 0.0, "mAP50-95": 0.0, "FPS": 0.0}
    raw = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "Precision": float(raw.get("Precision", 0.0)),
        "Recall": float(raw.get("Recall", 0.0)),
        "mAP50": float(raw.get("mAP50", 0.0)),
        "mAP50-95": float(raw.get("mAP50-95", raw.get("mAP50_95", 0.0))),
        "FPS": float(raw.get("FPS", 0.0)),
    }


def model_result_row(run: ModelRun, root: Path, params: int = 0) -> dict[str, Any]:
    weights = root / "runs" / "train" / run.run_name / "weights" / "best.pt"
    metrics_path = root / "runs" / "val" / run.run_name / "metrics.json"
    metrics = load_metrics(metrics_path)
    return {
        "Model": run.name,
        "Variant": run.variant,
        "Loss": run.loss,
        "Description": run.description,
        "Precision": round(metrics["Precision"], 4),
        "Recall": round(metrics["Recall"], 4),
        "mAP50": round(metrics["mAP50"], 4),
        "mAP50-95": round(metrics["mAP50-95"], 4),
        "FPS": round(metrics["FPS"], 2),
        "Params": params,
        "Weights": str(weights),
        "MetricsPath": str(metrics_path),
    }


def write_rows_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_results_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "| Model | Precision | Recall | mAP50 | mAP50-95 | FPS | Params |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['Model']} | {row['Precision']} | {row['Recall']} | {row['mAP50']} | "
            f"{row['mAP50-95']} | {row['FPS']} | {row['Params']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(path: Path) -> None:
    lines = [
        "# 路面标线缺失检测实验清单",
        "",
        "| ID | Phase | 名称 | 目标 | 关键产物 |",
        "|---|---|---|---|---|",
    ]
    for spec in EXPERIMENT_PLAN:
        lines.append(f"| {spec.exp_id} | {spec.phase} | {spec.title} | {spec.objective} | `{spec.artifact}` |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ablation_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    labels = [str(row["Model"]).replace("+", "\n+") for row in rows]
    map50 = [float(row["mAP50"]) for row in rows]
    recall = [float(row["Recall"]) for row in rows]
    precision = [float(row["Precision"]) for row in rows]
    x = range(len(rows))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.6), 4.8))
    if rows:
        ax.bar([i - width for i in x], precision, width, label="Precision")
        ax.bar(list(x), recall, width, label="Recall")
        ax.bar([i + width for i in x], map50, width, label="mAP50")
        ax.set_xticks(list(x), labels, rotation=0)
        ax.set_ylim(0, 1.05)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No experiment results yet", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title("Road-mark missing detection ablation metrics")
    ax.set_ylabel("score")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_complex_template(path: Path) -> None:
    fields = ["Scenario", "Model", "Data", "Weights", "Precision", "Recall", "mAP50", "mAP50-95", "FPS", "Notes"]
    scenarios = ["night", "shadow", "rain", "blur", "worn_missing", "occluded_missing"]
    rows = [
        {
            "Scenario": scenario,
            "Model": "",
            "Data": "",
            "Weights": "",
            "Precision": "",
            "Recall": "",
            "mAP50": "",
            "mAP50-95": "",
            "FPS": "",
            "Notes": "Fill with scenario validation result or pass --scenario-data name=path.yaml.",
        }
        for scenario in scenarios
    ]
    write_rows_csv(path, rows, fields)


def write_realtime_template(path: Path) -> None:
    fields = ["Source", "InputType", "Weights", "OutputDir", "FPS", "AvgLatencyMs", "Notes"]
    write_rows_csv(
        path,
        [
            {
                "Source": "",
                "InputType": "image/video/camera",
                "Weights": "",
                "OutputDir": "",
                "FPS": "",
                "AvgLatencyMs": "",
                "Notes": "Run EXP-08 with --source to populate road-mark missing detection outputs.",
            }
        ],
        fields,
    )
