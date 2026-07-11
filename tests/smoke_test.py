"""项目轻量自检。

运行:
    python tests/smoke_test.py
"""

from __future__ import annotations

import csv
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from losses import WiseIoU  # noqa: E402
from models.modules import BiFPN, C2fDCN, EMA  # noqa: E402
from roadmark_experiments.annotation import (  # noqa: E402
    annotation_status,
    export_reviewed_dataset,
    ingest_collected_images,
    select_annotation_candidates,
)
from roadmark_experiments.data_selection import SelectionOptions, prepare_dataset_from_raw  # noqa: E402
from roadmark_experiments.dataset_audit import audit_dataset  # noqa: E402
from roadmark_experiments.plan import all_experiment_ids, get_experiment, runs_for_experiment  # noqa: E402
from roadmark_experiments.zip_import import ZipImportOptions, import_rdd_txt_zips  # noqa: E402


def main() -> None:
    ema = EMA(32)
    x = torch.randn(2, 32, 64, 64)
    y = ema(x)
    assert y.shape == x.shape, y.shape

    bifpn = BiFPN([32, 64, 128], 64, num_layers=1)
    p3 = torch.randn(2, 32, 80, 80)
    p4 = torch.randn(2, 64, 40, 40)
    p5 = torch.randn(2, 128, 20, 20)
    out = bifpn([p3, p4, p5])
    assert [tuple(t.shape) for t in out] == [(2, 64, 80, 80), (2, 64, 40, 40), (2, 64, 20, 20)]

    c2f_dcn = C2fDCN(16, 32, n=1)
    dcn_out = c2f_dcn(torch.randn(1, 16, 32, 32))
    assert dcn_out.shape == (1, 32, 32, 32), dcn_out.shape

    loss_fn = WiseIoU()
    pred = torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 20.0, 20.0]])
    target = torch.tensor([[1.0, 1.0, 11.0, 11.0], [6.0, 6.0, 18.0, 18.0]])
    loss = loss_fn(pred, target)
    assert torch.isfinite(loss), loss

    assert get_experiment("EXP-07").title == "完整模型主对比"
    assert "EXP-00" in all_experiment_ids()
    assert any(run.variant == "ema" for run in runs_for_experiment("EXP-03"))

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_dir = root / "data"
        dataset = root / "dataset"
        (data_dir).mkdir()
        (dataset / "images" / "train").mkdir(parents=True)
        (dataset / "images" / "val").mkdir(parents=True)
        (dataset / "labels" / "train").mkdir(parents=True)
        (dataset / "labels" / "val").mkdir(parents=True)
        (dataset / "images" / "train" / "a.jpg").write_bytes(b"")
        (dataset / "images" / "val" / "b.jpg").write_bytes(b"")
        (dataset / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.2 0.1\n", encoding="utf-8")
        yaml_path = data_dir / "road_mark.yaml"
        yaml_path.write_text(
            "\n".join(
                [
                    "path: ../dataset",
                    "train: images/train",
                    "val: images/val",
                    "nc: 2",
                    "names:",
                    "  - lane_line",
                    "  - stop_line",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        outputs = audit_dataset(yaml_path, root / "runs" / "experiments")
        assert outputs["stats"].exists()
        assert outputs["distribution"].exists()
        assert '"val": [' in outputs["missing"].read_text(encoding="utf-8")

        raw = root / "raw_pool"
        (raw / "images").mkdir(parents=True)
        (raw / "labels").mkdir(parents=True)
        for index, class_id in enumerate([0, 0, 1, 1], start=1):
            image = np.full((24, 32, 3), 50 + index, dtype=np.uint8)
            image_path = raw / "images" / f"sample_{index}.jpg"
            cv2.imwrite(str(image_path), image)
            (raw / "labels" / f"sample_{index}.txt").write_text(
                f"{class_id} 0.5 0.5 0.2 0.2\n", encoding="utf-8"
            )
        cv2.imwrite(str(raw / "images" / "bad.jpg"), np.zeros((20, 20, 3), dtype=np.uint8))

        prepared = prepare_dataset_from_raw(
            raw_root=raw,
            output_root=root / "prepared",
            source_data_yaml=yaml_path,
            report_dir=root / "selection_reports",
            options=SelectionOptions(train_ratio=0.5, seed=1, force=True),
        )
        assert prepared.selected_count == 4
        assert prepared.rejected_count == 1
        assert prepared.data_yaml.exists()
        assert (root / "prepared" / "images" / "train").exists()
        assert (root / "prepared" / "labels" / "val").exists()

        zip_root = root / "zip_raw" / "georgia"
        zip_root.mkdir(parents=True)
        zip_path = zip_root / "georgia_txt.zip"
        image = np.full((64, 64, 3), 120, dtype=np.uint8)
        image_path = root / "sample.jpg"
        cv2.imwrite(str(image_path), image)
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr("data.yaml", "train: train/images\nval: valid/images\nnc: 2\nnames: ['D00', 'D10']\n")
            archive.write(image_path, "train/images/sample.jpg")
            archive.writestr("train/labelTxt/sample.txt", "10 20 30 20 30 40 10 40 D10 0\n")
            archive.write(image_path, "valid/images/sample_val.jpg")
            archive.writestr("valid/labels/sample_val.txt", "0 0.5 0.5 0.25 0.25\n")
        imported = import_rdd_txt_zips(
            raw_root=root / "zip_raw",
            output_root=root / "zip_dataset",
            data_yaml=root / "zip_data" / "road_mark.yaml",
            report_dir=root / "zip_reports",
            options=ZipImportOptions(force=True),
            class_names=["D00", "D10"],
        )
        assert imported.imported_images == 2
        converted_label = root / "zip_dataset" / "labels" / "train" / "georgia" / "sample.txt"
        assert converted_label.read_text(encoding="utf-8").startswith("1 0.312500 0.468750 0.312500 0.312500")

        annotation_source = root / "annotation_source" / "images" / "train"
        annotation_source.mkdir(parents=True)
        for index in range(4):
            candidate_image = np.zeros((96, 128, 3), dtype=np.uint8)
            cv2.line(candidate_image, (20 + index, 95), (55 + index, 35), (255, 255, 255), 4)
            cv2.imwrite(str(annotation_source / f"road_{index}.jpg"), candidate_image)
        annotation_workspace = root / "annotations"
        selection = select_annotation_candidates(
            annotation_source.parent.parent,
            annotation_workspace,
            max_images=4,
            random_fraction=0.5,
            seed=1,
        )
        with selection.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            annotation_rows = list(csv.DictReader(handle))
        assert len(annotation_rows) == 4

        collected = root / "collected_batch"
        collected.mkdir()
        (collected / "duplicate.jpg").write_bytes((annotation_source / "road_0.jpg").read_bytes())
        cv2.imwrite(str(collected / "new.jpg"), np.full((96, 128, 3), 100, dtype=np.uint8))
        cv2.imwrite(str(collected / "too_small.jpg"), np.full((20, 20, 3), 100, dtype=np.uint8))
        ingested = ingest_collected_images(
            collected,
            annotation_workspace,
            source_batch="batch_001",
            scene="urban",
            weather="sunny",
            time_of_day="day",
            min_width=64,
            min_height=64,
        )
        assert ingested.imported_images == 1
        assert ingested.duplicate_images == 1
        assert ingested.rejected_images == 1
        assert annotation_status(annotation_workspace)["total"] == 5

        with selection.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            annotation_rows = list(csv.DictReader(handle))
        annotation_rows = annotation_rows[:4]
        for index, row in enumerate(annotation_rows):
            row["status"] = "positive" if index < 2 else "negative"
            row["box_count"] = "1" if index < 2 else "0"
            label_path = annotation_workspace / "labels" / f"{row['candidate_id']}.txt"
            label_path.write_text("0 0.5 0.5 0.2 0.2\n" if index < 2 else "", encoding="utf-8")
        with selection.manifest.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=annotation_rows[0].keys())
            writer.writeheader()
            writer.writerows(annotation_rows)
        exported = export_reviewed_dataset(
            annotation_workspace,
            root / "roadmark_dataset",
            root / "roadmark_data" / "road_mark_missing.yaml",
            train_ratio=0.5,
            seed=1,
        )
        assert exported.train_images > 0 and exported.val_images > 0
        assert exported.train_images + exported.val_images == 4
        exported_yaml = yaml.safe_load(exported.data_yaml.read_text(encoding="utf-8"))
        assert exported_yaml["nc"] == 1 and exported_yaml["names"] == ["road_mark_missing"]

    print("Smoke test passed.")


if __name__ == "__main__":
    main()

