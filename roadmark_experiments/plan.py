"""Canonical experiment definitions used by experiment.py."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentSpec:
    exp_id: str
    phase: str
    title: str
    objective: str
    artifact: str


@dataclass(frozen=True)
class ModelRun:
    name: str
    variant: str
    loss: str
    exp_ids: tuple[str, ...]
    description: str

    @property
    def run_name(self) -> str:
        return f"exp_{self.variant}_{self.loss}"


EXPERIMENT_PLAN: tuple[ExperimentSpec, ...] = (
    ExperimentSpec("EXP-00", "Phase 0", "数据集与类别协议", "固定数据来源、类别编号和 YOLO 标注格式", "dataset_stats.csv"),
    ExperimentSpec("EXP-01", "Phase 1", "YOLO26 Clean Baseline", "建立标线缺失检测基线", "runs/train/exp_baseline_ciou"),
    ExperimentSpec("EXP-02", "Phase 1", "复杂环境增强与鲁棒性评估", "评估夜间、阴影、雨天、磨损、遮挡等场景", "complex_env_results.csv"),
    ExperimentSpec("EXP-03", "Phase 2", "EMA 注意力模块实验", "验证注意力模块对低对比度标线缺失区域的收益", "runs/train/exp_ema_ciou"),
    ExperimentSpec("EXP-04", "Phase 2", "BiFPN 多尺度融合实验", "验证多尺度融合对远距离和小目标缺失标线的收益", "runs/train/exp_ema_bifpn_ciou"),
    ExperimentSpec("EXP-05", "Phase 2", "C2f-DCN 形变建模实验", "验证不规则缺失边界和弯曲标线建模能力", "models/yolo26_roadmark_full.yaml"),
    ExperimentSpec("EXP-06", "Phase 3", "Wise-IoU 定位损失实验", "验证标线缺失区域框的定位质量", "runs/train/exp_full_wise_iou"),
    ExperimentSpec("EXP-07", "Phase 3", "完整模型主对比", "汇总 baseline、模块消融和完整模型", "experiment_results.csv"),
    ExperimentSpec("EXP-08", "Phase 4", "实时推理与视频验证", "验证图片、视频和摄像头输入", "realtime_results.csv"),
    ExperimentSpec("EXP-09", "Phase 4", "PyQt5 Demo 与结果展示", "完成可视化交互演示", "demo_checklist.md"),
)


ABLATION_RUNS: tuple[ModelRun, ...] = (
    ModelRun("YOLO26", "baseline", "ciou", ("EXP-01", "EXP-03", "EXP-07"), "baseline"),
    ModelRun("YOLO26+EMA", "ema", "ciou", ("EXP-03", "EXP-04", "EXP-07"), "加入 EMA 注意力"),
    ModelRun("YOLO26+EMA+BiFPN", "ema_bifpn", "ciou", ("EXP-04", "EXP-05", "EXP-07"), "加入 EMA 与 BiFPN"),
    ModelRun("YOLO26+C2f-DCN+EMA+BiFPN", "full", "ciou", ("EXP-05", "EXP-06"), "完整结构，仍使用 CIoU"),
    ModelRun("YOLO26+C2f-DCN+EMA+BiFPN+WiseIoU", "full", "wise_iou", ("EXP-06", "EXP-07"), "完整模型"),
)


def all_experiment_ids() -> list[str]:
    return [spec.exp_id for spec in EXPERIMENT_PLAN]


def get_experiment(exp_id: str) -> ExperimentSpec:
    for spec in EXPERIMENT_PLAN:
        if spec.exp_id == exp_id:
            return spec
    available = ", ".join(all_experiment_ids())
    raise KeyError(f"未知实验编号: {exp_id}. 可选: {available}")


def runs_for_experiment(exp_id: str) -> list[ModelRun]:
    if exp_id == "EXP-07":
        return [run for run in ABLATION_RUNS if run.loss == "wise_iou" or run.variant != "full"] + [
            run for run in ABLATION_RUNS if run.variant == "full" and run.loss == "ciou"
        ]
    return [run for run in ABLATION_RUNS if exp_id in run.exp_ids]
