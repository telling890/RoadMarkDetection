"""Ultralytics YOLO26 接入辅助。

本文件不直接修改 site-packages，而是在运行时注册/补丁：
1. 注册 EMA、C2fDCN、BiFPN，使 YOLO26 YAML 能识别这些模块；
2. 为 BiFPN 这种三输入多输出 Neck 增加 parse_model 分支；
3. 可选地把 Ultralytics 默认 CIoU 框回归损失替换为 Wise-IoU。

如果你准备修改 YOLO26 源码，README 中也给出了对应的永久修改位置。
"""

from __future__ import annotations

import ast
import contextlib
from copy import deepcopy
from typing import Any

import torch
import torch.nn.functional as F

from losses.wise_iou import WiseIoU
from models.modules import BiFPN, C2fDCN, EMA


def _existing_classes(module: Any, names: list[str]) -> set[type]:
    """从 Ultralytics 模块中按名称取类，兼容不同版本类名有无差异。"""

    return {getattr(module, name) for name in names if hasattr(module, name)}


def _resolve_module(name: str, tasks_module: Any) -> type:
    """解析 YAML 中的模块名。"""

    if "nn." in name:
        return getattr(torch.nn, name[3:])
    if "torchvision.ops." in name:
        import torchvision

        return getattr(torchvision.ops, name[16:])
    custom = {"EMA": EMA, "C2fDCN": C2fDCN, "BiFPN": BiFPN}
    if name in custom:
        return custom[name]
    return getattr(tasks_module, name)


def register_custom_modules() -> None:
    """注册 YOLO26 改进模块并替换 parse_model。

    该函数需要在 `YOLO(model_yaml)` 之前调用。
    """

    import ultralytics.nn.tasks as tasks

    if getattr(tasks, "_roadmark_custom_registered", False):
        return

    tasks.EMA = EMA
    tasks.C2fDCN = C2fDCN
    tasks.BiFPN = BiFPN
    tasks._roadmark_original_parse_model = tasks.parse_model
    tasks.parse_model = _build_patched_parse_model(tasks)
    _patch_dcn_safe_model_info(tasks)
    tasks._roadmark_custom_registered = True


def _patch_dcn_safe_model_info(tasks: Any) -> None:
    """Skip THOP FLOPs profiling for DCN models on Windows.

    torchvision DeformConv2d can raise a process-level access violation when THOP
    deep-copies and profiles the model on Windows. This does not affect normal
    forward/backward execution, so report parameter counts without FLOPs.
    """

    if getattr(tasks.model_info, "_roadmark_dcn_safe", False):
        return
    original_model_info = tasks.model_info

    def safe_model_info(model, detailed: bool = False, verbose: bool = True, imgsz: int = 640):
        if not any(isinstance(module, C2fDCN) for module in model.modules()):
            return original_model_info(model, detailed=detailed, verbose=verbose, imgsz=imgsz)
        if not verbose:
            return None
        leaves = [module for module in model.modules() if not module._modules]
        params = sum(parameter.numel() for parameter in model.parameters())
        gradients = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        tasks.LOGGER.info(
            f"DCN model summary: {len(leaves):,} layers, {params:,} parameters, "
            f"{gradients:,} gradients (FLOPs profiling skipped on Windows)"
        )
        return len(leaves), params, gradients, 0.0

    safe_model_info._roadmark_dcn_safe = True  # type: ignore[attr-defined]
    safe_model_info._roadmark_original = original_model_info  # type: ignore[attr-defined]
    tasks.model_info = safe_model_info


def _build_patched_parse_model(tasks: Any):
    """生成扩展版 parse_model。

    逻辑基本沿用 Ultralytics 8.4.x 的 parse_model，只额外处理：
    - C2fDCN: 作为可重复基础模块；
    - EMA: 单输入同尺寸注意力模块；
    - BiFPN: 三输入 [P3, P4, P5]，输出 list[P3, P4, P5]。
    """

    def parse_model(d: dict[str, Any], ch: int, verbose: bool = True):  # noqa: C901, PLR0912, PLR0915
        d = deepcopy(d)
        LOGGER = tasks.LOGGER
        colorstr = tasks.colorstr
        make_divisible = tasks.make_divisible
        Conv = tasks.Conv

        legacy = True
        max_channels = float("inf")
        nc, act, scales, end2end = (d.get(x) for x in ("nc", "activation", "scales", "end2end"))
        reg_max = d.get("reg_max", 16)
        depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
        scale = d.get("scale")
        if scales:
            if not scale:
                scale = next(iter(scales.keys()))
                LOGGER.warning(f"no model scale passed. Assuming scale='{scale}'.")
            depth, width, max_channels = scales[scale]

        if act:
            Conv.default_act = eval(act)  # noqa: S307 - 与 Ultralytics 原实现一致
            if verbose:
                LOGGER.info(f"{colorstr('activation:')} {act}")

        if verbose:
            LOGGER.info(
                f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}"
            )

        base_modules = _existing_classes(
            tasks,
            [
                "Classify",
                "Conv",
                "ConvTranspose",
                "GhostConv",
                "Bottleneck",
                "GhostBottleneck",
                "SPP",
                "SPPF",
                "C2fPSA",
                "C2PSA",
                "DWConv",
                "Focus",
                "BottleneckCSP",
                "C1",
                "C2",
                "C2f",
                "C3k2",
                "RepNCSPELAN4",
                "ELAN1",
                "ADown",
                "AConv",
                "SPPELAN",
                "C2fAttn",
                "C3",
                "C3TR",
                "C3Ghost",
                "DWConvTranspose2d",
                "C3x",
                "RepC3",
                "PSA",
                "SCDown",
                "C2fCIB",
                "A2C2f",
            ],
        ) | {torch.nn.ConvTranspose2d, C2fDCN}
        repeat_modules = _existing_classes(
            tasks,
            [
                "BottleneckCSP",
                "C1",
                "C2",
                "C2f",
                "C3k2",
                "C2fAttn",
                "C3",
                "C3TR",
                "C3Ghost",
                "C3x",
                "RepC3",
                "C2fPSA",
                "C2fCIB",
                "C2PSA",
                "A2C2f",
            ],
        ) | {C2fDCN}
        detect_modules = _existing_classes(
            tasks,
            [
                "Detect",
                "WorldDetect",
                "YOLOEDetect",
                "Segment",
                "Segment26",
                "YOLOESegment",
                "YOLOESegment26",
                "Pose",
                "Pose26",
                "OBB",
                "OBB26",
            ],
        )
        segment_modules = _existing_classes(tasks, ["Segment", "YOLOESegment", "Segment26", "YOLOESegment26"])

        ch = [ch]
        layers, save, c2 = [], [], ch[-1]
        for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
            if isinstance(m, str):
                m = _resolve_module(m, tasks)

            for j, a in enumerate(args):
                if isinstance(a, str):
                    with contextlib.suppress(ValueError, SyntaxError):
                        args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

            n = n_ = max(round(n * depth), 1) if n > 1 else n
            if m in base_modules:
                c1, c2 = ch[f], args[0]
                if c2 != nc:
                    c2 = make_divisible(min(c2, max_channels) * width, 8)

                if hasattr(tasks, "C2fAttn") and m is tasks.C2fAttn:
                    args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                    args[2] = int(
                        max(round(min(args[2], max_channels // 2 // 32)) * width, 1)
                        if args[2] > 1
                        else args[2]
                    )

                args = [c1, c2, *args[1:]]
                if m in repeat_modules:
                    args.insert(2, n)
                    n = 1
                if hasattr(tasks, "C3k2") and m is tasks.C3k2:
                    legacy = False
                    if scale in "mlx":
                        args[3] = True
                if hasattr(tasks, "A2C2f") and m is tasks.A2C2f:
                    legacy = False
                    if scale in "lx":
                        args.extend((True, 1.2))
                if hasattr(tasks, "C2fCIB") and m is tasks.C2fCIB:
                    legacy = False
            elif m is EMA:
                c2 = ch[f]
                args = [c2, *args]
            elif m is BiFPN:
                if not isinstance(f, list) or len(f) != 3:
                    raise ValueError("BiFPN 在 YAML 中必须使用 from=[P3, P4, P5] 三个输入层。")
                in_channels = [ch[x] for x in f]
                out_channels = int(args[0]) if args else max(in_channels)
                num_layers = int(args[1]) if len(args) > 1 else 1
                args = [in_channels, out_channels, num_layers, *args[2:]]
                c2 = out_channels
            elif hasattr(tasks, "AIFI") and m is tasks.AIFI:
                args = [ch[f], *args]
            elif m in _existing_classes(tasks, ["HGStem", "HGBlock"]):
                c1, cm, c2 = ch[f], args[0], args[1]
                args = [c1, cm, c2, *args[2:]]
                if hasattr(tasks, "HGBlock") and m is tasks.HGBlock:
                    args.insert(4, n)
                    n = 1
            elif hasattr(tasks, "ResNetLayer") and m is tasks.ResNetLayer:
                c2 = args[1] if args[3] else args[1] * 4
            elif m is torch.nn.BatchNorm2d:
                args = [ch[f]]
            elif hasattr(tasks, "Concat") and m is tasks.Concat:
                c2 = sum(ch[x] for x in f)
            elif m in detect_modules:
                args.extend([reg_max, end2end, [ch[x] for x in f]])
                if m in segment_modules:
                    args[2] = make_divisible(min(args[2], max_channels) * width, 8)
                m.legacy = legacy
            elif hasattr(tasks, "v10Detect") and m is tasks.v10Detect:
                args.append([ch[x] for x in f])
            elif hasattr(tasks, "ImagePoolingAttn") and m is tasks.ImagePoolingAttn:
                args.insert(1, [ch[x] for x in f])
            elif hasattr(tasks, "RTDETRDecoder") and m is tasks.RTDETRDecoder:
                args.insert(1, [ch[x] for x in f])
            elif hasattr(tasks, "CBLinear") and m is tasks.CBLinear:
                c2 = args[0]
                c1 = ch[f]
                args = [c1, c2, *args[1:]]
            elif hasattr(tasks, "CBFuse") and m is tasks.CBFuse:
                c2 = ch[f[-1]]
            elif m in _existing_classes(tasks, ["TorchVision", "Index"]):
                c2 = args[0]
                args = [*args[1:]]
            else:
                c2 = ch[f] if isinstance(f, int) else ch[f[-1]]

            module = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
            module_type = str(m)[8:-2].replace("__main__.", "")
            module.np = sum(x.numel() for x in module.parameters())
            module.i, module.f, module.type = i, f, module_type
            if verbose:
                LOGGER.info(f"{i:>3}{f!s:>20}{n_:>3}{module.np:10.0f}  {module_type:<45}{args!s:<30}")
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
            layers.append(module)
            if i == 0:
                ch = []
            ch.append(c2)
        return torch.nn.Sequential(*layers), sorted(save)

    return parse_model


def patch_wise_iou_loss() -> None:
    """将 Ultralytics BboxLoss 的 CIoU 替换为 Wise-IoU。

    调用时机：创建 YOLO 模型后、`model.train(...)` 前。该补丁保留官方 DFL/L1 分支。
    """

    import ultralytics.utils.loss as yolo_loss

    if getattr(yolo_loss.BboxLoss.forward, "_roadmark_wise_iou", False):
        return

    original_forward = yolo_loss.BboxLoss.forward

    def wise_forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ):
        if not fg_mask.any():
            zero = pred_bboxes.sum() * 0.0
            return zero, zero

        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        wise_iou = getattr(self, "_roadmark_wise_iou_loss", None)
        if wise_iou is None:
            wise_iou = WiseIoU(box_format="xyxy", reduction="none")
            self._roadmark_wise_iou_loss = wise_iou
        wise_iou = wise_iou.to(pred_bboxes.device)
        wise_iou.train(self.training)

        loss_iou_raw = wise_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask]).view(-1, 1)
        loss_iou = (loss_iou_raw * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = yolo_loss.bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]
            ) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = yolo_loss.bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True)
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl

    wise_forward._roadmark_wise_iou = True  # type: ignore[attr-defined]
    wise_forward._roadmark_original_forward = original_forward  # type: ignore[attr-defined]
    yolo_loss.BboxLoss.forward = wise_forward
