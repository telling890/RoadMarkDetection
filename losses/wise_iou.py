"""Wise-IoU 损失。

Wise-IoU 用动态聚焦机制降低异常框/低质量样本对回归的干扰，比固定 CIoU 更适合雨天、
遮挡、磨损等复杂道路环境中的标线框回归。
"""

from __future__ import annotations

import torch
from torch import nn


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """[x, y, w, h] -> [x1, y1, x2, y2]，x/y 为中心点。"""

    x, y, w, h = boxes.unbind(dim=-1)
    half_w, half_h = w / 2, h / 2
    return torch.stack((x - half_w, y - half_h, x + half_w, y + half_h), dim=-1)


def bbox_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    eps: float = 1e-7,
    box_format: str = "xyxy",
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算 IoU 和 Wise-IoU 需要的中心距离/外接框对角线比例。"""

    if box_format == "xywh":
        boxes1 = xywh_to_xyxy(boxes1)
        boxes2 = xywh_to_xyxy(boxes2)
    elif box_format != "xyxy":
        raise ValueError("box_format 仅支持 'xyxy' 或 'xywh'。")

    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1.unbind(dim=-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2.unbind(dim=-1)

    inter_w = (torch.minimum(b1_x2, b2_x2) - torch.maximum(b1_x1, b2_x1)).clamp(min=0)
    inter_h = (torch.minimum(b1_y2, b2_y2) - torch.maximum(b1_y1, b2_y1)).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = (b1_x2 - b1_x1).clamp(min=0) * (b1_y2 - b1_y1).clamp(min=0)
    area2 = (b2_x2 - b2_x1).clamp(min=0) * (b2_y2 - b2_y1).clamp(min=0)
    union = area1 + area2 - inter_area + eps
    iou = inter_area / union

    b1_cx, b1_cy = (b1_x1 + b1_x2) / 2, (b1_y1 + b1_y2) / 2
    b2_cx, b2_cy = (b2_x1 + b2_x2) / 2, (b2_y1 + b2_y2) / 2
    center_distance = (b1_cx - b2_cx).pow(2) + (b1_cy - b2_cy).pow(2)

    cw = torch.maximum(b1_x2, b2_x2) - torch.minimum(b1_x1, b2_x1)
    ch = torch.maximum(b1_y2, b2_y2) - torch.minimum(b1_y1, b2_y1)
    enclosing_distance = cw.pow(2) + ch.pow(2) + eps
    distance_ratio = center_distance / enclosing_distance
    return iou.clamp(0, 1), distance_ratio


class WiseIoU(nn.Module):
    """Wise-IoU v3 风格动态聚焦损失。

    Args:
        box_format: 输入框格式，'xyxy' 或 'xywh'；
        momentum: running mean 更新动量；
        gamma/delta: 动态聚焦曲线参数；
        reduction: 'none'、'mean' 或 'sum'。
    """

    def __init__(
        self,
        box_format: str = "xyxy",
        momentum: float = 0.01,
        gamma: float = 1.9,
        delta: float = 3.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction 仅支持 'none'、'mean'、'sum'。")
        self.box_format = box_format
        self.momentum = momentum
        self.gamma = gamma
        self.delta = delta
        self.reduction = reduction
        self.register_buffer("iou_mean", torch.tensor(1.0))

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        iou, distance_ratio = bbox_iou(pred_boxes, target_boxes, box_format=self.box_format)
        iou_loss = 1.0 - iou

        if self.training and iou_loss.numel() > 0:
            batch_mean = iou_loss.detach().mean()
            self.iou_mean.mul_(1.0 - self.momentum).add_(batch_mean * self.momentum)

        # Wise-IoU 的距离注意项：中心越偏离，惩罚越大；detach 避免梯度被距离项过度放大。
        distance_weight = torch.exp(distance_ratio.detach())

        # 动态非单调聚焦：质量极差样本不会无限主导训练，适合复杂噪声场景。
        outlier_degree = (iou_loss.detach() / self.iou_mean.clamp(min=1e-6)).clamp(min=0)
        focusing_weight = self.delta * torch.pow(self.gamma, outlier_degree - self.delta)
        loss = focusing_weight * distance_weight * iou_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def wise_iou_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    box_format: str = "xyxy",
    reduction: str = "mean",
) -> torch.Tensor:
    """函数式 Wise-IoU 损失接口。"""

    return WiseIoU(box_format=box_format, reduction=reduction)(pred_boxes, target_boxes)

