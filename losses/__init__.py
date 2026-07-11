"""损失函数扩展包。"""

from .wise_iou import WiseIoU, bbox_iou, wise_iou_loss

__all__ = ["WiseIoU", "bbox_iou", "wise_iou_loss"]

