"""EMA 注意力模块。

EMA（Efficient Multi-scale Attention）通过分组建模、横纵向上下文池化与多尺度权重融合，
增强网络对车道线、停止线、人行横道、箭头等道路标线区域的关注。

输入:  [B, C, H, W]
输出:  [B, C, H, W]
尺寸保持不变，可插入 Backbone 或 Neck 的任意同通道位置。
"""

from __future__ import annotations

import math

import torch
from torch import nn


class EMA(nn.Module):
    """Efficient Multi-scale Attention。

    Args:
        channels: 输入/输出通道数；
        factor: 分组数上限。实际分组数会自动调整，确保 channels 能被整除。
    """

    def __init__(self, channels: int, factor: int = 8) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("EMA 的 channels 必须为正整数。")

        self.channels = channels
        self.groups = self._valid_groups(channels, factor)
        group_channels = channels // self.groups

        self.conv1x1 = nn.Conv2d(group_channels, group_channels, kernel_size=1, bias=True)
        self.conv3x3 = nn.Conv2d(group_channels, group_channels, kernel_size=3, padding=1, bias=True)
        self.group_norm = nn.GroupNorm(1, group_channels)
        self.softmax = nn.Softmax(dim=-1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _valid_groups(channels: int, factor: int) -> int:
        """选择能整除 channels 的最大分组数，避免通道不能整除导致运行错误。"""

        factor = max(1, min(int(factor), channels))
        for groups in range(factor, 0, -1):
            if channels % groups == 0:
                return groups
        return math.gcd(channels, factor) or 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        grouped = x.reshape(b * self.groups, c // self.groups, h, w)

        # 横向/纵向上下文池化分别聚合车道线长条结构在两个方向上的响应。
        x_h = grouped.mean(dim=3, keepdim=True)
        x_w = grouped.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        attn_h, attn_w = torch.split(hw, [h, w], dim=2)
        attn_w = attn_w.permute(0, 1, 3, 2)

        x1 = self.group_norm(grouped * self.sigmoid(attn_h) * self.sigmoid(attn_w))
        x2 = self.conv3x3(grouped)

        # 双分支交叉生成空间权重，兼顾局部纹理和全局上下文。
        x11 = self.softmax(self.avg_pool(x1).reshape(b * self.groups, 1, -1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.avg_pool(x2).reshape(b * self.groups, 1, -1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(
            b * self.groups, 1, h, w
        )

        out = grouped * self.sigmoid(weights)
        return out.reshape(b, c, h, w)

