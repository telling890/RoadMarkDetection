"""BiFPN Neck 模块。

BiFPN 用可学习权重进行 P3/P4/P5 双向特征融合，适合道路标线中的：
- 远距离细车道线；
- 小尺寸道路箭头；
- 多尺度人行横道和导流线。

输入:  [P3, P4, P5]
输出:  [P3_out, P4_out, P5_out]
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class SeparableConvBlock(nn.Module):
    """深度可分离卷积块，减少 Neck 融合带来的额外参数量。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pointwise(self.depthwise(x))))


class WeightedFeatureFusion(nn.Module):
    """BiFPN 加权特征融合。

    每条输入边都有一个可学习非负权重，归一化后加权求和，避免简单相加时强特征淹没弱特征。
    """

    def __init__(self, num_inputs: int, eps: float = 1e-4) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))
        self.eps = eps

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != self.weight.numel():
            raise ValueError(f"期望 {self.weight.numel()} 个输入特征，实际收到 {len(features)} 个。")
        weight = F.relu(self.weight)
        weight = weight / (weight.sum() + self.eps)
        out = 0.0
        for index, feature in enumerate(features):
            out = out + weight[index] * feature
        return out


class BiFPNLayer(nn.Module):
    """单层 P3/P4/P5 双向路径聚合。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.p4_td_fuse = WeightedFeatureFusion(2)
        self.p3_td_fuse = WeightedFeatureFusion(2)
        self.p4_out_fuse = WeightedFeatureFusion(3)
        self.p5_out_fuse = WeightedFeatureFusion(2)

        self.p3_td_conv = SeparableConvBlock(channels)
        self.p4_td_conv = SeparableConvBlock(channels)
        self.p4_out_conv = SeparableConvBlock(channels)
        self.p5_out_conv = SeparableConvBlock(channels)

    @staticmethod
    def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="nearest")

    @staticmethod
    def _downsample_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(x, kernel_size=3, stride=2, padding=1) if x.shape[-2:] != ref.shape[-2:] else x

    def forward(
        self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 自顶向下：P5 -> P4 -> P3，增强高层语义对远距离标线的指导。
        p4_td = self.p4_td_conv(self.p4_td_fuse([p4, self._resize_like(p5, p4)]))
        p3_td = self.p3_td_conv(self.p3_td_fuse([p3, self._resize_like(p4_td, p3)]))

        # 自底向上：P3 -> P4 -> P5，把细粒度边缘信息回流给高层。
        p4_out = self.p4_out_conv(
            self.p4_out_fuse([p4, p4_td, self._downsample_like(p3_td, p4)])
        )
        p5_out = self.p5_out_conv(self.p5_out_fuse([p5, self._downsample_like(p4_out, p5)]))
        return p3_td, p4_out, p5_out


class BiFPN(nn.Module):
    """P3/P4/P5 三尺度 BiFPN。

    Args:
        in_channels: 三个输入特征的通道数，例如 [256, 512, 1024]；
        out_channels: 融合后的统一通道数；
        num_layers: BiFPN 重复层数。
    """

    def __init__(
        self,
        in_channels: Sequence[int] = (256, 512, 1024),
        out_channels: int = 256,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if len(in_channels) != 3:
            raise ValueError("BiFPN 仅支持 P3/P4/P5 三个输入特征。")

        self.in_channels = list(map(int, in_channels))
        self.out_channels = int(out_channels)
        self.lateral_convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(c, self.out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.out_channels),
                nn.SiLU(inplace=True),
            )
            for c in self.in_channels
        )
        self.layers = nn.ModuleList(BiFPNLayer(self.out_channels) for _ in range(num_layers))

    def forward(self, features: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if len(features) != 3:
            raise ValueError("BiFPN.forward 需要输入 [P3, P4, P5]。")

        p3, p4, p5 = [conv(feature) for conv, feature in zip(self.lateral_convs, features)]
        for layer in self.layers:
            p3, p4, p5 = layer(p3, p4, p5)
        return [p3, p4, p5]

