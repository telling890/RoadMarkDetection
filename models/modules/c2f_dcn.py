"""C2f-DCN 模块。

该文件实现道路标线检测项目中的 Backbone 增强模块：
1. Deformable Convolution：通过学习采样偏移量提升弯曲/磨损标线的建模能力；
2. C2f 结构：参考 YOLO 系列的轻量跨阶段特征聚合方式，保留较好的速度。

模块输入/输出均为 [B, C, H, W]，可直接放入 YOLO26 Backbone 中替换部分 C3k2/C2f。
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

try:
    from torchvision.ops import DeformConv2d
except Exception as exc:  # pragma: no cover - 只有依赖缺失时触发
    DeformConv2d = None
    _DEFORM_IMPORT_ERROR = exc
else:
    _DEFORM_IMPORT_ERROR = None


def _make_divisible(value: int, divisor: int = 8) -> int:
    """将通道数对齐到 divisor，避免 Tensor Core/卷积实现效率下降。"""

    return int((value + divisor / 2) // divisor * divisor)


class ConvBNAct(nn.Module):
    """标准 Conv + BN + SiLU，小模块内复用，避免依赖 Ultralytics 私有类。"""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        padding = k // 2 if p is None else p
        self.conv = nn.Conv2d(c1, c2, k, s, padding, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DeformableConv(nn.Module):
    """DCNv2 风格可变形卷积。

    TorchVision 的 DeformConv2d 需要额外输入 offset 和 mask，本类内部用普通卷积预测：
    - offset: 每个卷积核采样点在 x/y 两个方向上的偏移；
    - mask: 每个采样点的调制权重，经过 sigmoid 约束到 [0, 1]。
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        deform_groups: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        if DeformConv2d is None:
            raise ImportError(
                "C2f-DCN 需要 torchvision.ops.DeformConv2d，请先安装 torchvision。"
            ) from _DEFORM_IMPORT_ERROR

        padding = k // 2 if p is None else p
        offset_channels = 2 * deform_groups * k * k
        mask_channels = deform_groups * k * k

        self.offset = nn.Conv2d(c1, offset_channels, kernel_size=k, stride=s, padding=padding)
        self.mask = nn.Conv2d(c1, mask_channels, kernel_size=k, stride=s, padding=padding)
        self.dcn = DeformConv2d(
            c1,
            c2,
            kernel_size=k,
            stride=s,
            padding=padding,
            groups=g,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

        # offset 初始化为 0，mask 初始化为 0.5，使初始状态接近普通卷积，训练更稳定。
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        nn.init.zeros_(self.mask.weight)
        nn.init.zeros_(self.mask.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset(x)
        mask = torch.sigmoid(self.mask(x))
        return self.act(self.bn(self.dcn(x, offset, mask)))


class BottleneckDCN(nn.Module):
    """带可变形卷积的瓶颈块。

    先用 1x1 卷积压缩/整理通道，再用 DCN 捕获弯曲标线和细长结构。
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
    ) -> None:
        super().__init__()
        hidden = _make_divisible(int(c2 * e), 8)
        self.cv1 = ConvBNAct(c1, hidden, k=1, s=1)
        self.cv2 = DeformableConv(hidden, c2, k=3, s=1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2fDCN(nn.Module):
    """C2f + DCN 复合模块。

    参数与 Ultralytics C2f 风格保持接近，方便在 YOLO26 YAML 中替换：
    Args:
        c1: 输入通道数；
        c2: 输出通道数；
        n: BottleneckDCN 重复次数；
        shortcut: 是否使用残差连接；
        g: 分组卷积数；
        e: 隐藏层扩展比例。
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ) -> None:
        super().__init__()
        self.c = _make_divisible(int(c2 * e), 8)
        self.cv1 = ConvBNAct(c1, 2 * self.c, k=1, s=1)
        self.cv2 = ConvBNAct((2 + n) * self.c, c2, k=1, s=1)
        self.m = nn.ModuleList(
            BottleneckDCN(self.c, self.c, shortcut=shortcut, g=g, e=1.0) for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，输出尺寸与普通 C2f/C3 类模块一致。"""

        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(block(y[-1]) for block in self.m)
        return self.cv2(torch.cat(y, dim=1))


def module_param_count(module: nn.Module) -> int:
    """统计模块参数量，供实验脚本展示。"""

    return sum(p.numel() for p in module.parameters())


def flatten_modules(modules: Iterable[nn.Module]) -> list[nn.Module]:
    """将模块迭代器转为列表，便于调试打印。"""

    return list(modules)

