from typing import List

import torch
from torch import nn

from mmdet3d.models.builder import FUSERS

__all__ = ["LidarCameraFusion"]


@FUSERS.register_module()
class LidarCameraFusion(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(sum(in_channels), out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(out_channels, out_channels // 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(True),
            nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        x = self.conv1(torch.cat(inputs, dim=1))
        x = self.conv2(x) * x
        return x
