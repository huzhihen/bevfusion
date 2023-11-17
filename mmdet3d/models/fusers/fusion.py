from typing import List

import torch
from torch import nn

from mmdet3d.models.builder import FUSERS

__all__ = ["LidarCameraFusion"]


class LidarCameraCrossAttention(nn.Module):
    def __init__(self, camera_channel, lidar_channel, reduction=16):
        super(LidarCameraCrossAttention, self).__init__()
        self.camera_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.camera_fc = nn.Sequential(
            nn.Linear(camera_channel, reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, lidar_channel, bias=False),
            nn.Sigmoid()
        )
        self.lidar_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.lidar_fc = nn.Sequential(
            nn.Linear(lidar_channel, reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, camera_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, camera_feature_x, lidar_feature_x):
        camera_b, camera_c, _, _ = camera_feature_x.size()
        lidar_b, lidar_c, _, _ = lidar_feature_x.size()
        camera_feature_y = self.camera_avg_pool(camera_feature_x).view(camera_b, camera_c)
        camera_feature_y = self.camera_fc(camera_feature_y).view(camera_b, lidar_c, 1, 1)
        lidar_feature_y = self.lidar_avg_pool(lidar_feature_x).view(lidar_b, lidar_c)
        lidar_feature_y = self.lidar_fc(lidar_feature_y).view(lidar_b, camera_c, 1, 1)
        return camera_feature_x * lidar_feature_y.expand_as(camera_feature_x),\
               lidar_feature_x * camera_feature_y.expand_as(lidar_feature_x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1      = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1    = nn.ReLU()
        self.fc2      = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out     = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1   = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out    = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class LidarCameraFusion_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(LidarCameraFusion_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out


@FUSERS.register_module()
class LidarCameraFusion(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cross_attention = LidarCameraCrossAttention(in_channels[0], in_channels[1])
        self.fusion_conv = self._make_layer(block=LidarCameraFusion_Block,
                                            input_channels=sum(in_channels),
                                            output_channels=out_channels,
                                            block_nums=2)

    def _make_layer(self, block, input_channels, output_channels, block_nums=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(block_nums - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        inputs = self.cross_attention(inputs[0], inputs[1])
        x = self.fusion_conv(torch.cat(inputs, dim=1))
        return x
