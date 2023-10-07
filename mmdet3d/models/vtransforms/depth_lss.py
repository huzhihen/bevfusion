from typing import Tuple

import torch
import torch.nn.functional as F
from torch.cuda.amp.autocast_mode import autocast
from mmcv.runner import force_fp32
from torch import nn

from mmdet3d.models.builder import VTRANSFORMS

from .base import BaseDepthTransform

__all__ = ["DepthLSSTransform"]


class HoriConv(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, cat_dim=0):
        """HoriConv that reduce the image feature
            in height dimension and refine it.

        Args:
            in_channels (int): in_channels
            mid_channels (int): mid_channels
            out_channels (int): output channels
            cat_dim (int, optional): channels of position
                embedding. Defaults to 0.
        """
        super().__init__()

        self.merger = nn.Sequential(
            nn.Conv2d(in_channels + cat_dim, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True),
        )

        self.reduce_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    @autocast(False)
    def forward(self, x, pe=None):
        if pe is not None:
            x = self.merger(torch.cat([x, pe], 1))
        else:
            x = self.merger(x)
        x = x.max(dim=2)[0]
        x = self.reduce_conv(x)
        x = self.conv1(x) + x
        x = self.conv2(x) + x
        x = self.out_conv(x)
        return x


class DepthReducer(nn.Module):

    def __init__(self, img_channels, mid_channels):
        """Module that compresses the predicted
            categorical depth in height dimension

        Args:
            img_channels (int): in_channels
            mid_channels (int): mid_channels
        """
        super().__init__()
        self.vertical_weighter = nn.Sequential(
            nn.Conv2d(img_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=3, stride=1, padding=1),
        )

    @autocast(False)
    def forward(self, feat, depth):
        vert_weight = self.vertical_weighter(feat).softmax(dim=2)
        depth = (depth * vert_weight).sum(dim=2)
        return depth


@VTRANSFORMS.register_module()
class DepthLSSTransform(BaseDepthTransform):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )
        self.dtransform = nn.Sequential(
            nn.Conv2d(1, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.etransform = nn.Sequential(
            nn.Conv2d(4, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channels + 64 + 64, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, self.D + self.C, 1),
        )
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

        # self.horiconv = HoriConv(self.C, 512, self.C)
        # self.depth_reducer = DepthReducer(self.C, self.C)

    def get_downsampled_gt_depth(self, gt_depths, downsample=8):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N, H // downsample, downsample, W // downsample, downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, downsample * downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0, 1e5 * torch.ones_like(gt_depths), gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // downsample, W // downsample)

        gt_depths = (gt_depths - (1 - 0.5)) / 0.5
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0), gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:, 1:]
        return gt_depths.float()

    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds, loss_depth_weight=3.0):
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum())
        return depth_loss * loss_depth_weight

    @force_fp32()
    def get_cam_feats(self, x, d, e):
        B, N, C, fH, fW = x.shape

        d = d.view(B * N, *d.shape[2:])
        e = e.view(B * N, *e.shape[2:])
        x = x.view(B * N, C, fH, fW)

        d = self.dtransform(d)
        e = self.etransform(e)
        x = torch.cat([d, e, x], dim=1)
        x = self.depthnet(x)

        depth = x[:, : self.D].softmax(dim=1)

        # x = x[:, self.D: (self.D + self.C)]
        # return x, depth
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)
        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x, depth

    def forward(self, *args, **kwargs):
        x = super().forward(*args, **kwargs)
        final_x = self.downsample(x[0]), x[1]
        return final_x
        # x = super().forward(*args, **kwargs)
        # x = self.downsample(x)
        # return x