from typing import Tuple

import torch
from mmcv.runner import force_fp32
from torch import nn
from torch.nn import functional as F

from mmdet3d.ops import bev_pool

__all__ = ["BaseTransform", "BaseDepthTransform"]

def boolmask2idx(mask):
    # A utility function, workaround for ONNX not supporting 'nonzero'
    return torch.nonzero(mask).squeeze(1).tolist()

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor(
        [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]
    )
    return dx, bx, nx


class BaseTransform(nn.Module):
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
        use_points='lidar', 
        depth_input='scalar',
        height_expand=False,
        add_depth_features=False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.feature_size = feature_size
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound
        self.use_points = use_points
        assert use_points in ['radar', 'lidar']
        self.depth_input=depth_input
        assert depth_input in ['scalar', 'one-hot']
        self.height_expand = height_expand
        self.add_depth_features = add_depth_features

        dx, bx, nx = gen_dx_bx(self.xbound, self.ybound, self.zbound)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.C = out_channels
        self.frustum = self.create_frustum()
        self.D = self.frustum.shape[0]
        # self.bev_anchors = self.create_bev_anchors()
        self.fp16_enabled = False

    @force_fp32()
    def create_frustum(self):
        iH, iW = self.image_size
        fH, fW = self.feature_size

        ds = (
            torch.arange(*self.dbound, dtype=torch.float)
            .view(-1, 1, 1)
            .expand(-1, fH, fW)
        )
        D, _, _ = ds.shape

        xs = (
            torch.linspace(0, iW - 1, fW, dtype=torch.float)
            .view(1, 1, fW)
            .expand(D, fH, fW)
        )
        ys = (
            torch.linspace(0, iH - 1, fH, dtype=torch.float)
            .view(1, fH, 1)
            .expand(D, fH, fW)
        )

        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    @force_fp32()
    def create_bev_anchors(self, ds_rate=1):
        x_coords = ((torch.linspace(
            self.xbound[0],
            self.xbound[1] - self.xbound[2] * ds_rate,
            self.nx[0] // ds_rate,
            dtype=torch.float,
        ) + self.xbound[2] * ds_rate / 2).view(self.nx[0] // ds_rate, 1).expand(
            self.nx[0] // ds_rate,
            self.nx[1] // ds_rate))

        y_coords = ((torch.linspace(
            self.ybound[0],
            self.ybound[1] - self.ybound[2] * ds_rate,
            self.nx[1] // ds_rate,
            dtype=torch.float,
        ) + self.ybound[2] * ds_rate / 2).view(1, self.nx[1] // ds_rate).expand(
            self.nx[0] // ds_rate,
            self.nx[1] // ds_rate))

        anchors = torch.stack([x_coords, y_coords]).permute(1, 2, 0)
        return nn.Parameter(anchors, requires_grad=False)

    @force_fp32()
    def get_geometry(
        self,
        camera2lidar_rots,
        camera2lidar_trans,
        intrins,
        post_rots,
        post_trans,
        **kwargs,
    ):
        B, N, _ = camera2lidar_trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = (
            torch.inverse(post_rots)
            .view(B, N, 1, 1, 1, 3, 3)
            .matmul(points.unsqueeze(-1))
        )
        # cam_to_lidar
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )
        combine = camera2lidar_rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)

        if "extra_rots" in kwargs:
            extra_rots = kwargs["extra_rots"]
            points = (
                extra_rots.view(B, 1, 1, 1, 1, 3, 3)
                .repeat(1, N, 1, 1, 1, 1, 1)
                .matmul(points.unsqueeze(-1))
                .squeeze(-1)
            )
        if "extra_trans" in kwargs:
            extra_trans = kwargs["extra_trans"]
            points += extra_trans.view(B, 1, 1, 1, 1, 3).repeat(1, N, 1, 1, 1, 1)

        return points

    def get_cam_feats(self, x):
        raise NotImplementedError

    @force_fp32()
    def bev_pool(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
                for ix in range(B)
            ]
        )
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]

        x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1)

        return final

    @force_fp32()
    def forward(
        self,
        img,
        points,
        radar, 
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        **kwargs,
    ):
        rots = camera2ego[..., :3, :3]
        trans = camera2ego[..., :3, 3]
        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        lidar2ego_rots = lidar2ego[..., :3, :3]
        lidar2ego_trans = lidar2ego[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )
        mats_dict = {
            'intrin_mats': camera_intrinsics, 
            'ida_mats': img_aug_matrix, 
            'bda_mat': lidar_aug_matrix,
            'sensor2ego_mats': camera2ego, 
        }
        x = self.get_cam_feats(img, mats_dict)

        use_depth = False
        if type(x) == tuple:
            x, depth = x 
            use_depth = True
        
        x = self.bev_pool(geom, x)

        if use_depth:
            return x, depth 
        else:
            return x



class BaseDepthTransform(BaseTransform):
    @force_fp32()
    def get_proj_mat(self, geom=None, mats_dict=None):
        """Create the Ring Matrix and Ray Matrix

        Args:
            mats_dict (dict, optional): dictionary that
                contains intrin- and extrin- parameters.
            Defaults to None.

        Returns:
            tuple: Ring Matrix in [B, D, L, L] and Ray Matrix in [B, W, L, L]
        """

        bev_size = int(self.nx[0])  # only consider square BEV
        geom_sep = (geom - (self.bx - self.dx / 2.0)) / self.dx
        geom_sep = geom_sep.mean(3).permute(0, 1, 3, 2, 4).contiguous()
        B, Nc, W, D, _ = geom_sep.shape
        geom_sep = geom_sep.long().view(B, Nc * W, D, -1)[..., :2]

        invalid1 = torch.logical_or((geom_sep < 0)[..., 0], (geom_sep < 0)[..., 1])
        invalid2 = torch.logical_or((geom_sep > (bev_size - 1))[..., 0],
                                    (geom_sep > (bev_size - 1))[..., 1])
        geom_sep[(invalid1 | invalid2)] = int(bev_size / 2)
        geom_idx = geom_sep[..., 1] * bev_size + geom_sep[..., 0]

        geom_uni = self.bev_anchors[None].repeat([B, 1, 1, 1])
        B, L, L, _ = geom_uni.shape

        circle_map = geom_uni.new_zeros((B, D, L * L))
        ray_map = geom_uni.new_zeros((B, Nc * W, L * L))
        for b in range(B):
            for dir in range(Nc * W):
                ray_map[b, dir, geom_idx[b, dir]] += 1
            for d in range(D):
                circle_map[b, d, geom_idx[b, :, d]] += 1
        null_point = int((bev_size / 2) * (bev_size + 1))
        circle_map[..., null_point] = 0
        ray_map[..., null_point] = 0
        circle_map = circle_map.view(B, D, L * L)
        ray_map = ray_map.view(B, -1, L * L)
        circle_map /= circle_map.max(1)[0].clip(min=1)[:, None]
        ray_map /= ray_map.max(1)[0].clip(min=1)[:, None]

        return circle_map, ray_map

    @force_fp32()
    def reduce_and_project(self, feature, depth, geom, mats_dict):
        """reduce the feature and depth in height
            dimension and make BEV feature

        Args:
            feature (Tensor): image feature in [B, C, H, W]
            depth (Tensor): Depth Prediction in [B, D, H, W]
            mats_dict (dict): dictionary that contains intrin-
                and extrin- parameters

        Returns:
            Tensor: BEV feature in B, C, L, L
        """
        depth = self.depth_reducer(feature, depth)
        B = mats_dict['intrin_mats'].shape[0]
        feature = self.horiconv(feature)
        depth = depth.permute(0, 2, 1).reshape(B, -1, self.D)
        feature = feature.permute(0, 2, 1).reshape(B, -1, self.C)
        circle_map, ray_map = self.get_proj_mat(geom, mats_dict)

        proj_mat = depth.matmul(circle_map)
        proj_mat = (proj_mat * ray_map).permute(0, 2, 1)
        img_feat_with_depth = proj_mat.matmul(feature)
        img_feat_with_depth = img_feat_with_depth.permute(0, 2, 1).reshape(
            B, -1, *self.nx[:2])

        return img_feat_with_depth

    @force_fp32()
    def forward(
        self,
        img,
        points,
        radar, 
        sensor2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        cam_intrinsic,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        **kwargs,
    ):
        rots = sensor2ego[..., :3, :3]
        trans = sensor2ego[..., :3, 3]
        intrins = cam_intrinsic[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        lidar2ego_rots = lidar2ego[..., :3, :3]
        lidar2ego_trans = lidar2ego[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        if self.use_points == 'radar':
            points = radar

        if self.height_expand:
            for b in range(len(points)):
                points_repeated = points[b].repeat_interleave(8, dim=0)
                points_repeated[:, 2] = torch.arange(0.25, 2.25, 0.25).repeat(points[b].shape[0])
                points[b] = points_repeated

        batch_size = len(points)
        depth_in_channels = 1 if self.depth_input=='scalar' else self.D
        if self.add_depth_features:
            depth_in_channels += points[0].shape[1]

        depth = torch.zeros(batch_size, img.shape[1], depth_in_channels, *self.image_size, device=points[0].device)


        for b in range(batch_size):
            cur_coords = points[b][:, :3]
            cur_img_aug_matrix = img_aug_matrix[b]
            cur_lidar_aug_matrix = lidar_aug_matrix[b]
            cur_lidar2image = lidar2image[b]

            # inverse aug
            cur_coords -= cur_lidar_aug_matrix[:3, 3]
            cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
                cur_coords.transpose(1, 0)
            )
            # lidar2image
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)
            # get 2d coords
            dist = cur_coords[:, 2, :]
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

            # imgaug
            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)

            # normalize coords for grid sample
            cur_coords = cur_coords[..., [1, 0]]

            on_img = (
                (cur_coords[..., 0] < self.image_size[0])
                & (cur_coords[..., 0] >= 0)
                & (cur_coords[..., 1] < self.image_size[1])
                & (cur_coords[..., 1] >= 0)
            )
            for c in range(on_img.shape[0]):
                masked_coords = cur_coords[c, on_img[c]].long()
                masked_dist = dist[c, on_img[c]]

                if self.depth_input == 'scalar':
                    depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist
                elif self.depth_input == 'one-hot':
                    # Clamp depths that are too big to D
                    # These can arise when the point range filter is different from the dbound. 
                    masked_dist = torch.clamp(masked_dist, max=self.D-1)
                    depth[b, c, masked_dist.long(), masked_coords[:, 0], masked_coords[:, 1]] = 1.0

                if self.add_depth_features:
                    depth[b, c, -points[b].shape[-1]:, masked_coords[:, 0], masked_coords[:, 1]] = points[b][boolmask2idx(on_img[c])].transpose(0,1)

        step = 7
        B, N, C, H, W = depth.size()
        depth_tmp = depth.reshape(B * N, C, H, W)
        pad = (step - 1) // 2
        depth_tmp = F.pad(depth_tmp, [pad, pad, pad, pad], mode='constant', value=0)
        patches = depth_tmp.unfold(dimension=2, size=step, step=1)
        patches = patches.unfold(dimension=3, size=step, step=1)
        max_depth, _ = patches.reshape(B, N, C, H, W, -1).max(dim=-1)

        step = float(step)
        shift_list = [[step / H, 0.0 / W], [-step / H, 0.0 / W], [0.0 / H, step / W], [0.0 / H, -step / W]]
        max_depth_tmp = max_depth.reshape(B * N, C, H, W)
        output_list = []
        for shift in shift_list:
            transform_matrix = torch.tensor([[1, 0, shift[0]], [0, 1, shift[1]]]).unsqueeze(0).repeat(B * N, 1, 1).cuda()
            grid = F.affine_grid(transform_matrix, max_depth_tmp.shape).float()
            output = F.grid_sample(max_depth_tmp, grid, mode='nearest').reshape(B, N, C, H, W)
            output = max_depth - output
            output_mask = ((output == max_depth) == False)
            output = output * output_mask
            output_list.append(output)
        grad = torch.cat(output_list, dim=2)

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]
        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        mats_dict = {
            'intrin_mats': intrins, 
            'ida_mats': img_aug_matrix, 
            'bda_mat': lidar_aug_matrix,
            'sensor2ego_mats': sensor2ego, 
        }
        x = self.get_cam_feats(img, depth, grad, mats_dict)

        use_depth = False
        if type(x) == tuple:
            x, depth = x 
            use_depth = True
        
        x = self.bev_pool(geom, x)

        if use_depth:
            return x, depth 
        else:
            return x
        # x = self.reduce_and_project(x, depth, geom, mats_dict)
        # return x

