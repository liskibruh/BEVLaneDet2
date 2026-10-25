import logging
import torch
from torch import nn
from ..utils.common_modules import Residual

from mmcv.cnn import ConvModule
from mmengine.model import BaseModule, kaiming_init, constant_init, Sequential
from mmengine.logging import print_log
from mmdet.registry import MODELS

@MODELS.register_module()
class FCTransform_(BaseModule):
    def __init__(self, image_featmap_size, space_featmap_size, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        ic, ih, iw = image_featmap_size
        sc, sh, sw = space_featmap_size
        
        self.image_featmpa_size = image_featmap_size
        self.space_featmap_size = space_featmap_size
        
        self.fc_transform = Sequential(
            nn.Linear(ih*iw, sh*sw),
            nn.ReLU(),
            nn.Linear(sh*sw, sh*sw),
            nn.ReLU()
        )
        
        self.conv1 = ConvModule(
            in_channels=ic,
            out_channels=sc,
            kernel_size=1,
            stride=1,
            bias=False,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')
        )
        
        self.residual = Residual(
            module= ConvModule(
                in_channels=sc,
                out_channels=sc,
                kernel_size=1,
                stride=1,
                bias=False,
                norm_cfg=dict(type='BN'),
                act_cfg=None
            )
        )
    
    def forward(self, x):
        x = x.view(list(x.size()[:2]) + [self.image_featmpa_size[1] * self.image_featmpa_size[2], ]) # (B,C,H,W) -> (B,C,H*W)
        bev_view = self.fc_transform(x)
        bev_view = bev_view.view(list(bev_view.size()[:2]) + [self.space_featmap_size[1], self.space_featmap_size[2]])
        bev_view = self.conv1(bev_view)
        bev_view = self.residual(bev_view)
        
        return bev_view
    

@MODELS.register_module()
class FCTransform_Mod(BaseModule):
    def __init__(self, image_featmap_size, space_featmap_size, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        ic, ih, iw = image_featmap_size
        sc, sh, sw = space_featmap_size
        
        self.image_featmpa_size = image_featmap_size
        self.space_featmap_size = space_featmap_size
        
        self.fc_transform = Sequential(
            nn.Linear(ih*iw, sh*sw),
            nn.ReLU(),
            nn.Linear(sh*sw, sh*sw),
            nn.ReLU()
        )
        
        self.conv1 = ConvModule(
            in_channels=ic,
            out_channels=sc,
            kernel_size=1,
            stride=1,
            bias=False,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')
        )
        
        self.residual = Residual(
            module= ConvModule(
                in_channels=sc,
                out_channels=sc,
                kernel_size=1,
                stride=1,
                bias=False,
                norm_cfg=dict(type='BN'),
                act_cfg=None
            )
        )
    
    def forward(self, x):
        x = x.view(list(x.size()[:2]) + [self.image_featmpa_size[1] * self.image_featmpa_size[2], ]) # (B,C,H,W) -> (B,C,H*W)
        bev_view = self.fc_transform(x)
        bev_view = bev_view.view(list(bev_view.size()[:2]) + [self.space_featmap_size[1], self.space_featmap_size[2]])
        bev_view = self.conv1(bev_view)
        bev_view = self.residual(bev_view)
        
        return bev_view
    
@MODELS.register_module()    
class ExplicitVoxelProjection(BaseModule):
    def __init__(
        self,
        z_range,
        roi_x,
        roi_y,
        grid_res,
    ):
        super().__init__()

        self.z_range = z_range
        self.roi_x = roi_x
        self.roi_y = roi_y
        self.grid_res = grid_res

        self.num_grids_x = int(
            (roi_x[1] - roi_x[0]) / grid_res[0]
        )
        self.num_grids_y = int(
            (roi_y[1] - roi_y[0]) / grid_res[1]
        )
        self.num_grids_z = int(
            (2 * z_range) / grid_res[2]
        )

        self.num_voxels = (
            self.num_grids_x
            * self.num_grids_y
            * self.num_grids_z
        )

    def forward(
        self,
        x,
        voxel_proj_idx,
        voxel_valid_mask,
    ):
        """
        Args:
            x:
                (B, C, Hf, Wf), backbone feature map.

            voxel_proj_idx:
                (B, 2, N), projected feature-map coordinates.
                Index 0 is u (column); index 1 is v (row).

            voxel_valid_mask:
                (B, N), True for valid projections.

        Returns:
            voxel_feats:
                (B, C, Nx, Nz, Ny)
            voxel_mask:
                (B, 1, Nx, Nz, Ny)
        """
        B, C, Hf, Wf = x.shape

        assert voxel_proj_idx.ndim == 3
        assert voxel_proj_idx.shape == (
            B, 2, self.num_voxels
        ), (
            f"Expected voxel_proj_idx shape "
            f"({B}, 2, {self.num_voxels}), "
            f"got {tuple(voxel_proj_idx.shape)}"
        )

        assert voxel_valid_mask.shape == (
            B, self.num_voxels
        )

        idx = voxel_proj_idx.to(
            device=x.device,
            dtype=torch.long,
        )

        valid = voxel_valid_mask.to(
            device=x.device,
            dtype=torch.bool,
        )

        u = idx[:, 0, :]
        v = idx[:, 1, :]

        u = u.clamp(0, Wf - 1)
        v = v.clamp(0, Hf - 1)

        x_flat = x.flatten(start_dim=2)

        linear_idx = v * Wf + u

        gathered = torch.gather(
            x_flat,
            dim=2,
            index=linear_idx.unsqueeze(1).expand(-1, C, -1),
        )

        gathered = gathered * valid.unsqueeze(1).to(
            dtype=gathered.dtype
        )

        voxel_feats = gathered.reshape(
            B,
            C,
            self.num_grids_x,
            self.num_grids_y,
            self.num_grids_z,
        )

        voxel_feats = voxel_feats.permute(
            0, 1, 2, 4, 3
        ).contiguous()

        voxel_mask = valid.reshape(
            B,
            self.num_grids_x,
            self.num_grids_y,
            self.num_grids_z,
        )

        voxel_mask = voxel_mask.permute(
            0, 1, 3, 2
        ).unsqueeze(1).contiguous()

        return voxel_feats, voxel_mask