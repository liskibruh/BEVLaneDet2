import torch
from torch import nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer, build_upsample_layer

def naive_init_module(mod):
    for m in mod.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, 1)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return mod

class Residual(nn.Module):
    def __init__(self, module, downsample=None):
        super().__init__()
        self.module = module
        self.downsample = downsample
        self.relu = nn.ReLU()
    
    def forward(self, x):
        identity = x
        out = self.module(x)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        
        return self.relu(out)

def masked_mean_height(voxel_feats, voxel_mask):
    """
    Args:
        voxel_feats: (B, C, Nx, Nz, Ny)
        voxel_mask:  (B, 1, Nx, Nz, Ny)
                    True/1 means valid.

    Returns:
        bev_feats:   (B, C, Nx, Ny)
    """
    if voxel_mask.ndim == 4:
        voxel_mask = voxel_mask.unsqueeze(1)

    mask = voxel_mask.to(
        device=voxel_feats.device,
        dtype=voxel_feats.dtype,
    )

    if mask.shape[0] != voxel_feats.shape[0]:
        raise ValueError("Batch dimensions of features and mask do not match.")

    if mask.shape[2:] != voxel_feats.shape[2:]:
        raise ValueError(
            f"Feature shape {tuple(voxel_feats.shape)} and "
            f"mask shape {tuple(mask.shape)} are incompatible."
        )

    masked_feats = voxel_feats * mask
    feat_sum = masked_feats.sum(dim=3)
    valid_count = mask.sum(dim=3)
    bev_feats = feat_sum / valid_count.clamp_min(1.0)

    return bev_feats

class LearnedHeightCompression(nn.Module):
    def __init__(self, channels=64, out_channels=64, num_z=28):
        super().__init__()
        self.compress=ConvModule(
            in_channels=channels * num_z + num_z,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'),
            inplace=True
        )
    
    def forward(self, voxel_feats, voxel_mask):
        
        B, C, Nx, Nz, Ny = voxel_feats.shape
        
        mask = voxel_mask.to(
            device=voxel_feats.device,
            dtype=voxel_feats.dtype
        )
        
        feats = voxel_feats * mask
        feats = feats.permute(0, 1, 3, 2, 4).contiguous()
        feats = feats.reshape(B, C*Nz, Nx, Ny)
        
        mask_channels = mask[:, 0].permute(0, 2, 1, 3).contiguous()
        feats = torch.cat([feats, mask_channels], dim=1)
        
        return self.compress(feats)

class GlobalSpatialRelationModule(nn.Module):
    def __init__(self, in_shape, out_shape, in_ch, context_ch, out_ch):
        super().__init__()
        
        in_h, in_w = in_shape
        self.out_h, self.out_w = out_shape
        
        self.fc_spatial = nn.Sequential(
            nn.Linear(in_h*in_w, self.out_h*self.out_w),
            nn.ReLU(inplace=True),
            # nn.Linear(self.out_h*self.out_w, self.out_h*self.out_w),
            # nn.ReLU(inplace=True)
            )
                
        self.pointwise_spatial = ConvModule(
            in_channels=in_ch,
            out_channels=context_ch,
            kernel_size=1,
            norm_cfg=dict(type='BN'),
            act_cfg=None
        )
                
        # self.pointwise_bev = ConvModule(
        #     in_channels=in_ch,
        #     out_channels=64,
        #     kernel_size=1,
        #     norm_cfg=dict(type='BN'),
        #     act_cfg=None
        # )
        
        self.fuse = ConvModule(
            in_channels=in_ch + context_ch,
            out_channels=out_ch,
            kernel_size=1,
            norm_cfg=dict(type='BN'),
            act_cfg=None
        )        
    
    def forward(self, bev_feats):
        B, C, X, Y = bev_feats.shape
        
        # bev_feats = self.pointwise_bev(bev_feats)

        spatial_feats = bev_feats.reshape(B, C, X*Y).to(device=bev_feats.device, dtype=bev_feats.dtype)
        spatial_feats = self.fc_spatial(spatial_feats)
        spatial_feats = spatial_feats.reshape(B, C, self.out_h, self.out_w)
        spatial_feats = self.pointwise_spatial(spatial_feats)
        
        spatial_feats = F.interpolate(
            spatial_feats,
            size=(X,Y),
            mode='bilinear',
            align_corners=False
        )
        
        combined = torch.concat([bev_feats, spatial_feats], dim=1)
        
        return bev_feats + self.fuse(combined)