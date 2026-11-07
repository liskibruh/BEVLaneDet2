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

class SimpleSpatialLinear(nn.Module):
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
        
        # spatial_feats = F.interpolate(
        #     spatial_feats,
        #     size=(X,Y),
        #     mode='bilinear',
        #     align_corners=False
        # )
        
        combined = torch.concat([bev_feats, spatial_feats], dim=1)
        
        return self.fuse(combined)

class SimpleSpatialLinear2(nn.Module):
    def __init__(self, in_shape, in_channels, context_ch=8, out_channels=64):
        super().__init__()

        in_planes = in_shape[0] * in_shape[1]
        out_planes = (in_shape[0] // 2) * (in_shape[1] // 2)

        self.spatial = nn.Sequential(
            nn.Linear(in_planes, out_planes),
            nn.ReLU()
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, context_ch, kernel_size=1),
            nn.BatchNorm2d(context_ch)
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(context_ch, context_ch, kernel_size=1),
            nn.BatchNorm2d(context_ch)
        )
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels+context_ch, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, bev_feats):
        B, C, H, W = bev_feats.shape

        x = bev_feats.reshape(B, C, -1).to(device=bev_feats.device, dtype=bev_feats.dtype)
        x = self.spatial(x)
        x = x.reshape(B, C, H // 2, W // 2)

        x = self.downsample(x)

        x = nn.functional.interpolate(
            x,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )

        x = self.pointwise(x)
        
        x = torch.cat([bev_feats, x], dim=1)
        
        x = self.final_conv(x)

        return x

class CBAMSpatialModule(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, kernel_size=7):
        super().__init__()
        
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False),
            # nn.BatchNorm2d(out_channels)
        )
        
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, bev_feats):
        mean = torch.mean(bev_feats, dim=1, keepdim=True)
        max, _ = torch.max(bev_feats, dim=1, keepdim=True)
        
        cated = torch.cat([mean, max], dim=1)
        reduced = self.reduce(cated)
        attn = self.sigmoid(reduced)
        
        return bev_feats * attn
    
    
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        # The Shared MLP structure matching the paper
        self.mlp = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_planes // ratio, in_planes, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        # Flatten spatial dims to feed to the Linear layers
        avg_out = self.mlp(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        max_out = self.mlp(self.max_pool(x).view(b, c)).view(b, c, 1, 1)
        
        # Element-wise summation of shared weights
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # In planes is strictly 2 (1 for avg, 1 for max)
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate along the channel axis
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM_Block(nn.Module):
    def __init__(self, channels, ratio=16, kernel_size=7):
        super(CBAM_Block, self).__init__()
        
        self.downscale = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        
        self.ca = ChannelAttention(channels, ratio)
        self.sa = SpatialAttention(kernel_size)
        
        self.downsample = nn.Conv2d(128, 64, 1)

    def forward(self, x):
        
        x = self.downscale(x)
        
        # 1. Apply Channel Attention first
        x = x * self.ca(x)
        # 2. Apply Spatial Attention sequentially downstream
        x = x * self.sa(x)
        
        x = nn.functional.interpolate(
            x,
            scale_factor=2,
            mode='bilinear',
            align_corners=False
        )
        
        x = self.downsample(x)
        
        return x