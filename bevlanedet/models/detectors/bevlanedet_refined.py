import torch
from torch import nn
from ..utils.common_modules import Residual, masked_mean_height, LearnedHeightCompression, \
    SimpleSpatialLinear, SimpleSpatialLinear2, CBAMSpatialModule, CBAM_Block
# from ...utils.cluster import embedding_post
# from ...evaluation.utils.post_process_utils import bev_instance2points_with_offset_z

from mmengine.model import kaiming_init, constant_init, Sequential
from mmengine.logging import print_log
from mmcv.cnn import ConvModule, build_conv_layer
from mmdet.models import BaseDetector
from mmdet.registry import MODELS


class FuseBackboneFeats(nn.Module):
    def __init__(self, feat16_in, feat16_out, feat32_in, feat32_out, out):
        super().__init__()
        
        self.feat16_downsample = ConvModule(
            in_channels=feat16_in,
            out_channels=feat16_out,
            kernel_size=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')
        )
        
        self.feat32_downsample = ConvModule(
            in_channels=feat32_in,
            out_channels=feat32_out,
            kernel_size=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')
        )
        
        self.fuser = ConvModule(
            in_channels = feat16_out + feat32_out,
            out_channels = out,
            kernel_size=1,
            norm_cfg=dict(type='BN'),
            act_cfg=None
        )
    
    def forward(self, feat16, feat32):
        feat16 = self.feat16_downsample(feat16)
        feat32 = self.feat32_downsample(feat32)
        
        feat32 = nn.functional.interpolate(
            feat32,
            scale_factor=2,
            mode='bilinear',
            align_corners=False)
        
        feat = torch.concat([feat16, feat32], dim=1)
        
        return self.fuser(feat)

@MODELS.register_module()
class BEVLaneDetVoxUpscaledRefined(BaseDetector):
    def __init__(self,
                data_preprocessor,
                backbone,
                view_transformer,
                fv_lane_head,
                bev_lane_head,
                loss_seg,
                loss_embed,
                loss_offset,
                loss_z,
                bev_loss_seg_weight=3,
                bev_loss_embed_weight=0.5,
                bev_loss_offset_weight=60,
                bev_loss_z_weight=30,
                fv_loss_seg_weight=3,
                fv_loss_embed_weight=0.5,
                bb_out_ch=512,
                bev_feat_ch=1792,
                init_cfg=None):
        
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        
        self.backbone = MODELS.build(backbone)
        
        self.img16_upsample = ConvModule(
            in_channels=1024,
            out_channels=2048,
            kernel_size=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'),
        )
        
        self.backbone_reduce = ConvModule(
            in_channels=2048,
            out_channels=bb_out_ch,
            kernel_size=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'),
        )
        
        self.ch_reduce = ConvModule(
            in_channels=bb_out_ch,
            out_channels=64,
            kernel_size=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')   # todo: try removing this act fn (mobilenetv2)
        )
        
        self.downscale = Sequential(
            ConvModule(
                in_channels=128,
                out_channels=bev_feat_ch,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU')
            ),
            ConvModule(
                in_channels=bev_feat_ch,
                out_channels=bev_feat_ch,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU')
            )
        )
        
        # self.downscale = ConvModule(
        #     in_channels=512,
        #     out_channels=bev_feat_ch,
        #     kernel_size=3,
        #     stride=2,
        #     padding=1,
        #     norm_cfg=dict(type='BN'),
        #     act_cfg=dict(type='ReLU')
        # )
        
        # MODELS
        self.view_transformer = MODELS.build(view_transformer)
        self.fv_lane_head = MODELS.build(fv_lane_head)
        self.bev_lane_head = MODELS.build(bev_lane_head)
        
        # LOSSES
        self.loss_bce_logits = MODELS.build(loss_seg.bce_loss)
        self.loss_soft_iou = MODELS.build(loss_seg.iou_loss)
        self.loss_push_pull = MODELS.build(loss_embed)
        self.loss_bce_logits_offset = MODELS.build(loss_offset)
        self.loss_mse = MODELS.build(loss_z)
        
        self.learned_height_compressor = LearnedHeightCompression(channels=64, out_channels=128, num_z=28)
        self.spatial_relation_module = SimpleSpatialLinear(in_shape=(50, 12), out_shape=(50, 12), in_ch=128, context_ch=8, out_ch=128)
        # self.cbam_spatial_module = CBAMSpatialModule(in_channels=2, out_channels=1, kernel_size=7)
        # self.cbam_module = CBAM_Block(channels=128, ratio=16, kernel_size=7)
        
        self.bev_loss_seg_weight = bev_loss_seg_weight
        self.bev_loss_embed_weight = bev_loss_embed_weight
        self.bev_loss_offset_weight = bev_loss_offset_weight
        self.bev_loss_z_weight = bev_loss_z_weight
        self.fv_loss_seg_weight = fv_loss_seg_weight
        self.fv_loss_embed_weight = fv_loss_embed_weight
        
        
    def extract_feat(self, input):
        feat = self.backbone(input)
        return feat
    
    def _forward(self, inputs):
        img = inputs['img']
        vox_proj_idx = inputs['voxel_proj_index']
        vox_valid_mask = inputs['voxel_valid_mask']
        
        img_16, img_32 = self.extract_feat(img)
        
        # multiscale feature fusion
        img_16 = self.img16_upsample(img_16)
        img_32 = nn.functional.interpolate(
            img_32,
            scale_factor=2,
            mode='bilinear',
            align_corners=False
        )
        img = img_16 + img_32
        
        img = self.backbone_reduce(img)
        
        img_32_bev = self.ch_reduce(img)
        
        voxel_feats, voxel_mask = self.view_transformer(img_32_bev, vox_proj_idx, vox_valid_mask)
        # B, C, Nx, Nz, Ny = voxel_feats.shape
        # bev_feats = masked_mean_height(voxel_feats, voxel_mask)
        
        bev_feats = self.learned_height_compressor(voxel_feats, voxel_mask)
        
        bev_feats = self.downscale(bev_feats)
        
        bev_feats = self.spatial_relation_module(bev_feats)
        # bev_feats = self.distance_based_spatial_module(bev_feats)
        # bev_feats = self.cbam_spatial_module(bev_feats)
        # bev_feats = self.cbam_module(bev_feats)
        
        if self.training:
            return self.bev_lane_head(bev_feats), self.fv_lane_head(img)
        else:
            return self.bev_lane_head(bev_feats)
    
    def loss(self, inputs, data_samples):
        losses = dict()
        bev_out, fv_out = self._forward(inputs)
        
        pred, emb, offset_y, z = bev_out
        pred_2d, emb_2d = fv_out
        
        bev_seg_gt = torch.stack([
            sample.bev_gt.sem_seg
            for sample in data_samples
        ])
        
        bev_instance_gt = torch.stack([
            sample.bev_gt.instance_seg
            for sample in data_samples
        ])
        
        bev_offset_gt = torch.stack([
            sample.bev_gt.y_offset_map
            for sample in data_samples
        ])
        
        bev_z_gt = torch.stack([
            sample.bev_gt.z_map
            for sample in data_samples
        ])
        
        loss_seg = self.loss_bce_logits(pred, bev_seg_gt) + self.loss_soft_iou(torch.sigmoid(pred), bev_seg_gt)
        loss_emb = self.loss_push_pull(emb, bev_instance_gt)
        loss_offset = self.loss_bce_logits_offset(offset_y.squeeze(1), bev_offset_gt.squeeze(1), weight=bev_seg_gt.squeeze(1)).unsqueeze(0)
        loss_z = self.loss_mse(bev_seg_gt * z, bev_z_gt).unsqueeze(0)
        loss_total = self.bev_loss_seg_weight * loss_seg + self.bev_loss_embed_weight * loss_emb
        loss_total = loss_total.unsqueeze(0)
        loss_offset = self.bev_loss_offset_weight * loss_offset
        loss_z = self.bev_loss_z_weight * loss_z
        
        fv_seg_gt = torch.stack([
            sample.image_gt_segment.sem_seg
            for sample in data_samples
        ])
        
        fv_instance_gt = torch.stack([
            sample.image_gt_instance.instance_seg
            for sample in data_samples
        ])
        
        loss_seg_2d = self.loss_bce_logits(pred_2d, fv_seg_gt) + self.loss_soft_iou(torch.sigmoid(pred_2d), fv_seg_gt)
        loss_emb_2d = self.loss_push_pull(emb_2d, fv_instance_gt)
        loss_total_2d = self.fv_loss_seg_weight * loss_seg_2d + self.fv_loss_embed_weight * loss_emb_2d
        loss_total_2d = loss_total_2d.unsqueeze(0)
        
        losses['loss_total_bev'] = loss_total
        losses['loss_total_fv'] = loss_total_2d
        losses['loss_offset_bev'] = loss_offset
        losses['loss_z_bev'] = loss_z
                
        return losses
    
    def predict(self, inputs, data_samples):
        
        bev_out = self._forward(inputs)
        
        pred_seg      = bev_out[0].detach().cpu()
        pred_emb      = bev_out[1].detach().cpu()
        pred_offset_y = torch.sigmoid(bev_out[2]).detach().cpu()
        pred_z        = bev_out[3].detach().cpu()
        
        
        for idx, sample in enumerate(data_samples):            
            sample.bev_pred = {
                'pred_seg': pred_seg[idx].unsqueeze(0),
                'pred_emb': pred_emb[idx].unsqueeze(0),
                'pred_offset_y': pred_offset_y[idx].unsqueeze(0),
                'pred_z': pred_z[idx].unsqueeze(0)
            }
            
        return data_samples