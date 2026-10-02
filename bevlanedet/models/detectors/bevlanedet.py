import logging
import torch
from torch import nn
from ..utils.common_modules import Residual

from mmengine.model import kaiming_init, constant_init, Sequential
from mmengine.logging import print_log
from mmcv.cnn import ConvModule
from mmdet.models import BaseDetector
from mmdet.registry import MODELS

@MODELS.register_module()
class BEVLaneDet(BaseDetector):
    def __init__(self,
                data_preprocessor,
                backbone,
                s32_view_transformer,
                s64_view_transformer,
                fv_lane_head,
                bev_lane_head,
                loss_seg,
                loss_embed,
                loss_offset,
                loss_z,
                bev_loss_seg_weight,
                bev_loss_embed_weight,
                bev_loss_offset_weight,
                bev_loss_z_weight,
                fv_loss_seg_weight,
                fv_loss_embed_weight,
                bb_out_ch=512,
                init_cfg=None):
        
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        
        self.backbone = MODELS.build(backbone)
        
        self.down = Residual(
            module = Sequential(
                ConvModule(
                    in_channels=bb_out_ch, 
                    out_channels=1024, 
                    kernel_size=3, 
                    stride=2, 
                    padding=1, 
                    norm_cfg=dict(type='BN'), 
                    act_cfg=dict(type='ReLU')
                    ),
                ConvModule(
                    in_channels=1024, 
                    out_channels=1024, 
                    kernel_size=3, 
                    stride=1, 
                    padding=1, 
                    norm_cfg=dict(type='BN'), 
                    act_cfg=None
                    )
            ),
            downsample = nn.Conv2d(bb_out_ch, 1024, kernel_size=3, stride=2, padding=1)
        )
        
        # MODELS
        self.s32_view_transformer = MODELS.build(s32_view_transformer)
        self.s64_view_transformer = MODELS.build(s64_view_transformer)
        self.fv_lane_head = MODELS.build(fv_lane_head)
        self.bev_lane_head = MODELS.build(bev_lane_head)
        
        # LOSSES
        self.loss_bce_logits = MODELS.build(loss_seg.bce_loss)
        self.loss_soft_iou = MODELS.build(loss_seg.iou_loss)
        self.loss_push_pull = MODELS.build(loss_embed)
        self.loss_bce = MODELS.build(loss_offset)
        self.loss_mse = MODELS.build(loss_z)
        
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
        img_s32 = self.extract_feat(inputs)[0]      # only last stage feature
        img_s64 = self.down(img_s32)
        bev_32 = self.s32_view_transformer(img_s32)
        bev_64 = self.s64_view_transformer(img_s64)
        bev = torch.cat([bev_32, bev_64], dim=1)
        
        if self.training:
            return self.bev_lane_head(bev), self.fv_lane_head(img_s32)
        else:
            return self.bev_lane_head(bev)
        
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
        loss_offset = self.loss_bce(bev_seg_gt * torch.sigmoid(offset_y), bev_offset_gt).unsqueeze(0)
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
        
        loss_seg_2d = self.loss_bce_logits(pred_2d, fv_seg_gt) + self.loss_soft_iou(torch.sigmoid(pred_2d), fv_instance_gt)
        loss_emb_2d = self.loss_push_pull(emb_2d, fv_instance_gt)
        loss_total_2d = self.fv_loss_seg_weight * loss_seg_2d + self.fv_loss_embed_weight * loss_emb_2d
        loss_total_2d = loss_total_2d.unsqueeze(0)
        
        losses['loss_total_bev'] = loss_total
        losses['loss_total_fv'] = loss_total_2d
        losses['loss_offset_bev'] = loss_offset
        losses['loss_z_bev'] = loss_z
        
        return losses
    
    def predict(self, inputs):
        bev_out = self._forward(inputs)
        
        pred, _, _, _ = bev_out
        
        return pred