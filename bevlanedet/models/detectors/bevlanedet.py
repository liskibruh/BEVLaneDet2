import logging
import torch
from torch import nn
from ..utils.common_modules import Residual
# from ...utils.cluster import embedding_post
# from ...evaluation.utils.post_process_utils import bev_instance2points_with_offset_z

from mmengine.model import kaiming_init, constant_init, Sequential
from mmengine.logging import print_log
from mmcv.cnn import ConvModule, build_conv_layer
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
                bev_loss_seg_weight=3,
                bev_loss_embed_weight=0.5,
                bev_loss_offset_weight=60,
                bev_loss_z_weight=30,
                fv_loss_seg_weight=3,
                fv_loss_embed_weight=0.5,
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
        self.loss_bce_logits_offset = MODELS.build(loss_offset)
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

@MODELS.register_module()
class BEVLaneDetMod(BaseDetector):
    def __init__(self,
                data_preprocessor,
                backbone,
                s32_view_transformer,
                # s64_view_transformer,
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
                bev_feat_ch=64,
                init_cfg=None):
        
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        
        self.backbone = MODELS.build(backbone)
        
        self.ch_reduce = ConvModule(
            in_channels=bb_out_ch,
            out_channels=64,
            kernel_size=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')
        )
        
        self.downscale = ConvModule(
            in_channels=64,
            out_channels=bev_feat_ch,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')
        )
        
        # MODELS
        self.s32_view_transformer = MODELS.build(s32_view_transformer)
        # self.s64_view_transformer = MODELS.build(s64_view_transformer)
        self.fv_lane_head = MODELS.build(fv_lane_head)
        self.bev_lane_head = MODELS.build(bev_lane_head)
        
        # LOSSES
        self.loss_bce_logits = MODELS.build(loss_seg.bce_loss)
        self.loss_soft_iou = MODELS.build(loss_seg.iou_loss)
        self.loss_push_pull = MODELS.build(loss_embed)
        self.loss_bce_logits_offset = MODELS.build(loss_offset)
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
        img_32 = self.extract_feat(inputs)[0]      # only last stage feature
        # img_s64 = self.down(img_s32)
        img_32_bev = self.ch_reduce(img_32)
        bev_feats = self.s32_view_transformer(img_32_bev)
        bev_feats = self.downscale(bev_feats)
        
        if self.training:
            return self.bev_lane_head(bev_feats), self.fv_lane_head(img_32)
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
    
@MODELS.register_module()
class BEVLaneDetVox(BaseDetector):
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
        
        self.ch_reduce = ConvModule(
            in_channels=bb_out_ch,
            out_channels=64,
            kernel_size=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')
        )
        
        self.downscale = ConvModule(
            in_channels=64,
            out_channels=bev_feat_ch,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')
        )
        
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
        
        self.bev_loss_seg_weight = bev_loss_seg_weight
        self.bev_loss_embed_weight = bev_loss_embed_weight
        self.bev_loss_offset_weight = bev_loss_offset_weight
        self.bev_loss_z_weight = bev_loss_z_weight
        self.fv_loss_seg_weight = fv_loss_seg_weight
        self.fv_loss_embed_weight = fv_loss_embed_weight
        
    def extract_feat(self, input):
        feat = self.backbone(input)
        return feat
    
    def masked_mean_height(self, voxel_feats, voxel_mask):
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
    
    def _forward(self, inputs):
        img = inputs['img']
        vox_proj_idx = inputs['voxel_proj_index']
        vox_valid_mask = inputs['voxel_valid_mask']
        
        img_32 = self.extract_feat(img)[0]      # only last stage feature
        img_32_bev = self.ch_reduce(img_32)
        voxel_feats, voxel_mask = self.view_transformer(img_32_bev, vox_proj_idx, vox_valid_mask)
        
        # B, C, Nx, Nz, Ny = voxel_feats.shape
        bev_feats = self.masked_mean_height(voxel_feats, voxel_mask)
        bev_feats = self.downscale(bev_feats)
        
        if self.training:
            return self.bev_lane_head(bev_feats), self.fv_lane_head(img_32)
        else:
            return self.bev_lane_head(bev_feats)
        
    def loss(self, inputs, data_samples):
        losses = dict()
        bev_out, fv_out = self._forward(inputs)
        
        pred, emb, offset_y, z = bev_out
        pred_2d, emb_2d = fv_out
        
        # print(f"pred_bev.shape: {pred.shape}")
        # print(f"emb_bev.shape: {emb.shape}")
        # print(f"offset_y_bev.shape: {offset_y.shape}")
        # print(f"pred_2d.shape: {pred_2d.shape}")
        # print(f"emb_2d.shape: {emb_2d.shape}")
        # exit()
        
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


        
        # pred_seg, emb, offset_y, z = bev_out
        
        
        
        # pred = (pred_seg, emb)
        # offset_y = offset_y[0][0]
        # z = z[0][0]
        
        # canvas, ids = embedding_post(pred, self.post_conf, emb_margin=self.post_emb_margin, min_cluster_size=self.post_min_cluster_size, canvas_color=False)
        # lines = bev_instance2points_with_offset_z(canvas, max_x=self.x_range[1], ms_per_pixel=(self.meter_per_pixel, self.meter_per_pixel), offset_y=offset_y, Z=z)
        
        # return 0.0