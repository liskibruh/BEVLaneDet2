import torch
import os
import cv2
import matplotlib
import numpy as np
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.registry import HOOKS

@HOOKS.register_module()
class FeatMapVisHook(Hook):
    def __init__(self,
                 target_layers,
                 interval=50,
                 channel_reduction='squeeze_mean',
                 topk=4,
                 alpha=0.5
                 ):
        self.target_layers = target_layers
        self.interval = interval
        self.channel_reduction = channel_reduction
        self.topk = topk
        self.alpha = alpha  
        
        self._handles = []
        self._captured_features = {}
        
        self.counter = 0
        
    def _get_forward_hook(self, layer_name):
        
        def hook_fn(module, inputs, output):
            feat = output[0] if isinstance(output, (list, tuple)) else output
            self._captured_features[layer_name] = feat[0].detach().cpu()\
                
        return hook_fn
    
    def before_run(self, runner):
        
        model = runner.model.modules() if is_model_wrapper(runner.model) else runner.model
        modules = dict(model.named_modules())
        
        for layer_name in self.target_layers:
            if layer_name not in modules:
                runner.logger.warning(
                    f"[FeatureMapVisHook] Layer {layer_name} not found in model. "
                    f"Available top-level modules: {list(model._modules().keys())}"
                )
                
                continue
                
            handle = modules[layer_name].register_forward_hook(
                self._get_forward_hook(layer_name)
            )
            
            self._handles.append(handle)
    
    def before_train_iter(self, runner, batch_idx, data_batch=None):
        self._captured_features.clear()
    
    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if runner.rank != 0 or (runner.iter + 1) % self.interval !=0:
            return
        
        if not self._captured_features:
            return
        
        overlaid_img = self._extract_img(data_batch)
        
        out_dir = os.path.join(runner.work_dir, 'featmaps_succ_exp')
        sample_dir = os.path.join(out_dir, f"sample{self.counter}")
        os.makedirs(sample_dir, exist_ok=True)
        
        
        for layer_name, feat_tensor in self._captured_features.items():
            drawn_img = self.visualize_activation(
                feat_tensor,
                overlaid_img,
                alpha=self.alpha,
                vmax=5.0
            )
            # drawn_img = runner.visualizer.draw_featmap(
            #     featmap=feat_tensor,
            #     overlaid_image=overlaid_img,
            #     channel_reduction=self.channel_reduction,
            #     topk=self.topk,
            #     alpha=self.alpha
            # )
        
            safe_name = layer_name.replace('.', '_')
            # runner.visualizer.add_image(
            #     name=f"featmap/{safe_name}",
            #     image=drawn_img,
            #     step=runner.iter+1
            # )
            
            path = os.path.join(sample_dir, f"{safe_name}_epoch{runner.epoch}_iter{runner.iter+1}.jpg")
            
            cv2.imwrite(path, drawn_img)
            
            self.counter += 1
                
    def after_run(self, runner):
        
        for handle in self._handles:
            handle.remove()
            
        self._handles.clear()
    
    # @staticmethod    
    # def visualize_activation(feat, overlaid_img=None, alpha=0.5):

    #     # Average across channels
    #     feat = feat.mean(dim=0)

    #     # Apply sigmoid
    #     feat = torch.sigmoid(feat)

    #     # Convert to [0, 255]
    #     feat = (feat.numpy() * 255).astype(np.uint8)

    #     # Resize to input image dimensions
    #     feat = cv2.resize(
    #         feat,
    #         (1024, 576),
    #         interpolation=cv2.INTER_LINEAR
    #     )

    #     # Apply fixed JET colormap
    #     heatmap = cv2.applyColorMap(feat, cv2.COLORMAP_JET)

    #     if overlaid_img is not None:
    #         heatmap = cv2.addWeighted(
    #             overlaid_img, 1 - alpha,
    #             heatmap, alpha,
    #             0
    #         )

    #     return heatmap
    @staticmethod
    def visualize_activation(
        feat,
        overlaid_img=None,
        alpha=0.5,
        vmax=5.0
    ):
        # feat: [C, H, W]

        # Average across channels
        feat = feat.float().mean(dim=0)

        # Fixed normalization (no sigmoid)
        feat = (feat / vmax).clamp(0, 1)

        # Convert to uint8
        feat = (feat.numpy() * 255).astype(np.uint8)

        # Resize to input image dimensions
        feat = cv2.resize(
            feat,
            (1, 1024, 576),
            interpolation=cv2.INTER_LINEAR
        )

        # Apply JET colormap
        heatmap = cv2.applyColorMap(
            feat,
            cv2.COLORMAP_JET
        )

        if overlaid_img is not None:
            heatmap = cv2.addWeighted(
                overlaid_img,
                1 - alpha,
                heatmap,
                alpha,
                0
            )

        return heatmap
        
    @staticmethod    
    def _extract_img(data_batch):
        if not isinstance(data_batch, dict) or 'inputs' not in data_batch:
            print(f"data_batch is not a dict or 'inputs' is not found in data_batch")
        
        inputs = data_batch['inputs']
        
        img_tensor = inputs['img'][0] if isinstance(inputs, dict) else inputs[0]
        
        if not isinstance(img_tensor, torch.Tensor) or img_tensor.ndim != 3:
            return None
        
        img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
        
        if img.dtype != np.uint8:
            img_min, img_max = img.min(), img.max()
            
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min) * 255.0
            
            img = img.astype(np.uint8)
        
        return img[..., ::-1].copy()


@HOOKS.register_module()
class FeatMapVisHookCustom(Hook):
    def __init__(self,
                 target_layers,
                 interval=50,
                 channel_reduction='squeeze_mean',
                 topk=4,
                 alpha=0.5
                 ):
        self.target_layers = target_layers
        self.interval = interval
        self.channel_reduction = channel_reduction
        self.topk = topk
        self.alpha = alpha  
        
        self._handles = []
        self._captured_features = {}
        
        self.counter = 0
        
        self.feature_limits = {
            'backbone.layer3': {
                'signed': 0.45,
                'magnitude': 0.45
            },
            'backbone.layer4': {
                'signed': 2.5,
                'magnitude': 2.5
            },
            'ch_reduce': {
                'signed': 1.5,
                'magnitude': 1.5
            },
            'learned_height_compressor': {
                'signed': 1.0,
                'magnitude': 1.0
            },
            'downscale': {
              'signed': 1.0,
              'magnitude': 1.0  
            },
            'spatial_relation_module': {
                'signed': 0.22,
                'magnitude': 2.2
            },
            'fv_lane_head.fv_up': {
                'signed': 19.0,
                'magnitude': 19.0
            },
            'fv_lane_head.head.neck': {
                'signed': 1.5,
                'magnitude': 1.5
            },
            'fv_lane_head.head.ms': {
                'signed': 23.0,
                'magnitude': 23.0
            },
            'fv_lane_head.head.me': {
                'signed': 8.0,
                'magnitude': 8.5
            },
            'bev_lane_head.bev_up': {
                'signed': 3.8,
                'magnitude': 3.8
            },
            'bev_lane_head.head.neck': {
                'signed': 1.4,
                'magnitude': 1.4
            },
            'bev_lane_head.head.ms': {
                'signed': 12.0,
                'magnitude': 12.0
            },
            'bev_lane_head.head.me': {
                'signed': 20.0,
                'magnitude': 20.5
            },
            'bev_lane_head.head.moff': {
                'signed': 2.5,
                'magnitude': 2.5
            },
            'bev_lane_head.head.mz': {
                'signed': 3.0,
                'magnitude': 3.0
            }
        }
        
    def _get_forward_hook(self, layer_name):
        
        def hook_fn(module, inputs, output):
            feat = output[0] if isinstance(output, (list, tuple)) else output
            self._captured_features[layer_name] = feat[0].detach().cpu()\
                
        return hook_fn
    
    def before_run(self, runner):
        
        model = runner.model.modules() if is_model_wrapper(runner.model) else runner.model
        modules = dict(model.named_modules())
        
        for layer_name in self.target_layers:
            if layer_name not in modules:
                runner.logger.warning(
                    f"[FeatureMapVisHook] Layer {layer_name} not found in model. "
                    f"Available top-level modules: {list(modules.keys())}"
                )
                
                continue
                
            handle = modules[layer_name].register_forward_hook(
                self._get_forward_hook(layer_name)
            )
            
            self._handles.append(handle)
    
    def before_train_iter(self, runner, batch_idx, data_batch=None):
        self._captured_features.clear()
    
    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if runner.rank != 0 or (runner.iter + 1) % self.interval !=0:
            return
        
        if not self._captured_features:
            return
        
        overlaid_img = self._extract_img(data_batch)

        out_dir = os.path.join(
            runner.work_dir,
            'featmaps_r34_openlane'
        )

        sample_dir = os.path.join(
            out_dir,
            f'epoch{runner.epoch + 1}_iter{runner.iter + 1}'
        )

        os.makedirs(sample_dir, exist_ok=True)

        for layer_name, feat_tensor in self._captured_features.items():
            
            print(f"\nLayer: {layer_name}")
            self.get_activation_stats(feat_tensor)

            is_bev = any(
                key in layer_name.lower()
                for key in ('bev', 'height_compressor', 'spatial_relation', 'downscale')
            )

            if is_bev:
                overlay = np.full(
                    (200, 48, 3),
                    255,
                    dtype=np.uint8
                )
            else:
                overlay = overlaid_img

            signed_map, magnitude_map = self.reduce_feature(feat_tensor)

            limits = self.feature_limits[layer_name]

            signed_img = self.visualize_activation(
                signed_map,
                limit=limits['signed'],
                signed=True,
                overlaid_img=overlay,
                alpha=self.alpha
            )

            magnitude_img = self.visualize_activation(
                magnitude_map,
                limit=limits['magnitude'],
                signed=False,
                overlaid_img=overlay,
                alpha=self.alpha
            )

            safe_name = layer_name.replace('.', '_')

            cv2.imwrite(
                os.path.join(sample_dir, f'{safe_name}_signed.png'),
                signed_img
            )

            cv2.imwrite(
                os.path.join(sample_dir, f'{safe_name}_magnitude.png'),
                magnitude_img
            )
                
    def after_run(self, runner):
        
        for handle in self._handles:
            handle.remove()
            
        self._handles.clear()
    
    @staticmethod
    def reduce_feature(feat):
        """
        Args:
            feat: [C, H, W] or [B, C, H, W]

        Returns:
            signed:    [H, W] signed channel mean
            magnitude: [H, W] mean absolute activation
        """

        if isinstance(feat, (tuple, list)):
            feat = feat[0]

        if feat.ndim == 4:
            feat = feat[0]

        if feat.ndim != 3:
            raise ValueError(
                f"Expected [C,H,W] or [B,C,H,W], got {feat.shape}"
            )

        feat = feat.detach().float().cpu()

        signed = feat.mean(dim=0)
        magnitude = feat.abs().mean(dim=0)

        return (
            signed.numpy(),
            magnitude.numpy()
        )


    @staticmethod
    def visualize_activation(
        spatial_map,
        limit,
        signed=False,
        overlaid_img=None,
        alpha=0.5
    ):
        """
        Visualizes a spatial activation map using a fixed range.

        signed=True:
            Fixed symmetric range [-limit, +limit].
            Coolwarm colormap.

        signed=False:
            Fixed range [0, limit].
            Viridis colormap.
        """

        if limit <= 0:
            raise ValueError(f"Invalid visualization limit: {limit}")

        spatial_map = np.nan_to_num(
            spatial_map,
            nan=0.0,
            posinf=limit,
            neginf=-limit
        )

        if signed:
            normalized = np.clip(
                spatial_map / limit,
                -1.0,
                1.0
            )

            normalized = (normalized + 1.0) / 2.0

            rgb = (
                matplotlib.colormaps['coolwarm'](normalized)[..., :3]
                * 255
            ).astype(np.uint8)

            heatmap = cv2.cvtColor(
                rgb,
                cv2.COLOR_RGB2BGR
            )

        else:
            normalized = np.clip(
                spatial_map / limit,
                0.0,
                1.0
            )

            gray = (normalized * 255).astype(np.uint8)

            heatmap = cv2.applyColorMap(
                gray,
                cv2.COLORMAP_VIRIDIS
            )

        if overlaid_img is not None:

            heatmap = cv2.resize(
                heatmap,
                (overlaid_img.shape[1], overlaid_img.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

            heatmap = cv2.addWeighted(
                overlaid_img,
                1 - alpha,
                heatmap,
                alpha,
                0
            )

        return heatmap
    
    @staticmethod
    def get_activation_stats(feat):

        feat = feat.detach().float().cpu()

        signed = feat.mean(dim=1) if feat.ndim == 4 else feat.mean(dim=0)
        magnitude = feat.abs().mean(dim=1) if feat.ndim == 4 else feat.abs().mean(dim=0)

        for name, tensor in [
            ('Signed', signed.abs()),
            ('Magnitude', magnitude)
        ]:
            values = tensor.flatten()

            quantiles = torch.quantile(
                values,
                torch.tensor([0.50, 0.90, 0.95, 0.99, 0.995, 0.999])
            )

            print(
                name,
                dict(zip(
                    ['50%', '90%', '95%', '99%', '99.5%', '99.9%'],
                    quantiles.tolist()
                ))
            )
        
    @staticmethod    
    def _extract_img(data_batch):
        if not isinstance(data_batch, dict) or 'inputs' not in data_batch:
            print(f"data_batch is not a dict or 'inputs' is not found in data_batch")
        
        inputs = data_batch['inputs']
        
        img_tensor = inputs['img'][0] if isinstance(inputs, dict) else inputs[0]
        
        if not isinstance(img_tensor, torch.Tensor) or img_tensor.ndim != 3:
            return None
        
        img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
        
        if img.dtype != np.uint8:
            img_min, img_max = img.min(), img.max()
            
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min) * 255.0
            
            img = img.astype(np.uint8)
        
        return img[..., ::-1].copy()