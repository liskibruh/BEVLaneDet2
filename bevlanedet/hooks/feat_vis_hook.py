import torch
import os
import cv2
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
        
        os.makedirs(out_dir, exist_ok=True)
        
        for layer_name, feat_tensor in self._captured_features.items():
            drawn_img = self.visualize_activation(
                feat_tensor,
                overlaid_img,
                alpha=self.alpha
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
            
            path = os.path.join(out_dir, f"{safe_name}_epoch{runner.epoch}_iter{runner.iter+1}.jpg")
            
            cv2.imwrite(path, drawn_img)
                
    def after_run(self, runner):
        
        for handle in self._handles:
            handle.remove()
            
        self._handles.clear()
    
    @staticmethod    
    def visualize_activation(feat, overlaid_img=None, alpha=0.5):

        # Average across channels
        feat = feat.mean(dim=0)

        # Apply sigmoid
        feat = torch.sigmoid(feat)

        # Convert to [0, 255]
        feat = (feat.numpy() * 255).astype(np.uint8)

        # Resize to input image dimensions
        feat = cv2.resize(
            feat,
            (1024, 576),
            interpolation=cv2.INTER_LINEAR
        )

        # Apply fixed JET colormap
        heatmap = cv2.applyColorMap(feat, cv2.COLORMAP_JET)

        if overlaid_img is not None:
            heatmap = cv2.addWeighted(
                overlaid_img, 1 - alpha,
                heatmap, alpha,
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