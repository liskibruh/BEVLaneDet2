import torch
import os
import cv2
import matplotlib
import numpy as np
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.registry import HOOKS

@HOOKS.register_module()
class GradientMonitorHook(Hook):
    def __init__(self, interval, layer_groups):
        self.interval = interval
        self.layer_groups = layer_groups
        self.enabled = False
        
        self.handles = []
        self.stats = {}
    
    def _make_grad_hook(self, name, group, param):
        def grad_hook(grad):
            g = grad.detach().float()
            w = param.detach().float()
            
            values = {
                'grad_sq': g.square().sum().item(),
                'grad_abs_sum': g.abs().sum().item(),
                'grad_max': g.abs().max().item(),
                'grad_zero': (g == 0).sum().item(),
                'numel': g.numel(),
                'weight_sq': w.square().sum().item()
            }
            
            if group not in self.stats:
                self.stats[group] = {
                    'grad_sq': 0.0,
                    'grad_abs_sum': 0.0,
                    'grad_max': 0.0,
                    'grad_zero': 0.0,
                    'numel': 0.0,
                    'weight_sq': 0.0
                }
            
            for key, value in values.items():
                self.stats[group][key] += value
            
            return grad

        return grad_hook
        
    
    def before_run(self, runner):
        model = runner.model
        
        if is_model_wrapper(model):
            model = model.module
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            group = 'other'
            for g in self.layer_groups:
                if name.startswith(g):
                    group = g
            
            handle = param.register_hook(
                self._make_grad_hook(name, group, param)
            )
            
            self.handles.append(handle)
    
    def before_train_iter(self, runner, batch_idx, data_batch=None):
        self.enabled = runner.rank != 0 or (runner.iter + 1) % self.interval == 0
        self.stats.clear()
    
    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        
        if not self.enabled:
            return
        
        logger = runner.logger
        
        for group, s in self.stats.items():
            grad_norm = s['grad_sq'] ** 0.5
            weight_norm = s['weight_sq'] ** 0.5
            
            mean_abs = s['grad_abs_sum'] / max(s['numel'], 1)
            zero_fraction = s['grad_zero'] / max(s['numel'], 1)
            ratio = grad_norm / (weight_norm + 1e-12)
            
            logger.info(
                f'{group:32s} | '
                f'grad_norm={grad_norm:.3e} | '
                f'mean_abs={mean_abs:.3e} | '
                f'max={s["grad_max"]:.3e} | '
                f'zero={zero_fraction:.3f} | '
                f'grad/weight={ratio:.3e}'
            )
    
        self.enabled = False
    
    def after_run(self, runner):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()