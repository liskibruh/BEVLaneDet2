import torch
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.registry import HOOKS

@HOOKS.register_module()
class GradientMonitorHook(Hook):
    def __init__(self, iter_interval, epoch_interval, layer_groups):
        self.iter_interval = iter_interval
        self.epoch_interval = epoch_interval
        self.layer_groups = layer_groups
        self.iter_enabled = False
        self.epoch_enabled = False
        
        self.handles = []
        self.stats = {}
        self.all_stats = {}
        
        self.ylimit2metric = {
            'all_grad_norms': (0, 90),
            'all_mean_abs': (0, 0.1),
            'all_zero_fractions': (0, 0.2),
            'all_ratios': (0, 2)
        }
        
        for k in layer_groups:
            self.all_stats[k] = {
                'all_xs': [],
                'all_grad_norms': [],
                'all_mean_abs': [],
                'all_zero_fractions': [],
                'all_ratios': []
            }
    
    def _make_grad_hook(self, name, group, param):
        def grad_hook(grad):
            
            if not self.iter_enabled:
                return grad
            
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
    def before_train_epoch(self, runner):
        self.epoch_enabled = runner.rank ==0 and (runner.epoch + 1) % self.epoch_interval == 0
    
    def before_train_iter(self, runner, batch_idx, data_batch=None):
        self.iter_enabled = runner.rank == 0 and (runner.iter + 1) % self.iter_interval == 0
        self.stats.clear()
    
    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        
        if not self.iter_enabled:
            return
        
        logger = runner.logger
        
        for group, s in self.stats.items():
            grad_norm = s['grad_sq'] ** 0.5
            weight_norm = s['weight_sq'] ** 0.5
            
            mean_abs = s['grad_abs_sum'] / max(s['numel'], 1)
            zero_fraction = s['grad_zero'] / max(s['numel'], 1)
            ratio = grad_norm / (weight_norm + 1e-12)
            
            if "other" not in group:
                self.all_stats[group]['all_xs'].append(runner.iter + 1)
                self.all_stats[group]['all_grad_norms'].append(grad_norm)
                self.all_stats[group]['all_mean_abs'].append(mean_abs)
                # self.all_stats[group]['all_max'].append(s['gra'])
                self.all_stats[group]['all_zero_fractions'].append(zero_fraction)
                self.all_stats[group]['all_ratios'].append(ratio)
            
            logger.info(
                f'{group:32s} | '
                f'grad_norm={grad_norm:.3e} | '
                f'mean_abs={mean_abs:.3e} | '
                f'max={s["grad_max"]:.3e} | '
                f'zero={zero_fraction:.3f} | '
                f'grad/weight={ratio:.3e}'
            )
        
        self.iter_enabled = False
    
    def after_train_epoch(self, runner):
        
        if not self.epoch_enabled:
            return
        
        out_dir = os.path.join(
            runner.work_dir,
            'grads_r34_exp'
        )
        
        sample_dir = os.path.join(
            out_dir,
            f'epoch{runner.epoch + 1}'
        )
        
        os.makedirs(sample_dir, exist_ok=True)
        safe_name = f'grad_stats_plot'
        
        
        for stat_name in self.all_stats[self.layer_groups[0]].keys():
            
            if stat_name == 'all_xs':
                continue
            
            plt.figure(figsize=(25, 16))
            
            for group, stat in self.all_stats.items():
                xs = stat['all_xs']
                ys = stat[stat_name]

                if xs and ys:
                    plt.plot(xs, ys, label=group)
            
            y_low, y_up = self.ylimit2metric[stat_name]
            plt.ylim(y_low, y_up)
            plt.title(f"{stat_name.replace('all_', '')} vs. Train Iters")
            plt.xlabel("train iters")
            plt.ylabel(f"{stat_name.replace('all_', '')}")
            plt.legend()
            plt.grid(True)
            
            plt.savefig(os.path.join(sample_dir, f"{safe_name}_{stat_name.replace('all_', '')}.jpg"), dpi=300, bbox_inches='tight')
            
            plt.close()
    
    def after_run(self, runner):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.all_stats.clear()
        
        