# import argparse

# from mmengine.config import Config
# from mmengine.runner import Runner
# from mmengine.registry import DefaultScope
# from mmdet.registry import MODELS


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('config')
#     return parser.parse_args()


# def main():
#     args = parse_args()

#     cfg = Config.fromfile(args.config)

#     print('\n========== DEBUG ==========')

#     # Check that custom_imports actually imported your package
#     import bevlanedet

#     print('bevlanedet:', bevlanedet)
#     print('BEVLaneDet class:', bevlanedet.BEVLaneDet)

#     # Check the mmdet registry
#     print('MODELS registry:', MODELS)
#     print('BEVLaneDet in MODELS:', MODELS.get('BEVLaneDet'))

#     # Check current scope
#     current_scope = DefaultScope.get_current_instance()
#     print('DefaultScope:', current_scope)

#     if current_scope is not None:
#         print('Scope name:', current_scope.scope_name)

#     print('cfg.default_scope:', cfg.get('default_scope', None))
#     print('cfg.custom_imports:', cfg.get('custom_imports', None))

#     print('===========================\n')

#     runner = Runner.from_cfg(cfg)
#     runner.train()


# if __name__ == '__main__':
#     main()


import argparse
import bevlanedet
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.registry import init_default_scope


def parse_args():
    parser = argparse.ArgumentParser(description='Train BEVLaneDet2')
    parser.add_argument(
        'config',
        help='Path to the training config file')
    parser.add_argument(
        '--work-dir',
        help='Directory to save logs and checkpoints')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from the latest checkpoint')
    parser.add_argument(
        '--amp',
        action='store_true',
        help='Enable AMP training')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    cfg = Config.fromfile(args.config)

    # Override work directory from command line
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    # Enable AMP if requested
    if args.amp:
        if 'optim_wrapper' not in cfg:
            cfg.optim_wrapper = dict(
                type='AmpOptimWrapper',
                loss_scale='dynamic'
            )
        else:
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'
            
    init_default_scope(cfg.get('default_scope', 'mmdet'))

    # Build runner from config
    runner = Runner.from_cfg(cfg)

    # Resume from latest checkpoint if requested
    if args.resume:
        runner.resume()

    # Start training
    runner.train()


if __name__ == '__main__':
    main()