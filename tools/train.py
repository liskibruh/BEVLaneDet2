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