import argparse
import bevlanedet
import torch

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.registry import init_default_scope
from mmengine.runner import checkpoint as mmengine_checkpoint

_original_load_from_local = (
    mmengine_checkpoint.CheckpointLoader._schemes['']
)

def load_from_local_compat(filename, map_location=None):
    return torch.load(
        filename,
        map_location=map_location,
        weights_only=False,
    )

mmengine_checkpoint.CheckpointLoader.register_scheme(
    prefixes='',
    loader=load_from_local_compat,
    force=True,
)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Debug BEVLaneDet evaluation pipeline'
    )

    parser.add_argument(
        'config',
        help='Path to the config file'
    )

    parser.add_argument(
        '--work-dir',
        help='Directory to save logs/results'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    cfg = Config.fromfile(args.config)

    # Optional work directory override
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    # Initialize MMDetection scope
    init_default_scope(
        cfg.get('default_scope', 'mmdet')
    )

    # Build runner
    runner = Runner.from_cfg(cfg)

    # Run evaluation
    runner.test()


if __name__ == '__main__':
    main()