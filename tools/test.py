# import argparse
# import bevlanedet

# from mmengine.config import Config
# from mmengine.runner import Runner
# from mmengine.registry import init_default_scope


# def parse_args():
#     parser = argparse.ArgumentParser(
#         description='Test BEVLaneDet2'
#     )

#     parser.add_argument(
#         'config',
#         help='Path to the config file'
#     )

#     parser.add_argument(
#         'checkpoint',
#         help='Path to the model checkpoint'
#     )

#     parser.add_argument(
#         '--work-dir',
#         help='Directory to save test results/logs'
#     )

#     return parser.parse_args()


# def main():
#     args = parse_args()

#     # ---------------------------------------------------------
#     # Load config
#     # ---------------------------------------------------------
#     cfg = Config.fromfile(args.config)

#     # ---------------------------------------------------------
#     # Override work directory
#     # ---------------------------------------------------------
#     if args.work_dir is not None:
#         cfg.work_dir = args.work_dir

#     # ---------------------------------------------------------
#     # Load checkpoint
#     # ---------------------------------------------------------
#     cfg.load_from = args.checkpoint

#     # ---------------------------------------------------------
#     # Initialize MMDetection default scope
#     # ---------------------------------------------------------
#     init_default_scope(
#         cfg.get('default_scope', 'mmdet')
#     )

#     # ---------------------------------------------------------
#     # Build runner
#     # ---------------------------------------------------------
#     runner = Runner.from_cfg(cfg)

#     # ---------------------------------------------------------
#     # Run test/evaluation
#     # ---------------------------------------------------------
#     runner.test()


# if __name__ == '__main__':
#     main()






import argparse
import bevlanedet

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.registry import init_default_scope


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