#!/usr/bin/env python3
"""Create and save an untrained policy whose I/O dimensions match a TOML config.

The saved directory can be used directly as `pretrained_name_or_path` in a
[policies.<name>] TOML block with `ds_repo_id` omitted, so the policy
controller derives feature shapes from the TOML topics instead of a dataset.

Example
-------
    python3 -m lerobot_ros.init_benchmark_policy \\
        --config robot.toml \\
        --policy-type act \\
        --output ./benchmark_policies/act_stub

Then add to robot.toml::

    [policies.benchmark_act]
    pretrained_name_or_path = "./benchmark_policies/act_stub"
    device = "cuda"
    # ds_repo_id omitted → features derived from [topics] in this config

And run the policy controller with::

    ros2 run lerobot_ros policy_controller --ros-args \\
        -p config:=robot.toml -p benchmark:=true
"""

import argparse
import os
import sys
import types

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

from lerobot_ros.config import load_toml_dict, parse_config
from lerobot_ros.core.frame_assembler import FrameAssembler

POLICY_TYPES = {
    "act": ACTConfig,
    "smolvla": SmolVLAConfig,
}


def _build_mock_meta(config):
    """Return a minimal ds_meta-like namespace from the TOML config features."""
    features = FrameAssembler(config.topics).get_feature_description()
    return types.SimpleNamespace(features=features, stats=None)


def main():
    parser = argparse.ArgumentParser(
        description="Instantiate an untrained policy matching TOML config features and save it to disk."
    )
    parser.add_argument("--config", required=True, help="Path to the TOML configuration file.")
    parser.add_argument(
        "--policy-type",
        required=True,
        choices=list(POLICY_TYPES),
        help="Policy architecture to instantiate.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory where the policy and processors will be saved.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device used when instantiating the policy (default: cpu).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: config file '{args.config}' not found.", file=sys.stderr)
        sys.exit(1)

    config = parse_config(load_toml_dict(args.config))
    mock_meta = _build_mock_meta(config)

    features = mock_meta.features
    print("Features derived from TOML config:")
    for name, desc in features.items():
        print(f"  {name}: dtype={desc['dtype']}, shape={desc['shape']}")

    # Build policy config with the target device; no pretrained weights.
    policy_cfg_cls = POLICY_TYPES[args.policy_type]
    policy_cfg = policy_cfg_cls(device=args.device)

    print(f"\nInstantiating untrained {args.policy_type.upper()} policy on {args.device}...")
    policy = make_policy(policy_cfg, ds_meta=mock_meta)
    policy.eval()

    print("Creating processors (identity normalization, no dataset stats)...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=None,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    output = os.path.abspath(args.output)
    os.makedirs(output, exist_ok=True)

    print(f"Saving to '{output}'...")
    policy.save_pretrained(output)
    preprocessor.save_pretrained(output)
    postprocessor.save_pretrained(output)

    print("\nDone. Add this block to your TOML config:")
    print()
    chunk = policy_cfg.chunk_size
    print(f'[policies.benchmark_{args.policy_type}]')
    print(f'pretrained_name_or_path = "{output}"')
    print(f'device = "{args.device}"')
    print(f'action_queue_size = {int(chunk * 0.8)}  # trigger re-infer at 80% of chunk_size={chunk}')
    print('# ds_repo_id is omitted — feature shapes are derived from [topics]')
    print()
    print("Then run the policy controller with:")
    print('  ros2 run lerobot_ros policy_controller --ros-args \\')
    print('      -p config:=<your_config>.toml -p benchmark:=true')


if __name__ == "__main__":
    main()
