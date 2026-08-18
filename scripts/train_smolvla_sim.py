#!/usr/bin/env python3
"""SmolVLA 训练脚本 - so101-mujoco sim twin 数据集

与 train_smolvla.py 的区别:
  - 无 rename_map (3 相机原生匹配 camera1/camera2/camera3)
  - 默认数据集 dobri420/pick-cube-so101-sim
  - 使用 lerobot-train 原生命令 (无需 patch)
"""

import argparse
import os
import subprocess
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("MUJOCO_GL", "egl")


def main():
    parser = argparse.ArgumentParser(description="Train SmolVLA on so101-mujoco sim twin")
    parser.add_argument("--dataset.repo_id", default="dobri420/pick-cube-so101-sim",
                        dest="dataset_repo_id", help="Dataset repo ID")
    parser.add_argument("--dataset.root", default=None,
                        dest="dataset_root", help="Local dataset root")
    parser.add_argument("--policy.path", default="lerobot/smolvla_base",
                        dest="policy_path", help="Pretrained model path")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--save_freq", type=int, default=5000)
    parser.add_argument("--log_freq", type=int, default=500)
    parser.add_argument("--output_dir", default="/data/checkpoints/smolvla-sim")
    parser.add_argument("--job_name", default="smolvla-sim-twin")
    parser.add_argument("--policy.repo_id", default="xieyucheng123/so101-smolvla-sim",
                        dest="policy_repo_id", help="HF Hub repo to push checkpoint")
    parser.add_argument("--policy.push_to_hub", default="false",
                        dest="push_to_hub", help="Push to HF Hub")
    parser.add_argument("--wandb.enable", default="false", dest="wandb_enable")
    parser.add_argument("--wandb.project", default="so101-smolvla-sim", dest="wandb_project")
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    cmd = [
        "python", "/workspace/scripts/lerobot_train_patched.py",
        f"--policy.path={args.policy_path}",
        f"--policy.device=cuda",
        f"--policy.repo_id={args.policy_repo_id}",
        f"--policy.push_to_hub={args.push_to_hub}",
        f"--dataset.repo_id={args.dataset_repo_id}",
        f"--batch_size={args.batch_size}",
        f"--steps={args.steps}",
        f"--save_freq={args.save_freq}",
        f"--log_freq={args.log_freq}",
        f"--output_dir={args.output_dir}",
        f"--job_name={args.job_name}",
        f"--wandb.enable={args.wandb_enable}",
        f"--wandb.project={args.wandb_project}",
    ]

    if args.dataset_root:
        cmd.append(f"--dataset.root={args.dataset_root}")

    print(f"=== SmolVLA Sim Twin Training ===")
    print(f"Policy: {args.policy_path}")
    print(f"Dataset: {args.dataset_repo_id}")
    print(f"Batch size: {args.batch_size}")
    print(f"Steps: {args.steps}")
    print(f"Output: {args.output_dir}")
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Training failed with return code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n=== Training Complete ===")
    print(f"Checkpoints saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
