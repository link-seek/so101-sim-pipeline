#!/usr/bin/env python3
"""SmolVLA 训练脚本 - 使用公开数据集微调 SmolVLA base 模型"""

import argparse
import os
import subprocess
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("MUJOCO_GL", "egl")


def main():
    parser = argparse.ArgumentParser(description="Train SmolVLA on SO101 public dataset")
    parser.add_argument("--policy.path", default="lerobot/smolvla_base",
                        dest="policy_path", help="Pretrained model path")
    parser.add_argument("--dataset.repo_id", default="lerobot/svla_so101_pickplace",
                        dest="dataset_repo_id", help="Dataset repo ID")
    parser.add_argument("--dataset.fps", type=int, default=None,
                        dest="dataset_fps", help="Dataset FPS (unused, kept for compatibility)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--save_freq", type=int, default=5000)
    parser.add_argument("--env_eval_freq", type=int, default=2000)
    parser.add_argument("--log_freq", type=int, default=500)
    parser.add_argument("--output_dir", default="/data/checkpoints/smolvla")
    parser.add_argument("--job_name", default="smolvla-pickplace-20k")
    parser.add_argument("--policy.repo_id", default="xieyucheng123/so101-smolvla",
                        dest="policy_repo_id", help="HF Hub repo to push checkpoint")
    parser.add_argument("--policy.push_to_hub", default="true",
                        dest="push_to_hub", help="Push to HF Hub")
    parser.add_argument("--wandb.enable", default="true", dest="wandb_enable")
    parser.add_argument("--wandb.project", default="so101-smolvla", dest="wandb_project")
    parser.add_argument("--rename_map", default=None,
                        help='Camera rename map JSON, e.g. {"observation.images.side":"observation.images.camera1"}')
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    cmd = [
        "lerobot-train",
        f"--policy.path={args.policy_path}",
        f"--policy.device=cuda",
        f"--policy.repo_id={args.policy_repo_id}",
        f"--policy.push_to_hub={args.push_to_hub}",
        f"--dataset.repo_id={args.dataset_repo_id}",
        f"--batch_size={args.batch_size}",
        f"--steps={args.steps}",
        f"--save_freq={args.save_freq}",
        f"--env_eval_freq={args.env_eval_freq}",
        f"--log_freq={args.log_freq}",
        f"--output_dir={args.output_dir}",
        f"--job_name={args.job_name}",
        f"--wandb.enable={args.wandb_enable}",
        f"--wandb.project={args.wandb_project}",
    ]

    if args.rename_map:
        cmd.append(f"--rename_map={args.rename_map}")

    print(f"=== SmolVLA Training ===")
    print(f"Policy: {args.policy_path}")
    print(f"Dataset: {args.dataset_repo_id}")
    print(f"Batch size: {args.batch_size}")
    print(f"Steps: {args.steps}")
    print(f"Output: {args.output_dir}")
    print(f"Push to: {args.policy_repo_id}")
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Training failed with return code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n=== Training Complete ===")
    print(f"Checkpoints saved to: {args.output_dir}")
    print(f"Model pushed to: https://huggingface.co/{args.policy_repo_id}")


if __name__ == "__main__":
    main()
