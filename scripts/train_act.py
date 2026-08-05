#!/usr/bin/env python3
"""ACT 策略训练 - 使用 LeRobot 训练框架"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-repo", default=os.environ.get("DATASET_REPO", "xieyucheng123/so101-sim-picklift"))
    parser.add_argument("--dataset-root", default="/data/pipeline/dataset")
    parser.add_argument("--output-dir", default="/data/pipeline/training")
    parser.add_argument("--train-steps", type=int, default=int(os.environ.get("TRAIN_STEPS", "10000")))
    parser.add_argument("--save-freq", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    import shutil
    shutil.rmtree(args.output_dir, ignore_errors=True)

    if args.hf_token:
        Path("~/.huggingface").expanduser().mkdir(exist_ok=True)
        Path("~/.cache/huggingface/token").expanduser().parent.mkdir(parents=True, exist_ok=True)
        Path("~/.cache/huggingface/token").expanduser().write_text(args.hf_token)

    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        "--policy.type=act",
        f"--dataset.repo_id={args.dataset_repo}",
        f"--dataset.root={args.dataset_root}",
        f"--output_dir={args.output_dir}",
        f"--steps={args.train_steps}",
        f"--save_freq={args.save_freq}",
        f"--batch_size={args.batch_size}",
        f"--policy.optimizer_lr={args.learning_rate}",
        "--policy.device=cuda",
        "--policy.use_amp=false",
        "--policy.push_to_hub=false",
        "--wandb.enable=false",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"Training failed with code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n=== Training Complete ===")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
