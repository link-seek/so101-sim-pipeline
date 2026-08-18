#!/usr/bin/env python3
"""so101-mujoco Sim Twin 评测脚本

封装 dyordan1/so101-mujoco 的 mujoco_policy.py --sweep，
适配流水线的输入输出格式。

两种模式:
  --sweep:  grid sweep 评测 (5 reach × 13 azim × 5 trials = 325 episodes)
  --record: 录制几个代表性放置的 MP4 视频
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

RESULTS_DIR = Path("/data/eval/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DYORDAN1_ROOT = Path("/workspace/so101-mujoco")


def find_checkpoint(checkpoint_arg):
    if checkpoint_arg and Path(checkpoint_arg).exists():
        return checkpoint_arg
    ckpt_dir = Path("/data/checkpoints/smolvla-sim/checkpoints")
    if ckpt_dir.exists():
        ckpts = sorted(ckpt_dir.glob("*/pretrained_model"), key=lambda p: p.name)
        if ckpts:
            return str(ckpts[-1])
    if checkpoint_arg:
        return checkpoint_arg
    print("ERROR: No checkpoint found")
    sys.exit(1)


def run_sweep(checkpoint, seconds=20.0, output_dir=None):
    print(f"\n=== MuJoCo Grid Sweep ===")
    print(f"Checkpoint: {checkpoint}")
    print(f"Seconds per episode: {seconds}")

    output_dir = output_dir or str(RESULTS_DIR)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["DATASETS_DIR"] = "/data/datasets"

    cmd = [
        sys.executable, str(DYORDAN1_ROOT / "mujoco_policy.py"),
        checkpoint,
        "--sweep",
        f"--seconds={seconds}",
    ]

    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(
        cmd, cwd=str(DYORDAN1_ROOT), env=env, capture_output=True, text=True
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-2000:])

    success_rate = 0.0
    for line in result.stdout.splitlines():
        if line.startswith("SUCCESS") and "=" in line:
            parts = line.split("=")
            if len(parts) >= 2:
                success_rate = float(parts[1].strip().rstrip("%")) / 100.0
            break

    report = {
        "checkpoint": checkpoint,
        "mode": "sweep",
        "success_rate": success_rate,
        "stdout": result.stdout[-5000:],
        "returncode": result.returncode,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    report_path = Path(output_dir) / "mujoco_sweep_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {report_path}")
    print(f"Success rate: {success_rate:.1%}")

    return success_rate


def run_record(checkpoint, seconds=20.0, output_dir=None):
    print(f"\n=== MuJoCo Record ===")
    print(f"Checkpoint: {checkpoint}")

    output_dir = output_dir or str(RESULTS_DIR / "videos")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["DATASETS_DIR"] = "/data/datasets"

    cmd = [
        sys.executable, str(DYORDAN1_ROOT / "mujoco_policy.py"),
        checkpoint,
        "--record", output_dir,
        f"--seconds={seconds}",
    ]

    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(DYORDAN1_ROOT), env=env, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-2000:])

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="so101-mujoco sim twin evaluation")
    parser.add_argument("--checkpoint", default="", help="Checkpoint path")
    parser.add_argument("--mode", choices=["sweep", "record"], default="sweep")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--output", default=str(RESULTS_DIR))
    args = parser.parse_args()

    checkpoint = find_checkpoint(args.checkpoint)

    if args.mode == "sweep":
        rate = run_sweep(checkpoint, args.seconds, args.output)
        print(f"\nFinal: {'SUCCESS' if rate > 0.5 else 'FAILED'} ({rate:.1%})")
    else:
        run_record(checkpoint, args.seconds, args.output)


if __name__ == "__main__":
    main()
