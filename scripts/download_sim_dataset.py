#!/usr/bin/env python3
"""下载 dyordan1 so101-mujoco sim twin 数据集

默认下载 dobri420/pick-cube-so101-sim (3 相机, LeRobot v2 格式)
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def main():
    from huggingface_hub import snapshot_download

    repo_id = os.environ.get("SIM_DATASET", "dobri420/pick-cube-so101-sim")
    if len(sys.argv) > 1 and sys.argv[1].startswith("--"):
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--repo", default=repo_id)
        ap.add_argument("--output", default="/data/datasets")
        args = ap.parse_args()
        repo_id = args.repo
        dest_root = args.output
    else:
        dest_root = sys.argv[2] if len(sys.argv) > 2 else "/data/datasets"

    dest = Path(dest_root) / repo_id.split("/")[-1]
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {repo_id} -> {dest}")
    snapshot_download(
        repo_id,
        repo_type="dataset",
        local_dir=str(dest),
        endpoint=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
    )

    info_path = dest / "meta" / "info.json"
    if info_path.exists():
        import json
        with open(info_path) as f:
            info = json.load(f)
        print(f"Dataset: {repo_id}")
        print(f"  episodes: {info.get('total_episodes')}")
        print(f"  frames: {info.get('total_frames')}")
        print(f"  fps: {info.get('fps')}")
        print(f"  cameras: {[k for k in info.get('features', {}) if 'images' in k]}")
    else:
        print(f"WARNING: {info_path} not found")

    print(f"Done: {dest}")


if __name__ == "__main__":
    main()
