#!/usr/bin/env python3
"""验证 ataghof/so101nexus-cube500-binary 数据集"""

import json
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np

def main():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    repo_id = "ataghof/so101nexus-cube500-binary"
    print(f"Loading dataset: {repo_id}")

    ds = LeRobotDataset(repo_id, root="/data/datasets/ataghof")
    print(f"Dataset loaded: {len(ds)} frames")
    print(f"Features: {list(ds.features.keys())}")

    # Check meta/stats
    meta_path = os.path.join(ds.root, "meta", "stats.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            stats = json.load(f)
        print(f"\n=== Action Stats ===")
        for key, val in stats.items():
            if "action" in key:
                print(f"{key}: {json.dumps(val, indent=2)[:500]}")

    # Check first frame
    frame = ds[0]
    print(f"\n=== Frame 0 keys: {list(frame.keys())} ===")

    for key in frame:
        val = frame[key]
        if hasattr(val, "shape"):
            print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
            if "image" in key.lower():
                import cv2
                fname = f"/data/eval/cam_check/{key.replace('.', '_')}_frame0.png"
                os.makedirs(os.path.dirname(fname), exist_ok=True)
                img = val.numpy() if hasattr(val, "numpy") else np.array(val)
                if img.dtype != np.uint8:
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                cv2.imwrite(fname, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                print(f"    -> saved to {fname}")
        else:
            print(f"  {key}: {val}")

    # Check action range across first 100 frames
    actions = []
    for i in range(min(100, len(ds))):
        f = ds[i]
        for key in f:
            if "action" in key.lower() and hasattr(f[key], "shape"):
                actions.append(f[key].numpy() if hasattr(f[key], "numpy") else np.array(f[key]))

    if actions:
        actions = np.stack(actions)
        print(f"\n=== Action range (first 100 frames) ===")
        print(f"  shape: {actions.shape}")
        print(f"  min: {actions.min(axis=0)}")
        print(f"  max: {actions.max(axis=0)}")
        print(f"  mean: {actions.mean(axis=0)}")
        print(f"  std: {actions.std(axis=0)}")

    # Check fps
    print(f"\n=== Dataset meta ===")
    print(f"  fps: {ds.fps}")
    print(f"  num_episodes: {ds.num_episodes}")

    print("\n=== Verification complete ===")

if __name__ == "__main__":
    main()
