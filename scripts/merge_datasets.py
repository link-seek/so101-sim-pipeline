#!/usr/bin/env python3
"""Merge multiple LeRobot v3 datasets (identical features) into one, via writer-API replay.

Why replay instead of file surgery: the v3 layout (chunked parquet + per-episode
mp4s + meta stats/tasks) is version-coupled; re-adding decoded frames through
LeRobotDataset.add_frame/save_episode lets the installed writer own the layout.
Used to combine per-leg collection shards (collect.yml matrix legs must NOT
share one output dir: save_dataset() rmtree()s it per leg).
"""
import argparse
import json
import os
import sys
from pathlib import Path

CAM_KEYS = [
    "observation.images.camera1",
    "observation.images.camera2",
    "observation.images.camera3",
]


def _import_lerobot_dataset():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        return LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        return LeRobotDataset


def load_episodes(root, repo_id):
    """Return (features, fps, robot_type, [(frames, task_str), ...]). Frames are decoded numpy."""
    import numpy as np
    import pyarrow.parquet as pq

    LeRobotDataset = _import_lerobot_dataset()
    ds = LeRobotDataset(repo_id=repo_id, root=root)
    meta_dir = Path(root) / "meta"
    tasks_tbl = pq.read_table(meta_dir / "tasks.parquet").to_pylist()
    task_of = {r["task_index"]: r["task"] for r in tasks_tbl}
    feats = json.load(open(meta_dir / "info.json"))["features"]
    fps = json.load(open(meta_dir / "info.json")).get("fps", 20)
    robot_type = json.load(open(meta_dir / "info.json")).get("robot_type", "so_follower")

    episodes = []
    hf = ds.hf_dataset
    epi = ds.episode_data_index
    for ep_idx in range(ds.num_episodes):
        s, e = int(epi["from"][ep_idx]), int(epi["to"][ep_idx])
        frames = []
        task_str = None
        for i in range(s, e):
            row = hf[i]
            if task_str is None:
                task_str = task_of.get(int(row.get("task_index", 0)), "")
            frames.append({k: np.asarray(row[k]) for k in CAM_KEYS + ["observation.state", "action"]})
        episodes.append((frames, task_str or ""))
    return feats, fps, robot_type, episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="comma-separated dataset roots")
    ap.add_argument("--output", required=True)
    ap.add_argument("--repo-id", default="libero_so101_expert")
    args = ap.parse_args()

    roots = [r for r in args.inputs.split(",") if r]
    assert len(roots) >= 2, "need >=2 inputs to merge"
    assert not Path(args.output).exists(), f"output exists (refuse to clobber): {args.output}"

    LeRobotDataset = _import_lerobot_dataset()
    all_eps, feats, fps, robot_type = [], None, None, None
    for r in roots:
        f, fp, rt, eps = load_episodes(r, args.repo_id)
        if feats is None:
            feats, fps, robot_type = f, fp, rt
        else:
            assert set(f) == set(feats), f"feature mismatch: {r}"
        print(f"[merge] {r}: {len(eps)} episodes")
        all_eps.extend(eps)
    print(f"[merge] total {len(all_eps)} episodes -> {args.output}")

    create_feats = {k: {"dtype": v["dtype"], "shape": list(v["shape"]),
                        "names": v.get("names") or (["height", "width", "channel"] if v["dtype"] in ("video", "image") else None)}
                    for k, v in feats.items() if k in CAM_KEYS + ["observation.state", "action"]}
    out = LeRobotDataset.create(
        repo_id=args.repo_id, fps=fps, features=create_feats,
        root=args.output, robot_type=robot_type, use_videos=True,
    )
    for frames, lang in all_eps:
        for fr in frames:
            out.add_frame({
                "observation.images.camera1": fr["observation.images.camera1"],
                "observation.images.camera2": fr["observation.images.camera2"],
                "observation.images.camera3": fr["observation.images.camera3"],
                "observation.state": fr["observation.state"],
                "action": fr["action"],
                "task": lang,
            })
        out.save_episode()
    print(f"[merge] done: {args.output}")

    check = LeRobotDataset(repo_id=args.repo_id, root=args.output)
    assert check.num_episodes == len(all_eps), (check.num_episodes, len(all_eps))
    print(f"[merge] verified: {check.num_episodes} episodes")


if __name__ == "__main__":
    main()
