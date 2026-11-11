#!/usr/bin/env python3
"""Merge multiple LeRobot v3 datasets (identical features) into one.

Reader path avoids LeRobotDataset(...) open (it phones home to the Hub for
repo refs, which fails for local-only repo_ids). Reads the v3 layout directly:
data parquet + per-frame PNGs + meta parquet/json, then replays through
LeRobotDataset.create/add_frame/save_episode (proven offline path).
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


CAM_KEYS = [
    "observation.images.camera1",
    "observation.images.camera2",
    "observation.images.camera3",
]
STATE_KEY = "observation.state"
ACTION_KEY = "action"


def _read_video_frames(root, cam, ep):
    """Fallback: decode episode mp4 (pyav). Probes common v3 layouts."""
    import av
    vdir = root / "videos" / cam
    cands = sorted(vdir.rglob(f"*episode-{ep:06d}*.mp4")) or sorted(vdir.rglob("*.mp4"))
    assert cands, f"no mp4 for {cam} ep {ep} under {vdir}"
    path = cands[ep % len(cands)] if len(cands) > 1 else cands[0]
    print(f"    [video-fallback] {cam} ep {ep}: {path}")
    frames = []
    with av.open(str(path)) as cont:
        for frame in cont.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    assert frames, f"empty video {path}"
    return frames


def read_dataset(root):
    root = Path(root)
    info = json.load(open(root / "meta" / "info.json"))
    feats = info["features"]
    fps = info.get("fps", 20)
    robot_type = info.get("robot_type", "so_follower")
    tasks = {r["task_index"]: r["task"] for r in pq.read_table(root / "meta" / "tasks.parquet").to_pylist()}

    data_files = sorted(glob.glob(str(root / "data" / "chunk-*" / "*.parquet")))
    assert data_files, f"no data parquet in {root}"
    table = None
    for f in data_files:
        t = pq.read_table(f)
        table = t if table is None else __import__("pyarrow").concat_tables([table, t])
    ep_idx = np.asarray(table.column("episode_index").to_pylist())
    episodes = []
    for ep in sorted(set(ep_idx.tolist())):
        mask = ep_idx == ep
        idxs = np.where(mask)[0]
        task_idx = int(table.column("task_index")[idxs[0]].as_py())
        import glob as _glob
        global _PROBED
        try:
            _PROBED
        except NameError:
            _PROBED = set()
        if str(root) not in _PROBED:
            _PROBED.add(str(root))
            print(f"[merge] layout probe for {root}:")
            for probe in sorted(_glob.glob(str(root / "images" / "*" / "*")))[:8]:
                print(f"    probe: {probe}")
            for probe in sorted(_glob.glob(str(root / "videos" / "*" / "*" / "*")))[:8]:
                print(f"    probe: {probe}")
        frames = []
        ep_dir = f"episode-{ep:06d}"
        for cam in CAM_KEYS:
            img_dir = root / "images" / cam / ep_dir
            pngs = sorted(img_dir.glob("frame-*.png")) or sorted(img_dir.glob("*.png"))
            if pngs:
                arrs = [np.asarray(Image.open(p).convert("RGB")) for p in pngs]
            else:
                arrs = _read_video_frames(root, cam, ep)
            frames.append(arrs)
        n = len(frames[0])
        assert all(len(a) == n for a in frames), "camera frame counts differ"
        states = np.asarray(table.column(STATE_KEY).to_pylist())[mask]
        acts = np.asarray(table.column(ACTION_KEY).to_pylist())[mask]
        assert len(states) == n and len(acts) == n, (len(states), len(acts), n)
        ep_frames = []
        for t in range(n):
            ep_frames.append({
                "camera1": frames[0][t], "camera2": frames[1][t], "camera3": frames[2][t],
                "state": np.asarray(states[t], dtype=np.float32),
            })
        episodes.append((ep_frames, np.asarray(acts, dtype=np.float32), tasks.get(task_idx, "")))
    return feats, fps, robot_type, episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--repo-id", default="libero_so101_expert")
    args = ap.parse_args()
    roots = [r for r in args.inputs.split(",") if r]
    assert len(roots) >= 2
    assert not Path(args.output).exists(), f"refuse to clobber {args.output}"

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    all_eps, feats, fps, robot_type = [], None, None, None
    for r in roots:
        f, fp, rt, eps = read_dataset(r)
        if feats is None:
            feats, fps, robot_type = f, fp, rt
        print(f"[merge] {r}: {len(eps)} episodes")
        all_eps.extend(eps)
    print(f"[merge] total {len(all_eps)} episodes -> {args.output}")

    create_feats = {k: {"dtype": v["dtype"], "shape": list(v["shape"]),
                        "names": v.get("names") or (["height", "width", "channel"] if v["dtype"] in ("video", "image") else None)}
                    for k, v in feats.items() if k in CAM_KEYS + [STATE_KEY, ACTION_KEY]}
    out = LeRobotDataset.create(
        repo_id=args.repo_id, fps=fps, features=create_feats,
        root=args.output, robot_type=robot_type, use_videos=True,
    )
    for frames, acts, lang in all_eps:
        for t in range(len(acts)):
            out.add_frame({
                "observation.images.camera1": frames[t]["camera1"],
                "observation.images.camera2": frames[t]["camera2"],
                "observation.images.camera3": frames[t]["camera3"],
                "observation.state": frames[t]["state"],
                "action": acts[t],
                "task": lang,
            })
        out.save_episode()
    print("[merge] done")


if __name__ == "__main__":
    main()
