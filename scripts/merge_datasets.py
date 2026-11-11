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
    """Read a v3 dataset: parquet gives episode boundaries, mp4s give frames (one file per chunk)."""
    import av
    import glob as _glob
    root = Path(root)
    info = json.load(open(root / "meta" / "info.json"))
    feats = info["features"]
    fps = info.get("fps", 20)
    robot_type = info.get("robot_type", "so_follower")
    tasks = {r["task_index"]: r["task"] for r in pq.read_table(root / "meta" / "tasks.parquet").to_pylist()}

    data_files = sorted(_glob.glob(str(root / "data" / "chunk-*" / "*.parquet")))
    assert data_files, f"no data parquet in {root}"
    import pyarrow as pa
    table = pa.concat_tables([pq.read_table(f) for f in data_files])
    ep_idx = np.asarray(table.column("episode_index").to_pylist())
    ep_ids = sorted(set(ep_idx.tolist()))
    counts = [(ep_idx == ep).sum() for ep in ep_ids]
    print(f"[merge] {root}: {len(ep_ids)} episodes, {len(table)} rows")

    frames_by_cam = {}
    for cam in CAM_KEYS:
        vdir = root / "videos" / cam
        mp4s = []
        for chunk in sorted(vdir.glob("chunk-*")):
            mp4s.extend(sorted(chunk.glob("*.mp4")))
        assert mp4s, f"no mp4 under {vdir}"
        frames = []
        for m in mp4s:
            with av.open(str(m)) as cont:
                frames.extend(f.to_ndarray(format="rgb24") for f in cont.decode(video=0))
        print(f"[merge] {root} {cam}: {len(mp4s)} mp4s -> {len(frames)} frames")
        assert len(frames) == len(table), (cam, len(frames), len(table))
        frames_by_cam[cam] = frames

    states = np.asarray(table.column(STATE_KEY).to_pylist())
    acts = np.asarray(table.column(ACTION_KEY).to_pylist())
    task_of_row = np.asarray(table.column("task_index").to_pylist())
    episodes, off = [], 0
    for ep, n in zip(ep_ids, counts):
        ep_frames = []
        for t in range(n):
            i = off + t
            ep_frames.append({
                "camera1": frames_by_cam[CAM_KEYS[0]][i],
                "camera2": frames_by_cam[CAM_KEYS[1]][i],
                "camera3": frames_by_cam[CAM_KEYS[2]][i],
                "state": np.asarray(states[i], dtype=np.float32),
            })
        episodes.append((ep_frames, np.asarray(acts[off:off + n], dtype=np.float32),
                         tasks.get(int(task_of_row[off]), "")))
        off += n
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
    out.finalize()
    print("[merge] finalized; verifying by reopen...")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset as _LDS2
    check = _LDS2(repo_id=args.repo_id, root=args.output)
    assert check.num_episodes == len(all_eps), (check.num_episodes, len(all_eps))
    print(f"[merge] verified: {check.num_episodes} episodes, meta on disk, done")


if __name__ == "__main__":
    main()
