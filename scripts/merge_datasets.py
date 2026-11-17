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


def read_dataset(root, only=()):
    """Read a v3 dataset with INDEXED per-episode video decode (peak = 1 episode).

    Build order: parquet (row counts per episode) -> per-cam mp4 frame counts
    (header-only, no decode) -> global frame offsets -> decode ONLY the frame
    ranges of kept episodes. only: lowercase task substrings to keep.
    """
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
    task_of_row = np.asarray(table.column("task_index").to_pylist())
    counts = {ep: int((ep_idx == ep).sum()) for ep in ep_ids}
    # global row offset of each episode
    off_of, _off = {}, 0
    for ep in ep_ids:
        off_of[ep] = _off; _off += counts[ep]
    total_rows = len(table)
    # task string of each episode (from its first row)
    lang_of = {}
    for ep in ep_ids:
        first = int(np.argmax(ep_idx == ep))
        lang_of[ep] = tasks.get(int(task_of_row[first]), "")
    keep = [ep for ep in ep_ids if (not only) or any(s in lang_of[ep].lower() for s in only)]
    print(f"[merge] {root}: {len(keep)}/{len(ep_ids)} episodes kept ({sum(counts[e] for e in keep)}/{total_rows} rows)")

    states = np.asarray(table.column(STATE_KEY).to_pylist())
    acts = np.asarray(table.column(ACTION_KEY).to_pylist())
    del table
    import gc; gc.collect()

    # per-cam: list of (mp4 path, nframes) + cumulative offsets, header-only probe
    cam_index = {}
    for cam in CAM_KEYS:
        vdir = root / "videos" / cam
        mp4s = []
        for chunk in sorted(vdir.glob("chunk-*")):
            mp4s.extend(sorted(chunk.glob("*.mp4")))
        assert mp4s, f"no mp4 under {vdir}"
        entries, acc = [], 0
        for m in mp4s:
            with av.open(str(m)) as cont:
                n = sum(1 for _ in cont.decode(video=0))
            entries.append((m, acc, n)); acc += n
        print(f"[merge] {root} {cam}: {len(mp4s)} mp4s -> {acc} frames")
        assert acc == total_rows, (cam, acc, total_rows)
        cam_index[cam] = entries

    def decode_range(cam, start, n):
        """Decode global frames [start, start+n) for one camera."""
        out = []
        need_end = start + n
        for (m, base, cnt) in cam_index[cam]:
            if base + cnt <= start or base >= need_end:
                continue
            with av.open(str(m)) as cont:
                for i, frame in enumerate(cont.decode(video=0)):
                    g = base + i
                    if g >= start and g < need_end:
                        out.append(frame.to_ndarray(format="rgb24"))
                    if g + 1 >= need_end:
                        break
            if len(out) >= n:
                break
        assert len(out) == n, (cam, start, n, len(out))
        return out

    episodes = []
    for ep in keep:
        n, off = counts[ep], off_of[ep]
        by_cam = {cam: decode_range(cam, off, n) for cam in CAM_KEYS}
        ep_frames = []
        for k in range(n):
            i = off + k
            ep_frames.append({
                "camera1": by_cam[CAM_KEYS[0]][k],
                "camera2": by_cam[CAM_KEYS[1]][k],
                "camera3": by_cam[CAM_KEYS[2]][k],
                "state": np.asarray(states[i], dtype=np.float32),
            })
        episodes.append((ep_frames, np.asarray(acts[off:off + n], dtype=np.float32), lang_of[ep]))
        del ep_frames, by_cam
        gc.collect()
    del states, acts
    gc.collect()
    return feats, fps, robot_type, episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--repo-id", default="libero_so101_expert")
    ap.add_argument("--tasks", default="",
                    help="comma-separated substrings; only episodes whose task contains one are kept")
    ap.add_argument("--resume", action="store_true",
                    help="append into existing dataset at output (LeRobotDataset.resume) instead of create")
    args = ap.parse_args()
    roots = [r for r in args.inputs.split(",") if r]
    assert len(roots) >= 1
    if not args.resume:
        assert not Path(args.output).exists(), f"refuse to clobber {args.output}"
    else:
        assert Path(args.output).exists(), f"nothing to resume at {args.output}"

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    # Stream: read one input, write its episodes immediately, then free.
    # (Loading all inputs at once OOMs the box past ~100 episodes; exit 137.)
    if args.resume:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset as _LDSR
        out = _LDSR.resume(repo_id=args.repo_id, root=args.output)
        feats = out.features
        print(f"[merge] resumed {args.output}: {out.num_episodes} episodes already on disk")
        total = out.num_episodes
    else:
        out = None
        total = 0
    first = roots[0]
    import pyarrow.parquet as _pq
    _info = json.load(open(Path(first) / "meta" / "info.json"))
    feats, fps, robot_type = _info["features"], _info.get("fps", 20), _info.get("robot_type", "so_follower")
    create_feats = {k: {"dtype": v["dtype"], "shape": list(v["shape"]),
                        "names": v.get("names") or (["height", "width", "channel"] if v["dtype"] in ("video", "image") else None)}
                    for k, v in feats.items() if k in CAM_KEYS + [STATE_KEY, ACTION_KEY]}
    if out is None:
        out = LeRobotDataset.create(
        repo_id=args.repo_id, fps=fps, features=create_feats,
        root=args.output, robot_type=robot_type, use_videos=True,
        # Bound writer memory: small chunks + serial per-episode encode.
        # (Defaults buffer whole-dataset chunks and fork N parallel encoders;
        # 492 eps OOM-killed at finalize with exit 137.)
        data_files_size_in_mb=100, video_files_size_in_mb=100,
        )
    only = [s.strip().lower() for s in args.tasks.split(",") if s.strip()]
    for r in roots:
        _, _, _, eps = read_dataset(r, tuple(only))
        if only:
            before = len(eps)
            eps = [(f, a, l) for (f, a, l) in eps if any(s in l.lower() for s in only)]
            print(f"[merge] {r}: {len(eps)}/{before} episodes after task filter {only}")
        else:
            print(f"[merge] {r}: {len(eps)} episodes -> writing...")
        for frames, acts, lang in eps:
            for t in range(len(acts)):
                out.add_frame({
                    "observation.images.camera1": frames[t]["camera1"],
                    "observation.images.camera2": frames[t]["camera2"],
                    "observation.images.camera3": frames[t]["camera3"],
                    "observation.state": frames[t]["state"],
                    "action": acts[t],
                    "task": lang,
                })
            out.save_episode(parallel_encoding=False)
            total += 1
        del eps
        import gc; gc.collect()
        print(f"[merge] {r} done, running total {total}")
    print(f"[merge] total {total} episodes -> {args.output}")
    out.finalize()
    print("[merge] finalized; verifying by reopen...")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset as _LDS2
    check = _LDS2(repo_id=args.repo_id, root=args.output)
    assert check.num_episodes == total, (check.num_episodes, total)
    print(f"[merge] verified: {check.num_episodes} episodes, meta on disk, done")


if __name__ == "__main__":
    main()
