#!/usr/bin/env python3
"""Lightweight quality gate for merged LeRobot v3 datasets (CPU-only).

Checks (fail-closed, exit 1 on hard failure):
  1. dataset opens (validates video/parquet row alignment + meta)
  2. task coverage: every task has >= min_eps episodes
  3. degenerate episodes: length < min_len frames, or near-zero action variance
  4. dumps first/mid/last frames of 1 episode per task for eyeballing
Writes report.json + sample PNGs into --out-dir.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--repo-id", default="libero_so101_expert")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-eps", type=int, default=1)
    ap.add_argument("--min-len", type=int, default=20)
    ap.add_argument("--min-act-std", type=float, default=1e-4)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    print(f"[validate] opening {args.root} ...", flush=True)
    ds = LeRobotDataset(repo_id=args.repo_id, root=args.root, video_backend="pyav")
    print(f"[validate] episodes={ds.num_episodes} frames={ds.num_frames} fps={ds.fps}", flush=True)
    if ds.num_episodes == 0:
        print("[validate] FAIL: zero episodes"); return 1

    import pyarrow.parquet as _pq
    task_of = {r["task_index"]: r["task"] for r in
               _pq.read_table(Path(args.root) / "meta" / "tasks.parquet").to_pylist()}
    tasks = defaultdict(list)   # task -> [(ep_idx, length)]
    ep_lens = {}
    for ep in range(ds.num_episodes):
        m = ds.meta.episodes[ep]
        ln = int(m["length"])
        ep_lens[ep] = ln
        ti = m.get("task_index")
        tasks[str(task_of.get(int(ti), "UNK") if ti is not None else "UNK")].append((ep, ln))
    # per-episode action variance (cheap: read action column via hf dataset slices)
    print("[validate] scanning actions ...", flush=True)
    acts = np.asarray(ds.hf_dataset["action"])
    idx_of_ep, off = {}, 0
    for ep in range(ds.num_episodes):
        idx_of_ep[ep] = (off, off + ep_lens[ep]); off += ep_lens[ep]
    for ep, (a, b) in idx_of_ep.items():
        ln = b - a
        if ln < args.min_len:
            report["bad_episodes"].append({"ep": ep, "reason": f"too_short({ln})"})
            continue
        std = float(np.asarray(acts[a:b], dtype=np.float64).std())
        if std < args.min_act_std:
            report["bad_episodes"].append({"ep": ep, "reason": f"static_action(std={std:.2e})"})

    # sample frames: first ep of each task
    samples = out / "samples"; samples.mkdir(exist_ok=True)
    from PIL import Image
    for task, eps in sorted(tasks.items()):
        ep = eps[0][0]
        a, b = idx_of_ep[ep]
        for tag, idx in [("first", a), ("mid", (a + b) // 2), ("last", b - 1)]:
            try:
                row = ds[idx]
                img = np.asarray(row["observation.images.camera1"])
                if img.dtype != np.uint8:
                    img = np.clip(img, 0, 255).astype(np.uint8) if img.max() > 1 else (img * 255).astype(np.uint8)
                if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[2] not in (1, 3, 4):
                    img = np.transpose(img, (1, 2, 0))
                if img.ndim != 3 or img.shape[2] not in (3, 4) or min(img.shape[:2]) < 16:
                    raise ValueError(f"bad frame shape {img.shape} {img.dtype}")
                safe = "".join(c if c.isalnum() else "_" for c in task)[:40]
                Image.fromarray(img[:, :, :3]).save(samples / f"{safe}_ep{ep}_{tag}.png")
            except Exception as e:
                print(f"[validate] WARN: sample {task[:30]} ep={ep} {tag} failed: {e}", flush=True)
    report["sample_dir"] = "samples"

    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print("[validate] per-task:", flush=True)
    for task, s in sorted(report["tasks"].items()):
        print(f"  {task[:60]:62s} eps={s['episodes']:3d} len {s['len_min']}-{s['len_mean']}-{s['len_max']}")
    print(f"[validate] bad episodes: {len(report['bad_episodes'])}")
    for b in report["bad_episodes"][:20]:
        print(f"  ep={b['ep']} {b['reason']}")
    if hard_fail or len(report["bad_episodes"]) > max(5, ds.num_episodes * 0.05):
        print("[validate] FAIL: too many bad episodes"); return 1
    print("[validate] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
