#!/usr/bin/env python3
"""Warp 后端数据收集 - 使用向量化环境批量收集"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


RESULTS_DIR = Path("/data/pipeline/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def create_dataset(repo_id, root, fps=30):
    features = {
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["shoulder_pan", "shoulder_lift", "elbow", "wrist", "gripper_rotation", "gripper"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["shoulder_pan", "shoulder_lift", "elbow", "wrist", "gripper_rotation", "gripper"],
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.overhead": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channel"],
        },
    }
    from lerobot.configs.video import RGBEncoderConfig
    rgb_encoder = RGBEncoderConfig(vcodec="h264", preset="fast", g=30, crf=30)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=root,
        use_videos=True,
        rgb_encoder=rgb_encoder,
    )
    print(f"Dataset created: {repo_id} at {root}")
    return ds


def extract_obs(obs, env_idx):
    """从 Warp 观测中提取 state, wrist_camera, overhead_camera"""
    if isinstance(obs, dict):
        state = obs.get("state", obs.get("observation.state"))
        wrist = obs.get("wrist_camera", obs.get("observation.images.wrist"))
        overhead = obs.get("overhead_camera", obs.get("observation.images.overhead"))

        if state is not None and hasattr(state, "shape") and state.ndim > 1:
            state = state[env_idx]
        if wrist is not None and hasattr(wrist, "shape") and wrist.ndim > 3:
            wrist = wrist[env_idx]
        if overhead is not None and hasattr(overhead, "shape") and overhead.ndim > 3:
            overhead = overhead[env_idx]

        state = state.cpu().numpy() if isinstance(state, torch.Tensor) else np.array(state)
        wrist = wrist.cpu().numpy() if isinstance(wrist, torch.Tensor) else np.array(wrist)
        overhead = overhead.cpu().numpy() if isinstance(overhead, torch.Tensor) else np.array(overhead)

        if wrist.dtype != np.uint8:
            wrist = (wrist * 255).clip(0, 255).astype(np.uint8) if wrist.max() <= 1.0 else wrist.astype(np.uint8)
        if overhead.dtype != np.uint8:
            overhead = (overhead * 255).clip(0, 255).astype(np.uint8) if overhead.max() <= 1.0 else overhead.astype(np.uint8)

        return state.astype(np.float32), wrist, overhead

    raise ValueError(f"Unexpected obs type: {type(obs)}")


def collect_warp(num_episodes, num_envs, max_steps, repo_id, root, task="WarpPickLift-v1"):
    """使用 Warp 向量化环境收集数据"""
    import so101_nexus
    import so101_nexus.warp
    import gymnasium as gym

    config = so101_nexus.PickConfig(
        observations=[
            so101_nexus.JointPositions(),
            so101_nexus.WristCamera(),
            so101_nexus.OverheadCamera(),
        ]
    )

    print(f"Creating Warp env: num_envs={num_envs}, task={task}")
    env = gym.make_vec(task, num_envs=num_envs, device="cuda", config=config)
    obs, info = env.reset()

    print(f"obs type: {type(obs)}")
    if isinstance(obs, dict):
        for k, v in obs.items():
            shape = v.shape if hasattr(v, "shape") else "N/A"
            dtype = v.dtype if hasattr(v, "dtype") else "N/A"
            print(f"  obs['{k}']: shape={shape}, dtype={dtype}")

    import shutil
    shutil.rmtree(root, ignore_errors=True)
    dataset = create_dataset(repo_id, root)

    episode_frames = [[] for _ in range(num_envs)]
    episode_rewards = [0.0] * num_envs
    episode_success = [False] * num_envs
    episode_steps = [0] * num_envs

    collected = 0
    successes = 0
    start_time = time.time()

    while collected < num_episodes:
        actions = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(actions)

        for i in range(num_envs):
            if collected >= num_episodes:
                break

            r = reward[i].item() if isinstance(reward, torch.Tensor) else reward[i]
            t = terminated[i].item() if isinstance(terminated, torch.Tensor) else terminated[i]
            tr = truncated[i].item() if isinstance(truncated, torch.Tensor) else truncated[i]

            episode_rewards[i] += r
            episode_steps[i] += 1

            try:
                state, wrist, overhead = extract_obs(obs, i)
                action_np = actions[i].cpu().numpy() if isinstance(actions, torch.Tensor) else np.array(actions[i])
                episode_frames[i].append((
                    action_np.astype(np.float32).copy(),
                    state.copy(),
                    wrist.copy(),
                    overhead.copy(),
                ))
            except Exception as e:
                if episode_steps[i] == 1:
                    print(f"  WARN: obs extraction failed for env {i}: {e}")

            succ = False
            if isinstance(info, dict):
                succ_key = None
                for key in ["success", "is_success"]:
                    if key in info:
                        succ_key = key
                        break
                if succ_key:
                    s = info[succ_key]
                    succ = s[i].item() if isinstance(s, torch.Tensor) else bool(s[i])

            if succ:
                episode_success[i] = True

            if t or tr or episode_steps[i] >= max_steps:
                if episode_success[i]:
                    succ = True

                for action, state, wrist_img, overhead_img in episode_frames[i]:
                    frame = {
                        "action": action,
                        "observation.state": state,
                        "observation.images.wrist": wrist_img,
                        "observation.images.overhead": overhead_img,
                        "task": "pick_and_lift",
                    }
                    dataset.add_frame(frame)
                dataset.save_episode()

                collected += 1
                if episode_success[i]:
                    successes += 1

                elapsed = time.time() - start_time
                print(f"[Episode {collected}/{num_episodes}] env={i}, success={episode_success[i]}, "
                      f"reward={episode_rewards[i]:.3f}, steps={episode_steps[i]}, "
                      f"elapsed={elapsed:.1f}s, rate={collected/elapsed:.2f} eps/s")

                episode_frames[i] = []
                episode_rewards[i] = 0.0
                episode_success[i] = False
                episode_steps[i] = 0

    env.close()
    elapsed = time.time() - start_time

    try:
        dataset.push_to_hub()
        print("Dataset pushed to HF Hub")
    except Exception as e:
        print(f"Warning: push_to_hub failed ({e}), data saved locally at {root}")

    report = {
        "backend": "warp",
        "num_episodes": collected,
        "num_envs": num_envs,
        "successes": successes,
        "success_rate": successes / collected if collected > 0 else 0,
        "elapsed": elapsed,
        "eps_per_sec": collected / elapsed if elapsed > 0 else 0,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    report_path = RESULTS_DIR / "warp_collect_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n=== Warp 数据收集完成 ===")
    print(f"Episodes: {collected}, Successes: {successes}, Rate: {successes/collected:.1%}")
    print(f"耗时: {elapsed:.1f}s, 速率: {collected/elapsed:.2f} eps/s")

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=os.environ.get("DATASET_REPO", "xieyucheng123/so101-sim-picklift-warp"))
    parser.add_argument("--root", default="/data/pipeline/dataset_warp")
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--task", default="WarpPickLift-v1")
    args = parser.parse_args()

    collect_warp(args.num_episodes, args.num_envs, args.max_steps,
                 args.repo_id, args.root, args.task)


if __name__ == "__main__":
    main()
