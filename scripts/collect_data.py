#!/usr/bin/env python3
"""SO101-Nexus 仿真数据收集 - 支持双 GPU 并行"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import multiprocessing as mp
import numpy as np
from pathlib import Path


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
    rgb_encoder = RGBEncoderConfig(vcodec="h264_nvenc", preset=1, g=30, crf=30)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=root,
        robot_type="so101",
        use_videos=True,
        rgb_encoder=rgb_encoder,
    )
    print(f"Dataset created: {repo_id} at {root}")
    return ds


def worker_fn(gpu_id, num_episodes, task, max_steps, result_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["MUJOCO_GL"] = "egl"

    import so101_nexus
    import so101_nexus.mujoco
    import gymnasium as gym

    config = so101_nexus.PickConfig(
        observations=[
            so101_nexus.JointPositions(),
            so101_nexus.WristCamera(),
            so101_nexus.OverheadCamera(),
        ]
    )
    env = gym.make(task, config=config)
    print(f"[GPU {gpu_id}] Worker started, collecting {num_episodes} episodes")

    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        success = False
        frames = []

        for step in range(max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            frames.append((
                action.astype(np.float32).copy(),
                obs["state"].astype(np.float32).copy(),
                obs["wrist_camera"].copy(),
                obs["overhead_camera"].copy(),
            ))

            if info.get("success", False):
                success = True
                break
            if terminated or truncated:
                break

        result_queue.put((gpu_id, ep, frames, success, episode_reward, step + 1))
        print(f"[GPU {gpu_id}] Episode {ep+1}/{num_episodes}: success={success}, reward={episode_reward:.3f}, steps={step+1}")

    env.close()
    result_queue.put((gpu_id, -1, None, None, None, None))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=os.environ.get("DATASET_REPO", "xieyucheng123/so101-sim-picklift"))
    parser.add_argument("--root", default="/data/pipeline/dataset")
    parser.add_argument("--num-episodes", type=int, default=int(os.environ.get("NUM_EPISODES", "50")))
    parser.add_argument("--task", default="MuJoCoPickLift-v1")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--num-gpus", type=int, default=2)
    args = parser.parse_args()

    import shutil
    shutil.rmtree(args.root, ignore_errors=True)

    dataset = create_dataset(args.repo_id, args.root)

    num_gpus = args.num_gpus
    eps_per_gpu = [args.num_episodes // num_gpus + (1 if i < args.num_episodes % num_gpus else 0) for i in range(num_gpus)]

    result_queue = mp.Queue()
    workers = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=worker_fn, args=(gpu_id, eps_per_gpu[gpu_id], args.task, args.max_steps, result_queue))
        p.start()
        workers.append(p)

    total_collected = 0
    successes = 0
    active_workers = num_gpus

    while active_workers > 0:
        gpu_id, ep_idx, frames, success, reward, steps = result_queue.get()
        if ep_idx == -1:
            active_workers -= 1
            continue

        for action, state, wrist, overhead in frames:
            frame = {
                "action": action,
                "observation.state": state,
                "observation.images.wrist": wrist,
                "observation.images.overhead": overhead,
                "task": "pick_and_lift",
            }
            dataset.add_frame(frame)
        dataset.save_episode()

        total_collected += 1
        successes += int(success)
        print(f"[Main] Saved episode {total_collected}/{args.num_episodes} (from GPU {gpu_id}): success={success}, reward={reward:.3f}, steps={steps}")

    for p in workers:
        p.join()

    try:
        dataset.push_to_hub()
        print("Dataset pushed to HF Hub")
    except Exception as e:
        print(f"Warning: push_to_hub failed ({e}), data saved locally at {args.root}")

    print(f"\n=== Collection Complete ===")
    print(f"Episodes: {total_collected}, Successes: {successes}, Rate: {successes/total_collected:.1%}")


if __name__ == "__main__":
    main()
