#!/usr/bin/env python3
"""SO101-Nexus 仿真数据收集 - 生成 LeRobot 格式数据集"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import numpy as np
from pathlib import Path

import so101_nexus
import so101_nexus.mujoco  # trigger gym registration
import gymnasium as gym
from lerobot.datasets.lerobot_dataset import LeRobotDataset


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
            "names": None,
        },
        "observation.images.overhead": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": None,
        },
    }
    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=root,
        robot_type="so101",
        use_videos=True,
    )
    print(f"Dataset created: {repo_id} at {root}")
    return ds


def collect_episode(env, dataset, max_steps=200):
    obs, info = env.reset()
    episode_reward = 0.0
    success = False

    for step in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        frame = {
            "action": action.astype(np.float32),
            "observation.state": obs["state"].astype(np.float32),
            "observation.images.wrist": obs["wrist_camera"],
            "observation.images.overhead": obs["overhead_camera"],
        }
        dataset.add_frame(frame)

        if info.get("success", False):
            success = True
            break
        if terminated or truncated:
            break

    dataset.save_episode(task=f"PickLift success={success}")
    return success, episode_reward, step + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=os.environ.get("DATASET_REPO", "xieyucheng123/so101-sim-picklift"))
    parser.add_argument("--root", default="/data/pipeline/dataset")
    parser.add_argument("--num-episodes", type=int, default=int(os.environ.get("NUM_EPISODES", "50")))
    parser.add_argument("--task", default="MuJoCoPickLift-v1")
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()

    Path(args.root).mkdir(parents=True, exist_ok=True)

    config = so101_nexus.PickConfig(
        observations=[
            so101_nexus.JointPositions(),
            so101_nexus.WristCamera(),
            so101_nexus.OverheadCamera(),
        ]
    )
    env = gym.make(args.task, config=config)

    dataset = create_dataset(args.repo_id, args.root)

    successes = 0
    for ep in range(args.num_episodes):
        success, reward, steps = collect_episode(env, dataset, args.max_steps)
        successes += success
        print(f"Episode {ep+1}/{args.num_episodes}: success={success}, reward={reward:.3f}, steps={steps}")

    dataset.consolidate()
    dataset.push_to_hub()
    env.close()

    print(f"\n=== Collection Complete ===")
    print(f"Episodes: {args.num_episodes}, Successes: {successes}, Rate: {successes/args.num_episodes:.1%}")


if __name__ == "__main__":
    main()
