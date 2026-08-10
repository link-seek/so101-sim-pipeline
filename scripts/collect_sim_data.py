#!/usr/bin/env python3
"""仿真数据采集 - 使用 so101_nexus 硬编码轨迹 + domain randomization"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import gymnasium as gym
import so101_nexus.mujoco
from so101_nexus import (
    PickConfig,
    PickAndPlaceConfig,
    StackCubeConfig,
    CubeObject,
    sim_qpos_to_dataset_row,
    SO101_JOINT_NAMES,
)


PICK_AND_PLACE_KEYFRAMES = [
    np.array([0.0, -1.57, 1.57, 0.66, 0.0, 0.5]),
    np.array([0.0, -1.20, 1.20, 0.50, 0.0, 0.5]),
    np.array([0.0, -1.05, 1.05, 0.30, 0.0, 0.5]),
    np.array([0.0, -1.05, 1.05, 0.30, 0.0, -0.8]),
    np.array([0.0, -1.20, 1.20, 0.50, 0.0, -0.8]),
    np.array([0.3, -1.20, 1.20, 0.50, 0.0, -0.8]),
    np.array([0.3, -1.05, 1.05, 0.30, 0.0, -0.8]),
    np.array([0.3, -1.05, 1.05, 0.30, 0.0, 0.5]),
    np.array([0.3, -1.20, 1.20, 0.50, 0.0, 0.5]),
]

PICK_LIFT_KEYFRAMES = [
    np.array([0.0, -1.57, 1.57, 0.66, 0.0, 0.5]),
    np.array([0.0, -1.20, 1.20, 0.50, 0.0, 0.5]),
    np.array([0.0, -1.05, 1.05, 0.30, 0.0, 0.5]),
    np.array([0.0, -1.05, 1.05, 0.30, 0.0, -0.8]),
    np.array([0.0, -1.20, 1.20, 0.50, 0.0, -0.8]),
    np.array([0.0, -0.80, 0.80, 0.20, 0.0, -0.8]),
]

STACK_CUBE_KEYFRAMES = [
    np.array([0.0, -1.57, 1.57, 0.66, 0.0, 0.5]),
    np.array([-0.15, -1.20, 1.20, 0.50, 0.0, 0.5]),
    np.array([-0.15, -1.05, 1.05, 0.30, 0.0, 0.5]),
    np.array([-0.15, -1.05, 1.05, 0.30, 0.0, -0.8]),
    np.array([-0.15, -1.20, 1.20, 0.50, 0.0, -0.8]),
    np.array([0.15, -1.20, 1.20, 0.50, 0.0, -0.8]),
    np.array([0.15, -1.05, 1.05, 0.30, 0.0, -0.8]),
    np.array([0.15, -1.05, 1.05, 0.30, 0.0, 0.5]),
    np.array([0.15, -1.20, 1.20, 0.50, 0.0, 0.5]),
    np.array([0.0, -1.20, 1.20, 0.50, 0.0, 0.5]),
    np.array([0.0, -1.05, 1.05, 0.30, 0.0, 0.5]),
    np.array([0.0, -1.05, 1.05, 0.30, 0.0, -0.8]),
    np.array([0.0, -1.20, 1.20, 0.50, 0.0, -0.8]),
    np.array([0.15, -1.20, 1.20, 0.50, 0.0, -0.8]),
    np.array([0.15, -0.90, 0.90, 0.40, 0.0, -0.8]),
    np.array([0.15, -0.90, 0.90, 0.40, 0.0, 0.5]),
    np.array([0.15, -1.20, 1.20, 0.50, 0.0, 0.5]),
]

ENV_CONFIGS = {
    "MuJoCoPickAndPlace-v1": {
        "keyframes": PICK_AND_PLACE_KEYFRAMES,
        "config_class": PickAndPlaceConfig,
        "task_desc": "pick up the cube and place it on the target",
    },
    "MuJoCoPickLift-v1": {
        "keyframes": PICK_LIFT_KEYFRAMES,
        "config_class": PickConfig,
        "task_desc": "pick up the object and lift it",
    },
    "MuJoCoStackCube-v1": {
        "keyframes": STACK_CUBE_KEYFRAMES,
        "config_class": StackCubeConfig,
        "task_desc": "stack the cube on top of the other cube",
    },
}

COLORS = ["red", "green", "blue", "yellow", "orange", "purple"]
GROUND_COLORS = ["gray", "white", "black"]


def make_random_config(env_id, seed):
    rng = random.Random(seed)
    config_class = ENV_CONFIGS[env_id]["config_class"]

    common_kwargs = {
        "ground_colors": rng.sample(GROUND_COLORS, 1),
        "robot_init_qpos_noise": 0.02,
    }

    if env_id == "MuJoCoPickAndPlace-v1":
        return PickAndPlaceConfig(
            cube_colors=rng.sample(COLORS, min(3, len(COLORS))),
            target_colors=rng.sample(COLORS, min(2, len(COLORS))),
            **common_kwargs,
        )
    elif env_id == "MuJoCoPickLift-v1":
        return PickConfig(
            objects=CubeObject(color=rng.choice(COLORS)),
            n_distractors=rng.randint(0, 2),
            **common_kwargs,
        )
    elif env_id == "MuJoCoStackCube-v1":
        return StackCubeConfig(
            cube_a_colors=rng.sample(COLORS, min(2, len(COLORS))),
            cube_b_colors=rng.sample(COLORS, min(2, len(COLORS))),
            **common_kwargs,
        )
    return config_class()


def interpolate_keyframes(keyframes, steps_per_segment):
    actions = []
    for i in range(len(keyframes) - 1):
        start, end = keyframes[i], keyframes[i + 1]
        for t in range(steps_per_segment):
            alpha = t / steps_per_segment
            actions.append(start + alpha * (end - start))
    actions.append(keyframes[-1])
    return actions


def collect_episode(env, actions, task_desc):
    obs, info = env.reset()
    frames = []
    success = False

    for i, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(action)

        state = sim_qpos_to_dataset_row(np.asarray(obs["joint_pos"] if "joint_pos" in obs else obs[:6]))
        action_ds = sim_qpos_to_dataset_row(np.asarray(action))

        frame = {
            "observation.state": state,
            "action": action_ds,
            "reward": float(reward),
            "success": bool(info.get("success", False)),
            "done": bool(terminated or truncated),
            "task": task_desc,
            "frame_index": i,
        }

        if "wrist" in obs:
            frame["observation.images.wrist"] = obs["wrist"]
        if "overhead" in obs:
            frame["observation.images.overhead"] = obs["overhead"]

        frames.append(frame)

        if info.get("success", False):
            success = True
        if terminated or truncated:
            break

    return frames, success


def collect_dataset(
    env_ids,
    episodes_per_env,
    steps_per_segment,
    output_repo_id,
    push_to_hub,
    hf_token,
    seed_base=42,
):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from so101_nexus.teleop.dataset import FieldSelection, build_features

    all_episodes = []
    total_success = 0
    total_episodes = 0

    for env_id in env_ids:
        config = ENV_CONFIGS[env_id]
        task_desc = config["task_desc"]
        keyframes = config["keyframes"]
        actions = interpolate_keyframes(keyframes, steps_per_segment)

        print(f"\n=== {env_id} ===")
        print(f"Task: {task_desc}")
        print(f"Episodes: {episodes_per_env}")
        print(f"Steps per episode: {len(actions)}")

        for ep in range(episodes_per_env):
            seed = seed_base + ep
            env_config = make_random_config(env_id, seed)
            env = gym.make(env_id, config=env_config, render_mode="rgb_array", control_mode="pd_joint_pos")

            frames, success = collect_episode(env, actions, task_desc)
            all_episodes.append(frames)
            total_success += int(success)
            total_episodes += 1

            status = "OK" if success else "FAIL"
            print(f"  Episode {ep+1}/{episodes_per_env}: {len(frames)} frames, {status}")

            env.close()

    success_rate = total_success / total_episodes if total_episodes > 0 else 0
    print(f"\n=== Collection Complete ===")
    print(f"Total episodes: {total_episodes}")
    print(f"Success rate: {success_rate:.1%}")
    print(f"Total frames: {sum(len(ep) for ep in all_episodes)}")

    action_features = {f"{name}.pos": float for name in SO101_JOINT_NAMES}
    follower_features = {
        **action_features,
        "wrist": (480, 640, 3),
        "overhead": (480, 640, 3),
    }
    features = build_features(FieldSelection(), follower_features, action_features)

    dataset = LeRobotDataset.create(
        repo_id=output_repo_id,
        fps=30,
        features=features,
        robot_type="sim_so_follower",
        use_videos=True,
    )

    for episode_frames in all_episodes:
        for frame in episode_frames:
            dataset.add_frame(frame)
        dataset.save_episode()

    dataset.finalize()

    if push_to_hub:
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
        dataset.push_to_hub()
        print(f"Dataset pushed to: https://huggingface.co/datasets/{output_repo_id}")

    return success_rate, total_episodes


def main():
    parser = argparse.ArgumentParser(description="Collect simulation data with scripted policies")
    parser.add_argument("--envs", nargs="+",
                        default=["MuJoCoPickAndPlace-v1", "MuJoCoPickLift-v1", "MuJoCoStackCube-v1"],
                        help="Environments to collect from")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Episodes per environment")
    parser.add_argument("--steps-per-segment", type=int, default=50,
                        help="Interpolation steps between keyframes")
    parser.add_argument("--output-repo-id", default="xieyucheng123/so101-sim-dataset",
                        help="Output LeRobot dataset repo ID")
    parser.add_argument("--push-to-hub", action="store_true", default=True)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    print(f"=== Simulation Data Collection ===")
    print(f"Environments: {args.envs}")
    print(f"Episodes per env: {args.episodes}")
    print(f"Output: {args.output_repo_id}")

    success_rate, total = collect_dataset(
        env_ids=args.envs,
        episodes_per_env=args.episodes,
        steps_per_segment=args.steps_per_segment,
        output_repo_id=args.output_repo_id,
        push_to_hub=args.push_to_hub,
        hf_token=args.hf_token,
    )

    print(f"\nFinal success rate: {success_rate:.1%} ({total} episodes)")


if __name__ == "__main__":
    main()
