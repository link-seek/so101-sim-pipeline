#!/usr/bin/env python3
"""渲染评测视频 - 在 SO101-Nexus 仿真中录制策略执行视频"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import cv2

import so101_nexus
import so101_nexus.mujoco
import gymnasium as gym
from lerobot.policies.act.modeling_act import ACTPolicy

RESULTS_DIR = Path("/data/pipeline/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def render_video(env, policy, num_episodes=3, max_steps=200):
    all_frames = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        frames = []

        for step in range(max_steps):
            state = torch.from_numpy(obs["state"]).float().unsqueeze(0).to(policy.device)
            wrist = torch.from_numpy(obs["wrist_camera"]).float().permute(2, 0, 1).unsqueeze(0).to(policy.device) / 255.0
            overhead = torch.from_numpy(obs["overhead_camera"]).float().permute(2, 0, 1).unsqueeze(0).to(policy.device) / 255.0

            obs_dict = {
                "observation.state": state,
                "observation.images.wrist": wrist,
                "observation.images.overhead": overhead,
            }

            with torch.no_grad():
                action = policy.select_action(obs_dict)

            action_np = action.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action_np)

            frame = np.hstack([obs["overhead_camera"], obs["wrist_camera"]])
            frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            if info.get("success", False) or terminated or truncated:
                break

        all_frames.extend(frames)
        print(f"Episode {ep+1}: {len(frames)} frames")

    video_path = str(RESULTS_DIR / "eval_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, 30, (1280, 480))
    for frame in all_frames:
        writer.write(frame)
    writer.release()
    print(f"Video saved: {video_path} ({len(all_frames)} frames)")
    return video_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", "/data/pipeline/training/checkpoints/last/pretrained_model"))
    parser.add_argument("--task", default="MuJoCoPickLift-v1")
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()

    config = so101_nexus.PickConfig(
        observations=[
            so101_nexus.JointPositions(),
            so101_nexus.WristCamera(),
            so101_nexus.OverheadCamera(),
        ]
    )
    env = gym.make(args.task, config=config)
    policy = ACTPolicy.from_pretrained(args.model_path)
    policy.to("cuda")
    policy.eval()
    render_video(env, policy, args.num_episodes, args.max_steps)
    env.close()


if __name__ == "__main__":
    main()
