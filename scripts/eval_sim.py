#!/usr/bin/env python3
"""SO101-Nexus 仿真评测 - 加载训练好的 ACT 策略在仿真中评测"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import so101_nexus
import so101_nexus.mujoco
import gymnasium as gym
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig


RESULTS_DIR = Path("/data/pipeline/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_policy(model_path, device="cuda"):
    print(f"Loading policy from {model_path}")
    policy = ACTPolicy.from_pretrained(model_path)
    policy.to(device)
    policy.eval()
    print(f"Policy loaded: {type(policy).__name__}")
    return policy


def run_eval(env, policy, num_episodes, max_steps=300):
    results = []
    all_frames = []
    device = next(policy.parameters()).device

    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        success = False
        steps = 0

        for step in range(max_steps):
            state = torch.from_numpy(obs["state"]).float().unsqueeze(0).to(device)
            wrist = torch.from_numpy(obs["wrist_camera"]).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            overhead = torch.from_numpy(obs["overhead_camera"]).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

            obs_dict = {
                "observation.state": state,
                "observation.images.wrist": wrist,
                "observation.images.overhead": overhead,
            }

            with torch.no_grad():
                action = policy.select_action(obs_dict)

            action_np = action.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action_np)
            episode_reward += reward
            steps += 1

            if info.get("success", False):
                success = True
                break
            if terminated or truncated:
                break

        results.append({
            "episode": ep + 1,
            "success": success,
            "reward": float(episode_reward),
            "steps": steps,
        })
        print(f"Episode {ep+1}/{num_episodes}: success={success}, reward={episode_reward:.3f}, steps={steps}")

    success_rate = sum(r["success"] for r in results) / len(results)
    avg_reward = sum(r["reward"] for r in results) / len(results)
    avg_steps = sum(r["steps"] for r in results) / len(results)

    report = {
        "num_episodes": num_episodes,
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "avg_steps": avg_steps,
        "episodes": results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    report_path = RESULTS_DIR / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n=== Evaluation Complete ===")
    print(f"Success rate: {success_rate:.1%}, Avg reward: {avg_reward:.3f}, Avg steps: {avg_steps:.1f}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", "/data/pipeline/training/checkpoints/last/pretrained_model"))
    parser.add_argument("--task", default="MuJoCoPickLift-v1")
    parser.add_argument("--num-episodes", type=int, default=int(os.environ.get("NUM_EVAL_EPISODES", "20")))
    parser.add_argument("--max-steps", type=int, default=300)
    args = parser.parse_args()

    config = so101_nexus.PickConfig(
        observations=[
            so101_nexus.JointPositions(),
            so101_nexus.WristCamera(),
            so101_nexus.OverheadCamera(),
        ]
    )
    env = gym.make(args.task, config=config)
    policy = load_policy(args.model_path)
    report = run_eval(env, policy, args.num_episodes, args.max_steps)
    env.close()


if __name__ == "__main__":
    main()
