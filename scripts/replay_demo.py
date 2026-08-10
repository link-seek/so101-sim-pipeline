#!/usr/bin/env python3
"""仿真回放演示 - 用训练好的 SmolVLA 策略在 so101_nexus 仿真中执行任务并录像"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import gymnasium as gym
import so101_nexus.mujoco
from so101_nexus import PickAndPlaceConfig, CubeObject, sim_qpos_to_dataset_row

RESULTS_DIR = Path("/data/eval/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_policy(checkpoint, device="cuda"):
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    print(f"Loading SmolVLA policy from {checkpoint}...")
    policy = SmolVLAPolicy.from_pretrained(checkpoint)
    policy.to(device)
    policy.eval()
    print(f"Policy loaded successfully")
    return policy


def predict_action(policy, obs, task_description, device="cuda"):
    import torch

    joint_pos = obs["joint_pos"] if "joint_pos" in obs else np.asarray(obs[:6])

    frame = {
        "observation.state": torch.from_numpy(joint_pos).float().to(device),
        "task": task_description,
    }

    if "wrist" in obs:
        img = obs["wrist"]
        if img.ndim == 3:
            img = np.transpose(img, (2, 0, 1))
        frame["observation.images.wrist"] = torch.from_numpy(img).byte().to(device)
    if "overhead" in obs:
        img = obs["overhead"]
        if img.ndim == 3:
            img = np.transpose(img, (2, 0, 1))
        frame["observation.images.overhead"] = torch.from_numpy(img).byte().to(device)

    with torch.no_grad():
        action = policy.predict_action(frame)

    if isinstance(action, dict):
        action = action["action"]
    if isinstance(action, torch.Tensor):
        action = action.cpu().numpy()

    if action.ndim == 2:
        action = action[0]

    return action


def run_replay(env_id, checkpoint, task_description, max_steps, output_path, device="cuda"):
    env = gym.make(env_id, render_mode="rgb_array", control_mode="pd_joint_pos")
    obs, info = env.reset()

    policy = load_policy(checkpoint, device)

    frames = []
    step_results = []
    success = False

    print(f"\n=== Replay Demo ===")
    print(f"Environment: {env_id}")
    print(f"Task: {task_description}")
    print(f"Max steps: {max_steps}")
    print()

    for step in range(max_steps):
        try:
            action = predict_action(policy, obs, task_description, device)
        except Exception as e:
            print(f"Step {step}: prediction error: {e}")
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        is_success = info.get("success", False)
        if is_success:
            success = True

        step_results.append({
            "step": step,
            "reward": float(reward),
            "success": is_success,
        })

        if step % 50 == 0:
            print(f"Step {step}/{max_steps}: reward={reward:.4f}, success={is_success}")

        if terminated or truncated:
            print(f"Episode ended at step {step}: terminated={terminated}, truncated={truncated}")
            break

    env.close()

    print(f"\n=== Replay Complete ===")
    print(f"Total steps: {len(step_results)}")
    print(f"Success: {success}")
    print(f"Frames: {len(frames)}")

    save_video(frames, output_path, fps=30)

    report = {
        "env_id": env_id,
        "checkpoint": checkpoint,
        "task": task_description,
        "total_steps": len(step_results),
        "success": success,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    report_path = Path(output_path).with_suffix(".json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to: {report_path}")

    return success


def save_video(frames, output_path, fps=30):
    import cv2

    if not frames:
        print("No frames to save")
        return

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame in frames:
        if frame.ndim == 3 and frame.shape[2] == 3:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            bgr = frame
        writer.write(bgr)

    writer.release()
    print(f"Video saved to: {output_path} ({len(frames)} frames, {w}x{h})")


def main():
    parser = argparse.ArgumentParser(description="SmolVLA simulation replay demo")
    parser.add_argument("--checkpoint", default="xieyucheng123/so101-smolvla",
                        help="SmolVLA checkpoint (HF repo ID or local path)")
    parser.add_argument("--env", default="MuJoCoPickAndPlace-v1",
                        help="so101_nexus environment ID")
    parser.add_argument("--task", default="pick up the red cube and place it on the green circle",
                        help="Language instruction for the policy")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--output", default="/data/eval/results/replay_pickplace.mp4")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    success = run_replay(
        env_id=args.env,
        checkpoint=args.checkpoint,
        task_description=args.task,
        max_steps=args.max_steps,
        output_path=args.output,
        device=args.device,
    )

    print(f"\nFinal result: {'SUCCESS' if success else 'FAILED'}")


if __name__ == "__main__":
    main()
