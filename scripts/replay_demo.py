#!/usr/bin/env python3
"""仿真回放演示 - 用训练好的 SmolVLA 策略在 so101_nexus 仿真中执行任务并录像

使用社区标准推理管线:
  build_inference_frame / prepare_observation_for_inference
  → preprocess → select_action → postprocess
  → sim_qpos_to_dataset_row / dataset_row_to_sim_qpos 单位转换
"""

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
from so101_nexus import (
    PickAndPlaceConfig,
    WristCamera,
    OverheadCamera,
    JointPositions,
    sim_qpos_to_dataset_row,
    SO101_JOINT_NAMES,
)

RESULTS_DIR = Path("/data/eval/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_policy_and_processors(checkpoint, device="cuda"):
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors

    print(f"Loading SmolVLA policy from {checkpoint}...")
    policy = SmolVLAPolicy.from_pretrained(checkpoint)
    policy.to(device)
    policy.eval()

    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    print(f"Policy and processors loaded successfully")
    return policy, preprocess, postprocess


def build_dataset_features():
    from so101_nexus.teleop.dataset import FieldSelection, build_features

    action_features = {f"{name}.pos": float for name in SO101_JOINT_NAMES}
    follower_features = {
        **action_features,
        "wrist": (480, 640, 3),
        "overhead": (480, 640, 3),
    }
    features = build_features(FieldSelection(), follower_features, action_features)
    return features


def predict_action(policy, preprocess, postprocess, obs, task_description, device="cuda"):
    import torch
    from lerobot.policies.utils import prepare_observation_for_inference

    joint_pos = obs["state"] if "state" in obs else np.asarray(obs[:6])
    state_ds = sim_qpos_to_dataset_row(np.asarray(joint_pos))

    frame = {
        "observation.state": np.asarray(state_ds, dtype=np.float32),
    }

    if "wrist_camera" in obs:
        frame["observation.images.camera1"] = np.asarray(obs["wrist_camera"], dtype=np.float32)
    if "overhead_camera" in obs:
        frame["observation.images.camera2"] = np.asarray(obs["overhead_camera"], dtype=np.float32)
    if "wrist_camera" in obs and "overhead_camera" in obs:
        frame["observation.images.camera3"] = np.asarray(obs["overhead_camera"], dtype=np.float32)

    frame = prepare_observation_for_inference(
        frame, device, task=task_description, robot_type=""
    )
    frame = preprocess(frame)

    with torch.inference_mode():
        action = policy.select_action(frame)

    action = postprocess(action)

    if isinstance(action, dict):
        action = action["action"]
    if isinstance(action, torch.Tensor):
        action = action.squeeze().cpu().numpy()

    return action


def dataset_row_to_sim_qpos(row):
    from so101_nexus import SO101_GRIPPER_LIMITS_RAD
    import math

    values = np.asarray(row, dtype=np.float64).copy()
    lower, upper = SO101_GRIPPER_LIMITS_RAD
    deg2rad = math.pi / 180.0
    sim = values * deg2rad
    sim[-1] = lower + (values[-1] / 100.0) * (upper - lower)
    return sim


def run_replay(env_id, checkpoint, task_description, max_steps, output_path, device="cuda"):
    env_config = PickAndPlaceConfig(
        observations=[JointPositions(), WristCamera(width=640, height=480), OverheadCamera(width=640, height=480)],
    )
    env = gym.make(env_id, config=env_config, render_mode="rgb_array", control_mode="pd_joint_pos")
    obs, info = env.reset()

    policy, preprocess, postprocess = load_policy_and_processors(checkpoint, device)

    frames = []
    step_results = []
    success = False
    prediction_errors = 0
    first_action = None

    print(f"\n=== Replay Demo ===")
    print(f"Environment: {env_id}")
    print(f"Task: {task_description}")
    print(f"Max steps: {max_steps}")
    print()

    for step in range(max_steps):
        try:
            action_ds = predict_action(
                policy, preprocess, postprocess, obs, task_description, device
            )
            action_rad = dataset_row_to_sim_qpos(action_ds)
            if first_action is None:
                first_action = action_rad.copy()
                print(f"First action (rad): {action_rad}")
                print(f"First action (ds):  {action_ds}")
        except Exception as e:
            if prediction_errors == 0:
                print(f"Step {step}: FIRST prediction error: {e}")
            prediction_errors += 1
            action_rad = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action_rad)

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

        if step % 10 == 0:
            current_state = obs["state"] if "state" in obs else np.asarray(obs[:6])
            print(f"Step {step}/{max_steps}: reward={reward:.4f}, success={is_success}")
            print(f"  state(rad): {np.asarray(current_state).round(4)}")
            print(f"  action(rad): {action_rad.round(4)}")
            print(f"  action(ds):  {action_ds.round(4)}")

        if terminated or truncated:
            print(f"Episode ended at step {step}: terminated={terminated}, truncated={truncated}")
            break

    env.close()

    print(f"\n=== Replay Complete ===")
    print(f"Total steps: {len(step_results)}")
    print(f"Success: {success}")
    print(f"Frames: {len(frames)}")
    print(f"Prediction errors: {prediction_errors}/{len(step_results)}")
    if prediction_errors == 0:
        print(f"Model controlled robot for ALL steps (no random fallback)")

    save_video(frames, output_path, fps=30)

    report = {
        "env_id": env_id,
        "checkpoint": checkpoint,
        "task": task_description,
        "total_steps": len(step_results),
        "success": success,
        "prediction_errors": prediction_errors,
        "model_in_control": prediction_errors == 0,
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
    parser.add_argument("--task", default="Pick up the red cube and place it on the blue circle.",
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
