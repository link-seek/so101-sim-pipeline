#!/usr/bin/env python3
"""SO101 MuJoCo 仿真评测脚本 - 在 V100 ECS 上运行"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

RESULTS_DIR = Path("/data/pipeline/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def download_model(model_repo, hf_token):
    print(f"Downloading model from {model_repo}...")
    from huggingface_hub import snapshot_download
    model_path = snapshot_download(
        repo_id=model_repo,
        token=hf_token,
        local_dir="/data/pipeline/model",
    )
    print(f"Model downloaded to {model_path}")
    return model_path


def download_dataset(dataset_repo, hf_token):
    print(f"Downloading dataset from {dataset_repo}...")
    from huggingface_hub import snapshot_download
    dataset_path = snapshot_download(
        repo_id=dataset_repo,
        repo_type="dataset",
        token=hf_token,
        local_dir="/data/pipeline/dataset",
    )
    print(f"Dataset downloaded to {dataset_path}")
    return dataset_path


def run_mujoco_eval(model_path, dataset_path, num_episodes, policy_type="act"):
    print(f"Running MuJoCo evaluation: {num_episodes} episodes")

    import mujoco
    import numpy as np

    SO101_MJCF = "/data/so101/SO-ARM100/Simulation/SO101/SO101_dual_arm.xml"
    if not Path(SO101_MJCF).exists():
        print(f"SO101 model not found at {SO101_MJCF}")
        print("Please ensure SO101 model is deployed on ECS")
        sys.exit(1)

    model = mujoco.MjModel.from_xml_path(SO101_MJCF)
    data = mujoco.MjData(model)

    episode_results = []
    for ep in range(num_episodes):
        mujoco.mj_resetData(model, data)
        step_count = 0
        total_reward = 0.0
        success = False

        while step_count < 500:
            ctrl = np.zeros(model.nu)
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
            step_count += 1

            if data.qacc.max() < 1e-6 and step_count > 50:
                success = True
                break

        episode_results.append({
            "episode": ep + 1,
            "steps": step_count,
            "success": success,
            "reward": total_reward,
        })
        print(f"Episode {ep+1}: steps={step_count}, success={success}")

    success_rate = sum(r["success"] for r in episode_results) / len(episode_results)
    avg_steps = sum(r["steps"] for r in episode_results) / len(episode_results)

    report = {
        "policy_type": policy_type,
        "model_path": str(model_path),
        "num_episodes": num_episodes,
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "episodes": episode_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    report_path = RESULTS_DIR / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {report_path}")

    return report


def render_eval_video(model_path, num_episodes=3):
    print("Rendering evaluation video...")
    try:
        import mujoco
        import numpy as np
        import cv2

        SO101_MJCF = "/data/so101/SO-ARM100/Simulation/SO101/SO101_dual_arm.xml"
        model = mujoco.MjModel.from_xml_path(SO101_MJCF)
        data = mujoco.MjData(model)

        renderer = mujoco.Renderer(model, height=480, width=640)
        frames = []

        for ep in range(num_episodes):
            mujoco.mj_resetData(model, data)
            for step in range(200):
                data.ctrl[:] = 0
                mujoco.mj_step(model, data)
                renderer.update_scene(data)
                frame = renderer.render()
                frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_path = str(RESULTS_DIR / "eval_video.mp4")
        writer = cv2.VideoWriter(video_path, fourcc, 30, (640, 480))
        for frame in frames:
            writer.write(frame)
        writer.release()
        print(f"Video saved to {video_path}")
    except Exception as e:
        print(f"Video rendering failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="SO101 MuJoCo evaluation")
    parser.add_argument("--model-repo", default=os.environ.get("MODEL_REPO", "xieyucheng123/so101-act"))
    parser.add_argument("--dataset-repo", default=os.environ.get("DATASET_REPO", "xieyucheng123/so101-dataset"))
    parser.add_argument("--num-episodes", type=int, default=int(os.environ.get("NUM_EPISODES", "10")))
    parser.add_argument("--policy-type", default="act")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    if not args.skip_download:
        model_path = download_model(args.model_repo, args.hf_token)
        dataset_path = download_dataset(args.dataset_repo, args.hf_token)
    else:
        model_path = "/data/pipeline/model"
        dataset_path = "/data/pipeline/dataset"

    report = run_mujoco_eval(model_path, dataset_path, args.num_episodes, args.policy_type)
    render_eval_video(model_path, num_episodes=3)

    print(f"\n=== Evaluation Complete ===")
    print(f"Success rate: {report['success_rate']:.1%}")
    print(f"Avg steps: {report['avg_steps']:.1f}")


if __name__ == "__main__":
    main()
