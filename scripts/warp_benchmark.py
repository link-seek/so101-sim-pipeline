#!/usr/bin/env python3
"""Warp 后端渲染测试和吞吐量基准测试"""

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


def test_warp_rendering():
    """测试 Warp 带相机渲染是否正常工作（验证 bug 修复）"""
    import so101_nexus
    import so101_nexus.warp
    import gymnasium as gym

    print("=== Warp 渲染测试 ===")
    config = so101_nexus.PickConfig(
        observations=[
            so101_nexus.JointPositions(),
            so101_nexus.WristCamera(),
            so101_nexus.OverheadCamera(),
        ]
    )

    env = gym.make_vec("WarpPickLift-v1", num_envs=4, device="cuda", config=config)
    obs, info = env.reset()

    print(f"obs type: {type(obs)}")
    if isinstance(obs, dict):
        for k, v in obs.items():
            print(f"  obs['{k}']: type={type(v).__name__}, shape={v.shape if hasattr(v, 'shape') else 'N/A'}, dtype={v.dtype if hasattr(v, 'dtype') else 'N/A'}")
    elif isinstance(obs, torch.Tensor):
        print(f"  obs shape: {obs.shape}, dtype: {obs.dtype}")
    else:
        print(f"  obs: {obs}")

    print(f"info type: {type(info)}")
    if isinstance(info, dict):
        for k, v in info.items():
            print(f"  info['{k}']: type={type(v).__name__}")

    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

    if isinstance(obs, dict):
        for k, v in obs.items():
            if "image" in k or "camera" in k or "wrist" in k or "overhead" in k:
                sample = v[0] if v.ndim > 3 else v
                print(f"  {k} sample: min={sample.min().item():.1f}, max={sample.max().item():.1f}, mean={sample.float().mean().item():.1f}")
                img = sample.cpu().numpy().astype(np.uint8) if sample.dtype != torch.uint8 else sample.cpu().numpy()
                img_path = RESULTS_DIR / f"warp_{k}_sample.png"
                try:
                    from PIL import Image
                    if img.ndim == 3 and img.shape[-1] == 3:
                        Image.fromarray(img).save(img_path)
                        print(f"  Saved {img_path}")
                except Exception as e:
                    print(f"  Could not save image: {e}")
    elif isinstance(obs, torch.Tensor):
        print(f"  obs sample: min={obs.min().item():.4f}, max={obs.max().item():.4f}, mean={obs.float().mean().item():.4f}")

    env.close()
    print("Warp 渲染测试通过！\n")
    return True


def benchmark_warp(num_envs_list, max_steps=500):
    """测量不同 num_envs 下的吞吐量"""
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

    results = []
    for num_envs in num_envs_list:
        print(f"--- num_envs={num_envs} ---")
        try:
            env = gym.make_vec("WarpPickLift-v1", num_envs=num_envs, device="cuda", config=config)
            obs, info = env.reset()

            warmup = 10
            for _ in range(warmup):
                obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

            torch.cuda.synchronize()
            start = time.time()
            for _ in range(max_steps):
                obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            torch.cuda.synchronize()
            elapsed = time.time() - start

            env_steps_per_sec = (num_envs * max_steps) / elapsed
            fps = max_steps / elapsed

            results.append({
                "num_envs": num_envs,
                "env_steps_per_sec": env_steps_per_sec,
                "fps": fps,
                "elapsed": elapsed,
            })
            print(f"  {env_steps_per_sec:.0f} env-steps/s, {fps:.1f} fps, {elapsed:.2f}s")
            env.close()

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"num_envs": num_envs, "error": str(e)})

    return results


def benchmark_mujoco(max_steps=500):
    """测量 MuJoCo 单环境吞吐量作为对比"""
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

    print("--- MuJoCo single env ---")
    env = gym.make("MuJoCoPickLift-v1", config=config)
    obs, info = env.reset()

    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

    start = time.time()
    for _ in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
    elapsed = time.time() - start

    env_steps_per_sec = max_steps / elapsed
    print(f"  {env_steps_per_sec:.0f} env-steps/s, {elapsed:.2f}s")
    env.close()
    return {"num_envs": 1, "env_steps_per_sec": env_steps_per_sec, "elapsed": elapsed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=str, default="1,4,16,64,128")
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args()

    num_envs_list = [int(x) for x in args.num_envs.split(",")]

    print("=== Warp 后端渲染 + 吞吐量基准测试 ===\n")

    rendering_ok = test_warp_rendering()

    print("=== Warp 吞吐量基准测试（带相机）===\n")
    warp_results = benchmark_warp(num_envs_list, args.max_steps)

    print("\n=== MuJoCo 对比基准（带相机）===\n")
    mujoco_result = benchmark_mujoco(args.max_steps)

    report = {
        "rendering_ok": rendering_ok,
        "warp_results": warp_results,
        "mujoco_result": mujoco_result,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    report_path = RESULTS_DIR / "warp_benchmark.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n=== 基准测试完成 ===")
    print(f"报告已保存到 {report_path}")

    if warp_results and mujoco_result:
        best_warp = max((r.get("env_steps_per_sec", 0) for r in warp_results if "env_steps_per_sec" in r), default=0)
        mujoco_fps = mujoco_result.get("env_steps_per_sec", 0)
        if mujoco_fps > 0 and best_warp > 0:
            print(f"Warp 最佳: {best_warp:.0f} env-steps/s")
            print(f"MuJoCo 单环境: {mujoco_fps:.0f} env-steps/s")
            print(f"加速比: {best_warp / mujoco_fps:.1f}x")


if __name__ == "__main__":
    main()
