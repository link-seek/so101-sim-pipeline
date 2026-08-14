#!/usr/bin/env python3
"""PPO 评估脚本 — 加载训练好的 best_agent.pt，在 MuJoCo 后端做确定性评估。

输出:
  - eval_result.json: {success_rate, avg_reward, avg_steps, num_episodes, per_episode: [...]}
  - eval_video.mp4:   第一个 episode 的渲染视频

用法:
    python eval_ppo.py --checkpoint /data/checkpoints/ppo/best_agent.pt
    python eval_ppo.py --checkpoint /data/checkpoints/ppo/best_agent.pt --num-episodes 50
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
from torch import nn

import gymnasium as gym

import so101_nexus.mujoco  # noqa: F401
from so101_nexus import observations_from_feature_names


_STAT_BOUND = 1e6


def _finite(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=_STAT_BOUND, neginf=-_STAT_BOUND)


_LAYER_INIT_STD = float(np.sqrt(2))


def layer_init(layer, std=_LAYER_INIT_STD, bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, act_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))


def _env_cls_from_spec(env_id: str, attr: str):
    import importlib
    entry = getattr(gym.spec(env_id), attr)
    if not isinstance(entry, str):
        raise ValueError(f"{env_id} has no string {attr}")
    module_path, class_name = entry.split(":")
    return getattr(importlib.import_module(module_path), class_name)


def _mujoco_config(mujoco_env_id: str, observations):
    return _env_cls_from_spec(mujoco_env_id, "entry_point").default_config_cls(
        observations=observations
    )


def write_video(frames, path, fps=30):
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("[warn] imageio not installed; skipping video.")
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)
    return path


def main():
    parser = argparse.ArgumentParser(description="PPO evaluation on MuJoCo backend")
    parser.add_argument("--checkpoint", required=True, help="Path to best_agent.pt")
    parser.add_argument("--env-id", default="WarpPickLift-v1", help="Warp env id (eval uses MuJoCo equivalent)")
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--control-mode", default="pd_joint_delta_pos")
    parser.add_argument("--episode-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", default="/data/ppo/results", help="Output directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    names = ckpt.get("env_state_names")
    observations = None if not names else observations_from_feature_names(names)

    mujoco_id = args.env_id.replace("Warp", "MuJoCo")
    probe = gym.make(
        mujoco_id, control_mode=args.control_mode, max_episode_steps=args.episode_length,
        **({} if observations is None else {"config": _mujoco_config(mujoco_id, observations)}),
    )
    obs_shape = probe.observation_space.shape
    act_shape = probe.action_space.shape
    obs_dim = int(np.prod(obs_shape))
    act_dim = int(np.prod(act_shape))
    probe.close()

    agent = Agent(obs_dim, act_dim, args.hidden_dim).to(device)
    agent.load_state_dict(ckpt["model"])
    agent.eval()

    obs_mean = ckpt["obs_mean"].to(device).float()
    obs_var = ckpt["obs_var"].to(device).float()

    env = gym.make(
        mujoco_id, control_mode=args.control_mode,
        render_mode="rgb_array", max_episode_steps=args.episode_length,
        **({} if observations is None else {"config": _mujoco_config(mujoco_id, observations)}),
    )

    def norm(o):
        t = torch.as_tensor(o, dtype=torch.float32, device=device)
        return (((t - obs_mean) / torch.sqrt(obs_var + 1e-8)).clamp(-10.0, 10.0)).unsqueeze(0)

    print(f"[eval] env={mujoco_id} episodes={args.num_episodes} checkpoint={args.checkpoint}", flush=True)

    returns, succs, lens = [], [], []
    per_episode = []
    first_frames = None
    start_time = time.time()

    with torch.no_grad():
        for ep in range(args.num_episodes):
            obs, _ = env.reset(seed=args.seed + ep)
            ep_ret, ep_len, done = 0.0, 0, False
            ever_succ = False
            frames = [] if ep == 0 else None
            while not done:
                a = agent.actor_mean(norm(obs)).squeeze(0).cpu().numpy()
                obs, r, term, trunc, info = env.step(a)
                ep_ret += float(r)
                ep_len += 1
                ever_succ = ever_succ or bool(info.get("success", False))
                done = bool(term or trunc)
                if frames is not None:
                    frames.append(env.render())
            returns.append(ep_ret)
            succs.append(float(ever_succ))
            lens.append(ep_len)
            per_episode.append({"episode": ep, "success": ever_succ, "reward": ep_ret, "steps": ep_len})
            if ep == 0:
                first_frames = frames
            status = "SUCCESS" if ever_succ else "FAIL"
            print(f"  [Episode {ep+1}/{args.num_episodes}] {status} reward={ep_ret:.3f} steps={ep_len}", flush=True)

    env.close()
    elapsed = time.time() - start_time

    result = {
        "success_rate": float(np.mean(succs)),
        "avg_reward": float(np.mean(returns)),
        "avg_steps": float(np.mean(lens)),
        "num_episodes": args.num_episodes,
        "env_id": mujoco_id,
        "checkpoint": args.checkpoint,
        "elapsed_s": elapsed,
        "per_episode": per_episode,
    }

    os.makedirs(args.output, exist_ok=True)
    result_path = os.path.join(args.output, "eval_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    if first_frames:
        video_path = os.path.join(args.output, "eval_video.mp4")
        write_video(first_frames, video_path, fps=30)
        print(f"[video] saved to {video_path}", flush=True)

    print(
        f"\n[done] success_rate={result['success_rate']:.3f} "
        f"avg_reward={result['avg_reward']:.3f} "
        f"avg_steps={result['avg_steps']:.1f} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    print(f"[result] saved to {result_path}", flush=True)


if __name__ == "__main__":
    main()
