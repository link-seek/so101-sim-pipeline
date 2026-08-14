#!/usr/bin/env python3
"""PPO 训练脚本 — 基于 SO101-Nexus 官方 ppo_warp.py 适配流水线。

CleanRL 风格 MLP actor-critic，GPU 并行 Warp 环境。
WarpPickLift-v1 官方 seed-validated: 0.973 mean (5 seed, 30M steps, RTX 5090)。

用法:
    python train_ppo.py --env-id WarpPickLift-v1 --total-timesteps 30000000
    python train_ppo.py --env-id WarpPickLift-v1 --total-timesteps 30000000 --num-envs 512 --seed 3
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import importlib

import numpy as np
import torch
from torch import nn

from so101_nexus import observations_from_feature_names, PickConfig
from so101_nexus._reproducibility import seed_everything


_STAT_BOUND = 1e6


def _finite(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=_STAT_BOUND, neginf=-_STAT_BOUND)


class RunningMeanStd:
    def __init__(self, shape, device, epsilon=1e-4):
        self.mean = torch.zeros(shape, dtype=torch.float64, device=device)
        self.var = torch.ones(shape, dtype=torch.float64, device=device)
        self.count = epsilon

    def update(self, x: torch.Tensor) -> None:
        x = _finite(x.to(torch.float64))
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / tot
        self.var = m2 / tot
        self.count = tot


class ObsNormalizer:
    def __init__(self, obs_dim, device, enabled=True):
        self.rms = RunningMeanStd((obs_dim,), device)
        self.enabled = enabled

    def __call__(self, obs: torch.Tensor, update: bool = True) -> torch.Tensor:
        if not self.enabled:
            return obs
        obs = _finite(obs)
        if update:
            self.rms.update(obs)
        normed = (obs - self.rms.mean.float()) / torch.sqrt(self.rms.var.float() + 1e-8)
        return normed.clamp(-10.0, 10.0)


class RewardScaler:
    def __init__(self, num_envs, device, gamma, enabled=True):
        self.ret = torch.zeros(num_envs, dtype=torch.float64, device=device)
        self.rms = RunningMeanStd((), device)
        self.gamma = gamma
        self.enabled = enabled

    def __call__(self, reward: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return reward
        reward = _finite(reward)
        self.ret = self.ret * self.gamma + reward.to(torch.float64)
        self.rms.update(self.ret)
        scaled = reward / torch.sqrt(self.rms.var.float() + 1e-8)
        self.ret = self.ret * (1.0 - done.to(torch.float64))
        return scaled


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

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, *, generator=None):
        mean = self.actor_mean(x)
        logstd = self.actor_logstd.clamp(-5.0, 0.0).expand_as(mean)
        std = torch.exp(logstd)
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
            action = mean + std * noise
        logprob = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        return action, logprob, entropy, self.critic(x).squeeze(-1)


def _env_cls_from_spec(env_id: str, attr: str):
    import gymnasium as gym
    import so101_nexus.mujoco
    import so101_nexus.warp  # noqa: F401

    entry = getattr(gym.spec(env_id), attr)
    if not isinstance(entry, str):
        raise ValueError(f"{env_id} has no string {attr}")
    module_path, class_name = entry.split(":")
    return getattr(importlib.import_module(module_path), class_name)


def _resolve_env_cls(env_id: str):
    return _env_cls_from_spec(env_id, "vector_entry_point")


def _mujoco_config(mujoco_env_id: str, observations, lift_threshold=None):
    kwargs = {}
    if observations is not None:
        kwargs["observations"] = observations
    if lift_threshold is not None:
        kwargs["lift_threshold"] = lift_threshold
    return _env_cls_from_spec(mujoco_env_id, "entry_point").default_config_cls(**kwargs)


def _fixed_horizon(env_cls):
    class _FixedHorizon(env_cls):
        def _compute_reward_terminated(self, *args, **kwargs):
            reward, success, info = super()._compute_reward_terminated(*args, **kwargs)
            return reward, torch.zeros_like(success), info

    _FixedHorizon.__name__ = f"FixedHorizon{env_cls.__name__}"
    return _FixedHorizon


def _make_envs(env_id, num_envs, device, seed, *, control_mode="pd_joint_delta_pos",
               episode_length=512, terminate_on_success=False, observations=None,
               lift_threshold=None):
    env_cls = _resolve_env_cls(env_id)
    config = None
    if observations is not None or lift_threshold is not None:
        kwargs = {}
        if observations is not None:
            kwargs["observations"] = observations
        if lift_threshold is not None:
            kwargs["lift_threshold"] = lift_threshold
        config = env_cls.default_config_cls(**kwargs)
    if not terminate_on_success:
        env_cls = _fixed_horizon(env_cls)
    return env_cls(
        num_envs=num_envs, config=config, control_mode=control_mode,
        device=str(device), max_episode_steps=episode_length, seed=seed,
    )


def evaluate_mujoco(agent, obs_norm, device, *, env_id, control_mode, episode_length,
                    eval_episodes, seed, capture_video, observations=None, lift_threshold=None):
    import gymnasium as gym
    import so101_nexus.mujoco  # noqa: F401

    mujoco_id = env_id.replace("Warp", "MuJoCo")
    config_kwargs = {}
    if observations is not None or lift_threshold is not None:
        config_kwargs["config"] = _mujoco_config(mujoco_id, observations, lift_threshold)
    env = gym.make(
        mujoco_id, control_mode=control_mode,
        render_mode="rgb_array" if capture_video else None,
        max_episode_steps=episode_length,
        **config_kwargs,
    )
    mean = obs_norm.rms.mean.to(device).float()
    var = obs_norm.rms.var.to(device).float()

    def norm(o):
        t = torch.as_tensor(o, dtype=torch.float32, device=device)
        return (((t - mean) / torch.sqrt(var + 1e-8)).clamp(-10.0, 10.0)).unsqueeze(0)

    returns, succs, lens = [], [], []
    frames = None
    with torch.no_grad():
        for ep in range(eval_episodes):
            obs, _ = env.reset(seed=seed + 1000 + ep)
            ep_ret, ep_len, done = 0.0, 0, False
            ever_succ = False
            capture = capture_video and ep == 0
            if capture:
                frames = []
            while not done:
                a = agent.actor_mean(norm(obs)).squeeze(0).cpu().numpy()
                obs, r, term, trunc, info = env.step(a)
                ep_ret += float(r)
                ep_len += 1
                ever_succ = ever_succ or bool(info.get("success", False))
                done = bool(term or trunc)
                if capture and frames is not None:
                    frames.append(env.render())
            returns.append(ep_ret)
            succs.append(float(ever_succ))
            lens.append(ep_len)
    env.close()
    metrics = {
        "eval/return": float(np.mean(returns)),
        "eval/success_rate": float(np.mean(succs)),
        "eval/ep_len": float(np.mean(lens)),
    }
    return metrics, frames


def write_video(frames, path, fps=30):
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("[warn] imageio not installed; skipping video.")
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)
    return path


def _save(agent, obs_norm, path, step, success):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": agent.state_dict(),
            "obs_mean": obs_norm.rms.mean.cpu(),
            "obs_var": obs_norm.rms.var.cpu(),
            "step": step,
            "success": success,
        },
        path,
    )


def train(*, env_id="WarpPickLift-v1", num_envs=1024, num_steps=16,
           total_timesteps=30_000_000, learning_rate=3e-4, anneal_lr=True,
           gamma=0.99, gae_lambda=0.95, num_minibatches=32, update_epochs=10,
           norm_adv=True, clip_coef=0.2, clip_vloss=True,
           ent_coef=0.03, ent_coef_final=0.005, vf_coef=0.5, max_grad_norm=0.5,
           target_kl=None, norm_obs=True, norm_reward=True, hidden_dim=256,
           control_mode="pd_joint_delta_pos", episode_length=512,
           terminate_on_success=False, success_bonus=0.0, stagger_resets=True,
           capture_video=False, eval_freq=0, eval_episodes=5,
           device="cuda", seed=1, torch_deterministic=True,
           save_dir=None, log_freq=1, log=False, lift_threshold=None):
    batch_size = num_envs * num_steps
    if num_minibatches < 1 or num_minibatches > batch_size:
        raise ValueError(f"num_minibatches must be in [1, {batch_size}], got {num_minibatches}")
    if batch_size % num_minibatches != 0:
        raise ValueError(f"batch_size ({batch_size}) must be divisible by num_minibatches ({num_minibatches})")
    if total_timesteps < batch_size:
        raise ValueError(f"total_timesteps ({total_timesteps}) must be >= batch_size ({batch_size})")
    minibatch_size = batch_size // num_minibatches
    num_updates = total_timesteps // batch_size

    seed_everything(seed, deterministic=torch_deterministic)
    dev = torch.device(device)
    np_rng = np.random.default_rng(seed)
    policy_rng = torch.Generator(device=dev).manual_seed(seed + 1)
    stagger_rng = torch.Generator(device=dev).manual_seed(seed + 2)

    envs = _make_envs(
        env_id, num_envs, dev, seed,
        control_mode=control_mode, episode_length=episode_length,
        terminate_on_success=terminate_on_success, lift_threshold=lift_threshold,
    )
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))

    agent = Agent(obs_dim, act_dim, hidden_dim).to(dev)
    optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    obs_norm = ObsNormalizer(obs_dim, dev, enabled=norm_obs)
    rew_scaler = RewardScaler(num_envs, dev, gamma, enabled=norm_reward)

    obs_buf = torch.zeros((num_steps, num_envs, obs_dim), device=dev)
    act_buf = torch.zeros((num_steps, num_envs, act_dim), device=dev)
    logp_buf = torch.zeros((num_steps, num_envs), device=dev)
    rew_buf = torch.zeros((num_steps, num_envs), device=dev)
    done_buf = torch.zeros((num_steps, num_envs), device=dev)
    val_buf = torch.zeros((num_steps, num_envs), device=dev)

    ep_ret = torch.zeros(num_envs, device=dev)
    ep_len = torch.zeros(num_envs, device=dev)
    ep_succeeded = torch.zeros(num_envs, dtype=torch.bool, device=dev)
    ret_hist: deque = deque(maxlen=400)
    succ_hist: deque = deque(maxlen=400)
    len_hist: deque = deque(maxlen=400)

    global_step = 0
    start_time = time.time()
    best_success = -1.0

    next_obs_raw, _ = envs.reset(seed=seed)
    if stagger_resets:
        envs._elapsed = torch.randint(
            0, episode_length, (num_envs,), generator=stagger_rng, device=dev
        )
    next_obs = obs_norm(next_obs_raw.to(dev), update=True)
    next_done = torch.zeros(num_envs, device=dev)

    pg_loss = v_loss = entropy_loss = torch.tensor(0.0)
    approx_kl = torch.tensor(0.0)
    clipfracs: list[float] = []
    succ_rate = mean_ret = mean_len = 0.0

    for update in range(1, num_updates + 1):
        if anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = frac * learning_rate
        ent_now = ent_coef + (ent_coef_final - ent_coef) * ((update - 1.0) / num_updates)

        hold_sum = 0.0
        for step in range(num_steps):
            global_step += num_envs
            obs_buf[step] = next_obs
            done_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs, generator=policy_rng)
            act_buf[step] = action
            logp_buf[step] = logprob
            val_buf[step] = value

            next_obs_raw, reward, terminated, truncated, info = envs.step(action)
            next_obs_raw = _finite(next_obs_raw.to(dev))
            reward = _finite(reward.to(dev))
            terminated = terminated.to(dev)
            done = (terminated | truncated.to(dev)).float()
            succ = info["success"].to(dev).bool()
            hold_sum += float(succ.float().mean())

            first_success = succ & ~ep_succeeded
            shaped = reward + success_bonus * first_success.float()
            rew_buf[step] = rew_scaler(shaped, done)

            ep_ret += reward
            ep_len += 1
            ep_succeeded |= succ
            done_mask = done.bool()
            if bool(done_mask.any()):
                idx = done_mask.nonzero(as_tuple=True)[0]
                for r_, l_, s_ in zip(
                    ep_ret[idx].tolist(), ep_len[idx].tolist(), ep_succeeded[idx].tolist(),
                    strict=False,
                ):
                    ret_hist.append(r_)
                    len_hist.append(l_)
                    succ_hist.append(float(s_))
                ep_ret[idx] = 0.0
                ep_len[idx] = 0.0
                ep_succeeded[idx] = False

            next_obs = obs_norm(next_obs_raw, update=True)
            next_done = done

        with torch.no_grad():
            next_value = agent.get_value(next_obs).squeeze(-1)
            advantages = torch.zeros_like(rew_buf)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - done_buf[t + 1]
                    nextvalues = val_buf[t + 1]
                delta = rew_buf[t] + gamma * nextvalues * nextnonterminal - val_buf[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + val_buf

        b_obs = obs_buf.reshape(-1, obs_dim)
        b_logp = logp_buf.reshape(-1)
        b_act = act_buf.reshape(-1, act_dim)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)
        b_val = val_buf.reshape(-1)

        b_inds = np.arange(batch_size)
        clipfracs = []
        for _epoch in range(update_epochs):
            np_rng.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                mb = b_inds[start : start + minibatch_size]
                _, newlogp, entropy, newval = agent.get_action_and_value(b_obs[mb], b_act[mb])
                logratio = newlogp - b_logp[mb]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > clip_coef).float().mean().item())

                mb_adv = b_adv[mb]
                if norm_adv and mb_adv.numel() > 1:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                if clip_vloss:
                    v_unclipped = (newval - b_ret[mb]) ** 2
                    v_clipped = b_val[mb] + torch.clamp(newval - b_val[mb], -clip_coef, clip_coef)
                    v_clipped = (v_clipped - b_ret[mb]) ** 2
                    v_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()
                else:
                    v_loss = 0.5 * ((newval - b_ret[mb]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_now * entropy_loss + vf_coef * v_loss

                optimizer.zero_grad(set_to_none=True)
                if torch.isfinite(loss):
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                    optimizer.step()

            if target_kl is not None and approx_kl > target_kl:
                break

        succ_rate = float(np.mean(succ_hist)) if succ_hist else 0.0
        mean_ret = float(np.mean(ret_hist)) if ret_hist else 0.0
        mean_len = float(np.mean(len_hist)) if len_hist else 0.0
        hold_frac = hold_sum / num_steps
        sps = int(global_step / (time.time() - start_time))

        if succ_rate > best_success and len(succ_hist) >= 100:
            best_success = succ_rate
            if save_dir:
                _save(agent, obs_norm, f"{save_dir}/best_agent.pt", global_step, succ_rate)

        if log and (update % log_freq == 0 or update == num_updates):
            print(
                f"[step {global_step}] success={succ_rate:.3f} return={mean_ret:.1f} "
                f"ent={ent_now:.4f} SPS={sps}",
                flush=True,
            )

        if capture_video and eval_freq > 0 and (update % eval_freq == 0 or update == num_updates):
            try:
                ev, frames = evaluate_mujoco(
                    agent, obs_norm, dev, env_id=env_id, control_mode=control_mode,
                    episode_length=episode_length, eval_episodes=eval_episodes,
                    seed=seed, capture_video=True, lift_threshold=lift_threshold,
                )
            except Exception as e:
                print(f"[eval {global_step}] skipped: {e}", flush=True)
                ev, frames = {}, None
            if log:
                print(f"[eval {global_step}] {ev}", flush=True)
            if frames and save_dir:
                write_video(frames, f"{save_dir}/videos/eval_{global_step}.mp4", fps=30)

    if save_dir:
        _save(agent, obs_norm, f"{save_dir}/agent.pt", global_step, succ_rate)

    envs.close()
    elapsed = time.time() - start_time
    return {
        "iterations": num_updates,
        "policy_loss": float(pg_loss.item()),
        "value_loss": float(v_loss.item()),
        "mean_return": float(np.mean(ret_hist)) if ret_hist else float("nan"),
        "episodes": len(ret_hist),
        "success_rate": succ_rate,
        "best_success": max(best_success, 0.0),
        "total_timesteps": total_timesteps,
        "env_id": env_id,
        "seed": seed,
        "sps": sps,
        "elapsed_s": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="PPO training for SO101-Nexus Warp envs")
    parser.add_argument("--env-id", default="WarpPickLift-v1")
    parser.add_argument("--total-timesteps", type=int, default=30_000_000)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--num-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-minibatches", type=int, default=32)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.03)
    parser.add_argument("--ent-coef-final", type=float, default=0.005)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--control-mode", default="pd_joint_delta_pos")
    parser.add_argument("--episode-length", type=int, default=512)
    parser.add_argument("--save-dir", default="/data/checkpoints/ppo")
    parser.add_argument("--result-path", default="/data/ppo/results/train_result.json")
    parser.add_argument("--eval-freq", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--capture-video", action="store_true")
    parser.add_argument("--log-freq", type=int, default=10)
    parser.add_argument("--lift-threshold", type=float, default=None,
                        help="Min lift height for success (default: env default 0.05)")
    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cfg] env={args.env_id} device={device} total_timesteps={args.total_timesteps}", flush=True)

    stats = train(
        env_id=args.env_id,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        ent_coef_final=args.ent_coef_final,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        hidden_dim=args.hidden_dim,
        control_mode=args.control_mode,
        episode_length=args.episode_length,
        capture_video=args.capture_video,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        device=str(device),
        seed=args.seed,
        save_dir=args.save_dir,
        log_freq=args.log_freq,
        log=True,
        lift_threshold=args.lift_threshold,
    )

    print(
        f"[done] best_success={stats['best_success']:.3f} "
        f"final_success={stats['success_rate']:.3f} "
        f"elapsed={stats['elapsed_s']:.1f}s sps={stats['sps']}",
        flush=True,
    )

    os.makedirs(os.path.dirname(args.result_path), exist_ok=True)
    with open(args.result_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[result] saved to {args.result_path}", flush=True)


if __name__ == "__main__":
    main()
