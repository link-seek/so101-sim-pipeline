# Ch2：PPO 纯仿真 Baseline — 最简路径，100% 成功

> SO101 仿真评测教程 · 第二章  
> 实战项目：[link-seek/so101-sim-pipeline](https://github.com/link-seek/so101-sim-pipeline)  
> 关联 Discussion：[#2 PPO 纯仿真流水线设计](https://github.com/link-seek/so101-sim-pipeline/discussions/2)

---

## 1. 为什么从 PPO 开始

在进入复杂的 VLA 调试之前，我们先用 PPO 建立一个"确定能成功"的 baseline。这有三个价值：

1. **验证基础设施**：如果 PPO 都跑不通，说明 ECS / Docker / GPU 环境有问题，不用浪费时间调 VLA
2. **验证环境可学性**：确认 WarpPickLift 任务在 V100 上可以收敛，不是环境本身有问题
3. **建立性能上限参考**：PPO 100% 的成功率为后续 VLA 的 47% 提供对比锚点

PPO 是最简路径：不需要训练数据、不需要相机、不需要语言指令，只需要 reward 信号。

---

## 2. PPO 算法速览

### 2.1 核心思想

PPO（Proximal Policy Optimization）是一种 on-policy 强化学习算法：

```
策略 π(a|s) → 在环境执行 → 收集 (s, a, r, s') → 更新 π → 重复
```

关键创新是 **Clipped Surrogate Objective**：限制策略更新幅度，避免一步更新太大导致崩溃。

### 2.2 CleanRL 风格

我们用的是 [CleanRL](https://github.com/vwxyzjn/cleanrl) 风格的实现——单文件、无框架抽象、所有超参数一目了然：

```python
# scripts/train_ppo.py 核心结构（简化）
def train(args):
    env = make_warp_env(args.env_id, args.num_envs)  # 1024 并行环境
    
    agent = Agent(obs_dim, action_dim, hidden_dim=256)
    optimizer = Adam(agent.parameters(), lr=3e-4)
    
    for update in range(num_updates):
        # 1. Rollout: 收集 16 steps × 1024 envs = 16384 transitions
        obs, actions, rewards, next_obs = rollout(env, agent, num_steps=16)
        
        # 2. GAE: 计算 advantage
        advantages = compute_gae(rewards, values, gamma=0.99, gae_lambda=0.95)
        
        # 3. Policy update: 10 epochs × 32 minibatches
        for epoch in range(10):
            for batch in minibatches(obs, actions, advantages, 32):
                loss = clipped_surrogate_loss(agent, batch, clip_coef=0.2)
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()
        
        # 4. Evaluate
        if update % eval_freq == 0:
            success = evaluate(env, agent, 10 episodes)
            if success > best_success:
                save_checkpoint(agent, "best_agent.pt")
```

### 2.3 Warp GPU 并行环境

PPO 快的秘诀是 **1024 个环境同时 step**：

```python
# so101_nexus Warp 环境
env = WarpPickLiftVecEnv(
    num_envs=1024,      # 1024 个并行环境
    episode_length=512, # 固定 horizon
    lift_threshold=0.15 # 抬起 15cm 判定成功
)

# 一次 step 同时推进 1024 个环境
obs, reward, done = env.step(actions)  # actions shape: (1024, 6)
```

V100 上 1024 并行环境的 SPS（steps per second）≈ 5784，意味着每秒收集 5784 个 transition。

---

## 3. 超参数详解

这些是 so101_nexus 官方在 RTX 5090 上调优的值，我们直接复用：

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_envs` | 1024 | GPU 并行环境数，越多越快但显存越大 |
| `num_steps` | 16 | 每次 update 的 rollout 长度 |
| `total_timesteps` | 30,000,000 | 总训练步数（~1831 updates） |
| `learning_rate` | 3e-4 | 学习率，线性退火到 0 |
| `gamma` | 0.99 | 折扣因子，越接近 1 越重视长期回报 |
| `gae_lambda` | 0.95 | GAE 参数，balance bias/variance |
| `num_minibatches` | 32 | 小批量数，1024×16/32 = 512 samples/batch |
| `update_epochs` | 10 | 每次 rollout 重复更新 10 轮 |
| `clip_coef` | 0.2 | PPO clip 参数，限制策略更新幅度 |
| `ent_coef` | 0.03 → 0.005 | 熵系数，warm-start 鼓励探索，后期收敛 |
| `vf_coef` | 0.5 | value loss 权重 |
| `max_grad_norm` | 0.5 | 梯度裁剪，防爆炸 |
| `hidden_dim` | 256 | MLP 隐藏层维度 |
| `episode_length` | 512 | 固定 horizon，避免 reach-farming |
| `control_mode` | pd_joint_delta_pos | 控制模式：关节位置增量 |

### 关键设计：固定 episode_length

不设固定 horizon 时，策略可能学会"伸手但不抓"来刷 reward（reach-farming）。固定 512 步强制策略在有限步内完成任务。

---

## 4. 实战：端到端运行

### 4.1 拉取镜像

```bash
docker pull swr.cn-north-4.myhuaweicloud.com/link-seek/so101-ppo:latest
```

### 4.2 启动训练

```bash
docker run --gpus all \
  -v /data:/data \
  swr.cn-north-4.myhuaweicloud.com/link-seek/so101-ppo:latest \
  python train_ppo.py \
    --env_id WarpPickLift-v1 \
    --total_timesteps 30000000 \
    --num_envs 1024 \
    --seed 1
```

### 4.3 训练过程

```bash
# 查看训练日志
docker logs <container_id> -f | grep -E "(success|SPS|iter)"
```

典型训练曲线：

```
iter 100  | success: 0.00 | SPS: 5784 | ent_coef: 0.028
iter 300  | success: 0.15 | SPS: 5801 | ent_coef: 0.025
iter 500  | success: 0.42 | SPS: 5768 | ent_coef: 0.022
iter 800  | success: 0.78 | SPS: 5792 | ent_coef: 0.018
iter 1200 | success: 0.93 | SPS: 5780 | ent_coef: 0.012
iter 1500 | success: 0.98 | SPS: 5785 | ent_coef: 0.008
iter 1831 | success: 0.965 | best: 0.995 | ent_coef: 0.005
```

### 4.3 评估

训练完成后自动触发评估：

```python
# scripts/eval_ppo.py（简化）
def evaluate(checkpoint_path, env_id, num_episodes=50):
    agent = load_checkpoint(checkpoint_path)
    env = make_mujoco_env(env_id)  # MuJoCo 后端，确定性
    
    successes = []
    for ep in range(num_episodes):
        obs = env.reset(seed=1000 + ep)  # 固定 seed，可复现
        for step in range(512):
            action = agent.act(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if done:
                break
        successes.append(info["success"])
    
    return {
        "success_rate": sum(successes) / num_episodes,
        "avg_reward": ...,
        "avg_steps": ...,
    }
```

### 4.4 结果

| 指标 | 值 |
|------|-----|
| **success_rate** | **1.000** (50/50) |
| avg_reward | 1.440 |
| avg_steps | 50.0 |
| 评估耗时 | 13.8s |

---

## 5. v2：提高 lift_threshold

v1 的 `lift_threshold=0.05`（5cm）太低——策略学会微抬 5cm 就判定成功，视频里看不出抬起动作。

### 修改

```bash
# 添加 --lift-threshold 参数
docker run --gpus all \
  -v /data:/data \
  swr.cn-north-4.myhuaweicloud.com/link-seek/so101-ppo:latest \
  python train_ppo.py \
    --env_id WarpPickLift-v1 \
    --total_timesteps 30000000 \
    --lift_threshold 0.15
```

### 结果对比

| 指标 | v1 (5cm) | v2 (15cm) |
|------|----------|-----------|
| best_success | 0.995 | 0.9925 |
| eval success_rate | **100%** | **98%** (49/50) |
| avg_reward | 1.440 | 1.511 |
| avg_steps | 50.0 | 58.4 |

**分析**：
- success_rate 仅降 2%，说明策略能可靠抓取并抬起 15cm
- avg_reward 提高因为更高 threshold 给更高 reward 信号
- avg_steps 增加因为需要更多步骤完成更高抬起
- 唯一失败 episode (34) 跑满 512 步未成功

### 与官方对比

| 环境 | GPU | 耗时 | SPS | best_success | eval |
|------|-----|------|-----|-------------|------|
| 官方 RTX 5090 | sm_90 | 24.5 min | ~20k | 0.973 (5 seed) | - |
| **本项目 V100** | sm_70 | **86 min** | **5784** | **0.995** | **1.000** |

V100 慢 3.5x 但结果更好（可能 seed 运气）。算法不依赖 GPU 架构，只是速度差异。

---

## 6. 产物归档

```
obs://so101-sim-pipeline/ppo/
├── train_result.json     # 训练指标
├── eval_result.json      # 评估指标
├── eval_video.mp4        # 评估视频回放
└── checkpoints/
    └── best_agent.pt     # 策略权重 (591KB)
```

`train_result.json` 示例：

```json
{
  "env_id": "WarpPickLift-v1",
  "total_timesteps": 30000000,
  "best_success": 0.995,
  "final_success": 0.965,
  "elapsed_s": 5186,
  "sps": 5784,
  "seed": 1
}
```

---

## 踩坑复盘

### 坑 1：lift_threshold 太低导致"假成功"

**现象**：success_rate=100%，但视频里物体几乎没动。

**根因**：`lift_threshold=0.05` 意味着抬起 5cm 就算成功。策略学会微抬 5cm 骗过 reward。

**修复**：提高到 0.15（15cm），视频中有明显抬起动作。

**教训**：reward 设计要和任务语义对齐。"抬起"应该是肉眼可见的高度，不是 5cm。

### 坑 2：mujoco-warp 版本 bug

**现象**：`undefined _magnetometer_0, _cam_projection_0` NVRTC 编译错误。

**根因**：so101_nexus 限制 `mujoco-warp<3.10`，3.9.x 的 sensor.py 有代码生成 bug。

**修复**：`pip install mujoco-warp>=3.10.0.1,<3.12`

**教训**：遇到底层编译错误，先查依赖版本约束的 changelog 和 GitHub issues。

---

## 思考题

1. **为什么 PPO success_rate 能到 100% 而 VLA 只有 47%？**  
   提示：PPO 直接在评估环境中训练（on-policy），VLA 从固定数据集学习（off-policy），泛化性受限。

2. **`episode_length=512` 固定 horizon 有什么意义？**  
   提示：不固定的话策略可以"伸手但不抓"无限刷 reward（reach-farming）。

3. **如果换 PickAndPlace 任务（需要放置），PPO 还能收敛吗？**  
   提示：PickAndPlace 比 PickLift 更难（需要抓+移+放），reward 更稀疏。官方标注 pending。

4. **V100 SPS 5784 vs RTX 5090 ~20k，为什么结果一致？**  
   提示：PPO 是 on-policy 算法，收敛性取决于超参数和 reward 设计，不依赖 GPU 架构。GPU 只影响速度。

---

> **上一章**：[Ch1 基础设施](so101-tutorial-ch1-infrastructure.md) | **下一章**：[Ch3 SmolVLA 仿真训练入门](so101-tutorial-ch3-vla-intro.md)
