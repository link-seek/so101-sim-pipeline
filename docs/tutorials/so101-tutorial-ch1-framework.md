# Ch1：仿真与评测技术框架 — 理解你运行的每一行代码

> SO101 仿真评测教程 · 第一章  
> 实战项目：[link-seek/so101-sim-pipeline](https://github.com/link-seek/so101-sim-pipeline)  
> **目标**：在动手搭建之前，先建立完整的知识框架——当你点击 "Run workflow" 时，清楚地知道每一步在做什么、为什么需要。

---

## 1. 仿真环境：机器人的虚拟世界

### 1.1 什么是物理仿真引擎

物理仿真引擎是一个"虚拟物理世界"——它用数学模型模拟现实世界的物理规律，让机器人在不接触真实硬件的情况下"活"在一个虚拟空间里。

```
真实世界                          仿真世界
┌──────────────┐                ┌──────────────────┐
│ 机器人 (金属)  │                │ 机器人 (URDF/XML) │
│ 桌面 (木材)    │   ──建模──→   │ 桌面 (几何体)      │
│ 物体 (塑料)    │                │ 物体 (碰撞体)      │
│ 重力/摩擦/接触 │                │ 重力/摩擦/接触模型 │
│ 相机 (CMOS)    │                │ 渲染器 (EGL/OpenGL)│
└──────────────┘                └──────────────────┘
```

我们用的是 **MuJoCo**（Multi-Joint Dynamics with Contact），机器人学领域最常用的物理引擎之一：

| 特性 | MuJoCo | 说明 |
|------|--------|------|
| 物理后端 | 刚体动力学 + 凸接触 | 精确的关节约束和接触力计算 |
| 场景定义 | MJCF (XML 格式) | 声明式描述机器人、物体、相机 |
| 渲染 | EGL / OpenGL 无头渲染 | 不需要显示器，服务器上可渲染 |
| 并行 | Warp GPU 后端 | 1024 个环境同时在 GPU 上 step |
| 速度 | ~5784 SPS (V100) | 比真实时间快 100x+ |

### 1.2 SO-101 在仿真中的建模

SO-101 是一个 6-DOF 机械臂，在 MuJoCo 中通过 URDF/MJCF 描述：

```
机器人模型 (so101_nexus)
├── 6 个旋转关节 (shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3)
├── 1 个夹爪 (gripper, 范围 0-45)
├── 桌面 (1m × 0.5m)
├── 目标物体 (红色方块, 3cm³)
├── 目标位置 (蓝色圆圈)
└── 相机
    ├── overhead (俯视, 全局视角)
    └── wrist (腕部, 机器人视角)
```

### 1.3 观测空间（Observation Space）

机器人在每一步"看到"的东西。这是策略的输入：

```python
observation = {
    # 本体感觉 (Proprioception) — 关节角度, 6 维
    "observation.state": [0.12, -0.45, 0.89, -0.32, 0.15, -0.08],

    # 视觉 (Vision) — 相机图像
    "observation.images.overhead": [480, 640, 3],  # 俯视 RGB
    "observation.images.wrist":     [480, 640, 3],  # 腕部 RGB

    # 语言指令 (Language, VLA 专属)
    "language_instruction": "Pick up the red cube and place it on the blue circle",
}
```

| 观测类型 | 维度 | PPO 用 | VLA 用 | 说明 |
|----------|------|--------|--------|------|
| 关节角度 | 6 | ✅ | ✅ | 机器人知道自己各关节的位置 |
| 俯视图像 | 480×640×3 | ❌ | ✅ | 全局视角，看到桌面全貌 |
| 腕部图像 | 480×640×3 | ❌ | ✅ | 机器人视角，看到夹爪附近 |
| 语言指令 | text | ❌ | ✅ | 告诉模型要做什么任务 |

**关键区别**：PPO 只用关节角度（向量），VLA 用图像+语言。这就是 PPO 不能迁移真机（没有视觉）而 VLA 可以的原因。

### 1.4 动作空间（Action Space）

机器人在每一步"决定"要做的事。这是策略的输出：

```python
action = [a0, a1, a2, a3, a4, a5]  # 6 维
# a0-a5: 6 个关节的目标角度 (radians)
# 控制模式: pd_joint_delta_pos (位置增量控制)
#   → 下一帧关节目标 = 当前关节 + action × scale
```

| 控制模式 | 含义 | 我们的选择 |
|----------|------|-----------|
| `pd_joint_delta_pos` | 关节位置增量 | ✅ PPO + VLA 都用 |
| `pd_joint_pos` | 关节绝对位置 | ❌ |
| `torque` | 关节力矩 | ❌ |

### 1.5 Reward 函数（RL 专属）

PPO 训练时，环境在每一步给出一个 reward 信号，告诉策略"做得好不好"：

```
PickLift 任务 reward 分解:

  reward = w1 × reach_reward      # 接近物体
         + w2 × grasp_reward      # 抓住物体 (gripper 闭合 + 接触)
         + w3 × lift_reward       # 抬起物体 (高度 > threshold)
         - w4 × action_penalty    # 动作平滑性惩罚

  成功判定: 物体高度 > lift_threshold (0.15m)
```

VLA 不需要 reward——它从演示数据中学习"看到这个场景时，人类/专家会做什么动作"。

### 1.6 Episode：一次完整尝试

一个 Episode 是机器人从开始到结束的一次完整尝试：

```
Episode 生命周期:
  env.reset()          → 物体放到随机位置, 机器人回到初始位姿
  env.step(action_1)   → 执行第 1 步
  env.step(action_2)   → 执行第 2 步
  ...
  env.step(action_N)   → 成功 (done=True) 或超时 (N=512)
  
  统计: success (bool), total_reward (float), total_steps (int)
```

| 参数 | PPO | VLA | 说明 |
|------|-----|-----|------|
| episode_length | 512 | 300 | 最大步数，超时算失败 |
| 评估 episodes | 50 | 325 | 评估时跑多少个 episode |

---

## 2. 两种学习范式：RL vs BC

我们的项目同时用了两种完全不同的学习范式。理解它们的区别是理解整个项目的基础。

### 2.1 强化学习（RL）— PPO 路线

```
         reward 信号
             ↑
策略 π ────→ 环境 ────→ 新状态 + reward
  │                              │
  └────────── 更新策略 ←──────────┘
```

- **输入**：关节角度（向量）
- **学习信号**：reward（环境给出）
- **不需要数据**：策略自己探索，试错学习
- **训练循环**：在环境中执行 → 收集 (state, action, reward) → 更新策略 → 重复
- **结果**：MLP 策略，100% 成功，但无视觉，不能迁移真机

### 2.2 行为克隆（BC）— VLA 路线

```
演示数据: [(图像, 语言, action), ...]
             │
策略 π ────→ 模仿 ────→ 预测 action ≈ 专家 action
  │                        │
  └──── 更新策略 ←─── MSE Loss ──┘
```

- **输入**：图像 + 语言指令
- **学习信号**：演示数据（人类/专家的动作）
- **需要数据**：从演示数据中学习"看到这个场景时该做什么"
- **训练循环**：从数据集采样 batch → 前向预测 → MSE Loss → 反向传播 → 重复
- **结果**：VLA 策略，47% 成功，有视觉，有迁移真机潜力

### 2.3 对比总结

| | RL (PPO) | BC (SmolVLA) |
|--|----------|--------------|
| 学习信号 | Reward | 演示数据 |
| 需要数据 | ❌ | ✅ |
| 输入 | 关节向量 | 图像 + 语言 |
| 训练环境 | 仿真 (on-policy) | 数据集 (off-policy) |
| 评估环境 | 同一仿真 | 同一仿真 |
| 能迁移真机 | ❌ 无视觉 | ✅ 有视觉 |
| 成功率 | 100% | 47% |
| 训练时间 | 86 min | ~60 min |

**为什么两种都用了？** PPO 验证基础设施和环境可学性，VLA 验证视觉方案可行性。它们验证的东西不同，互为补充。

---

## 3. 策略评估：如何判断"学会了"

### 3.1 训练指标 vs 评估指标

这是初学者最容易混淆的概念：

| | 训练指标 | 评估指标 |
|--|---------|---------|
| **PPO** | reward 曲线, success_rate (训练环境) | success_rate (独立评估环境) |
| **VLA** | MSE Loss (action 预测精度) | success_rate (仿真环境任务完成率) |
| **用途** | 监控训练是否收敛 | 判断模型是否真的学会了任务 |
| **陷阱** | Loss 低 ≠ 性能好 | — |

**核心原则**：训练指标只告诉你"学习过程在推进"，评估指标才告诉你"模型是否真的学会了"。

### 3.2 评估的完整流程

无论 PPO 还是 VLA，评估的逻辑是一样的：

```python
def evaluate(checkpoint_path, env_config, eval_config):
    # 1. 加载训练好的策略
    policy = load_policy(checkpoint_path)

    # 2. 创建评估环境（独立于训练环境）
    env = make_env(env_config)  # MuJoCo 后端

    # 3. 运行 N 个 episode
    results = []
    for ep in range(eval_config.num_episodes):
        obs = env.reset(seed=eval_config.seeds[ep])  # 固定 seed
        for step in range(eval_config.max_steps):
            action = policy.act(obs)       # 策略推理
            obs, reward, done, info = env.step(action)  # 环境执行
            if done:
                break
        results.append({
            "success": info["success"],
            "reward": info["total_reward"],
            "steps": step,
        })

    # 4. 统计指标
    return {
        "success_rate": sum(r["success"] for r in results) / len(results),
        "avg_reward": mean(r["reward"] for r in results),
        "avg_steps": mean(r["steps"] for r in results),
    }
```

### 3.3 两种评估策略

| | 确定性评估 (PPO) | Grid Sweep (VLA) |
|--|------------------|-------------------|
| **初始条件** | 50 个固定 seed | 5 距离 × 13 角度 × 5 试验 = 325 |
| **覆盖范围** | 随机采样 | 系统扫描工作空间 |
| **结果** | 单个 success_rate | 热力图 (成功率 vs 位置) |
| **适用场景** | 策略稳定, 泛化好 | 策略不稳定, 需找盲区 |
| **耗时** | ~14s | ~30min |

Grid Sweep 的价值在于能发现**训练数据的覆盖盲区**——中心区域 60-100% 但边缘 ~0%，这是单一指标看不出来的。（详见 Ch6 评测方法论）

---

## 4. 系统架构全景图

### 4.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    GitHub Actions (控制层)                    │
│  workflow dispatch → start-ecs → pipeline → stop-ecs         │
└──────────┬──────────────────────────────────┬───────────────┘
           │                                  │
           ▼                                  ▼
┌─────────────────────┐          ┌──────────────────────────┐
│  华为云 ECS (计算层)  │          │  华为云 OBS (存储层)       │
│  V100 32GB GPU       │          │  checkpoint/              │
│  self-hosted runner  │          │  eval_result.json         │
│  ┌─────────────────┐│          │  eval_video.mp4           │
│  │  Docker 容器     ││          └──────────────────────────┘
│  │  ┌───────────┐  ││                     ▲
│  │  │ 训练脚本   │  ││─── checkpoint ──────┘
│  │  │ 评估脚本   │  ││─── eval result ─────┘
│  │  └───────────┘  ││
│  └─────────────────┘│
└─────────────────────┘
           ▲
           │
┌──────────┴──────────────────────────────────────────────────┐
│              HuggingFace Hub (数据层)                         │
│  dobri420/pick-cube-so101-sim  (训练数据集)                  │
│  lerobot/smolvla_base          (预训练模型)                  │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 数据流

训练和评估是两个独立阶段，通过 checkpoint 连接：

```
阶段 1: 训练
  数据集 (HF Hub) ──→ Docker 容器 ──→ 训练脚本 ──→ checkpoint
  dobri420/pick-cube    so101-train     train_smolvla    /data/ckpt/15000

阶段 2: 评估
  checkpoint ──→ Docker 容器 ──→ 评估脚本 ──→ 评估结果
  /data/ckpt/     so101-mujoco    eval_mujoco     eval_result.json
                                  grid_sweep      eval_video.mp4

阶段 3: 归档
  checkpoint + eval_result + eval_video ──→ OBS (长期存储)
```

### 4.3 8 个 Workflow 职责矩阵

| Workflow | 触发方式 | 做什么 | 用哪个镜像 |
|----------|---------|--------|-----------|
| `ci.yml` | push/PR | 代码检查 + 格式化 | 无 (ubuntu-latest) |
| `docker-build.yml` | push to main | 构建 5 个 Docker 镜像 | 无 (docker buildx) |
| `download-driver.yml` | 手动 | 下载 GPU 驱动到 ECS | 无 |
| `train.yml` | 手动 | 通用训练入口 | so101-train |
| `evaluate.yml` | 手动 | 通用评估入口 | so101-eval |
| `ppo-pipeline.yml` | 手动 | **PPO 完整流水线** | so101-ppo |
| `so101-mujoco-pipeline.yml` | 手动 | **VLA MuJoCo 完整流水线** | so101-train + so101-mujoco |
| `vla-pipeline.yml` | 手动 | VLA 训练+回放 (旧版) | so101-train |

**你最常用的是这两个**：
- `ppo-pipeline.yml` — PPO 训练 + 评估 + 归档
- `so101-mujoco-pipeline.yml` — VLA 训练 + grid sweep 评估 + 归档

### 4.4 5 个 Docker 镜像职责矩阵

| 镜像 | 包含什么 | 不包含什么 | 用于 |
|------|---------|-----------|------|
| `so101-train` | lerobot + smolvla + so101_nexus | mujoco-warp, CleanRL | VLA 训练 + 回放 |
| `so101-ppo` | so101_nexus[warp] + CleanRL | lerobot, smolvla | PPO 训练 + 评估 |
| `so101-mujoco` | robot_descriptions + mujoco_env | lerobot, warp | Sim twin 评估 + grid sweep |
| `so101-eval` | vla-eval + SQLite | 训练相关 | LIBERO benchmark 评估 |
| `so101-model-server` | 模型推理服务 | 训练相关 | HTTP/ZMQ 推理服务 |

**为什么不做一个大镜像？** 依赖冲突（so101_nexus vs robot_descriptions）、镜像太大（15GB vs 5-8GB）、改一行全量重建。（详见 Ch2 基础设施）

---

## 5. 流水线详解：点击 Run 之后发生了什么

这是本章节的核心——逐步拆解流水线，让你知道每次点击 "Run workflow" 后系统在做什么。

### 5.1 VLA 流水线 (`so101-mujoco-pipeline.yml`)

```
你点击 "Run workflow"
  │
  ├─ Step 1: start-ecs (ubuntu-latest, ~2min)
  │   │
  │   ├─ 安装华为云 CLI (pip install hcloud)
  │   ├─ 调用 hcloud API 启动 ECS 7f39cb83
  │   ├─ sleep 120s 等待 GPU 驱动就绪
  │   └─ ECS 启动, runner ecs-0002 自动注册到 GitHub
  │
  ├─ Step 2: pipeline (self-hosted V100, ~90min)
  │   │
  │   ├─ 2a. 拉取 Docker 镜像 (~2min)
  │   │   docker pull swr.cn-north-4.myhuaweicloud.com/link-seek/so101-train:latest
  │   │   docker pull swr.cn-north-4.myhuaweicloud.com/link-seek/so101-mujoco:latest
  │   │
  │   ├─ 2b. 训练阶段 (~60min)
  │   │   │
  │   │   ├─ 启动训练容器
  │   │   │   docker run --gpus all so101-train ...
  │   │   │
  │   │   ├─ 从 HF Hub 下载数据集 (dobri420/pick-cube-so101-sim)
  │   │   ├─ 加载预训练模型 (lerobot/smolvla_base)
  │   │   │
  │   │   ├─ 训练循环 (15K steps):
  │   │   │   for step in range(15000):
  │   │   │       batch = sample_batch(dataset)        # 采样
  │   │   │       pred = policy(batch.obs, batch.lang)  # 前向
  │   │   │       loss = mse(pred, batch.action)        # 损失
  │   │   │       loss.backward(); optimizer.step()     # 反向
  │   │   │       if step % 5000 == 0: save_checkpoint()# 存盘
  │   │   │
  │   │   └─ 输出: /data/checkpoints/15000/ (模型权重)
  │   │
  │   ├─ 2c. 评估阶段 (~30min)
  │   │   │
  │   │   ├─ 启动评估容器
  │   │   │   docker run --gpus all so101-mujoco ...
  │   │   │
  │   │   ├─ 加载 checkpoint (Step 2b 的输出)
  │   │   ├─ 创建 MuJoCo 评估环境
  │   │   │
  │   │   ├─ Grid Sweep (325 episodes):
  │   │   │   for reach in [0.15, 0.18, 0.20, 0.22, 0.25]:    # 5 距离
  │   │   │     for azimuth in range(-90, 91, 15):             # 13 角度
  │   │   │       for trial in range(5):                        # 5 试验
  │   │   │         obs = env.reset(reach, azimuth, seed=trial)
  │   │   │         for step in range(300):
  │   │   │           action = policy.select_action(obs)        # 推理
  │   │   │           obs, reward, done, info = env.step(action)# 执行
  │   │   │           if done: break
  │   │   │         record(success=info["success"])
  │   │   │
  │   │   ├─ 统计: 153/325 = 47% success_rate
  │   │   └─ 输出: eval_result.json + eval_video.mp4 + heatmap.png
  │   │
  │   └─ 2d. OBS 归档 (~2min)
  │       ├─ 上传 checkpoint → obs://so101-sim-pipeline/vla/checkpoints/
  │       ├─ 上传 eval_result → obs://so101-sim-pipeline/vla/results/
  │       └─ 上传 eval_video  → obs://so101-sim-pipeline/vla/results/
  │
  └─ Step 3: stop-ecs (ubuntu-latest, if:always(), ~1min)
      └─ 调用 hcloud API 关闭 ECS (无论成功失败都关)
```

### 5.2 PPO 流水线 (`ppo-pipeline.yml`)

PPO 流水线结构相同，但训练和评估逻辑不同：

| 阶段 | VLA 流水线 | PPO 流水线 |
|------|-----------|-----------|
| 训练输入 | 数据集 (图像+语言) | 环境 (reward 信号) |
| 训练循环 | BC: 采样→前向→MSE→反向 | RL: rollout→GAE→clip→更新 |
| 训练步数 | 15K steps | 30M steps (1024 envs 并行) |
| 训练时间 | ~60 min | ~86 min |
| 评估方式 | Grid sweep 325 episodes | 确定性 50 episodes |
| 评估时间 | ~30 min | ~14 sec |
| Docker 镜像 | so101-train + so101-mujoco | so101-ppo (单镜像) |
| 总时间 | ~95 min | ~90 min |

### 5.3 时间线总结

```
T+0min    ── 你点击 Run
T+2min    ── ECS 启动完成, runner 上线
T+4min    ── Docker 镜像拉取完成
T+4min    ── 训练开始
T+64min   ── 训练结束, checkpoint 保存
T+64min   ── 评估开始
T+94min   ── 评估结束, 47% success_rate
T+96min   ── OBS 归档完成
T+97min   ── ECS 关闭, 流水线完成
T+97min   ── 你收到 GitHub Actions 通知
```

---

## 6. 关键设计决策

理解"为什么这样设计"比记住"怎么用"更重要。

### 6.1 为什么用 GitHub Actions 而不是直接 SSH 跑脚本？

| 方案 | 直接 SSH | GitHub Actions |
|------|---------|----------------|
| 触发方式 | 手动 SSH 连服务器 | 一键 Web UI / CLI |
| 参数传递 | 命令行参数 | workflow input 参数 |
| 日志 | 自己管理 | 自动记录, 可回溯 |
| 失败通知 | 自己写脚本 | 自动 email/通知 |
| 团队协作 | 需要共享 SSH key | 仓库权限即可 |
| 成本控制 | 容易忘关机 | stop-ecs if:always() |

**核心价值**：把训练-评估流程**工程化**，不是一次性脚本，而是可复现、可协作的流水线。

### 6.2 为什么用 Docker 而不是 conda 环境？

| 方案 | conda | Docker |
|------|-------|--------|
| 依赖隔离 | 环境可能互相污染 | 完全隔离 |
| 可复现 | "works on my machine" | 镜像一致, 任何机器结果同 |
| GPU 支持 | 需要手动配 CUDA | nvidia/cuda 基础镜像 |
| CI 集成 | 需要在 runner 上配环境 | docker pull 即可 |
| 代价 | 轻 | 镜像构建/拉取时间 |

**核心价值**：lerobot + so101_nexus + MuJoCo 依赖链很复杂，Docker 保证所有人环境一致。

### 6.3 为什么用 self-hosted runner 而不是 GitHub 托管 runner？

| 方案 | GitHub 托管 GPU runner | self-hosted |
|------|----------------------|-------------|
| 成本 | $0.16/min ≈ ¥1.15/min | ¥30/h ≈ ¥0.50/min |
| 86min 训练 | $13.76 ≈ ¥99 | ¥43 |
| GPU 型号 | 不可选 | V100 32GB |
| 排队 | 可能排队 | 独占 |
| 代价 | 贵 | 需要自己管理 ECS 生命周期 |

**核心价值**：V100 按需开关机，成本只有 GitHub 托管 runner 的 1/2，且独占不排队。

### 6.4 为什么训练和评估用不同的 Docker 镜像？

```
训练镜像 (so101-train)              评估镜像 (so101-mujoco)
├── lerobot 0.6.1                   ├── robot_descriptions
├── smolvla                         ├── mujoco_env
├── so101_nexus                     ├── 无训练依赖
└── 无 grid sweep 评估              └── 有 grid sweep 评估
```

**原因**：训练需要 lerobot + smolvla（模型训练框架），评估需要 mujoco_env + grid sweep（评估工具链）。两者依赖不同，合在一起会冲突或镜像过大。

### 6.5 为什么用 OBS 而不是 HF Hub 存评估结果？

| 方案 | HF Hub | OBS |
|------|--------|-----|
| 适用 | 模型/数据集 (结构化) | 任意文件 (视频/JSON) |
| 上传 | git LFS, 慢 | obsutil, 快 |
| 成本 | 免费但有限额 | 按存储量计费, 便宜 |
| 访问 | 公开 | 可配置权限 |

**核心价值**：评估产物是 JSON + MP4 + PNG，不是模型权重，OBS 更适合存这类非结构化文件。

---

## 7. 知识地图：后续章节如何展开

现在你有了完整的知识框架，后续章节在这个框架中填充实战细节：

```
本章 (Ch1 技术框架)
  │
  ├─→ Ch2 基础设施搭建
  │     如何搭建 §4 描述的架构 (ECS + Docker + Runner)
  │
  ├─→ Ch3 PPO Baseline
  │     §2.1 RL 范式的完整实战 (训练 → 评估 → 100%)
  │
  ├─→ Ch4 VLA 训练入门
  │     §2.2 BC 范式的完整实战 (数据 → 训练 → 回放)
  │
  ├─→ Ch5 Debug 实战
  │     从 0% 到 47% 的完整调试旅程
  │
  ├─→ Ch6 评测方法论
  │     §3 评估的深入探讨 (Grid Sweep + 指标设计)
  │
  └─→ Ch7 优化进阶
        如何从 47% 提升到 87%+
```

---

## 思考题

1. **PPO 的观测空间只有 6 维关节角度，为什么能在仿真中达到 100%？**  
   提示：仿真环境中物体位置可以从物理状态精确推断，不需要"看"。但真机没有这个 luxury。

2. **如果训练用 2 个相机，评估用 3 个相机，会怎样？**  
   提示：模型从未见过第 3 个相机的图像，多出来的输入会导致推理异常。这就是我们的 camera3 bug。

3. **Grid Sweep 325 episodes 中只有 47% 成功，但中心区域 80%+。如果你要改进，应该先改什么？**  
   提示：不是算法问题，是训练数据覆盖不足。补充边缘区域数据 > 调超参数。

4. **为什么评估环境要独立于训练环境（即使是同一个仿真）？**  
   提示：训练环境有探索噪声（entropy bonus / data sampling），评估环境要确定性，消除训练时的随机性。

5. **如果让你设计一个新的流水线（比如 RL fine-tuning VLA），你会怎么组织 Docker 镜像？**  
   提示：需要 so101-train（VLA 模型）+ so101-ppo（RL 环境）的能力，可能需要新镜像或组合。

---

> **上一章**：[序章](so101-tutorial-ch0-prologue.md) | **下一章**：[Ch2 搭建仿真训练基础设施](so101-tutorial-ch1-infrastructure.md)
