# Ch3：SmolVLA 仿真训练入门

> SO101 仿真评测教程 · 第三章  
> 实战项目：[link-seek/so101-sim-pipeline](https://github.com/link-seek/so101-sim-pipeline)  
> 关联 Discussion：[#1 SmolVLA 仿真回放验证](https://github.com/link-seek/so101-sim-pipeline/discussions/1)

---

## 1. VLA 模型全景

在深入 SmolVLA 之前，先了解 VLA 领域的主要模型：

### 1.1 主流 VLA 模型对比

| 模型 | 参数量 | 动作输出 | 硬件需求 | 推理速度 | 特点 |
|------|--------|----------|----------|----------|------|
| **SmolVLA** | 450M | 连续 chunk (50步) | 1x V100 16GB | 50ms/step | LeRobot 原生，轻量 |
| **Octo** | 93M | Diffusion | 1x RTX 3090 | 10-40 Hz | 最小，Diffusion Policy |
| **OpenVLA** | 7.5B | 离散 token | 2x A100 80GB | 3-8 Hz | 通用，需要大数据 |
| **RT-2** | 55B | 离散 token | TPU/H100 | ~1 Hz | Google，最大 |

### 1.2 架构差异

```
SmolVLA:
  图像 + 语言 → SmolVLM (视觉-语言) → Action Head → 50 步连续动作

RT-2/OpenVLA:
  图像 → Vision Encoder
  语言 → Language Encoder
  → 融合 → 离散化 → 逐步生成动作 token
```

### 1.3 为什么选 SmolVLA

| 约束 | SmolVLA 优势 |
|------|-------------|
| **V100 16GB** | 450M 参数刚好装下 |
| **LeRobot 生态** | 原生支持，无需适配 |
| **社区验证** | 多个 SO101 项目成功 (86-90%) |
| **训练速度** | 10-14 小时可完成 |

其他模型要么太大（OpenVLA 7B, RT-2 55B），要么不在 LeRobot 生态内。

---

## 2. VLA 模型 vs PPO

| | PPO (Ch2) | SmolVLA (本章) |
|--|-----------|----------------|
| 输入 | 关节状态 (向量) | 图像 + 语言指令 |
| 输出 | 关节增量 (向量) | 关节角度 (向量) |
| 训练方式 | RL（自探索） | BC（模仿演示） |
| 需要数据 | ❌ 不需要 | ✅ 需要演示数据 |
| 能迁移真机 | ❌ MLP 无视觉 | ✅ VLA 有视觉 |
| 成功率 | 100% | 47%（目前） |

VLA 的核心优势是**有视觉**——可以看到物体、理解场景，因此有迁移到真机的潜力。代价是需要演示数据，且训练更难。

---

## 3. SmolVLA 架构

```
┌─────────────────────────────────────┐
│           SmolVLA 模型              │
│                                     │
│  图像1 (overhead) ─┐                │
│  图像2 (wrist)    ─┤→ SmolVLM base  │
│  语言指令          ─┘   (视觉-语言)  │
│                          ↓          │
│                     Action Head     │
│                          ↓          │
│                     50 个未来 action │
│                   (action chunking)  │
└─────────────────────────────────────┘
```

### 7.1 SmolVLM Base

预训练的视觉-语言模型，理解图像内容和语言指令。我们用 `lerobot/smolvla_base` 作为初始权重，在自己的数据上 fine-tune。

### 7.2 Action Head

将 SmolVLM 的隐状态映射到 6 维动作 (5 关节 + 1 gripper)：

```python
action = action_head(hidden_state)  # shape: (batch, 50, 6)
```

输出 50 个未来 step 的 action（action chunking），每次推理后逐步执行。

### 7.3 Action Chunking

Action Chunking 是 SmolVLA 的核心特性：

| chunk_size | 成功率（论文消融） |
|------------|-------------------|
| 1（无 chunking） | 50.0% |
| 10 | 80.3% |
| 50（默认） | 最优 |

每次推理预测 50 个 action，逐步执行。好处是动作更连贯，减少高频推理的抖动。

---

## 4. LeRobot 数据格式

### 7.1 v3.0 格式

```
dataset/
├── meta/
│   ├── info.json          # 数据集元信息
│   ├── stats.json         # action/obs 统计信息（min/max/mean）
│   └── episodes.jsonl     # 每个 episode 的元数据
├── data/
│   └── train-00000.parquet  # 数据帧（action, obs, images）
└── videos/
    └── train-00000/        # 视频文件
```

每个 frame 包含：

```python
{
    "observation.state": [6 floats],           # 关节角度
    "observation.images.overhead": [480×640×3], # 俯视相机
    "observation.images.wrist": [480×640×3],    # 腕部相机
    "action": [6 floats],                      # 目标关节角度
    "language_instruction": "Pick up the red cube and place it on the blue circle",
}
```

### 7.2 rename_map

数据集的相机名可能和模型期望不一致，需要映射：

```json
{
  "observation.images.cam0": "observation.images.overhead",
  "observation.images.cam1": "observation.images.wrist"
}
```

这个看似简单的映射，在我们的项目中导致了第一个 bug（P0 相机不匹配，详见 Ch4）。

---

## 5. 数据集选择

我们尝试了 3 个数据集，每个都有不同的教训：

### 7.1 数据集对比

| 数据集 | 来源 | Episodes | 相机 | 视觉域 | 结果 |
|--------|------|----------|------|--------|------|
| `shattori/so101_pick_place_thor` | **真机**遥操作 | 100 | wrist+overhead (2) | 真机 | P0 修复但 P1 存在 |
| `ataghof/so101nexus-cube500-binary` | **仿真** scripted expert | 500 | cam0+cam1 (2) | 仿真≠评测环境 | 5 轮失败，终止 |
| `dobri420/pick-cube-so101-sim` | **仿真** sim twin | - | camera1/2/3 (3) | 仿真=评测环境 | ✅ 47% |

### 7.2 第一个数据集：shattori（真机）

```yaml
# vla-pipeline.yml
env:
  DATASET_REPO: shattori/so101_pick_place_thor
  RENAME_MAP: '{"wrist":"camera1", "overhead":"camera2"}'
```

**发现 P0**：训练用 `side+up` 相机，推理用 `wrist+overhead`，模型从未见过腕部视角。

**修复**：切换到 shattori 数据集（wrist+overhead），prediction errors → 0/300。

**发现 P1**：训练数据是真机照片，评测是 MuJoCo 渲染。视觉域不匹配，模型在真机图片上学到的特征在仿真中失效。

### 7.3 第二个数据集：ataghof（仿真，500 eps）

```yaml
env:
  DATASET_REPO: ataghof/so101nexus-cube500-binary
  RENAME_MAP: '{"observation.images.cam0":"observation.images.overhead", "observation.images.cam1":"observation.images.wrist"}'
  DATASET_FPS: 30
```

**优点**：500 episodes（vs shattori 100），仿真采集（vs 真机），已有 MolmoAct2 验证结果。

**结果**：5 轮训练 + 3 个 bug 修复后仍 Success=False。根因是数据采集环境 ≠ 评测环境。（详见 Ch4）

### 7.4 第三个数据集：dobri420（仿真 sim twin）

```yaml
# so101-mujoco-pipeline.yml
env:
  SIM_DATASET: dobri420/pick-cube-so101-sim
  # 无需 rename_map，3 相机原生匹配
```

**关键优势**：数据采集和评估在同一个 MuJoCo 场景，1:1 匹配。

**结果**：15K steps 训练，grid sweep 325 episodes，**47% 成功率**。

---

## 6. 训练流程

### 7.1 lerobot-train 命令

```bash
docker run --rm --gpus all --shm-size=8g \
  -v /data/checkpoints:/data/checkpoints \
  -v /data/datasets:/data/datasets \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -e HF_LEROBOT_HOME=/data/datasets \
  so101-train:latest \
  python /workspace/scripts/train_smolvla.py \
    --dataset.repo_id=dobri420/pick-cube-so101-sim \
    --policy.path=lerobot/smolvla_base \
    --steps=20000 \
    --batch_size=32 \
    --save_freq=5000 \
    --env_eval_freq=2000
```

### 7.2 训练脚本核心

```python
# scripts/train_smolvla.py（简化）
def train(args):
    # 1. 加载数据集
    dataset = LeRobotDataset(args.dataset_repo_id, root=args.data_dir)
    
    # 2. 加载预训练模型
    policy = SmolVLA.from_pretrained(args.policy_path)
    
    # 3. 训练循环
    optimizer = Adam(policy.parameters(), lr=1e-4)
    for step in range(args.steps):
        batch = sample_batch(dataset, args.batch_size)
        
        # 前向：图像 + 语言 → 预测 action
        pred_actions = policy(batch["observation"], batch["language"])
        
        # 损失：MSE between predicted and ground truth actions
        loss = mse_loss(pred_actions, batch["action"])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % args.save_freq == 0:
            save_checkpoint(policy, f"/data/checkpoints/{step}")
```

### 7.3 训练 Loss 曲线

以 ataghof 数据集 20K steps 为例：

```
Step    Loss     LR
500     0.461    8.3e-05
1000    0.292    9.5e-05
2000    0.204    7.3e-05
5000    0.119    3.3e-06   ← Phase 2 完成
10000   0.069    4.9e-05
15000   0.058    2.7e-05
20000   0.046    2.5e-06   ← Phase 3 完成
```

Loss 持续下降，但 0.046 远高于社区成功案例的 0.005-0.018。这暗示有问题（详见 Ch4）。

---

## 7. 仿真回放

### 7.1 回放流程

```python
# scripts/replay_demo.py（简化）
def replay(checkpoint_path, env_id="MuJoCoPickAndPlace-v1"):
    # 1. 加载训练好的策略
    policy = SmolVLA.from_pretrained(checkpoint_path)
    
    # 2. 创建仿真环境
    env = make_so101_nexus_env(env_id)
    obs = env.reset()
    
    # 3. 推理循环
    errors = 0
    for step in range(300):
        # 仿真状态 → 数据集格式
        ds_row = sim_qpos_to_dataset_row(obs)
        
        # 准备推理输入
        inference_input = prepare_observation_for_inference(obs)
        
        # 策略推理
        action = policy.select_action(inference_input)
        
        # 数据集格式 → 仿真控制
        sim_action = dataset_row_to_sim_qpos(action)
        
        # 执行
        obs, reward, done, info = env.step(sim_action)
        
        if info.get("prediction_error"):
            errors += 1
    
    return {"prediction_errors": errors, "success": info["success"]}
```

### 7.2 回放结果

| 阶段 | Prediction Errors | Success | Reward |
|------|-------------------|---------|--------|
| P0 修复前 | >0 | False | ≈0 |
| P0 修复后 | 0/300 ✅ | False | ≈0 |
| 20K 训练后 | 0/300 ✅ | False | ≈0 |
| 3 Bug 修复后 | 0/300 ✅ | False | ≈0 |
| Sim Twin (方案B) | 0/300 ✅ | **True (47%)** | >0 |

"0 errors 但 Success=False" 是一个关键信号——模型能控制机器人，但没学会完成任务。这引导我们去找系统性问题（Ch4）。

---

## 踩坑复盘

### 坑 1：P0 相机视角不匹配

**现象**：prediction errors > 0，模型输出不合理。

**根因**：训练数据用 `side+up` 相机，推理环境只有 `wrist+overhead`。模型从未见过腕部视角，推理时相当于盲猜。

**修复**：切换到相机匹配的数据集 + 配置 `rename_map`。

**教训**：训练和推理的观测空间必须一致。这是最基本的约束，但容易被忽略。

### 坑 2：P1 Sim-to-Real Visual Gap

**现象**：P0 修复后 0 errors，但 Success=False，reward ≈ 0。

**根因**：训练数据是真机照片（自然光、真实材质），评测是 MuJoCo 渲染（点光源、简单纹理）。模型在真机图片上学到的视觉特征在仿真中失效。

**修复**：切换到仿真采集的数据集（ataghof → dobri420）。

**教训**：**训练和评估必须在同一视觉域**。这是本教程最重要的一个原则。

### 坑 3：FPS 不匹配

**现象**：数据集 33Hz，环境 30Hz，时序不对齐。

**修复**：训练时 `--dataset.fps=30` 自动 resample。

**教训**：数据集和环境的时间频率要一致，否则 action 时序错乱。

---

## 思考题

1. **为什么 500 episodes 的 ataghof 失败了，但 sim twin 数据集成功了？**  
   提示：episodes 数量不是关键，数据-环境匹配才是。

2. **Action Chunking（chunk_size=50）为什么比逐步执行好？**  
   提示：想想高频推理的抖动问题，以及 50 步前瞻规划的连贯性。

3. **Loss 0.046 vs 社区 0.005，差距在哪里？**  
   提示：相机数（2 vs 3）、数据来源（scripted vs 遥操作）、batch_size（32 vs 64）。

4. **如果用 3 个相机训练，但推理时只有 2 个，会怎样？**  
   提示：这就是我们的 camera3 bug——训练时缺失 camera3，推理时不应提供。

---

> **上一章**：[Ch2 PPO Baseline](so101-tutorial-ch2-ppo-baseline.md) | **下一章**：[Ch4 Debug 实战](so101-tutorial-ch4-debug-journey.md)
