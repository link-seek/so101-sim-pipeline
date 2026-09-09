# Ch3：SmolVLA 仿真训练入门

> SO101 仿真评测教程 · 第三章

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

### 3.1 SmolVLM Base

预训练的视觉-语言模型，理解图像内容和语言指令。我们用 `lerobot/smolvla_base` 作为初始权重，在自己的数据上 fine-tune。

### 3.2 Action Head

将 SmolVLM 的隐状态映射到 6 维动作 (5 关节 + 1 gripper)：

```python
action = action_head(hidden_state)  # shape: (batch, 50, 6)
```

输出 50 个未来 step 的 action（action chunking），每次推理后逐步执行。

### 3.3 Action Chunking

Action Chunking 是 SmolVLA 的核心特性：

| chunk_size | 成功率（论文消融） |
|------------|-------------------|
| 1（无 chunking） | 50.0% |
| 10 | 80.3% |
| 50（默认） | 最优 |

每次推理预测 50 个 action，逐步执行。好处是动作更连贯，减少高频推理的抖动。

---

## 4. LeRobot 数据格式

### 4.1 v3.0 格式

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

### 4.2 rename_map

数据集的相机名可能和模型期望不一致，需要映射：

```json
{
  "observation.images.cam0": "observation.images.overhead",
  "observation.images.cam1": "observation.images.wrist"
}
```

这个看似简单的映射，在我们的项目中导致了第一个 bug（P0 相机不匹配，主场在 Ch4 前情提要）。

---

## 5. 候选数据集

训练需要演示数据。这里有 3 个候选——先认识它们是谁、长什么样。选谁、为什么选、结局如何，是下一章的故事，本章不判胜负。

### 5.1 候选一览

| 数据集 | 来源 | Episodes | 相机 | 视觉域 |
|--------|------|----------|------|--------|
| `shattori/so101_pick_place_thor` | **真机**遥操作 | 100 | wrist+overhead (2) | 真机 |
| `ataghof/so101nexus-cube500-binary` | **仿真** scripted expert | 500 | cam0+cam1 (2) | 仿真 |
| `dobri420/pick-cube-so101-sim` | **仿真** sim twin | - | camera1/2/3 (3) | 仿真 |

### 5.2 第一个候选：shattori（真机）

```yaml
# 训练配置：shattori 数据集
env:
  DATASET_REPO: shattori/so101_pick_place_thor
  RENAME_MAP: '{"wrist":"camera1", "overhead":"camera2"}'
```

真机遥操作采集，100 episodes，wrist+overhead 双摄。注意它的视觉域是**真机**，而我们的评测在 MuJoCo 里——这个组合后来出了故事，见 Ch4 前情提要（P1）。

### 5.3 第二个候选：ataghof（仿真，500 eps）

```yaml
env:
  DATASET_REPO: ataghof/so101nexus-cube500-binary
  RENAME_MAP: '{"observation.images.cam0":"observation.images.overhead", "observation.images.cam1":"observation.images.wrist"}'
  DATASET_FPS: 30
```

500 episodes（5 倍于 shattori），仿真采集，已有 MolmoAct2 验证结果（93% grasp）。账面上是最强的候选——我们拿它开局。开局之后发生的事，整章在 Ch4。

### 5.4 第三个候选：dobri420（仿真 sim twin）

```yaml
# 训练配置：dobri420 sim twin 数据集
env:
  SIM_DATASET: dobri420/pick-cube-so101-sim
  # 无需 rename_map，3 相机原生匹配
```

sim twin 数据：采集和评估在同一个 MuJoCo 场景，3 相机原生匹配，无需 `rename_map`。它什么时候出场、出场拿几分，见 Ch4。

---

## 6. 训练流程

### 6.1 lerobot-train 命令

> ⚠️ 已验证破碎（Discussion #20）：`so101-train:latest` 当前 torch/torchvision 版本错配，
> `RuntimeError: operator torchvision::nms does not exist`，训练一行跑不起来。
> 在镜像修复前，**走 Ch1 §4.3 的 `so101-mujoco` 路径**（已验证 153/325 = 47%）。

```bash
docker run --rm --gpus all --shm-size=8g \
  -v /data/checkpoints:/data/checkpoints \
  -v /data/datasets:/data/datasets \
  so101-train:latest \
  python /workspace/scripts/train_smolvla.py \
    --dataset.repo_id=dobri420/pick-cube-so101-sim \
    --policy.path=lerobot/smolvla_base \
    --steps=20000 \
    --batch_size=32 \
    --save_freq=5000 \
    --env_eval_freq=2000
```

### 6.2 训练脚本核心

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

### 6.3 训练 Loss 曲线（正常启动长这样）

以前 5K steps 为例，loss 单调下降就是训练跑起来了：

```
Step    Loss     LR
500     0.461    8.3e-05
1000    0.292    9.5e-05
2000    0.204    7.3e-05
5000    0.119    3.3e-06
```

只记住一件事：loss 下降 = 优化器在干活。但它不承诺任务会成——有人拿着这条曲线判过"验证通过"，错了（见 Ch4 Phase 2）。

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

### 7.2 回放输出怎么读

| 输出 | 含义 |
|------|------|
| prediction errors > 0 | 推理管线不通（常见于相机名没对齐，见 §4.2 `rename_map`） |
| 0/300 errors + Success=False | 管线通，但任务没完成——这是数据/训练问题，不是代码问题 |

回放只回答"管线通不通"，不回答"任务会不会成"。拿着这套工具连败 5 轮、又翻盘的故事，在 Ch4。

---

## 踩坑复盘

### 坑 1 / 坑 2：P0 相机不匹配、P1 视觉鸿沟

已搬入 Ch4：P0/P1 是破案故事的起点，在 Ch4 前情提要盒子里（那里是它们的主场）。工具层面的对应知识见本章 §4.2（`rename_map`）和 §5（视觉域列）。

### 坑 3：FPS 不匹配

**现象**：数据集 33Hz，环境 30Hz，时序不对齐。

**修复**：训练时 `--dataset.fps=30` 自动 resample。

**教训**：数据集和环境的时间频率要一致，否则 action 时序错乱。

---

## 思考题

1. **Action Chunking（chunk_size=50）为什么比逐步执行好？**  
   提示：想想高频推理的抖动问题，以及 50 步前瞻规划的连贯性。

2. **如果 `rename_map` 把两个相机写反了，回放会报什么？**  
   提示：看 §7.2 的输出对照表——errors 会先于 Success 告诉你。

3. **数据集 33Hz、环境 30Hz，不对齐直接训会怎样？**  
   提示：见坑 3，`--dataset.fps=30` 是干什么的。

4. **预告**：下一章我们拿着这套工具连败 5 轮。先猜一个：loss 一路下降、0 errors，但 Success 恒为 False——病根会在代码、数据、环境三者中的哪一个？为什么？

---

> **上一章**：[Ch2 PPO Baseline](so101-tutorial-ch2-ppo-baseline.md) | **下一章**：[Ch4 Debug 实战](so101-tutorial-ch4-debug-journey.md)
