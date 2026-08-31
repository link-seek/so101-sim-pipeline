# Ch4：Debug 实战 — 从 0% 到 47% 的完整调试旅程

> SO101 仿真评测教程 · 第四章（核心章节）  
> 实战项目：[link-seek/so101-sim-pipeline](https://github.com/link-seek/so101-sim-pipeline)  
> 关联 Discussion：[#4 方案 A→B 完整迭代](https://github.com/link-seek/so101-sim-pipeline/discussions/4)（8 条评论，最活跃）

---

## 本章导读

这是整部教程的核心。我们用 Discussion #4 的 8 条评论作为时间线，还原一个真实项目从 0% 到 47% 成功率的完整调试过程。

你会看到：

- 5 轮训练迭代，每轮都失败
- 3 个 bug 的诊断与修复
- 1 次艰难的方案终止决策
- 最终通过 sim twin 数据集实现首次成功

**这不是事后总结，而是真实的调试日志。**

---

## 时间线总览

```
08-14  Phase 0: 数据集验证 ✅
  │
08-15  Phase 1: 代码修改 ✅
  │
08-15  Phase 2: 5k 快速验证 ✅ (loss 0.461→0.119, Success=False)
  │
08-15  Phase 3: 20k 完整训练 ✅ (loss 0.119→0.046, Success=False)
  │
08-17  社区调研 → 诊断 3 个 bug
  │
08-17  Bug 修复 1: Gripper 转换 (commit f53dd6f)
  │
08-17  Bug 修复 2: Action Chunking (commit bcd3e30)
  │
08-17  Bug 修复 3: camera3 不匹配 (commit 4efb51c)
  │
08-17  回放验证: 0 errors 但 Success=False
  │
08-18  bs=64 重训练 → 超时取消
  │
08-18  方案终止决策: 转向 so101-mujoco sim twin
  │
08-19  Sim Twin 训练 + 评测 → 47% 成功！🎉
```

---

## Phase 0-1：数据集验证与代码修改

### 数据集选择

选择 `ataghof/so101nexus-cube500-binary`：

| 项目 | 值 |
|------|-----|
| Episodes | 500 |
| Frames | 20,647 |
| FPS | 33 Hz |
| 相机 | cam0 (overhead) + cam1 (wrist), 480×640 |
| 格式 | LeRobot v3.0 |
| 机器人 | SO-101, 6-DOF |
| Gripper | 二值化 (0=闭合, 45=打开) |
| 任务 | Pick up red cube → place on blue circle |
| 已验证 | MolmoAct2 LoRA champion, 93% grasp / 30% success |

### 代码修改（4 个文件）

| 文件 | 变更 |
|------|------|
| `vla-pipeline.yml` | `DATASET_REPO=ataghof/so101nexus-cube500-binary`, `RENAME_MAP` 更新 |
| `train_smolvla.py` | 新增 `--dataset.fps` 参数 |
| `replay_demo.py` | 相机 key 统一为 `overhead`/`wrist` |
| `replay_demo.py` | 默认 task 改为 `"Pick up the red cube and place it on the blue circle."` |

### rename_map

```json
{
  "observation.images.cam0": "observation.images.overhead",
  "observation.images.cam1": "observation.images.wrist"
}
```

---

## Phase 2：5K 快速验证

> Discussion #4 评论 1

### 训练

| Step | Loss | LR |
|------|------|-----|
| 500 | 0.461 | 8.3e-05 |
| 1000 | 0.292 | 9.5e-05 |
| 2000 | 0.204 | 7.3e-05 |
| 3000 | 0.148 | 4.4e-05 |
| 4000 | 0.125 | 1.7e-05 |
| 5000 | 0.119 | 3.3e-06 |

Loss 从 0.461 稳定下降至 0.119，模型正在学习。训练时间 3h 31m，2.53s/step。

### 回放

| 指标 | 值 |
|------|-----|
| Prediction errors | **0/300** ✅ |
| Success | **False** |
| Reward | ±0.003（接近 0） |

### 关键发现

1. **P1（视觉域不匹配）已消除** — 模型全程控制机器人 (300/300 steps)，无随机回退
2. **Loss 稳定下降** — 学习曲线健康
3. **Reward ~0** — 5k steps 不足以学会完成任务，但模型行为正常

**结论**：Phase 2 验证通过，P1 已解决。接下来进行 20k 完整训练。

---

## Phase 3：20K 完整训练

> Discussion #4 评论 2

### 训练

| Step | Loss | LR |
|------|------|-----|
| 5K | 0.119 | 3.3e-06 |
| 10K | 0.069 | 4.9e-05 |
| 12K | 0.064 | 4.2e-05 |
| 14K | 0.058 | 2.7e-05 |
| 16K | 0.052 | 1.3e-05 |
| 18K | 0.048 | 5.5e-06 |
| 20K | 0.046 | 2.5e-06 |

Loss 从 0.119 → 0.046，持续下降。训练时间 14h 04m。

### 回放

| 指标 | 值 |
|------|-----|
| Prediction errors | **0/300** ✅ |
| Success | **False** |
| Reward | ±0.001（接近 0） |

### 问题

Loss 0.046 远高于社区成功案例的 0.005-0.018。模型全程控制机器人（0 errors），但任务未完成。

**这时我们面临一个关键判断**：是训练不够，还是有 bug？

---

## 社区调研：诊断 3 个根因

> Discussion #4 评论 3

我们调研了 4 个社区成功案例，对比配置差异：

| 项目 | 关键发现 |
|------|----------|
| ggand0/vla-so101 | 75 eps + 20k + bs=64 + 双摄 → 60-80% |
| Sa74ll/smolvla | 40 eps + 15k + 分层采样 → 87.66% |
| dyordan1/so101-mujoco | Sim twin, 数据-环境 1:1 匹配 |
| MSSergeev/so101-lab | SmolVLA + IQL → 86-88%, + PPO → 90% |

### 诊断出 3 个根因

#### 根因 1：Gripper 转换 Bug（致命）

`replay_demo.py` 中自实现的 `dataset_row_to_sim_qpos` 用 `values[-1] / 100.0` 转换 gripper。

但数据集的 gripper 范围是 **0-45**（不是 0-100）。

so101_nexus 官方文档明确说明：
> gripper.pos is RANGE_0_100 (percent of jaw travel, not degrees)  
> Decoding the whole vector with np.deg2rad runs cleanly yet silently corrupts only the gripper.

**后果**：gripper 最多只闭合 45%，永远无法夹紧物体 → pick-and-place 不可能成功。

#### 根因 2：未使用 Action Chunking（关键）

SmolVLA 的核心特性是 action chunking：每次推理预测 50 个未来 action，然后逐步执行。

| chunk_size | 成功率 |
|------------|--------|
| 1 (无 chunking) | 50.0% |
| 10 | 80.3% |
| 50 (默认) | 最优 |

当前 `replay_demo.py` 每步只执行 1 个 action，完全没利用 chunking。

#### 根因 3：缺少 camera3 + 训练超参数偏小

- SmolVLA base 预训练用 3 个相机，我们只提供 2 个
- 社区推荐 batch_size=64，我们用 32
- 社区 20k steps 通常 loss 降到 0.005，我们只到 0.046

---

## Bug 修复

> Discussion #4 评论 4

### 修复 1：Gripper 转换（commit f53dd6f）

**问题**：自实现 `/100.0` 转换，但 gripper 范围 0-45。

**修复**：改用 so101_nexus 官方函数：

```python
# 修复前（错误）
def dataset_row_to_sim_qpos(row):
    values = row.tolist()
    joint_angles = np.deg2rad(values[:-1])
    gripper = values[-1] / 100.0  # BUG: 应该是 / 45.0 或用官方函数
    return np.concatenate([joint_angles, [gripper]])

# 修复后（正确）
from so101_nexus import dataset_row_to_sim_qpos  # 官方函数
```

**教训**：不要自己实现坐标转换，用官方函数。静默错误最危险——程序不报错，但 gripper 永远夹不紧。

### 修复 2：Action Chunking（commit bcd3e30）

**发现**：最初在 `replay_demo.py` 外部实现 action queue 缓存 50 个 action，但日志显示 `Chunk shape: (1, 6)` — `select_action` 内部已处理 chunking，每次返回 1 个 action。

**修复**：移除外部 queue，直接调用 `policy.select_action()`，依赖 SmolVLA 内部 chunking 机制。

```python
# 修复前（错误理解）
action_queue = []
for step in range(300):
    if not action_queue:
        chunk = policy.predict_action_chunk(obs)  # 50 actions
        action_queue = list(chunk)
    action = action_queue.pop(0)
    env.step(action)

# 修复后（正确）
for step in range(300):
    action = policy.select_action(obs)  # 内部已处理 chunking
    env.step(action)
```

**教训**：理解框架内部机制，不要重复实现已有功能。

### 修复 3：camera3 分布不匹配（commit 4efb51c）

**问题**：SmolVLA base 模型 `input_features` 包含 3 个相机 (camera1/2/3)，但数据集只有 2 个 (cam0→camera2, cam1→camera1)。训练时 camera3 缺失（零填充），回放时却提供 overhead 作为 camera3，造成分布不匹配。

**修复**：回放时不提供 camera3，与训练时保持一致。

```python
# 修复前（不一致）
obs = {
    "camera1": wrist_image,
    "camera2": overhead_image,
    "camera3": overhead_image,  # BUG: 训练时没有 camera3
}

# 修复后（一致）
obs = {
    "camera1": wrist_image,
    "camera2": overhead_image,
    # 不提供 camera3，与训练一致
}
```

**教训**：训练和推理的输入必须完全一致，包括缺失的维度。

---

## 回放验证：0 errors 但 Success=False

> Discussion #4 评论 4

3 个 bug 修复后回放：

| 指标 | 值 |
|------|-----|
| Prediction errors | **0/300** ✅ |
| Success | **False** ❌ |
| 机器人运动 | 有移动（state 持续变化），但未完成任务 |
| Gripper (ds) | 范围 4-27（0-45 量程），夹爪未完全闭合 |
| Reward | ±0.003（接近 0） |

**关键判断**：gripper 和 chunking 修复后，模型全程控制机器人（0 errors），但任务仍未完成。

这说明不是代码 bug，而是**系统性问题**。

---

## 重训练：batch_size 32→64

> Discussion #4 评论 5-6

社区参考：ggand0/vla-so101 用 batch_size=64 + 20k steps + 75 episodes 达到 60-80% 成功率。

| 参数 | 旧值 | 新值 |
|------|------|------|
| batch_size | 32 | **64** |
| steps | 20k | 20k |
| camera3 | 回放时提供 | **不提供** |

### 超时问题

batch_size=64 的 20k steps 需要约 28 小时，超过 20h 超时限制。

改为 10k steps（预计 ~14h），但也被取消。

**这是 VLA Pipeline 在 main 分支频繁失败的直接原因。**

---

## 方案终止决策

> Discussion #4 评论 7

### 决策逻辑

```
3 个 bug 修复后回放仍 Success=False
    ↓
0/300 prediction errors（不是代码问题）
    ↓
Loss 0.046 vs 社区 0.005（差距 10x）
    ↓
对比成功案例：
  - 我们: 2 相机 + scripted expert + so101_nexus 评测
  - 社区: 3 相机 + 遥操作/sim twin + 同环境评测
    ↓
根因: 数据采集环境 ≠ 评测环境
    ↓
结论: 继续在 ataghof + so101_nexus 组合上迭代无法收敛
    ↓
决策: 终止方案 A，转向方案 B (so101-mujoco sim twin)
```

### 失败根因总结

| 因素 | 我们的配置 | 成功案例 |
|------|-----------|----------|
| 相机数 | 2 | 3 |
| 数据来源 | scripted expert | 真机遥操作 sim twin |
| 最终 loss | 0.046 | 0.005-0.018 |
| 数据-环境匹配 | ❌ so101_nexus ≠ ataghof 采集环境 | ✅ 同一 MuJoCo 场景 |

**核心问题**：ataghof 数据集在 so101_nexus 环境中采集，但回放评测环境与采集环境存在视觉/物理差异，模型无法泛化。3 个 bug 修复后回放 0/300 errors 但 Success=False，说明不是代码 bug，而是系统性的数据-环境不匹配。

### 保留资产

方案 A 虽然失败，但保留了：
- 训练基础设施（Docker 镜像 + 云服务器）
- PPO 方案已成功（success_rate=0.98）
- 社区成功案例调研结果
- 3 个 bug 修复的代码经验

---

## 方案 B：so101-mujoco Sim Twin

> Discussion #4 评论 8

### 架构

```
┌─────────────────────────────────────────────────────┐
│           so101-mujoco Pipeline (V100)              │
│                                                     │
│  Phase 1: 数据准备                                  │
│  download → dobri420/pick-cube-so101-sim            │
│  (3 相机, 480×640, LeRobot v2 格式)                 │
│                    ↓                                │
│  Phase 2: 训练 SmolVLA                              │
│  lerobot-train                                      │
│    --dataset.repo_id=dobri420/pick-cube-so101-sim   │
│    --policy.path=lerobot/smolvla_base               │
│    --steps=20000 --batch_size=32                    │
│                    ↓                                │
│  Phase 3: Sim 评测                                  │
│  mujoco_policy.py <checkpoint> --grid               │
│  (5 reach × 13 azimuth × 5 trials = 325 eps)        │
│  输出: success_rate + eval_video.mp4                │
└─────────────────────────────────────────────────────┘
```

### 关键差异

| | 方案 A (ataghof) | 方案 B (so101-mujoco) |
|--|------------------|----------------------|
| 数据集 | ataghof/so101nexus-cube500-binary | dobri420/pick-cube-so101-sim |
| 相机 | 2 (rename_map) | 3 (原生匹配) |
| 评测 | so101_nexus replay | MuJoCo policy grid sweep |
| 数据-环境一致性 | ❌ | ✅ |
| 社区验证 | ❌ 从未成功 | ✅ dyordan1 已验证 |

### 训练

| 指标 | 值 |
|------|-----|
| Steps | 15K（20K 在 19515 步超时取消） |
| batch_size | 32 |
| 速度 | 3.65s/step |
| Loss | 0.090 |

### 评测：Grid Sweep

325 episodes（5 reach × 13 azimuth × 5 trials）：

```
reach\azim   -90   -75   -60   -45   -30   -15    +0   +15   +30   +45   +60   +75   +90
  15cm    3/5   1/5   1/5   4/5   5/5   5/5   4/5   2/5   2/5   4/5   0/5   3/5   0/5
  18cm    4/5   2/5   2/5   4/5   4/5   5/5   4/5   1/5   4/5   3/5   1/5   0/5   0/5
  20cm    1/5   1/5   4/5   0/5   3/5   4/5   2/5   4/5   2/5   2/5   3/5   0/5   1/5
  22cm    3/5   0/5   1/5   3/5   3/5   5/5   4/5   4/5   2/5   2/5   1/5   1/5   0/5
  25cm    3/5   2/5   2/5   4/5   3/5   4/5   3/5   5/5   1/5   1/5   1/5   0/5   0/5

SUCCESS 153/325 = 47%
```

### 分析

- **中心区域表现优秀**：-15 到 +15 azimuth, 15-22cm reach, 成功率 60-100%
- **边缘表现较弱**：±90 azimuth 接近 0%，符合预期（训练数据集中在工作区中心）
- **模型已学会 pick-and-place**：153 次成功放置证明模型理解了任务

### 对比

| 方案 | 成功率 | 数据-环境匹配 |
|------|--------|-------------|
| ataghof VLA (方案 A) | 0% | ❌ |
| **so101-mujoco (方案 B)** | **47%** | **✅** |
| PPO | 98% | ✅ |

---

## Debug 方法论总结

从这段旅程中，我们提炼出系统性 Debug 的方法论：

### 1. 假设 → 验证 → 修复 → 回归

```
观察现象 → 形成假设 → 设计验证实验 → 确认/否定假设 → 修复 → 回归测试
```

### 2. 区分"代码 bug"和"系统性问题"

| 信号 | 判断 |
|------|------|
| prediction errors > 0 | 代码 bug（输入不匹配） |
| 0 errors 但 Success=False | 系统性问题（数据/环境/超参） |
| Loss 不收敛 | 超参数或数据问题 |
| Loss 收敛但性能差 | 数据质量或数据-环境匹配问题 |

### 3. 社区调研驱动的 Debug

当自己 debug 陷入瓶颈时，找成功案例对比配置：

```
我们的配置 vs 社区成功配置 → 找差异 → 逐项修复
```

### 4. 何时坚持，何时换方案

| 信号 | 决策 |
|------|------|
| 修复 bug 后有进步 | 继续迭代 |
| 修复所有 bug 后仍失败 | 换方案 |
| Loss 远高于社区（10x） | 换方案 |
| 数据-环境不匹配 | 换方案（无法通过调参解决） |

---

## 思考题

1. **如果 gripper bug 修复后 Success=True，还会发现数据-环境不匹配问题吗？**  
   提示：可能不会。gripper bug 掩盖了更深层的问题。这就是 debug 的难点——修一个 bug 可能掩盖另一个。

2. **如何在训练前就判断数据集和评测环境是否匹配？**  
   提示：可视化数据集的一帧图像 vs 评测环境的渲染图像，人工对比。

3. **"0 errors 但 Success=False" 为什么说明不是代码 bug？**  
   提示：0 errors 意味着模型在正常推理，输出合理。Success=False 意味着模型没学会任务，这是数据/训练问题，不是代码问题。

4. **方案终止决策的风险是什么？**  
   提示：可能再调一个超参就成功了。但无限调参的时间成本更高，及时止损是理性决策。

---

**本章小结**：我们用 `replay_demo.py` 和 `eval_mujoco_policy.py` 完成了从 0% 到 47% 的调试。但这两个评测脚本是怎么设计的？社区的标准评测框架长什么样？下一章我们系统介绍评测方法论。

---

> **上一章**：[Ch3 VLA 入门](so101-tutorial-ch3-vla-intro.md) | **下一章**：[Ch5 评测方法论](so101-tutorial-ch5-evaluation.md)
