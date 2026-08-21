# Ch5：仿真评测方法论

> SO101 仿真评测教程 · 第五章  
> 实战项目：[link-seek/so101-sim-pipeline](https://github.com/link-seek/so101-sim-pipeline)

---

## 1. 为什么不能只看 Training Loss

训练 Loss 下降不代表模型性能好。我们在 ataghof 方案中就踩了这个坑：

| 训练阶段 | Loss | 回放 Success |
|----------|------|-------------|
| 5K steps | 0.119 | False |
| 20K steps | 0.046 | False |
| 社区成功案例 | 0.005-0.018 | True |

Loss 0.046 看起来"还行"，但和社区 0.005 差 10 倍。更关键的是，**Loss 只衡量 action 预测精度，不衡量任务完成度**。

模型可以完美预测每一步的 action（低 loss），但这些 action 串起来可能完不成任务（Success=False）。

### 正确的评测指标

| 指标 | 含义 | 重要性 |
|------|------|--------|
| **success_rate** | 任务完成率 | ⭐⭐⭐ 最重要 |
| **avg_reward** | 平均回报 | ⭐⭐ 辅助 |
| **avg_steps** | 完成任务的平均步数 | ⭐ 效率 |
| prediction errors | 推理异常次数 | ⭐ 调试用 |
| training loss | 训练损失 | ⭐ 仅参考 |

---

## 2. 确定性评估 vs 随机评估

### 2.1 确定性评估（PPO 用）

固定 seed，消除随机性，结果可复现：

```python
# scripts/eval_ppo.py
for ep in range(50):
    obs = env.reset(seed=1000 + ep)  # 固定 seed
    for step in range(512):
        action = agent.act(obs, deterministic=True)  # 确定性策略
        obs, reward, done, info = env.step(action)
```

**优点**：结果可复现，同一 checkpoint 永远得到同一 success_rate  
**缺点**：只覆盖 50 个固定初始条件，可能有偏差

### 2.2 Grid Sweep（VLA 用）

系统扫描参数空间，覆盖更全面的初始条件：

```python
# scripts/eval_mujoco_policy.py
reach_values = [0.15, 0.18, 0.20, 0.22, 0.25]      # 5 个距离
azimuth_values = range(-90, 91, 15)                  # 13 个角度
trials = 5                                           # 每个条件 5 次

for reach in reach_values:
    for azim in azimuth_values:
        for trial in range(trials):
            obs = env.reset(reach=reach, azimuth=azim, seed=trial)
            ...  # 评估
# 总计 5 × 13 × 5 = 325 episodes
```

**优点**：覆盖整个工作空间，能发现性能热点和冷区  
**缺点**：评估时间长（325 vs 50 episodes）

### 2.3 我们的选择

| 方法 | PPO | VLA |
|------|-----|-----|
| 评估方式 | 确定性 50 episodes | Grid sweep 325 episodes |
| 原因 | MLP 策略稳定，50 ep 够 | VLA 泛化性差，需扫描参数空间 |

---

## 3. Grid Sweep 详解

### 3.1 参数空间

```
reach (物体距离):  15cm, 18cm, 20cm, 22cm, 25cm
azimuth (物体角度): -90°, -75°, -60°, ..., +75°, +90°
trials (重复次数):  5
```

### 3.2 热力图

```
reach\azim   -90   -75   -60   -45   -30   -15    +0   +15   +30   +45   +60   +75   +90
  15cm    3/5   1/5   1/5   4/5   5/5   5/5   4/5   2/5   2/5   4/5   0/5   3/5   0/5
  18cm    4/5   2/5   2/5   4/5   4/5   5/5   4/5   1/5   4/5   3/5   1/5   0/5   0/5
  20cm    1/5   1/5   4/5   0/5   3/5   4/5   2/5   4/5   2/5   2/5   3/5   0/5   1/5
  22cm    3/5   0/5   1/5   3/5   3/5   5/5   4/5   4/5   2/5   2/5   1/5   1/5   0/5
  25cm    3/5   2/5   2/5   4/5   3/5   4/5   3/5   5/5   1/5   1/5   1/5   0/5   0/5

SUCCESS 153/325 = 47%
```

### 3.3 如何解读热力图

**中心区域（-15 到 +15 azimuth, 15-22cm reach）**：成功率 60-100%

这是机器人最舒适的工作区域，也是训练数据最密集的区域。

**边缘区域（±90 azimuth）**：成功率接近 0%

物体在机器人侧面极限位置，训练数据中很少覆盖。

**结论**：47% 是 325 个不同初始条件的平均，不是单一条件。中心区域已经可用，边缘需要更多数据覆盖。

### 3.4 可视化

```python
import matplotlib.pyplot as plt
import numpy as np

# 325 个结果转为 5×13 矩阵
results = np.array([
    [3,1,1,4,5,5,4,2,2,4,0,3,0],  # 15cm
    [4,2,2,4,4,5,4,1,4,3,1,0,0],  # 18cm
    [1,1,4,0,3,4,2,4,2,2,3,0,1],  # 20cm
    [3,0,1,3,3,5,4,4,2,2,1,1,0],  # 22cm
    [3,2,2,4,3,4,3,5,1,1,1,0,0],  # 25cm
]) / 5  # 转为成功率

plt.figure(figsize=(14, 4))
plt.imshow(results, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
plt.colorbar(label='Success Rate')
plt.xlabel('Azimuth (°)')
plt.ylabel('Reach (cm)')
plt.title('SO101 Pick-and-Place Grid Sweep')
plt.show()
```

---

## 4. 回放验证

### 4.1 何时用回放 vs Grid Sweep

| 方法 | 回放 (replay) | Grid Sweep |
|------|---------------|------------|
| Episodes | 1 (300 steps) | 325 |
| 耗时 | ~30s | ~30min |
| 用途 | 快速验证 | 正式评估 |
| 时机 | 每次训练后 | 关键 checkpoint |

回放是"快速 smoke test"，Grid Sweep 是"正式考试"。

### 4.2 回放流程

```python
# scripts/replay_demo.py
def replay(checkpoint, env_id="MuJoCoPickAndPlace-v1"):
    policy = load_policy(checkpoint)
    env = make_env(env_id)
    obs = env.reset()
    
    errors = 0
    frames = []
    rewards = []
    
    for step in range(300):
        # 推理
        action = policy.select_action(prepare_obs(obs))
        
        # 执行
        obs, reward, done, info = env.step(postprocess(action))
        
        # 记录
        frames.append(env.render())
        rewards.append(reward)
        if info.get("prediction_error"):
            errors += 1
    
    # 输出
    save_video(frames, "replay_pickplace.mp4")
    save_json({
        "prediction_errors": errors,
        "success": info["success"],
        "rewards": rewards,
    }, "replay_pickplace.json")
```

### 4.3 回放指标解读

| 指标 | 好的值 | 坏的值 | 含义 |
|------|--------|--------|------|
| prediction_errors | 0/300 | >0 | 模型是否正常推理 |
| success | True | False | 是否完成任务 |
| reward 趋势 | 上升 | 振荡/下降 | 是否在接近目标 |
| state 变化 | 持续 | 不变 | 机器人是否在动 |

---

## 5. 评测指标设计原则

### 原则 1：指标要和任务语义对齐

```python
# 好的指标：物体是否放到目标位置
success = np.linalg.norm(object_pos - target_pos) < 0.02

# 坏的指标：reward 信号（可能被 gaming）
success = reward > threshold  # 策略可能学会刷 reward
```

我们在 PPO 中就遇到这个问题：`lift_threshold=0.05` 时策略学会微抬 5cm 骗过 reward。

### 原则 2：覆盖足够的初始条件

50 episodes 可能只覆盖中心区域，grid sweep 325 episodes 能发现边缘问题。

### 原则 3：结果可复现

固定 seed，确保同一 checkpoint 永远得到同一结果。否则无法对比不同训练版本。

### 原则 4：归档完整

```json
{
  "checkpoint": "015000",
  "eval_config": {
    "method": "grid_sweep",
    "reach": [0.15, 0.18, 0.20, 0.22, 0.25],
    "azimuth": [-90, -75, ..., 90],
    "trials": 5
  },
  "results": {
    "success_rate": 0.47,
    "total_episodes": 325,
    "successes": 153,
    "per_condition": "..."
  },
  "eval_video": "eval_video.mp4"
}
```

---

## 踩坑复盘

### 坑 1：只看 Loss 误判性能

**现象**：Loss 0.046，以为"还不错"。

**实际**：社区 0.005，差 10x。Loss 掩盖了数据-环境不匹配的问题。

**教训**：Loss 只衡量 action 预测精度，不衡量任务完成度。必须做仿真评估。

### 坑 2：lift_threshold 影响成功率

**现象**：PPO v1 success_rate=100%，但视频里物体几乎没动。

**根因**：`lift_threshold=0.05` 太低，5cm 就算"抬起"。

**修复**：提高到 0.15，success_rate 降至 98%，但视频中有明显抬起动作。

**教训**：评测指标要和任务语义对齐，不能只看数字。

### 坑 3：Grid Sweep 暴露训练数据偏置

**现象**：VLA 47% 成功率，但中心区域 60-100%，边缘 ~0%。

**分析**：训练数据集中在工作区中心，边缘覆盖不足。

**教训**：Grid Sweep 能发现训练数据的覆盖盲区，单一指标会掩盖这个问题。

---

## 思考题

1. **为什么 grid sweep 要用 5 trials 而不是 1？**  
   提示：单次试验有随机性（物体初始姿态、接触动力学），5 次取平均更可靠。

2. **如果训练数据只覆盖中心区域，边缘 0% 是 bug 还是预期？**  
   提示：是预期。模型没有边缘数据，无法泛化。解决方案是补充边缘数据或数据增强。

3. **PPO 用 50 episodes，VLA 用 325 episodes，为什么？**  
   提示：PPO 在训练环境中 on-policy 学习，泛化性好。VLA 从固定数据集学习，泛化性受限，需要更全面评估。

4. **如何设计一个更好的评测指标，既考虑成功率又考虑效率？**  
   提示：可以加权 success_rate 和 1/avg_steps，或用 "在 N 步内完成的概率"。

---

> **上一章**：[Ch4 Debug 实战](so101-tutorial-ch4-debug-journey.md) | **下一章**：[Ch6 优化进阶](so101-tutorial-ch6-optimization.md)
