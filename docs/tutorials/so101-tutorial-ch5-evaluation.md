# Ch5：仿真评测方法论

> SO101 仿真评测教程 · 第五章  
> 实战项目：[link-seek/so101-sim-pipeline](https://github.com/link-seek/so101-sim-pipeline)  
> 参考框架：[Gymnasium](https://gymnasium.farama.org/) · [LeRobot `lerobot-eval`](https://github.com/huggingface/lerobot) · [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) · [LIBERO-PRO](https://github.com/sylvestf/LIBERO-plus)

---

## 1. 开源评测框架全景

机器人策略评测不是我们发明的——社区已有成熟框架。我们的项目站在它们肩膀上：

```
评测框架层次:

  Gymnasium (环境 API 标准)
    │
    ├── LeRobot lerobot-eval (VLA 通用评测)
    │     └── 我们的 replay_demo.py 遵循其推理管线
    │
    ├── LIBERO (VLA 标准 benchmark, 2.2k stars)
    │     └── LIBERO-PRO (鲁棒性扩展, 423 stars)
    │           └── 我们的 eval_vla.py 调用 vla-eval harness
    │
    ├── CleanRL (RL 评测范式)
    │     └── 我们的 eval_ppo.py 遵循其确定性评估模式
    │
    └── so101-mujoco (社区 sim twin)
          └── 我们的 eval_mujoco_policy.py 封装其 grid sweep
```

### 各框架对比

| 框架 | 用途 | 核心指标 | 我们的使用方式 |
|------|------|----------|---------------|
| **Gymnasium** | 环境 API 标准 | `info["success"]`, `reward` | 所有 eval 脚本的底层 API |
| **LeRobot `lerobot-eval`** | VLA 通用评测 | `pc_success`, `avg_sum_reward`, `avg_max_reward` | `replay_demo.py` 用其推理管线 |
| **LIBERO** | VLA benchmark | task success rate × 10 tasks | `eval_vla.py` 通过 vla-eval harness |
| **LIBERO-PRO** | VLA 鲁棒性 | 5 个扰动维度 | `eval_vla.py` 的 libero_pro_* benchmarks |
| **CleanRL** | RL 评估范式 | `success_rate`, `ep_return` | `eval_ppo.py` 的确定性评估 |
| **so101-mujoco** | SO101 sim twin | grid sweep success rate | `eval_mujoco_policy.py` 封装 |

---

## 2. Gymnasium：评测的通用语言

### 2.1 标准 API

所有评测脚本都基于 [Gymnasium](https://gymnasium.farama.org/) 的标准 API：

```python
import gymnasium as gym

env = gym.make("MuJoCoPickAndPlace-v1", render_mode="rgb_array")
obs, info = env.reset(seed=42)          # 重置环境, 返回初始观测
action = policy(obs)                     # 策略推理
obs, reward, terminated, truncated, info = env.step(action)  # 执行
# info["success"] → bool: 任务是否完成
```

**关键约定**（Gymnasium 标准）：
- `reset(seed)` → 固定 seed 保证可复现
- `step()` 返回 5 元组 `(obs, reward, terminated, truncated, info)`
- `info["success"]` → 任务完成判定（环境定义，非策略定义）
- `terminated` → 任务自然结束（成功或失败）
- `truncated` → 因步数上限被截断

### 2.2 为什么用 Gymnasium 而不是自己写

LeRobot、so101_nexus、LIBERO 全部基于 Gymnasium API。用标准 API 意味着：
- 同一策略可以在任何 Gymnasium 环境中评测
- 环境的 `info["success"]` 判定逻辑由环境作者维护，评测者不需要自己定义
- 向量化环境 (`gym.vector.VectorEnv`) 可以并行评测多个 episode

---

## 3. LeRobot `lerobot-eval`：VLA 评测的标准做法

### 3.1 框架概述

[LeRobot](https://github.com/huggingface/lerobot) 提供了 `lerobot-eval` 命令行工具，是 VLA 评测的社区标准：

```bash
lerobot-eval \
    --policy.path=lerobot/smolvla_base \
    --env.type=pusht \
    --eval.n_episodes=10 \
    --eval.batch_size=10 \
    --policy.device=cuda
```

### 3.2 评测配置（`EvalPipelineConfig`）

```python
@dataclass
class EvalConfig:
    n_episodes: int = 50        # 评测 episode 数
    batch_size: int = 0         # 0 = 自动选择(按 CPU 核数, 上限 64)
    # seed: int = 1000          # 起始 seed, 每个 episode 递增

@dataclass
class EvalPipelineConfig:
    env: EnvConfig              # 环境配置
    eval: EvalConfig            # 评测配置
    policy: PreTrainedConfig    # 策略配置
    seed: int = 1000            # 全局 seed
    rename_map: dict = {}       # 观测键名映射
```

### 3.3 LeRobot 的评测指标

`lerobot-eval` 输出的指标体系（`src/lerobot/scripts/lerobot_eval.py`）：

```python
info = {
    "per_episode": [
        {
            "episode_ix": i,
            "sum_reward": sum_reward,    # 累积 reward
            "max_reward": max_reward,     # 最大 reward
            "success": success,           # 是否成功
            "seed": seed,                 # 使用的 seed
        }
        for i, (...) in enumerate(...)
    ],
    "aggregated": {
        "avg_sum_reward": float(np.nanmean(sum_rewards)),   # 平均累积 reward
        "avg_max_reward": float(np.nanmean(max_rewards)),   # 平均最大 reward
        "pc_success": float(np.nanmean(all_successes) * 100),  # 成功率 (%)
        "eval_s": elapsed,                # 评测总耗时
        "eval_ep_s": elapsed / n_episodes,  # 每 episode 耗时
    },
}
```

| 指标 | LeRobot 名称 | 我们 `eval_ppo.py` 对应 | 含义 |
|------|-------------|----------------------|------|
| 成功率 | `pc_success` | `success_rate` | `mean(successes) * 100` |
| 平均累积回报 | `avg_sum_reward` | `avg_reward` | `mean(sum(rewards))` |
| 平均最大回报 | `avg_max_reward` | — | `mean(max(rewards))` |
| 每 episode 耗时 | `eval_ep_s` | `elapsed_s / num_episodes` | 效率指标 |

**我们的 `eval_ppo.py` 遵循了同样的指标设计**，只是命名不同。`avg_max_reward` 我们没追踪，因为 PPO 的 reward 语义和 VLA 不同。

### 3.4 我们的 `replay_demo.py` 如何使用 LeRobot 推理管线

```python
# scripts/replay_demo.py — 使用 LeRobot 标准推理管线
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference

# 加载策略 + 预处理器（LeRobot 标准）
policy = SmolVLAPolicy.from_pretrained(checkpoint)
preprocess, postprocess = make_pre_post_processors(policy.config, checkpoint, ...)

# 推理循环（遵循 LeRobot 管线）
frame = prepare_observation_for_inference(frame, device, task=task_description)
frame = preprocess(frame)
action = policy.select_action(frame)      # LeRobot 标准推理
action = postprocess(action)               # 后处理
```

这不是我们自己写的推理逻辑——是 LeRobot 的标准管线。`prepare_observation_for_inference` → `preprocess` → `select_action` → `postprocess` 是 LeRobot 定义的推理协议。

---

## 4. LIBERO：VLA 的标准 Benchmark

### 4.1 为什么需要 Benchmark

单任务评测（如我们的 grid sweep）只能验证特定任务。[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)（2.2k stars）提供了标准化的多任务评测套件：

| Benchmark | 任务数 | 评测重点 | 每任务 episodes |
|-----------|--------|----------|----------------|
| `libero_spatial` | 10 | 空间泛化（物体位置变化） | 50 |
| `libero_object` | 10 | 物体泛化（不同物体） | 50 |
| `libero_goal` | 10 | 目标泛化（不同目标位置） | 50 |

### 4.2 LIBERO-PRO：鲁棒性扩展

[LIBERO-PRO](https://github.com/sylvestf/LIBERO-plus)（423 stars）在 LIBERO 基础上增加 5 个扰动维度：

| Benchmark | 扰动类型 |
|-----------|---------|
| `libero_pro_swap` | 物体替换 |
| `libero_pro_object` | 物体属性变化 |
| `libero_pro_lan` | 语言指令变化 |
| `libero_pro_task` | 任务组合 |
| `libero_pro_env` | 环境变化 |

### 4.3 我们的 `eval_vla.py` 如何使用

```python
# scripts/eval_vla.py — 通过 vla-eval harness 运行 LIBERO
LIBERO_BENCHMARKS = ["libero_spatial", "libero_object", "libero_goal"]
LIBERO_PRO_BENCHMARKS = ["libero_pro_swap", "libero_pro_object", 
                          "libero_pro_lan", "libero_pro_task", "libero_pro_env"]

# 1. 启动模型推理服务
server_proc = start_model_server("smolvla_so101.yaml", checkpoint)

# 2. 逐 benchmark 运行
for name in benchmarks:
    run_benchmark(name)    # vla-eval run --config configs/benchmarks/{name}.yaml
    merge_results(name)    # vla-eval merge → merged.json

# 3. 汇总
summary = {
    "benchmarks": {
        "libero_spatial": {"success_rate": 0.72, "num_episodes": 500, "num_tasks": 10},
        ...
    }
}
```

benchmark 配置（`configs/benchmarks/libero_spatial.yaml`）：
```yaml
benchmarks:
  - benchmark: "vla_eval.benchmarks.libero.benchmark:LibEROBenchmark"
    subname: libero_spatial
    episodes_per_task: 50      # 每个任务 50 episodes
    params:
      suite: libero_spatial
      seed: 7
      num_steps_wait: 10
```

**关键区别**：LIBERO 评测的是**跨任务泛化能力**，grid sweep 评测的是**单任务工作空间覆盖**。两者互补。

---

## 5. 评测方法论：从理论到实践

### 5.1 训练指标 vs 评估指标

这是 RL/BC 社区的共识（参见 [Spinning Up](https://spinningup.openai.com/) 的评测章节）：

| | 训练指标 | 评估指标 |
|--|---------|---------|
| **PPO** | reward 曲线, entropy | `success_rate` (独立评估环境) |
| **VLA** | MSE Loss | `pc_success` (LeRobot), `success_rate` (grid sweep) |
| **用途** | 监控收敛 | 判断任务完成能力 |
| **陷阱** | Loss 低 ≠ 性能好 (BC) | — |

我们在 ataghof 方案中的教训：

| 训练阶段 | Loss | 回放 Success |
|----------|------|-------------|
| 5K steps | 0.119 | False |
| 20K steps | 0.046 | False |
| 社区成功案例 | 0.005-0.018 | True |

Loss 0.046 看起来"还行"，但和社区 0.005 差 10 倍。**Loss 只衡量 action 预测精度，不衡量任务完成度**——这是 BC 评测的核心原则。

### 5.2 确定性评估（CleanRL 范式）

[CleanRL](https://github.com/vwxyzjn/cleanrl) 确立了 RL 评测的标准做法：固定 seed + 确定性策略 + 足够多的 episodes。

我们的 `eval_ppo.py` 遵循这一范式：

```python
# scripts/eval_ppo.py — CleanRL 风格确定性评估
agent.eval()  # 关闭 dropout/batchnorm

for ep in range(50):
    obs, _ = env.reset(seed=12345 + ep)        # 固定 seed
    while not done:
        a = agent.actor_mean(norm(obs))         # 确定性策略 (用 mean, 不采样)
        obs, r, term, trunc, info = env.step(a)
        ever_succ = ever_succ or info.get("success", False)
```

**为什么用 `actor_mean` 而不是采样**：评估时要看策略的"真实水平"，不是"运气好时的水平"。训练时用 `mean + std * noise` 探索，评估时只用 `mean`。

### 5.3 Grid Sweep（社区 sim twin 范式）

Grid sweep 不是标准 RL 评测方法，而是机器人仿真社区（如 [dyordan1/so101-mujoco](https://github.com/dyordan1/so101-mujoco)）的实践——系统扫描工作空间初始条件：

```python
# scripts/eval_mujoco_policy.py 封装 dyordan1/so101-mujoco 的 --sweep
reach_values = [0.15, 0.18, 0.20, 0.22, 0.25]      # 5 个距离
azimuth_values = range(-90, 91, 15)                  # 13 个角度
trials = 5                                           # 每个条件 5 次
# 总计 5 × 13 × 5 = 325 episodes
```

**与 Gymnasium 标准评测的关系**：Grid sweep 本质上是 `n_episodes=325` 的评测，只是 episode 的初始条件不是随机采样而是网格采样。每个 episode 仍然遵循 `reset → step × N → check success` 的标准循环。

### 5.4 统计显著性：多少 episodes 才够

这是评测理论中常被忽略的问题。给定成功率 p，N 个 episodes 的标准误差：

```
SE = sqrt(p * (1-p) / N)
```

| N | p=0.47 | 95% CI | 含义 |
|---|--------|--------|------|
| 50 | 0.47 | ±0.14 | [33%, 61%] — 太宽 |
| 325 | 0.47 | ±0.055 | [41.5%, 52.5%] — 可接受 |
| 1000 | 0.47 | ±0.031 | [43.9%, 50.1%] — 好 |

**我们的选择**：
- PPO: 50 episodes — PPO 策略稳定（100% 或 0%），不需要多 episode
- VLA: 325 episodes — VLA 泛化性差，需要足够 episodes 才有统计意义
- LIBERO: 50 ep × 10 tasks = 500 episodes — 多任务评测，总量足够

---

## 6. Grid Sweep 详解

### 6.1 参数空间

```
reach (物体距离):  15cm, 18cm, 20cm, 22cm, 25cm
azimuth (物体角度): -90°, -75°, -60°, ..., +75°, +90°
trials (重复次数):  5
```

### 6.2 热力图

```
reach\azim   -90   -75   -60   -45   -30   -15    +0   +15   +30   +45   +60   +75   +90
  15cm    3/5   1/5   1/5   4/5   5/5   5/5   4/5   2/5   2/5   4/5   0/5   3/5   0/5
  18cm    4/5   2/5   2/5   4/5   4/5   5/5   4/5   1/5   4/5   3/5   1/5   0/5   0/5
  20cm    1/5   1/5   4/5   0/5   3/5   4/5   2/5   4/5   2/5   2/5   3/5   0/5   1/5
  22cm    3/5   0/5   1/5   3/5   3/5   5/5   4/5   4/5   2/5   2/5   1/5   1/5   0/5
  25cm    3/5   2/5   2/5   4/5   3/5   4/5   3/5   5/5   1/5   1/5   1/5   0/5   0/5

SUCCESS 153/325 = 47%
```

### 6.3 如何解读热力图

**中心区域（-15 到 +15 azimuth, 15-22cm reach）**：成功率 60-100%

这是机器人最舒适的工作区域，也是训练数据最密集的区域。

**边缘区域（±90 azimuth）**：成功率接近 0%

物体在机器人侧面极限位置，训练数据中很少覆盖。

**结论**：47% 是 325 个不同初始条件的平均，不是单一条件。中心区域已经可用，边缘需要更多数据覆盖。

### 6.4 可视化

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

## 7. 回放验证：LeRobot 推理管线的 smoke test

### 7.1 何时用回放 vs Grid Sweep vs LIBERO

| 方法 | 回放 (replay) | Grid Sweep | LIBERO |
|------|---------------|------------|--------|
| Episodes | 1 (300 steps) | 325 | 500 (10 tasks × 50) |
| 耗时 | ~30s | ~30min | ~2h |
| 用途 | 快速 smoke test | 单任务正式评估 | 跨任务泛化评估 |
| 时机 | 每次训练后 | 关键 checkpoint | 里程碑节点 |
| 框架 | LeRobot 推理管线 | so101-mujoco | vla-eval harness |

回放是"快速 smoke test"，Grid Sweep 是"单任务考试"，LIBERO 是"毕业考试"。

### 7.2 回放流程（遵循 LeRobot 推理管线）

```python
# scripts/replay_demo.py — 使用 LeRobot 标准推理管线
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference

policy = SmolVLAPolicy.from_pretrained(checkpoint)
preprocess, postprocess = make_pre_post_processors(policy.config, checkpoint)

for step in range(300):
    # LeRobot 标准推理管线
    frame = prepare_observation_for_inference(frame, device, task=task_description)
    frame = preprocess(frame)
    action = policy.select_action(frame)      # SmolVLA 推理 (含 action chunking)
    action = postprocess(action)
    
    # so101_nexus 官方单位转换
    action_rad = dataset_row_to_sim_qpos(action)  # 数据集 → 仿真
    obs, reward, _, _, info = env.step(action_rad)
```

### 7.3 回放指标解读

| 指标 | 好的值 | 坏的值 | 含义 |
|------|--------|--------|------|
| prediction_errors | 0/300 | >0 | 模型是否正常推理 |
| success | True | False | 是否完成任务 |
| reward 趋势 | 上升 | 振荡/下降 | 是否在接近目标 |
| state 变化 | 持续 | 不变 | 机器人是否在动 |

---

## 8. 评测指标设计原则

### 原则 1：指标要和任务语义对齐（Gymnasium `info["success"]`）

Gymnasium 环境的 `info["success"]` 由环境作者定义，评测者不应自己定义成功条件：

```python
# 好：用环境定义的 success
success = info["success"]  # 环境作者已经定义了什么是"完成"

# 坏：自己定义 success（可能和 reward 不对齐）
success = reward > threshold  # 策略可能学会刷 reward
```

我们在 PPO 中就遇到这个问题：`lift_threshold=0.05` 时 `info["success"]` 返回 True，但视频里物体几乎没动。修复方式是修改环境的 `lift_threshold` 参数，不是自己定义 success。

### 原则 2：覆盖足够的初始条件（统计显著性）

50 episodes 的 95% CI 约 ±14%，325 episodes 约 ±5.5%。选择 episode 数量要考虑：

```python
# 所需 episodes 数（95% CI 宽度 < w）
N > 1.96^2 * p * (1-p) / (w/2)^2

# 例：p=0.47, 想要 CI 宽度 < 10% (±5%)
N > 3.84 * 0.47 * 0.53 / 0.0025 = 382
```

### 原则 3：结果可复现（固定 seed）

LeRobot `lerobot-eval` 默认 `seed=1000`，我们的 `eval_ppo.py` 用 `seed=12345`。固定 seed 确保同一 checkpoint 永远得到同一结果，否则无法对比不同训练版本。

### 原则 4：归档完整（LeRobot `per_episode` 格式）

遵循 LeRobot 的 `per_episode` 格式，每个 episode 都记录完整信息：

```json
{
  "per_episode": [
    {"episode": 0, "success": true, "reward": 1.52, "steps": 48, "seed": 12345},
    {"episode": 1, "success": true, "reward": 1.48, "steps": 52, "seed": 12346},
    ...
  ],
  "aggregated": {
    "success_rate": 0.98,
    "avg_reward": 1.511,
    "avg_steps": 58.4,
    "elapsed_s": 13.8
  }
}
```

---

## 踩坑复盘

### 坑 1：只看 Loss 误判性能

**现象**：Loss 0.046，以为"还不错"。

**实际**：社区 0.005，差 10x。Loss 掩盖了数据-环境不匹配的问题。

**教训**：Loss 只衡量 action 预测精度，不衡量任务完成度。必须做仿真评估。这是 BC 评测的核心原则，LeRobot 的 `lerobot-eval` 也是先跑 rollout 再算 `pc_success`，不只看 loss。

### 坑 2：lift_threshold 影响成功率

**现象**：PPO v1 success_rate=100%，但视频里物体几乎没动。

**根因**：`lift_threshold=0.05` 太低，5cm 就算"抬起"。环境的 `info["success"]` 判定标准太宽松。

**修复**：修改环境参数 `lift_threshold=0.15`，success_rate 降至 98%，但视频中有明显抬起动作。

**教训**：评测指标要和任务语义对齐。Gymnasium 的 `info["success"]` 依赖环境配置，评测前要理解环境的成功判定逻辑。

### 坑 3：Grid Sweep 暴露训练数据偏置

**现象**：VLA 47% 成功率，但中心区域 60-100%，边缘 ~0%。

**分析**：训练数据集中在工作区中心，边缘覆盖不足。

**教训**：Grid Sweep 能发现训练数据的覆盖盲区，单一指标（如 LIBERO 的 `pc_success`）会掩盖这个问题。两种评测方法互补。

---

## 思考题

1. **LeRobot 的 `pc_success` 和我们的 `success_rate` 有什么区别？**  
   提示：只是命名和单位不同（百分比 vs 小数），计算方式相同：`mean(successes)`。

2. **为什么 grid sweep 要用 5 trials 而不是 1？**  
   提示：单次试验有随机性（物体初始姿态、接触动力学），5 次取平均更可靠。统计上 N=5 的 SE = sqrt(p(1-p)/5)。

3. **如果训练数据只覆盖中心区域，边缘 0% 是 bug 还是预期？**  
   提示：是预期。模型没有边缘数据，无法泛化。解决方案是补充边缘数据或数据增强。LIBERO 的 `libero_spatial` 就是为了测试这种空间泛化。

4. **PPO 用 50 episodes，VLA 用 325 episodes，为什么？**  
   提示：PPO 在训练环境中 on-policy 学习，泛化性好（MLP 不依赖视觉）。VLA 从固定数据集学习，泛化性受限，需要更全面评估。统计上 PPO 100% 时 SE=0，50 ep 足够。

5. **LIBERO 评测和 grid sweep 评测有什么互补性？**  
   提示：LIBERO 测跨任务泛化（不同物体/目标/语言），grid sweep 测单任务工作空间覆盖。一个模型可能 LIBERO 80% 但 grid sweep 边缘 0%。

---

> **上一章**：[Ch4 Debug 实战](so101-tutorial-ch4-debug-journey.md) | **下一章**：[Ch6 优化进阶](so101-tutorial-ch6-optimization.md)
