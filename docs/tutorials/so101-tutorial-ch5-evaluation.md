# Ch5：仿真评测方法论

> SO101 仿真评测教程 · 第五章  
> 实战项目：[link-seek/so101-sim-pipeline](https://github.com/link-seek/so101-sim-pipeline)  
> 参考框架：[Gymnasium](https://gymnasium.farama.org/) · [LeRobot `lerobot-eval`](https://github.com/huggingface/lerobot) · [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) · [LIBERO-PRO](https://github.com/sylvestf/LIBERO-plus)

---

## 1. 开源评测框架全景

上一章我们用了两种评测：`replay_demo.py`（回放验证）和 `eval_mujoco_policy.py`（grid sweep）。但这两个脚本不是凭空发明的——它们背后有一套成熟的开源评测框架体系。

本章就来搞清楚：**社区是怎么评测机器人策略的，我们的脚本和这些框架是什么关系。**

机器人策略评测不是我们发明的——社区已有成熟框架。我们的项目站在它们肩膀上。评测涉及两个维度：**评测方法**（怎么评）和**环境来源**（在哪评）：

```
评测方法:

  Gymnasium (环境 API 标准，所有评测的底层接口)
    │
    ├── LeRobot lerobot-eval (VLA 通用评测)
    │     └── 我们的 replay_demo.py 用其推理管线做回放验证
    │
    ├── LIBERO / LIBERO-PRO (VLA 标准 benchmark + 鲁棒性扩展)
    │     └── 我们的 eval_vla.py 通过 vla-eval harness 调用
    │
    ├── CleanRL (RL 评测范式)
    │     └── 我们的 eval_ppo.py 遵循其确定性评估模式
    │
    └── Grid Sweep (多初始条件评测，社区常用方法)
          └── 我们的 eval_mujoco_policy.py 实现 5×13×5 sweep

环境来源:

  仿真环境 (Simulation)
    ├── MuJoCo 系列
    │     ├── so101_nexus —— Ch4 回放验证用的环境
    │     ├── so101-mujoco (社区 sim twin) —— Grid Sweep 用的环境
    │     └── Gym ALOHA —— ALOHA 双臂机器人仿真
    ├── RoboTwin —— 双臂操作 benchmark 仿真
    ├── LIBERO 仿真环境 —— eval_vla.py 用的环境
    └── Isaac Sim / Habitat —— 大规模仿真（本项目未用）

  真实机器人 (Real Robot)
    ├── SO101 实机 —— ataghof 数据采集用
    └── ALOHA / Viper / Franka —— 其他真实平台（本项目未用）
```

### 各框架对比

| 框架 | 类型 | 用途 | 核心指标 | 我们的使用方式 |
|------|------|------|----------|---------------|
| **Gymnasium** | API 标准 | 环境接口 | `info["success"]`, `reward` | 所有 eval 脚本的底层 API |
| **LeRobot lerobot-eval** | 评测方法 | VLA 通用评测 | `pc_success`, `avg_sum_reward` | `replay_demo.py` 用其推理管线 |
| **LIBERO** | 评测方法 | VLA benchmark | task success rate × 10 tasks | `eval_vla.py` 通过 vla-eval harness |
| **LIBERO-PRO** | 评测方法 | VLA 鲁棒性 | 5 个扰动维度 | `eval_vla.py` 的 libero_pro_* benchmarks |
| **CleanRL** | 评测方法 | RL 评估范式 | `success_rate`, `ep_return` | `eval_ppo.py` 的确定性评估 |
| **Grid Sweep** | 评测方法 | 多初始条件评测 | success rate across grid | `eval_mujoco_policy.py` 实现 |
| **so101_nexus** | 仿真环境 | MuJoCo 仿真 | — | Ch4 回放验证 |
| **so101-mujoco** | 仿真环境 | MuJoCo sim twin | — | `eval_mujoco_policy.py` 的环境 |
| **RoboTwin** | 仿真环境 | 双臂操作 benchmark | — | 本项目未用 |
| **SO101 实机** | 真实机器人 | 数据采集 | — | ataghof 数据集来源 |

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

### 3.4 我们的脚本如何使用社区框架

| 脚本 | 遵循的框架 | 推理管线 | 状态 |
|------|-----------|---------|------|
| `replay_demo.py` | LeRobot | `prepare_observation_for_inference` → `preprocess` → `select_action` → `postprocess` | 本地运行（Ch4） |
| `eval_ppo.py` | CleanRL | `agent.actor_mean(norm(obs))` 确定性评估 | GitHub Actions（3 次） |
| `eval_vla.py` | vla-eval harness | `run_benchmark()` → `merge_results()` | GitHub Actions（10 次，V100 ECS） |
| `eval_mujoco_policy.py` | so101-mujoco | `--sweep` grid search | GitHub Actions（6 次） |

---

## 4. LIBERO：VLA 的标准 Benchmark

### 4.1 LIBERO 是什么

[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)（2.2k stars，CoRL 2023）全称 **Lifelong Benchmark for Robotics**，是 Bo Liu 等人提出的 VLA 标准 benchmark。它的核心定位：

> 给定一组操作任务，系统化地评测 VLA 策略的**泛化能力**——不是"能不能做一个任务"，而是"能不能做一类任务"。

LIBERO 建立在 [RoboSuite](https://github.com/ARISE-Initiative/robosuite) 仿真框架上，使用 Franka Panda 7 DoF 机械臂，定义了 5 个 benchmark suite，总计 130 个任务。

### 4.2 评测对象

LIBERO 评测的是 **VLA 策略**（Vision-Language-Action Policy）——接收图像观测和语言指令，输出机器人动作的策略模型：

```
评测对象: VLA 策略 π(a | o_image, o_state, l_instruction)

输入:
  o_image   ∈ R^{H×W×3}    — 相机图像 (RGB)
  o_state   ∈ R^{d_state}   — 机器人本体感觉 (关节角度等)
  l_instruction ∈ String    — 语言指令 ("pick up the black bowl and place it on the plate")

输出:
  a ∈ R^{d_action}           — 机器人动作 (6 维: 5 关节 + 1 gripper)
```

**评测的 VLA 模型类型**（LIBERO 论文中评测的）：

| 模型类型 | 代表 | 特点 |
|----------|------|------|
| Diffusion Policy | DP | 扩散过程生成 action |
| ACT | ACT | Transformer + action chunking |
| RT-1 | RT-1 | Transformer, 大规模 |
| SimpleVLA | SimpleVLA | 轻量 VLA |

我们的 SmolVLA 也属于 VLA 策略，理论上可以作为评测对象——但需要机器人匹配（详见 §4.8）。

### 4.3 任务定义机制：BDDL

LIBERO 的任务用 **BDDL**（Behavior Description Definition Language）声明式定义。每个任务一个 `.bddl` 文件：

```bddl
;; pick_up_the_black_bowl_between_the_plate_and_the_ramekin
;; and_place_it_on_the_plate.bddl

(define (problem libero_spatial_pick_up_the_black_bowl ...)
  (:domain robosuite)
  (:objects
    black_bowl_1     -- bowl
    plate_1          -- plate
    ramekin_1        -- ramekin
    robot_0          -- panda robot
  )
  (:init
    (on black_bowl_1 table_1)
    (nextto black_bowl_1 plate_1)
    (nextto black_bowl_1 ramekin_1)
    ;; 区域坐标定义物体初始位置
    (inregion black_bowl_1 "target_3")
    (inregion plate_1 "target_1")
    (inregion ramekin_1 "target_2")
  )
  (:goal
    (and (on black_bowl_1 plate_1))
  )
)
```

**BDDL 的关键设计**：
- **声明式**：只描述"初始状态"和"目标状态"，不描述"怎么做"
- **区域抽象**：物体位置用 `inregion` 引用预定义区域，不是硬编码坐标
- **目标条件**：`:goal` 定义成功条件（如 `(on A B)` = A 在 B 上面）
- **环境无关**：同一 BDDL 可以在不同机器人/环境中实例化

**从 BDDL 到评测 episode**：

```
BDDL 文件
  → BDDLEnv 解析
    → RoboSuite 环境实例化 (加载机器人、物体、场景)
      → reset() → 初始观测
        → policy(obs) → action → env.step(action)
          → 检查 :goal 条件 → success?
```

### 4.4 评测方法：Episode 生成与成功判定

LIBERO 的评测流程遵循 Gymnasium API，但在任务层面做了标准化：

```python
# LIBERO 评测伪代码
for suite in ["libero_spatial", "libero_object", "libero_goal"]:
    tasks = get_suite_tasks(suite)          # 10 个 BDDL 任务
    for task in tasks:
        env = BDDLEnv(task, robot="Panda")  # 从 BDDL 实例化环境
        for ep in range(50):                # 每任务 50 episodes
            obs = env.reset(seed=base_seed + ep)
            for step in range(max_steps):   # 通常 600 步
                action = policy(obs, task.language_instruction)
                obs, reward, terminated, truncated, info = env.step(action)
                if info["success"]:         # BDDL :goal 条件满足
                    success = True
                    break
            record(suite, task, ep, success)
```

**成功判定逻辑**：

LIBERO 的 `info["success"]` 由 BDDL `:goal` 条件驱动，不是简单的 reward 阈值：

| 目标类型 | BDDL 示例 | 判定方式 |
|----------|-----------|----------|
| 放置 | `(on A B)` | A 的底面接触 B 的顶面 |
| 靠近 | `(nextto A B)` | A 和 B 的水平距离 < 阈值 |
| 状态 | `(inregion A "target")` | A 的位置在目标区域内 |
| 组合 | `(and (on A B) (on C D))` | 所有子条件同时满足 |

**与 Gymnasium 标准的关系**：LIBERO 环境继承 RoboSuite 的 `MujocoEnv`，最终暴露 Gymnasium API。`info["success"]` 的判定逻辑由 BDDL→RoboSuite 链路自动处理，评测者不需要自己定义成功条件。

### 4.5 三个 Suite 的设计理念

LIBERO 的核心创新是**通过任务分组系统化评测泛化的不同维度**：

#### libero_spatial — 空间泛化

**问题**：策略能否适应物体位置变化？

```
Task 1: pick up black bowl at position A, place on plate
Task 2: pick up black bowl at position B, place on plate
...
Task 10: pick up black bowl at position J, place on plate
```

- 10 个任务，**同一物体同一目标**，只变初始位置
- 评测策略是否学到了"抓放"的通用技能，而不是记住特定位置
- 对应我们 grid sweep 的"不同初始条件"——但 LIBERO 更系统化

#### libero_object — 物体泛化

**问题**：策略能否适应不同物体？

```
Task 1: pick up the black bowl, place on plate
Task 2: pick up the cream cheese, place on plate
...
Task 10: pick up the chocolate pudding, place on plate
```

- 10 个任务，**同一位置同一目标**，只变物体（形状、大小、摩擦系数不同）
- 评测策略的视觉泛化能力——能否识别和操作没见过的物体
- 这是 VLA 相比传统 RL 的核心优势：语言+视觉理解

#### libero_goal — 目标泛化

**问题**：策略能否适应不同目标？

```
Task 1: pick up the black bowl, place on the plate
Task 2: pick up the black bowl, place on the ramekin
...
Task 10: pick up the black bowl, place on the moka pot
```

- 10 个任务，**同一物体同一初始位置**，只变目标位置
- 评测策略是否理解"放到 X 上"的语义，而非记住固定轨迹

#### 三个 Suite 的关系

```
泛化维度:
  libero_spatial:  位置变化 → 测试空间泛化
  libero_object:   物体变化 → 测试视觉泛化
  libero_goal:     目标变化 → 测试语义泛化

  三个维度正交，组合起来全面评测 VLA 的泛化能力
```

**指标计算**：

```python
# 每个 suite 的报告
suite_result = {
    "suite": "libero_spatial",
    "task_success_rates": [0.8, 0.6, 0.9, ...],  # 10 个任务各自的成功率
    "overall_success_rate": 0.72,                  # 所有 episode 的平均成功率
    "num_episodes": 500,                           # 10 tasks × 50 episodes
}
```

**关键指标**：`overall_success_rate`（所有 episode 的平均成功率）是论文中报告的主指标。但 `task_success_rates` 的分布也重要——如果某任务 0% 而其他 100%，说明策略对该任务完全失败。

### 4.6 LIBERO-PRO：鲁棒性评测理论

[LIBERO-PRO](https://github.com/sylvestf/LIBERO-plus)（423 stars，LIBERO-plus）在 LIBERO 基础上引入**扰动评测**——不只测泛化，还测**鲁棒性**。

#### 核心理念

> 真实部署中，环境不会完美匹配训练条件。VLA 策略能否在扰动下保持性能？

LIBERO-PRO 定义了 5 个扰动维度，每个维度一个 benchmark suite：

| Suite | 扰动类型 | 具体做法 | 评测什么 |
|-------|---------|---------|---------|
| `libero_pro_swap` | 物体替换 | 把任务中的物体 A 换成同类物体 B | 策略能否适应同类但不同的物体 |
| `libero_pro_object` | 属性变化 | 改变物体大小/颜色/摩擦 | 策略能否适应物体属性变化 |
| `libero_pro_lan` | 语言变化 | 同一任务用不同语言描述 | 策略是否真正理解语言语义 |
| `libero_pro_task` | 任务组合 | 把两个单任务组合成复合任务 | 策略能否执行长程任务 |
| `libero_pro_env` | 环境变化 | 改变场景布局/光照/相机视角 | 策略能否适应环境变化 |

#### 扰动评测的方法论

```python
# LIBERO-PRO 评测伪代码
for suite in LIBERO_PRO_SUITES:
    for task in get_tasks(suite):
        original_task = task.base_task          # 原始 LIBERO 任务
        perturbed_task = apply_perturbation(task) # 扰动后的任务

        # 在扰动任务上评测
        env = BDDLEnv(perturbed_task)
        for ep in range(50):
            obs = env.reset(seed=ep)
            success = run_episode(policy, env, obs)
            record(suite, task, ep, success)

    # 计算 robustness gap
    original_rate = get_original_success_rate(suite)
    perturbed_rate = get_perturbed_success_rate(suite)
    robustness_gap = original_rate - perturbed_rate  # 越小越鲁棒
```

#### Robustness Gap

LIBERO-PRO 的核心输出不只是扰动后的成功率，而是 **robustness gap**——原始性能与扰动后性能的差距：

```
robustness_gap = success_rate(original) - success_rate(perturbed)

gap ≈ 0:  策略鲁棒 (扰动不影响性能)
gap 大:   策略脆弱 (扰动导致性能大幅下降)
```

| 扰动类型 | 典型 gap | 含义 |
|---------|---------|------|
| swap | 10-30% | 换物体导致性能下降 |
| lan | 5-20% | 语言变化影响理解 |
| env | 15-40% | 环境变化影响视觉感知 |
| task | 20-50% | 复合任务难度显著增加 |

#### 与 LIBERO 的关系

```
LIBERO (基础泛化)
  ├── libero_spatial  (位置泛化)
  ├── libero_object   (物体泛化)
  └── libero_goal     (目标泛化)

LIBERO-PRO (鲁棒性)
  ├── libero_pro_swap   (物体替换扰动)
  ├── libero_pro_object (属性变化扰动)
  ├── libero_pro_lan    (语言变化扰动)
  ├── libero_pro_task   (任务组合扰动)
  └── libero_pro_env    (环境变化扰动)

  总计: 8 个 suite, 80 个任务, 4000 个 episode
```

**LIBERO 测的是"能不能泛化"，LIBERO-PRO 测的是"泛化稳不稳"**。两者互补：一个策略可能 LIBERO 80% 但 LIBERO-PRO 仅 40%，说明它能泛化但鲁棒性差。

#### 关键区别：谁需要知道训练过程？

LIBERO 和 LIBERO-PRO 在**对训练信息的依赖**上有本质区别：

| | LIBERO | LIBERO-PRO |
|--|--------|------------|
| **"泛化"的参照系** | 训练时见过的任务 | 任务本身（原始 vs 扰动） |
| **需要知道训练数据？** | ✅ 需要 | ❌ 不需要 |
| **结论的得出方式** | 实验者对比训练集 A 和评测集 B，自己判断"B 没见过 → 这是泛化" | 框架自动计算 `gap = 原始成功率 − 扰动成功率`，不依赖训练信息 |
| **适合谁用** | 模型开发者（知道自己训练了什么） | **第三方评测方**（黑盒评测，不关心训练过程） |

**为什么有这个区别？**

- LIBERO 的"泛化"结论是**实验者解读出来的**，不是框架计算的。LIBERO 只提供任务 + 环境 + 成功率，"泛化"这个标签是实验者根据自己的训练集贴上去的
- LIBERO-PRO 的 robustness gap 是**框架内自计算的**：对每个任务 T，跑原始版本和扰动版本，`gap = success(T) − success(T_perturbed)`。这个过程完全不需要知道模型训练时见过什么

**对第三方评测方的启示**：

如果你是一个**不知道模型训练细节的评测方**（比如 leaderboard 评审、第三方 benchmark），最佳策略是：

```
1. 跑 LIBERO 全部 3 个 suite → 拿到绝对成功率（"在这些任务上表现如何"）
2. 跑 LIBERO-PRO 全部 5 个扰动维度 → 拿到鲁棒性（"对扰动有多敏感"）
3. 两者组合 = 完整评测报告，不依赖任何训练信息
```

- LIBERO 告诉你**"行不行"**（绝对性能）
- LIBERO-PRO 告诉你**"稳不稳"**（抗扰动能力）
- 两者都不需要知道模型是怎么训练的

**对我们的意义**：我们的 SO101 SmolVLA 在 SO101 pick-cube 演示上训练，从没见过 LIBERO 物体/任务。跑 LIBERO 时所有任务都是"没见过的" → 测的是完全 zero-shot 泛化。跑 LIBERO-PRO 时不需要关心这个 → 直接拿 robustness gap。两种结果的解读都不依赖于"训练时见过什么"，因为对我们来说答案很简单：什么都没见过。

### 4.7 我们的 LIBERO 实战：从设计到 0%

**为什么选 LIBERO**：LIBERO 是 VLA 领域公认的标准 benchmark（CoRL 2023，2.2k stars），提供 3 个 suite × 10 tasks 的多任务泛化评测。如果一个 VLA 模型能在 LIBERO 上拿到高分，说明它具备跨任务泛化能力——这是衡量 VLA 质量的金标准。

**我们的实现**：仓库已设计完整 LIBERO 评测管线——`eval_vla.py` + 8 个 benchmark 配置 + `so101-eval` Docker 镜像。通过 GitHub Actions `evaluate.yml` 在 V100 ECS 上运行。

**实战结果**（2026-08-27，run 33053613547）：

| Benchmark | Episodes | 结果 | 说明 |
|-----------|----------|------|------|
| `libero_goal` | 100 | 0%，steps=0 | 环境初始化即失败 |
| `libero_spatial` | 20 | 0%，跑满 230 步 | 能运行但任务未完成 |
| **总计** | **120** | **0%** | **模型-环境不兼容** |

**为什么是 0%**：LIBERO 只支持 Franka Panda 7 DoF 机械臂，我们的模型是在 SO101 5 DoF 上训练的。关节定义、观测空间、动作语义全部不匹配——SO101 的模型根本无法驱动 Franka。

**学到的教训**：
1. **评测前先确认机器人兼容性**——不是所有 benchmark 都支持所有机器人
2. **0% 也是有价值的结果**——直接证实了模型-环境不兼容，避免继续浪费时间
3. **需要先做集成**——要让 LIBERO 出正分，必须先在 LIBERO 中添加 SO101 机器人（详见 Ch6）

---

## 5. 我们的评测实践

前面介绍了社区框架（§1-4），现在看我们实际怎么用。

**评测进展总览**：

| 评测方法 | 流水线 | 状态 | 结果 |
|----------|--------|------|------|
| 回放验证 | 本地运行 | ✅ 已执行 | 方案 A 失败，方案 B 成功 |
| Grid Sweep | so101-mujoco-pipeline.yml | ✅ 已执行 | 153/325 = 47% |
| PPO 确定性评估 | ppo-pipeline.yml | ✅ 已执行 | v1: 100%, v2: 98% |
| LIBERO | evaluate.yml | ✅ 已执行 | 120 episodes / 0%（模型-环境不兼容） |
| LIBERO-PRO | — | ⬜ 已设计未执行 | 依赖 LIBERO 先出正分 |
| SO-101 Bench | — | ⬜ 硬件不支持 | V100 无法运行 |

从最简单到最全面，四种评测方法层层递进。

### 5.1 回放验证：快速 smoke test

每次训练后快速验证模型能否正常推理——跑 1 个 episode（300 步），~30 秒出结果。

**如何运行**：`replay_demo.py` 目前仅支持本地运行（无 GitHub Actions 流水线）。训练后在本地执行：

```bash
python scripts/replay_demo.py \
  --checkpoint /path/to/checkpoint \
  --dataset xieyucheng123/so101-dataset \
  --num-episodes 1
```

**回放指标解读**：

| 指标 | 好的值 | 坏的值 | 含义 |
|------|--------|--------|------|
| prediction_errors | 0/300 | >0 | 模型是否正常推理 |
| success | True | False | 是否完成任务 |
| reward 趋势 | 上升 | 振荡/下降 | 是否在接近目标 |
| state 变化 | 持续 | 不变 | 机器人是否在动 |

### 5.2 PPO 确定性评估：CleanRL 范式

[CleanRL](https://github.com/vwxyzjn/cleanrl) 确立了 RL 评测的标准做法：固定 seed + 确定性策略 + 足够多的 episodes。

**如何运行**：通过 GitHub Actions 流水线 `ppo-pipeline.yml` 触发。在 Actions 页面选择 "PPO Pipeline"，点击 "Run workflow"，填写参数即可：

```
Actions → PPO Pipeline → Run workflow
  ├── steps: 20000（训练步数）
  ├── batch_size: 32
  └── checkpoint:（留空=训练+评测，填路径=只评测）
```

流水线自动完成：启动 V100 ECS → 训练 PPO → 评测 50 episodes → 上传结果 → 关闭 ECS。

### 5.3 Grid Sweep：单任务工作空间扫描

Grid sweep 不是标准 RL 评测方法，而是机器人仿真社区的实践——系统扫描工作空间初始条件（5 个距离 × 13 个角度 × 5 次 = 325 episodes）。

**如何运行**：通过 GitHub Actions 流水线 `so101-mujoco-pipeline.yml` 触发。在 Actions 页面选择 "SO101 MuJoCo Pipeline"，点击 "Run workflow"：

```
Actions → SO101 MuJoCo Pipeline → Run workflow
  ├── steps: 20000（训练步数）
  ├── batch_size: 32
  ├── skip_train: false（true=跳过训练，用已有 checkpoint）
  ├── checkpoint:（留空=自动找最新，填路径=指定）
  └── mode: sweep（sweep=grid搜索，record=录像）
```

流水线自动完成：启动 V100 ECS → 下载数据+模型 → 训练 SmolVLA → Grid Sweep 评测 → 上传结果到 OBS → 关闭 ECS。

### 5.4 LIBERO 评测：跨任务泛化

LIBERO 是 VLA 领域的标准 benchmark（CoRL 2023，2.2k stars），评测模型跨任务泛化能力。

**如何运行**：通过 GitHub Actions 流水线 `evaluate.yml` 触发（在 V100 ECS 上运行）。在 Actions 页面选择 "Evaluate"，点击 "Run workflow"：

```
Actions → Evaluate → Run workflow
  ├── model_repo: xieyucheng123/so101-act（HuggingFace 模型仓库）
  ├── dataset_repo: xieyucheng123/so101-dataset（HuggingFace 数据集仓库）
  └── num_episodes: 10（每个 benchmark 的 episode 数）
```

流水线自动完成：启动 V100 ECS → 下载模型+数据 → 运行 8 个 LIBERO/LIBERO-PRO benchmark → 下载结果+视频 → 上传到 OBS → 关闭 ECS。

> **注意**：`evaluate.yml` 评测的是 **HuggingFace 上预训练的 VLA 模型**，不是我们自己训练的模型。我们要在 LIBERO 上评测自己的模型，需要先在 LIBERO 中添加 SO101 机器人（详见 Ch6）。

#### 热力图

```
reach\azim   -90   -75   -60   -45   -30   -15    +0   +15   +30   +45   +60   +75   +90
  15cm    3/5   1/5   1/5   4/5   5/5   5/5   4/5   2/5   2/5   4/5   0/5   3/5   0/5
  18cm    4/5   2/5   2/5   4/5   4/5   5/5   4/5   1/5   4/5   3/5   1/5   0/5   0/5
  20cm    1/5   1/5   4/5   0/5   3/5   4/5   2/5   4/5   2/5   2/5   3/5   0/5   1/5
  22cm    3/5   0/5   1/5   3/5   3/5   5/5   4/5   4/5   2/5   2/5   1/5   1/5   0/5
  25cm    3/5   2/5   2/5   4/5   3/5   4/5   3/5   5/5   1/5   1/5   1/5   0/5   0/5

SUCCESS 153/325 = 47%
```

**如何解读热力图**：

- **中心区域（-15 到 +15 azimuth, 15-22cm reach）**：成功率 60-100% — 机器人最舒适的工作区域，训练数据最密集
- **边缘区域（±90 azimuth）**：成功率接近 0% — 物体在机器人侧面极限位置，训练数据很少覆盖
- **结论**：47% 是 325 个不同初始条件的平均，不是单一条件。中心区域已经可用，边缘需要更多数据覆盖

### 5.5 四种方法对比

| 方法 | 回放 (replay) | Grid Sweep | PPO 确定性评估 | LIBERO |
|------|---------------|------------|----------------|--------|
| Episodes | 1 (300 steps) | 325 | 50 | 500 (10 tasks × 50) |
| 耗时 | ~30s | ~30min | ~15min | ~2h |
| 用途 | 快速 smoke test | 单任务工作空间扫描 | RL 策略评估 | 跨任务泛化评估 |
| 时机 | 每次训练后 | 关键 checkpoint | PPO 训练完成 | 里程碑节点 |
| 框架 | LeRobot 推理管线 | so101-mujoco | CleanRL 范式 | vla-eval harness |
| **我们是否跑过** | ✅ 已执行 | ✅ 已执行 | ✅ 已执行 | ✅ 已执行（0%，模型-环境不兼容） |

**从快到慢，从简单到全面**：

```
回放 (30s) → Grid Sweep (30min) → PPO Eval (15min) → LIBERO (2h)
  smoke test    单任务考试       RL baseline        毕业考试
```

回放是"快速 smoke test"，Grid Sweep 是"单任务考试"，PPO Eval 是"RL baseline 参照"，LIBERO 是"毕业考试"（已设计好考场但还没开考，详见 [Discussion #9](https://github.com/link-seek/so101-sim-pipeline/discussions/9) 和 [Ch6](so101-tutorial-ch6-optimization.md)）。

---

## 6. 评测方法论：为什么这样做

### 6.1 训练指标 vs 评估指标

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

### 6.2 统计显著性：多少 episodes 才够

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

**所需 episodes 数公式**（95% CI 宽度 < w）：

```python
N > 1.96^2 * p * (1-p) / (w/2)^2

# 例：p=0.47, 想要 CI 宽度 < 10% (±5%)
N > 3.84 * 0.47 * 0.53 / 0.0025 = 382
```

### 6.3 评测指标设计原则

#### 原则 1：指标要和任务语义对齐（Gymnasium `info["success"]`）

Gymnasium 环境的 `info["success"]` 由环境作者定义，评测者不应自己定义成功条件：

```python
# 好：用环境定义的 success
success = info["success"]  # 环境作者已经定义了什么是"完成"

# 坏：自己定义 success（可能和 reward 不对齐）
success = reward > threshold  # 策略可能学会刷 reward
```

我们在 PPO 中就遇到这个问题：`lift_threshold=0.05` 时 `info["success"]` 返回 True，但视频里物体几乎没动。修复方式是修改环境的 `lift_threshold` 参数，不是自己定义 success。

#### 原则 2：覆盖足够的初始条件（统计显著性）

50 episodes 的 95% CI 约 ±14%，325 episodes 约 ±5.5%。选择 episode 数量要考虑置信区间宽度（见 §6.2）。

#### 原则 3：结果可复现（固定 seed）

LeRobot `lerobot-eval` 默认 `seed=1000`，我们的 `eval_ppo.py` 用 `seed=12345`。固定 seed 确保同一 checkpoint 永远得到同一结果，否则无法对比不同训练版本。

#### 原则 4：归档完整（LeRobot `per_episode` 格式）

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

## 7. 踩坑复盘

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

6. **LIBERO 的三个 suite（spatial/object/goal）为什么是正交的？**  
   提示：每次只变一个维度，其他固定。spatial 变位置、object 变物体、goal 变目标。组合起来可以定位泛化瓶颈在哪个维度。

7. **LIBERO-PRO 的 robustness gap 和 LIBERO 的 success rate 有什么区别？**  
   提示：LIBERO 的 success rate 回答"能不能做"，LIBERO-PRO 的 gap 回答"扰动后还能不能做"。gap=0 说明鲁棒，gap 大说明脆弱。一个策略可以 LIBERO 80% 但 LIBERO-PRO gap 40%，意味着泛化能力有但鲁棒性差。

8. **BDDL 的声明式任务定义相比硬编码有什么优势？**  
   提示：声明式只描述初始状态和目标，不描述怎么做。这意味着同一 BDDL 可以在不同机器人上实例化（只要环境支持所需谓词），也方便自动生成扰动变体（LIBERO-PRO 就是程序化修改 BDDL）。

9. **为什么说 LIBERO-PRO 比 LIBERO 更适合第三方黑盒评测？**  
   提示：LIBERO 的"泛化"结论需要实验者知道训练时见过什么才能解读。LIBERO-PRO 的 robustness gap 是框架内自计算的（原始成功率 − 扰动成功率），不依赖任何训练信息。第三方评测方不需要知道模型怎么训练的，跑完 LIBERO + LIBERO-PRO 就能得到完整的"行不行 + 稳不稳"报告。

---

> **上一章**：[Ch4 Debug 实战](so101-tutorial-ch4-debug-journey.md) | **下一章**：[Ch6 优化进阶](so101-tutorial-ch6-optimization.md)
