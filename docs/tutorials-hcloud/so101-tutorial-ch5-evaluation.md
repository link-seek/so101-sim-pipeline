# Ch5：仿真评测方法论

> SO101 仿真评测教程 · 第五章
> 参考框架：[Gymnasium](https://gymnasium.farama.org/) · [LeRobot `lerobot-eval`](https://github.com/huggingface/lerobot) · [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) · [LIBERO-PRO（harness 源码定义）](https://github.com/allenai/vla-evaluation-harness/blob/v0.4.0/src/vla_eval/benchmarks/libero_pro/benchmark.py)

---

## 1. 开源评测框架全景

上一章我们用了两种评测：`replay_demo.py`（回放验证）和 `eval_mujoco_policy.py`（grid sweep）。但这两个脚本不是凭空发明的——它们背后有一套成熟的开源评测框架体系。

本章就来搞清楚：**社区是怎么评测机器人策略的，我们的脚本和这些框架是什么关系。**

新手流程地图（一句话版）：回放（smoke）→ Grid Sweep（单任务考试）→ PPO 确定性评估（RL 参照）→ LIBERO（毕业考试）。四种方法的何时用、谁执行，见 §5.5 对照表；操作细节是 Ch1–Ch4 的内容，本章不重复——下面只讲框架关系与专业测评。

### 各框架对比

| 框架 | 类型 | 用途 | 核心指标 | 我们的使用方式 | 一句话感受 |
|------|------|------|----------|---------------|-------------|
| **Gymnasium** | API 标准 | 环境接口 | `info["success"]`, `reward` | 所有 eval 脚本的底层 API | 契约本身没坑，坑全在环境作者定的 success 对不对——PPO v1 的 100% 假成功就是例子 |
| **LeRobot lerobot-eval** | 评测方法 | VLA 通用评测 | `pc_success`, `avg_sum_reward` | `replay_demo.py` 用其推理管线 | 推理管线开箱即用；指标名和我们的对得上，seed 默认 1000 照抄就行 |
| **LIBERO** | 评测方法 | VLA benchmark | task success rate × 10 tasks | `eval_vla.py` 通过 vla-eval harness | 金标准但只认 Franka——跨身体直接 0%，跑之前先确认机器人，见 §4.7 |
| **LIBERO-PRO** | 评测方法 | VLA 鲁棒性 | robustness gap | `eval_vla.py` 的 libero_pro_*（5 种扰动，定义见 §4.6） | 非官方改题卷，定义以 AllenAI 源码为准，详见 §4.6 |
| **CleanRL** | 评测方法 | RL 评估范式 | `success_rate`, `ep_return` | `eval_ppo.py` 的确定性评估 | 范式最省心：固定 seed + 确定性策略 + 50eps，PPO 稳到不用算 CI |
| **Grid Sweep** | 评测方法 | 多初始条件评测 | success rate across grid | `eval_mujoco_policy.py` 实现 | 不是标准框架，是社区土办法——但只有它照出了覆盖盲区（边缘 ~0%），最爱的一张热力图 |
| **so101_nexus** | 仿真环境 | MuJoCo 仿真 | — | Ch4 回放验证 | 回放够用，但和 ataghof 采集环境不是一回事——Ch4 全章就是为这句话买的单 |
| **so101-mujoco** | 仿真环境 | MuJoCo sim twin | — | `eval_mujoco_policy.py` 的环境 | 救命的环境：训测同场，47% 从这来；三个坑（shm/XET/独占）见 Ch1 §4.3 |

---

## 2. Gymnasium：评测的通用语言

### 2.1 唯一的契约：`info["success"]`

所有评测脚本都基于 [Gymnasium](https://gymnasium.farama.org/) 的标准 API（`reset(seed)` / `step()` 五元组）。新手只需记住一条：**成功与否由环境的 `info["success"]` 说了算，评测者不自造标准**——自造就会刷出 PPO v1 那种 100% 假成功（见 §6.3 原则 1）。Ch2/Ch4 的脚本都是这个契约的实例，细节查文档。

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

### 3.2 配置与指标（一句话版）

`EvalPipelineConfig` = 环境 + 评测（`n_episodes`，默认 50）+ 策略 + 全局 seed（默认 1000）+ `rename_map`（Ch4 修过的那个）。输出分两层：`per_episode`（每回合 success/seed/reward）+ `aggregated`（成功率、平均回报、耗时）。

| 指标 | LeRobot 名称 | 我们 `eval_ppo.py` 对应 |
|------|-------------|----------------------|
| 成功率 | `pc_success` | `success_rate` |
| 平均累积回报 | `avg_sum_reward` | `avg_reward` |
| 每 episode 耗时 | `eval_ep_s` | `elapsed_s / num_episodes` |

**我们的 `eval_ppo.py` 遵循了同样的指标设计**，只是命名不同。`avg_max_reward` 我们没追踪，因为 PPO 的 reward 语义和 VLA 不同。

### 3.3 我们的脚本如何使用社区框架

| 脚本 | 遵循的框架 | 推理管线 | Docker 镜像 |
|------|-----------|---------|------------|
| `replay_demo.py` | LeRobot | `prepare_observation_for_inference` → `preprocess` → `select_action` → `postprocess` | so101-train |
| `eval_ppo.py` | CleanRL | `agent.actor_mean(norm(obs))` 确定性评估 | so101-ppo |
| `eval_vla.py` | vla-eval harness | `run_benchmark()` → `merge_results()` | so101-eval |
| `eval_mujoco_policy.py` | so101-mujoco | `--sweep` grid search | so101-mujoco |

---

## 4. LIBERO：VLA 的标准 Benchmark

先看关系图——官方只出了一套卷，另外两套都是第三方出的：

```
官方考纲 + 真题卷                        第三方模拟卷（都非官方）
┌─────────────────────┐    改题    ┌─────────────────────────┐
│ LIBERO（原版）       │ ─────────→ │ PRO（AllenAI，随        │
│ 考纲 + BDDL 真题     │            │ harness 附赠）           │
│ 130 tasks, Franka    │            │ 5 种改法，gap 判稳       │
└─────────────────────┘            └─────────────────────────┘
        │ 另出卷                              │ 数据随镜像发布
        ↓                                     ↓
┌─────────────────────┐            ┌─────────────────────────┐
│ Plus（Sylvest）      │            │ 我们的 harness           │
│ 照考纲另出一套全真卷  │            │ vla-eval==0.4.0          │
│ 7 维 10030 题        │            │ 跑 LIBERO 真题 +          │
│ 论文 2510.13626     │            │ PRO 改题（见§4.6）       │
└─────────────────────┘            └─────────────────────────┘
```

**读图路线**：§4.1–§4.5 讲左上真题卷长什么样（BDDL、判定、三个 suite）；§4.6 讲右上改题卷（5 种改法 + gap）。跑分的事 §4.7 和 §5 再讲，这里只讲关系不讲分。

### 4.1 LIBERO 是什么：官方真题卷（左上那盒）

这是官方出的唯一一套卷，维护者是 [Lifelong-Robot-Learning 团队](https://github.com/Lifelong-Robot-Learning/LIBERO)（Bo Liu 等，CoRL 2023，2.2k stars）。全称 **Lifelong Benchmark for Robotics**，核心定位：

> 给定一组操作任务，系统化地评测 VLA 策略的**泛化能力**——不是"能不能做一个任务"，而是"能不能做一类任务"。

LIBERO 建立在 [RoboSuite](https://github.com/ARISE-Initiative/robosuite) 仿真框架上，使用 Franka Panda 7 DoF 机械臂，定义了 5 个 benchmark suite，总计 130 个任务。

### 4.2 评测对象

LIBERO 评测的是 **VLA 策略**：输入图像 + 语言指令，输出机器人动作。我们的 SmolVLA 也是 VLA，理论上可测——实际卡在机器人不匹配（详见 §4.7）：LIBERO 只认 Franka，我们的模型是 SO101 的身体。

### 4.3 任务定义机制：BDDL

LIBERO 的任务用 **BDDL**（Behavior Description Definition Language）声明式定义。每个任务一个 `.bddl` 文件——只看骨架：

```bddl
(define (problem libero_spatial_pick_up_the_black_bowl ...)
  (:objects
    black_bowl_1  -- bowl
    plate_1       -- plate
    robot_0       -- panda robot)
  (:init
    (on black_bowl_1 table_1)
    (inregion black_bowl_1 "target_3"))  ;; 位置用区域抽象，不是硬编码坐标
  (:goal
    (and (on black_bowl_1 plate_1))))    ;; 成功条件：碗在盘子上
```

**BDDL 的关键设计**：
- **声明式**：只描述"初始状态"和"目标状态"，不描述"怎么做"——同一 BDDL 可在不同机器人上实例化，也方便程序化生成扰动变体（PRO 就是这么来的）
- **成功条件即判定**：`:goal` 就是评测时的 `info["success"]` 来源，评测者不自造标准（回扣 §2.1 的契约）

### 4.4 评测方法：Episode 生成与成功判定

流程就是 §4.3 骨架跑起来：每个 suite 10 个 BDDL 任务，每任务 N 个 episode（`reset(seed)` → policy 推理 → `step` → 查 `:goal`）。成功判定由 BDDL `:goal` 驱动，不是 reward 阈值：

| 目标类型 | BDDL 示例 | 判定方式 |
|----------|-----------|----------|
| 放置 | `(on A B)` | A 的底面接触 B 的顶面 |
| 靠近 | `(nextto A B)` | A 和 B 的水平距离 < 阈值 |
| 状态 | `(inregion A "target")` | A 的位置在目标区域内 |
| 组合 | `(and (on A B) (on C D))` | 所有子条件同时满足 |

链路 BDDL→RoboSuite→Gymnasium 全自动，评测者不定义成功条件——还是 §2.1 那条契约。

### 4.5 三个 Suite 的设计理念

LIBERO 的核心创新是**通过任务分组系统化评测泛化的不同维度**——10 个任务里只变一个维度，其他固定：

| Suite | 只变什么 | 例子 | 测什么 |
|-------|---------|------|--------|
| `libero_spatial` | 初始位置 | 同一个黑碗放 10 个不同位置 | 空间泛化：学的是"抓放"还是记住了位置 |
| `libero_object` | 物体 | 同一位置抓 10 种不同物体 | 视觉泛化：能否操作没见过的物体 |
| `libero_goal` | 目标位置 | 同一黑碗放 10 个不同目标上 | 语义泛化：懂不懂"放到 X 上" |

三个维度正交，组合起来定位泛化瓶颈。主指标 `overall_success_rate`（所有 episode 平均成功率）；`task_success_rates` 分布也看——某任务 0% 而其他 100% 说明该任务完全失败。

### 4.6 LIBERO-PRO：非官方改题卷（右上那盒）

官方只有 LIBERO 一家——PRO 是 AllenAI 在自家 harness（[allenai/vla-evaluation-harness](https://github.com/allenai/vla-evaluation-harness)）里配的扩展，定义以该仓库源码为准。我们镜像装的是 `vla-eval==0.4.0`，它的 `LIBEROProBenchmark`（`src/vla_eval/benchmarks/libero_pro/benchmark.py`）就是"拿真题改题"的 5 种改法（套件命名 `{base}_{perturbation}`，如 `libero_spatial_swap`，改完需预生成 BDDL + init-state（这批文件随 AllenAI 的 `libero-pro` 镜像发布）：**swap**=物体位置交换、**object**=换成新物体、**lan**=指令同义改写（语义不变）、**task**=目标重设计（成功条件变了）、**env**=整个环境替换。一句话：**LIBERO 测"能不能泛化"，PRO 测"泛化稳不稳"**——核心输出 robustness gap = 原始成功率 − 扰动成功率，gap 越小越鲁棒。

> **别混淆**：左下角另有一套全真卷 [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus)，同样非官方——维护者是 Sylvest 团队（7 个扰动维度、10030 tasks，论文 2510.13626），harness 里是独立的 `libero_plus` benchmark。下文 PRO 均指右上角的 5 种改法。

PRO 的 gap 是框架内自计算的，不依赖训练信息——第三方黑盒评测（不知道模型怎么训的）直接拿 gap 即可，这是它相对 LIBERO 最大的方法论拓展。

### 4.7 交卷：我们的四张答卷

```
Franka 原生 47/100（Ch6，管线通）｜跨身体 0/120（本节，身体不通）｜SO101 集成后 0/300（Ch8，模型不行）｜PRO 改题（还没跑，安排见 §5）
```

四张里本节讲第二张——SO101 模型跑 Franka 考场。

**为什么选 LIBERO**：LIBERO 是 VLA 领域公认的标准 benchmark（CoRL 2023，2.2k stars），提供 3 个 suite × 10 tasks 的多任务泛化评测。如果一个 VLA 模型能在 LIBERO 上拿到高分，说明它具备跨任务泛化能力——这是衡量 VLA 质量的金标准。

**我们的实现**：仓库已设计完整 LIBERO 评测管线——`eval_vla.py` + 8 个 benchmark 配置 + `so101-eval` Docker 镜像。模型和数据集存 OBS，计算在华为云 V100 ECS 上完成，用 `docker run` 一条命令即可运行。

**实战结果**（2026-08-27，run 33053613547）：

| Benchmark | Episodes | 结果 | 说明 |
|-----------|----------|------|------|
| `libero_goal` | 100 | 0%，steps=0 | 环境初始化即失败 |
| `libero_spatial` | 20 | 0%，跑满 230 步 | 能运行但任务未完成 |
| **总计** | **120** | **0%** | **模型-环境不兼容** |

**为什么是 0%**：LIBERO 只支持 Franka Panda 7 DoF 机械臂，我们的模型是在 SO101 6 DoF（5 arm + 1 gripper）上训练的。关节定义、观测空间、动作语义全部不匹配——SO101 的模型根本无法驱动 Franka。

**学到的教训**：
1. **评测前先确认机器人兼容性**——不是所有 benchmark 都支持所有机器人
2. **0% 也是有价值的结果**——直接证实了模型-环境不兼容，避免继续浪费时间
3. **需要先做集成**——要让 LIBERO 出正分，必须先在 LIBERO 中添加 SO101 机器人（已在 #9 完成，详见 Ch8；集成后 300eps 仍 0%，定性为模型视觉域问题而非集成 bug）

> **两把尺子**：本节是跨身体线的 0%（SO101 模型跑 Franka 环境）；同套 LIBERO 在 Franka 原生模型上的正分 **47/100** 见 Ch6 §4.3。两节对照读：管线是通的（Ch6），跨身体是不通的（本节）。

---

## 5. 我们的评测实践

前面是社区框架速览（§1-4，新手建立流程观）；下面是我们的实战记录。单源原则：操作命令以 Ch1–Ch4 首发章节为准，本节只保留结果与解读。

**评测进展总览**：

| 评测方法 | Docker 镜像 | 状态 | 结果 |
|----------|------------|------|------|
| 回放验证 | so101-train | ✅ 已执行 | 方案 A 失败，方案 B 成功 |
| Grid Sweep | so101-mujoco | ✅ 已执行 | 153/325 = 47% |
| PPO 确定性评估 | so101-ppo | ✅ 已执行 | v1: 100%, v2: 98% |
| LIBERO | so101-eval | ✅ 已执行 | 跨身体 120eps 0%（§4.7）；Franka 100eps 47%（Ch6） |
| LIBERO-PRO | — | ⬜ 已设计未执行 | 依赖 LIBERO 先出正分 |
| SO-101 Bench | — | ⬜ 硬件不支持 | V100 无法运行 |

从最简单到最全面，四种评测方法层层递进。

### 5.1 回放验证：快速 smoke test

每次训练后快速验证模型能否正常推理——跑 1 个 episode（300 步），~30 秒出结果（用法见 Ch3 §7，命令以 Ch3 为准）。

**回放指标解读**：

| 指标 | 好的值 | 坏的值 | 含义 |
|------|--------|--------|------|
| prediction_errors | 0/300 | >0 | 模型是否正常推理 |
| success | True | False | 是否完成任务 |
| reward 趋势 | 上升 | 振荡/下降 | 是否在接近目标 |
| state 变化 | 持续 | 不变 | 机器人是否在动 |

### 5.2 PPO 确定性评估：CleanRL 范式

[CleanRL](https://github.com/vwxyzjn/cleanrl) 范式：固定 seed + 确定性策略 + 足够多的 episodes。标准命令见 Ch2 §4.2（v1）/ §5（v2，需 `--lift-threshold 0.15`）；结果 v1 100%、v2 98%、#20 直跑复现 100%。

### 5.3 Grid Sweep：单任务工作空间扫描

系统扫描工作空间初始条件（5 reach × 13 azim × 5 trials = 325 episodes）。标准命令见 Ch1 §4.3（`--shm-size`、XET 关闭、缓存挂载三个坑随命令走，命令以 Ch1 为准）。结果 153/325 = 47%，完整复盘见 Ch4 尾声；热力图一句话读法：中心 60–100%，边缘 ~0%——平均数掩盖覆盖盲区。

### 5.4 LIBERO 评测：跨任务泛化

VLA 标准 benchmark，测跨任务泛化。跑法见 Ch6 §3.2（Franka 100eps 出 47% 正分）；跨身体 0% 见 §4.7；SO101 集成后 300eps 见 Ch8。

### 5.5 四种方法对比

| 方法 | 回放 (replay) | Grid Sweep | PPO 确定性评估 | LIBERO |
|------|---------------|------------|----------------|--------|
| Episodes | 1 (300 steps) | 325 | 50 | 100（Franka 实测 10 tasks × 10；官方协议另有 50/task 版，见 Ch6 §5.2） |
| 耗时 | ~30s | ~30min | ~15min | ~2h |
| 用途 | 快速 smoke test | 单任务工作空间扫描 | RL 策略评估 | 跨任务泛化评估 |
| 时机 | 每次训练后 | 关键 checkpoint | PPO 训练完成 | 里程碑节点 |
| 框架 | LeRobot 推理管线 | so101-mujoco | CleanRL 范式 | vla-eval harness |
| **我们是否跑过** | ✅ 已执行 | ✅ 已执行 | ✅ 已执行 | ✅ 已执行（Ch6 47% 出正分；跨身体/SO101 线 0%，见 §4.7/#9） |

**从快到慢，从简单到全面**：

```
回放 (30s) → Grid Sweep (30min) → PPO Eval (15min) → LIBERO (2h)
  smoke test    单任务考试       RL baseline        毕业考试
```

回放是"快速 smoke test"，Grid Sweep 是"单任务考试"，PPO Eval 是"RL baseline 参照"，LIBERO 是"毕业考试"（已开考：Ch6 47% 出正分，#9 SO101 300eps 跑通但 0% 定性为模型问题，详见 [Discussion #9](https://github.com/link-seek/so101-sim-pipeline/discussions/9) 和 [Ch8](so101-tutorial-ch8-custom-robot.md)）。

### 5.6 同一把尺子：0 → 47 → 45

> 数据来源：[Discussion #20](https://github.com/link-seek/so101-sim-pipeline/discussions/20)（2026-09-05）、[#18](https://github.com/link-seek/so101-sim-pipeline/discussions/18)（2026-09-07）。本节三个数**全是同一把尺子量的**：SO101 MuJoCo Grid Sweep，5 reach × 13 azim × 5 trials = 325 episodes。§4.7 的 LIBERO 0%（Franka 身体）是另一把尺子，不参与本节对比；Franka 考场的另一个 47%（47/100）见 Ch6 §4.3 防火墙，别混。

同一个评测网格，我们先后量出三个数：

| 阶段 | Checkpoint / 数据 | 分数 | 含义 |
|------|-------------------|------|------|
| 错任务数据 | libero_object 30eps 训出的变体，15K | **0/325 = 0%** | 数据任务不对，网格全灭 |
| 对任务数据 | pick-cube 数据，15K | **153/325 = 47%** | 与历史 run（`32221378632`）逐数相同，可复现 |
| 单纯加步数 | 同模型续到 20K | **145/325 = 45%** | 掉 8 个 episode，不再涨 |

三句话解读：

1. **0 的教训**：换任务数据 ≠ 变体。pick-cube 的能力只从 pick-cube 的数据来，30eps 的 libero_object 数据在网格上拿零分，不是因为步数不够，是因为学的东西不对（#18）。
2. **47 的含义**：对任务数据 + 15K，能复现历史分数。47% 是 325 个初始条件的平均：中心区域 60–100%，边缘接近 0%（见 §5.3 一句话读法）——单一数字掩盖了覆盖盲区。
3. **45 的诚实读法**：145 vs 153 只差 8 个 episode（2.5 个百分点）。在 N=325、p≈0.46 时标准误约 ±2.8%，这个差距**落在噪声带里**。所以结论只能写“平台/不再涨”，不能写“显著倒退”。原判语“过训/噪声平台”（#18）留的余地是对的——加步数路线死，不是因为它显著变差，而是因为它**不再变好**。

### 有效 vs 无效：本节的手段对照表

| 手段 | 证据 | 结论 |
|------|------|------|
| 任务对齐的数据 | 0/325 → 153/325，全场最大单项增益 | ✅ 最有效 |
| 更新量（batch × steps） | 官方配方 20k steps @ batch 64；我们 batch 8，步数相同但样本更新量差 8 倍（dyordan1/so101-mujoco README） | ✅ 先看更新量，不只看步数 |
| 单纯加步数（15K → 20K） | 153/325 → 145/325，平台 | ❌ 无效 |
| 换任务数据（libero_object） | 0/325 | ❌ 无效 |
| 补数据量（Sawyer 100-ep 对照） | 路线已停，见下 | ⏳ 取消，不展开 |

> **本节 takeaway**：在数据欠拟合时，**加对任务的数据 > 加步数；步数只负责走到收敛，跨不过数据的天花板**。跨实验对比前先问“同一把尺子吗”（N、身体、任务分布任一不同，数字直接比就是耍流氓）。
>
> **路线已停说明**：Sawyer 50→100ep 对照（10eps/task）于 2026-09-08 取消——10eps/task 仍远低于社区配方密度（50eps/单task），预期买不到能改变结论的数据，而重训需 ~43h 卡时（实测 8s/step，数据加载瓶颈），性价比不足。结论以 50-ep + ACT 为准（详见 Ch7 §2.4）。

---

## 6. 评测方法论：踩坑驱动的原则

### 6.1 训练指标 ≠ 评估指标

这是 RL/BC 社区的共识（参见 [Spinning Up](https://spinningup.openai.com/) 的评测章节）：

| | 训练指标 | 评估指标 |
|--|---------|---------|
| **PPO** | reward 曲线, entropy | `success_rate` (独立评估环境) |
| **VLA** | MSE Loss | `pc_success` (LeRobot), `success_rate` (grid sweep) |
| **用途** | 监控收敛 | 判断任务完成能力 |
| **陷阱** | Loss 低 ≠ 性能好 (BC) | — |

> **踩坑**：ataghof 方案中 Loss 0.046 看起来"还行"，但社区成功案例是 0.005，差 10 倍。Loss 只衡量 action 预测精度，不衡量任务完成度——这是 BC 评测的核心原则。必须做仿真评估，不能只看 Loss。

### 6.2 统计显著性：多少 episodes 才够

给定成功率 p，N 个 episodes 的标准误差 `SE = sqrt(p*(1-p)/N)`：

| N | p=0.47 时 95% CI | 含义 |
|---|--------|------|
| 50 | ±14% | 太宽，两个模型差 10% 也分不出 |
| 325 | ±5.5% | 可接受——这就是 grid sweep 用 325 的原因 |
| 1000 | ±3% | 好，但 V100 跑不起 |

**我们的选择**：PPO 50eps（策略稳定，100% 或 0%，SE≈0）；VLA 325eps；LIBERO 按官方协议（10eps/task，Franka 线见 Ch6 §5.3）。记住一条：**报成功率不报 N，就是耍流氓**——§5.6 的"平台≠倒退"全靠这条才立得住。

### 6.3 评测指标设计原则

#### 原则 1：指标要和任务语义对齐（Gymnasium `info["success"]`）

Gymnasium 环境的 `info["success"]` 由环境作者定义，评测者不应自己定义成功条件：

```python
# 好：用环境定义的 success
success = info["success"]  # 环境作者已经定义了什么是"完成"

# 坏：自己定义 success（可能和 reward 不对齐）
success = reward > threshold  # 策略可能学会刷 reward
```

> **踩坑**：PPO v1 的 `lift_threshold=0.05` 太低（5cm 就算"抬起"），导致 success_rate=100% 但视频里物体几乎没动。修复：修改环境参数 `lift_threshold=0.15`，success_rate 降至 98%，但视频中有明显抬起动作。教训：评测前要理解环境的成功判定逻辑。

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

### 6.4 Grid Sweep 暴露训练数据偏置

> **踩坑**：VLA 47% 成功率，但中心区域 60-100%，边缘 ~0%。训练数据集中在工作区中心，边缘覆盖不足。Grid Sweep 能发现训练数据的覆盖盲区，单一指标（如 LIBERO 的 `pc_success`）会掩盖这个问题。两种评测方法互补。

---

## 思考题

1. **LeRobot 的 `pc_success` 和我们的 `success_rate` 有什么区别？**  
   提示：只是命名和单位不同（百分比 vs 小数），计算方式相同：`mean(successes)`。

2. **PPO 用 50eps、VLA 用 325eps、grid sweep 每格 5 trials，N 都是怎么定的？**  
   提示：同一条公式 `SE = sqrt(p(1-p)/N)`。PPO 的 p≈1（SE≈0）；VLA 的 p≈0.5（最需要 N）；每格 5 trials 是"格子内平均"，325 是"格子间平均"，两层平均别混。

3. **如果训练数据只覆盖中心区域，边缘 0% 是 bug 还是预期？**  
   提示：是预期。模型没有边缘数据，无法泛化。解决方案是补充边缘数据或数据增强。LIBERO 的 `libero_spatial` 就是为了测试这种空间泛化。

4. **LIBERO 评测和 grid sweep 评测有什么互补性？**  
   提示：LIBERO 测跨任务泛化（不同物体/目标/语言），grid sweep 测单任务工作空间覆盖。一个模型可能 LIBERO 80% 但 grid sweep 边缘 0%。

5. **LIBERO 的三个 suite（spatial/object/goal）为什么是正交的？**  
   提示：每次只变一个维度，其他固定。spatial 变位置、object 变物体、goal 变目标。组合起来可以定位泛化瓶颈在哪个维度。

6. **LIBERO-PRO 的 robustness gap 和 LIBERO 的 success rate 有什么区别？**  
   提示：LIBERO 的 success rate 回答"能不能做"，LIBERO-PRO 的 gap 回答"扰动后还能不能做"。gap=0 说明鲁棒，gap 大说明脆弱。一个策略可以 LIBERO 80% 但 LIBERO-PRO gap 40%，意味着泛化能力有但鲁棒性差。

---

Ch5 讲了"怎么评测"——方法、指标、原则。但方法论需要落地：**谁的评测最容易跑通？** 下一章从 LIBERO 的默认机器人 Franka Panda 出发，验证开箱即用的评测流程。

> **上一章**：[Ch4 Debug 实战](so101-tutorial-ch4-debug-journey.md) | **下一章**：[Ch6 Franka 评测能力盘点](so101-tutorial-ch6-franka-baseline.md)
