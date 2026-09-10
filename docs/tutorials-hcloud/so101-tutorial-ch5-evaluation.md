# Ch5：仿真评测方法论

> SO101 仿真评测教程 · 第五章
> 参考框架：[Gymnasium](https://gymnasium.farama.org/) · [LeRobot `lerobot-eval`](https://github.com/huggingface/lerobot) · [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) · [LIBERO-PRO（harness 源码定义）](https://github.com/allenai/vla-evaluation-harness/blob/v0.4.0/src/vla_eval/benchmarks/libero_pro/benchmark.py)

---

## 1. 开源评测框架全景

上一章我们用了两种评测（都是 VLA 线的）：`replay_demo.py`（回放验证）和 `eval_mujoco_policy.py`（grid sweep）。但这两个脚本不是凭空发明的——它们背后有一套成熟的开源评测框架体系。

前四章是上手：跑通、拿分——跑的时候还没这套理论。本章是回头：讲清你们当时用的尺子，再给跑过的分下判定。

本章就来搞清楚：**社区是怎么评测机器人策略的，我们的脚本和这些框架是什么关系。**

新手流程地图（一句话版）：回放（smoke）→ Grid Sweep（单任务考试）→ PPO 确定性评估（RL 参照）→ LIBERO（毕业考试）。操作细节是 Ch1–Ch4 的内容，本章不重复——下面只讲框架关系。

### 各框架对比

| 框架 | 类型 | 用途 | 核心指标 | 我们的使用方式 | 一句话感受 |
|------|------|------|----------|---------------|-------------|
| **Gymnasium** | API 标准 | 环境接口 | `info["success"]`, `reward` | 所有 eval 脚本的底层 API | 契约本身没坑，坑全在环境作者定的 success 对不对——PPO v1 的 100% 假成功就是例子 |
| **LeRobot lerobot-eval** | 评测方法 | VLA 通用评测 | `pc_success`, `avg_sum_reward` | `replay_demo.py` 用其推理管线 | 推理管线开箱即用；指标名和我们的对得上，seed 默认 1000 照抄就行 |
| **LIBERO** | 评测方法 | VLA benchmark | task success rate × 10 tasks | `eval_vla.py` 通过 vla-eval harness | 金标准但只认 Franka——先纸上对身体，见 §4.8 |
| **LIBERO-PRO** | 评测方法 | VLA 鲁棒性 | robustness gap | `eval_vla.py` 的 libero_pro_*（5 种扰动，定义见 §4.6） | 非官方改题卷，定义以 AllenAI 源码为准，详见 §4.6 |
| **CleanRL** | 评测方法 | RL 评估范式 | `success_rate`, `ep_return` | `eval_ppo.py` 的确定性评估 | 范式最省心：固定 seed + 确定性策略 + 50eps，PPO 稳到不用算 CI |
| **Grid Sweep** | 评测方法 | 多初始条件评测 | success rate across grid | `eval_mujoco_policy.py` 实现 | 不是标准框架，是社区土办法——但只有它照出了覆盖盲区（边缘 ~0%），最爱的一张热力图 |
| **so101_nexus** | 仿真环境 | MuJoCo 仿真 | — | Ch4 回放验证 | 回放够用，但和 ataghof 采集环境不是一回事——Ch4 全章就是为这句话买的单 |
| **so101-mujoco** | 仿真环境 | MuJoCo sim twin | — | `eval_mujoco_policy.py` 的环境 | 救命的环境：训测同场，47% 从这来；三个坑（shm/XET/独占）见 Ch1 §4.3 |

---

## 2. Gymnasium：评测的通用语言

### 2.1 唯一的契约：`info["success"]`

所有评测脚本都基于 [Gymnasium](https://gymnasium.farama.org/) 的标准 API（`reset(seed)` / `step()` 五元组）。新手只需记住一条：**成功与否由环境的 `info["success"]` 说了算，评测者不自造标准**——自造就会刷出 PPO v1 那种 100% 假成功（见 Ch2）。Ch2/Ch4 的脚本都是这个契约的实例，细节查文档。

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

**读图路线**：§4.1–§4.5 讲左上真题卷长什么样（BDDL、判定、三个 suite）；§4.6 讲右上改题卷（5 种改法 + gap），§4.7 讲左下全真卷（另出卷，和 PRO 的区别）。§4.8 收束：LIBERO 不支持 SO101，所以下一章从官方机器人开始。

### 4.1 LIBERO 是什么：官方真题卷（左上那盒）

这是官方出的唯一一套卷，维护者是 [Lifelong-Robot-Learning 团队](https://github.com/Lifelong-Robot-Learning/LIBERO)（Bo Liu 等，CoRL 2023，2.2k stars）。全称 **Lifelong Benchmark for Robotics**，核心定位：

> 给定一组操作任务，系统化地评测 VLA 策略的**泛化能力**——不是"能不能做一个任务"，而是"能不能做一类任务"。

LIBERO 建立在 [RoboSuite](https://github.com/ARISE-Initiative/robosuite) 仿真框架上，使用 Franka Panda 7 DoF 机械臂，定义了 5 个 benchmark suite，总计 130 个任务。

### 4.2 评测对象

LIBERO 评测的是 **VLA 策略**：输入图像 + 语言指令，输出机器人动作。我们的 SmolVLA 也是 VLA，理论上可测——实际卡在机器人不匹配（详见 §4.8）：LIBERO 只认 Franka，我们的模型是 SO101 的身体。

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

### 4.7 LIBERO-Plus：非官方全真卷（左下那盒）

同样非官方——维护者是 Sylvest 团队，论文 [arXiv:2510.13626](https://arxiv.org/abs/2510.13626)《LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models》。7 个扰动维度（物体布局、相机视角、机器人初始状态、语言指令、光照、背景纹理、传感器噪声），10030 tasks；harness 里是独立的 `libero_plus` benchmark。

和 PRO 的区别：**改题 vs 另出题**，生产线完全不同——

|  | PRO（右上） | Plus（左下） |
|---|---|---|
| 出法 | 拿真题改：每道有母题，名字暴露血缘（`libero_spatial_swap` = spatial 卷 swap 来的） | 照考纲另出：新题无母题，与 130 道真题无对应关系 |
| 输出 | robustness gap（母题分 − 改题分） | 独立分数 |
| 动机 | 测"扰动后稳不稳" | 补真题没有的维度——LIBERO 真题机位固定，想考"换个角度还认不认识"只能新出，改无可改 |

论文最狠的发现：相机视角和机器人初始状态最致命（95% 跌到 30% 以下），语言改写基本无感——模型根本不看指令。

工具链归属：本教程的链（`vla-eval==0.4.0`）自带 PRO；Plus 是另一套包（要替换 `libero`），不在同一条链上。只出镜，不参演。

### 4.8 小结：LIBERO 不支持 SO101——下一章从官方机器人开始

LIBERO 只认原生机器人（RoboSuite 自带的，如 Panda 和 Sawyer，名单以 Ch7 实测为准）。SO101 不在其中：6 自由度对 7 自由度，关节、动作、观测全对不上——纸上即可判死刑，连试都不用试。**评测前先纸上对身体**，这是本章留下的唯一一条行动规则。

所以路线是：先用官方机器人（Franka）把评测管线校准（下一章 Ch6），再谈 SO101 的集成（Ch8）。

---


## 思考题

1. **LIBERO 评测和 grid sweep 评测有什么互补性？**  
   提示：LIBERO 测跨任务泛化（不同物体/目标/语言），grid sweep 测单任务工作空间覆盖。一个模型可能 LIBERO 80% 但 grid sweep 边缘 0%。

2. **LIBERO 的三个 suite（spatial/object/goal）为什么是正交的？**  
   提示：每次只变一个维度，其他固定。spatial 变位置、object 变物体、goal 变目标。组合起来可以定位泛化瓶颈在哪个维度。

3. **LIBERO-PRO 的 robustness gap 和 LIBERO 的 success rate 有什么区别？**  
   提示：LIBERO 的 success rate 回答"能不能做"，LIBERO-PRO 的 gap 回答"扰动后还能不能做"。gap=0 说明鲁棒，gap 大说明脆弱。一个策略可以 LIBERO 80% 但 LIBERO-PRO gap 40%，意味着泛化能力有但鲁棒性差。

---

Ch5 把框架关系讲清了，结论只有一个：**LIBERO 只认原生机器人，SO101 不在其中**。下一章从 LIBERO 的默认机器人 Franka Panda 出发，先把评测管线校准。

> **上一章**：[Ch4 Debug 实战](so101-tutorial-ch4-debug-journey.md) | **下一章**：[Ch6 Franka 评测能力盘点](so101-tutorial-ch6-franka-baseline.md)
