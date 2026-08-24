# Ch6：落地 LIBERO 评测实战

> SO101 仿真评测教程 · 第六章  
> 实战项目：[link-seek/so101-sim-pipeline](https://github.com/link-seek/so101-sim-pipeline)  
> 详细调研：[Discussion #9 — SO101 跑 LIBERO 评测：现状分析与落地方案](https://github.com/link-seek/so101-sim-pipeline/discussions/9)

---

## 1. 为什么这一章讲 LIBERO 落地

前五章我们完成了从基础设施搭建到 PPO/VLA 训练到 Debug 实战到评测方法论的完整闭环。但评测一直停留在**单任务**层面——Grid Sweep 只测一个 pick-and-place 任务的工作空间覆盖。

**LIBERO 是 VLA 领域的标准 benchmark**（2.2k stars），提供 3 个 suite × 10 tasks 的多任务泛化评测。我们的仓库已设计好完整管线（`eval_vla.py` + 8 个 benchmark 配置 + `so101-eval` 镜像），但从未运行过。

这一章讲**如何把这套设计跑起来**——从数据准备到训练到评测到 CI 集成的完整落地路径。

> **详细调研结论**（模型-环境不兼容分析、社区调研、标准化路径对比）见 [Discussion #9](https://github.com/link-seek/so101-sim-pipeline/discussions/9)。本章聚焦**实操步骤**。

---

## 2. 落地方案总览

```
核心思路: 在 LIBERO 数据上训练新 SmolVLA → 用已有 eval_vla.py 评测

Phase 1: 数据准备 (1-2 天)
  ├── 下载 LIBERO 演示数据 (10 tasks × 50 demos/suite)
  ├── 编写 libero_to_lerobot.py 转换脚本
  └── 上传到 HF Hub
       ↓
Phase 2: 训练 (3-5 天, V100)
  ├── 训练 smolvla_libero_spatial (4-6h)
  ├── 训练 smolvla_libero_object (4-6h)
  └── 训练 smolvla_libero_goal (4-6h)
       ↓
Phase 3: 评测 (1 天)
  ├── 配置 smolvla_libero.yaml
  ├── 运行 LIBERO 3 suites (1500 episodes)
  └── 运行 LIBERO-PRO 5 suites (2500 episodes)
       ↓
Phase 4: CI 集成 (1 天)
  ├── 更新 evaluate.yml workflow
  └── 结果归档到 OBS
```

**总工期**: 6-9 天 | **V100 资源**: 30-40 GPU 小时 | **成本**: ¥900-1200

---

## 3. Phase 1：数据准备

### 3.1 下载 LIBERO 演示数据

LIBERO 每个 suite 自带 50 个演示 episode，格式为 RoboSuite 的 HDF5：

```bash
# 安装 LIBERO
pip install libero

# 下载数据
python -c "
from libero.libero import get_demo, benchmark
# 3 个 suite, 每个 10 tasks
for suite_name in ['libero_spatial', 'libero_object', 'libero_goal']:
    suite = benchmark.get_benchmark(suite_name)
    for task_id in range(10):
        task = suite.get_task(task_id)
        demos = get_demo(suite_name, task_id)  # 50 个 HDF5 轨迹
        print(f'{suite_name} task {task_id}: {len(demos)} demos')
"
```

每个 demo 包含：
- `qpos`: Franka 关节位置 (7 维)
- `qvel`: Franka 关节速度 (7 维)  
- `action`: 动作 (7 维)
- `obs`: 观测（`agentview_image`, `robot0_eye_in_hand_image` 等）

### 3.2 转换到 LeRobot 格式

LIBERO 用 RoboSuite HDF5，LeRobot 用自己的 v2 格式。需要写转换脚本：

```python
# scripts/libero_to_lerobot.py — 需要新建
import h5py
import numpy as np
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def convert_suite(suite_name: str, output_dir: str):
    """将 LIBERO suite 转换为 LeRobot 数据集"""
    dataset = LeRobotDataset.create(
        repo_id=f"link-seek/{suite_name}_lerobot",
        root=Path(output_dir),
        fps=20,
    )

    for task_id in range(10):
        demos = get_demo(suite_name, task_id)
        for demo_idx, demo in enumerate(demos):
            # 添加 episode
            for step in range(len(demo["actions"])):
                frame = {
                    "observation.state": demo["qpos"][step],          # Franka 关节状态
                    "action": demo["actions"][step],                   # Franka 动作
                    "observation.images.head": demo["obs"]["agentview_image"][step],
                    "observation.images.wrist": demo["obs"]["robot0_eye_in_hand_image"][step],
                    "task": f"{suite_name}_task_{task_id}",
                }
                dataset.add_frame(frame)
            dataset.save_episode()

    dataset.push_to_hub()
    print(f"Converted {suite_name}: {len(dataset)} episodes")

for suite in ["libero_spatial", "libero_object", "libero_goal"]:
    convert_suite(suite, f"outputs/{suite}_lerobot")
```

**关键点**：
- LIBERO 用 Franka Panda，观测/动作空间是 Franka 的——训练的模型控制 Franka，不是 SO101
- 这是**新模型**，和现有 `xieyucheng123/so101-smolvla` 完全不同
- 两个相机视角：`agentview`（第三人称）+ `robot0_eye_in_hand`（腕部相机）

### 3.3 上传到 HF Hub

```bash
# 上传转换后的数据集
huggingface-cli upload link-seek/libero_spatial_lerobot outputs/libero_spatial_lerobot
huggingface-cli upload link-seek/libero_object_lerobot outputs/libero_object_lerobot
huggingface-cli upload link-seek/libero_goal_lerobot outputs/libero_goal_lerobot
```

---

## 4. Phase 2：训练 LIBERO SmolVLA

### 4.1 复用现有训练管线

我们已有 `so101-train` Docker 镜像和 `so101-mujoco-pipeline.yml` workflow，只需替换数据集：

```bash
# 方式 1: Docker 直接训练
docker run --gpus all so101-train:latest \
    python -m lerobot.scripts.train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=link-seek/libero_spatial_lerobot \
    --output_dir=outputs/smolvla_libero_spatial \
    --steps=20000 \
    --batch_size=64 \
    --save_freq=2000 \
    --log_freq=500
```

### 4.2 GitHub Actions 训练

```bash
# 方式 2: 通过 workflow 触发
gh workflow run vla-pipeline.yml \
    -f dataset=link-seek/libero_spatial_lerobot \
    -f steps=20000 \
    -f batch_size=64
```

### 4.3 训练参数对比

| 参数 | SO101 训练 | LIBERO 训练 | 说明 |
|------|-----------|-------------|------|
| dataset | SO101 sim twin | LIBERO demos | 不同数据源 |
| steps | 15k | 20k | LIBERO 任务更多，需更多步 |
| batch_size | 32 | 64 | 社区推荐 |
| cameras | 3 (head + left + right) | 2 (agentview + wrist) | LIBERO 只有两个视角 |
| action_dim | 7 (SO101) | 7 (Franka) | 维度碰巧相同但语义不同 |
| save_freq | 5000 | 2000 | 更细粒度 checkpoint |

### 4.4 资源估算

| Suite | Episodes | Steps | V100 时间 | 显存 |
|-------|----------|-------|----------|------|
| libero_spatial | 500 (10×50) | 20k | 4-6h | ~14GB |
| libero_object | 500 | 20k | 4-6h | ~14GB |
| libero_goal | 500 | 20k | 4-6h | ~14GB |
| **总计** | **1500** | **60k** | **12-18h** | — |

**超时处理**：GitHub Actions 限制 6h/job。每个 suite 单独一个 job，或用 nohup 在 ECS 上直接跑：

```bash
# ECS 上后台训练，不受 Actions 超时限制
nohup docker run --gpus all so101-train:latest \
    python -m lerobot.scripts.train ... > train.log 2>&1 &
```

### 4.5 上传 Checkpoints

```bash
# 训练完成后上传到 HF Hub
huggingface-cli upload link-seek/smolvla_libero_spatial outputs/smolvla_libero_spatial
huggingface-cli upload link-seek/smolvla_libero_object outputs/smolvla_libero_object
huggingface-cli upload link-seek/smolvla_libero_goal outputs/smolvla_libero_goal
```

---

## 5. Phase 3：运行评测

### 5.1 配置模型服务

新建 `configs/model_servers/smolvla_libero.yaml`：

```yaml
# configs/model_servers/smolvla_libero.yaml
checkpoint: link-seek/smolvla_libero_spatial
model_class: SmolVLA

# LIBERO 观测格式（和 SO101 不同）
state_key: "observation.state"
image_keys:
  - "observation.images.head"     # agentview
  - "observation.images.wrist"    # wrist camera

# LIBERO 动作格式
action_key: "action"
chunk_size: 10

device: cuda
```

**与 `smolvla_so101.yaml` 的区别**：
- `checkpoint` → 新训练的 LIBERO 模型
- `image_keys` → LIBERO 的两个相机（不是 SO101 的三个）
- 不需要 `dataset_row_to_sim_qpos` 单位转换

### 5.2 运行 LIBERO 评测

```bash
# 运行 3 个 LIBERO suite
python scripts/eval_vla.py \
    --model-config configs/model_servers/smolvla_libero.yaml \
    --benchmarks libero_spatial,libero_object,libero_goal
```

评测流程（`eval_vla.py` 已实现）：
1. 启动模型推理服务（model server）
2. 逐 benchmark 运行：`vla-eval run --config configs/benchmarks/{name}.yaml`
3. 合并结果：`vla-eval merge → merged.json`
4. 汇总 success_rate

### 5.3 运行 LIBERO-PRO 鲁棒性评测

```bash
# 运行 5 个 LIBERO-PRO suite
python scripts/eval_vla.py \
    --model-config configs/model_servers/smolvla_libero.yaml \
    --benchmarks libero_pro_swap,libero_pro_object,libero_pro_lan,libero_pro_task,libero_pro_env
```

### 5.4 评测规模与预期

| Benchmark | Suites | Tasks | Episodes | V100 时间 | 预期成功率 |
|-----------|--------|-------|----------|----------|-----------|
| LIBERO | 3 | 30 | 1,500 | 2-3h | 60-80% |
| LIBERO-PRO | 5 | 50 | 2,500 | 3-4h | 30-50% |
| **总计** | **8** | **80** | **4,000** | **5-7h** | — |

**预期结果**（基于社区报告）：
- `libero_spatial`: 70-80%（空间泛化相对容易）
- `libero_object`: 60-70%（物体泛化中等）
- `libero_goal`: 50-60%（目标泛化较难）
- `libero_pro_*`: 30-50%（鲁棒性评测更难）

### 5.5 结果格式

`eval_vla.py` 输出遵循 LeRobot 的 `per_episode` 格式：

```json
{
  "benchmarks": {
    "libero_spatial": {
      "success_rate": 0.72,
      "num_episodes": 500,
      "num_tasks": 10,
      "per_task": [
        {"task": 0, "success_rate": 0.80, "episodes": 50},
        {"task": 1, "success_rate": 0.64, "episodes": 50},
        ...
      ]
    }
  }
}
```

---

## 6. Phase 4：CI 集成

### 6.1 更新 evaluate.yml

```yaml
# .github/workflows/evaluate.yml — 添加 LIBERO 评测 job
evaluate-libero:
  needs: [build-eval-image]
  runs-on: [self-hosted, gpu]
  steps:
    - uses: actions/checkout@v4
    - name: Run LIBERO evaluation
      run: |
        docker run --gpus all so101-eval:latest \
          python scripts/eval_vla.py \
          --model-config configs/model_servers/smolvla_libero.yaml \
          --benchmarks libero_spatial,libero_object,libero_goal
    - name: Archive results
      run: |
        # 上传结果到 OBS
        obsutil cp results/ obs://so101-eval/libero/$(date +%Y%m%d)/
```

### 6.2 触发评测

```bash
# 手动触发
gh workflow run evaluate.yml -f benchmark=libero

# 或在 PR 中自动触发（添加 label）
# .github/workflows/evaluate.yml
# on:
#   pull_request:
#     types: [labeled]
# jobs:
#   evaluate-libero:
#     if: contains(github.event.label.name, 'eval-libero')
```

---

## 7. 评测阶梯：落地后的完整体系

```
评测阶梯 (全部 V100 可运行):

  Level 0: 回放验证 (replay_demo.py)
    └── 30s, 1 episode, smoke test

  Level 1: PPO 确定性评估 (eval_ppo.py)
    └── 15min, 50 episodes, 单任务

  Level 2: Grid Sweep (eval_mujoco_policy.py)
    └── 30min, 325 episodes, 单任务工作空间覆盖

  Level 3: LIBERO (eval_vla.py)          ← 本章落地
    └── 2-3h, 1500 episodes, 跨任务泛化

  Level 4: LIBERO-PRO (eval_vla.py)      ← 本章落地
    └── 3-4h, 2500 episodes, 鲁棒性

  Level 5: SO-101 Bench (Isaac Lab)      ← 需 RTX GPU
    └── SO101 数字孪生, 硬件不支持
```

**LIBERO 评测落地后**，我们拥有从 smoke test 到跨任务泛化到鲁棒性的完整评测阶梯。

---

## 8. 技术风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| LIBERO 数据转换格式不匹配 | 训练失败 | 参考 LeRobot 已有的数据转换脚本，逐步验证 |
| SmolVLA 在 LIBERO 上性能差 | 评测结果低 | 预期行为——LIBERO 是 benchmark 不是训练任务，低分也有价值 |
| V100 显存不足 (16GB) | 训练 OOM | 用 bs=32 或梯度累积 |
| vla-eval harness 版本不兼容 | 评测脚本报错 | 锁定版本，参考 allenai 兼容矩阵 |
| GitHub Actions 超时 (6h limit) | 训练被截断 | 拆分 job 或 nohup 后台运行 |
| LIBERO 安装依赖冲突 | 环境搭建失败 | 在 Docker 镜像中固定版本 |

---

## 9. 长期方向：SO101 版 LIBERO

当前方案是**训练 Franka 模型跑 LIBERO 的 Franka 环境**。长期可以**自建 SO101 版多任务 benchmark**：

1. **在 MuJoCo 中定义 10 个 SO101 pick-and-place 任务** — 不同物体、不同目标位置、不同初始条件
2. **复用 `eval_mujoco_policy.py` 框架** — 改为多任务模式
3. **定义 `info["success"]` 判定逻辑** — 参照 LIBERO 的任务完成条件
4. **生成演示数据** — 用 PPO 策略或脚本策略生成 50 demos/task

这样**SO101 模型可以直接跑多任务泛化评测**，不需要训练 Franka 模型，也不依赖 Isaac Lab。

---

## 10. 社区生态与持续优化

### 10.1 关键参考项目

| 项目 | 与 LIBERO 落地的关系 |
|------|---------------------|
| [LeRobot](https://github.com/huggingface/lerobot) | 训练框架 + SmolVLA + 数据格式 |
| [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | Benchmark 数据 + 评测环境 |
| [vla-evaluation-harness](https://github.com/allenai/vla-evaluation-harness) | 评测调度框架 |
| [LIBERO-PRO](https://github.com/sylvestf/LIBERO-plus) | 鲁棒性扩展 |
| [SO-101 Bench](https://github.com/5hadytru/so101_bench) | SO101 专用 benchmark（需 RTX GPU） |

### 10.2 训练优化方向

LIBERO 评测落地后，可以继续优化模型性能：

| 方向 | 当前 | 目标 | 方法 |
|------|------|------|------|
| 超参调优 | bs=64, 20k steps | 更高 | 网格搜索 lr, batch_size |
| 数据增强 | 无 | DART 噪声 | 执行时加噪声，记录干净 action |
| RL fine-tuning | 纯 BC | BC + PPO | 在 BC 基础上在线 RL 探索 |
| 分层采样 | 随机 | 分层 | 确保每个 batch 覆盖不同任务阶段 |

---

## 踩坑预警

### 坑 1：LIBERO 版本兼容

LIBERO 依赖 RoboSuite，版本不同可能导致 API 变化。在 Docker 镜像中固定版本：

```dockerfile
# Dockerfile.eval
RUN pip install libero==0.1.0 robosuite==1.4.1
```

### 坑 2：相机图像格式

LIBERO 的图像是 `(H, W, 3)` uint8，LeRobot 期望的格式可能不同。转换时注意通道顺序和归一化。

### 坑 3：Action Chunking 与 LIBERO

SmolVLA 的 action chunking（预测未来 10 步）在 LIBERO 中可能需要调整。LIBERO 的 episode 较短（~300 steps），chunk_size 太大会导致最后几步无 action。

---

## 思考题

1. **为什么不能直接用 SO101 模型跑 LIBERO 环境？**  
   提示：机器人不同——SO101（6 DoF）vs Franka（7 DoF），关节定义、观测空间、动作语义都不匹配。详见 [Discussion #9](https://github.com/link-seek/so101-sim-pipeline/discussions/9)。

2. **在 LIBERO 数据上训练的模型，能用于 SO101 真机吗？**  
   提示：不能。模型学的是 Franka 动作空间，和 SO101 不兼容。但训练方法论可迁移。

3. **LIBERO 评测和 Grid Sweep 评测有什么互补性？**  
   提示：LIBERO 测跨任务泛化（不同物体/目标/语言），Grid Sweep 测单任务工作空间覆盖。一个模型可能 LIBERO 80% 但 Grid Sweep 边缘 0%。

4. **为什么 LIBERO-PRO 成功率比 LIBERO 低很多？**  
   提示：LIBERO-PRO 测鲁棒性——物体替换、属性变化、语言变化、任务组合、环境变化。模型需要更强的泛化能力。

5. **自建 SO101 版 LIBERO 的核心挑战是什么？**  
   提示：定义有意义的任务套件（不同难度、不同泛化维度）+ 生成高质量演示数据 + 定义合理的 success 判定逻辑。

---

> **上一章**：[Ch5 评测方法论](so101-tutorial-ch5-evaluation.md) | **下一章**：[附录](so101-tutorial-appendix.md)
