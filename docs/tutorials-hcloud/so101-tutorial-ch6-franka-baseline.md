# Ch6：Franka 评测能力盘点 — LIBERO 开箱即用实战

> SO101 仿真评测教程 · 第六章
> 实战讨论：[Discussion #14 — Ch6: Franka 评测能力盘点 — LIBERO 开箱即用实战](https://github.com/link-seek/so101-sim-pipeline/discussions/14)

---

## 1. 为什么从 Franka 开始

Ch5 讲了评测方法论，但方法论需要落地。**谁的评测最容易跑通？**

答案是 Franka Panda——LIBERO 的默认机器人。选择 Franka 作为起点的原因：

| 条件 | Franka | SO101 |
|------|--------|-------|
| LIBERO 原生支持 | ✅ 12 种 RoboSuite 机器人之一 | ❌ 需要自行集成 |
| 配置修改 | 零修改 | 需要改镜像 |
| 模型兼容性 | OBS 上有大量预训练模型 | 只有我们自己训练的模型 |
| 社区基线 | 完善（LIBERO 论文原版结果） | 无 |

**这一章的目标**：用 Franka 跑通 LIBERO 评测全流程，验证平台能力，建立基线。

---

## 2. LIBERO Benchmark 全景

LIBERO 是 VLA 领域的标准 benchmark（CoRL 2023，2.2k stars），提供 3 个 suite × 10 个任务：

### 2.1 三个 Suite

| Suite | 测试维度 | 10 个任务示例 |
|-------|----------|--------------|
| `libero_spatial` | 空间泛化（不同位置/姿态） | put the bowl on the plate, put the eggplant in the basket... |
| `libero_object` | 物体泛化（不同物体） | put the tomato in the basket, put the carrot on the plate... |
| `libero_goal` | 目标泛化（不同目标状态） | open the drawer, close the drawer, push the plate... |

每个 suite 的 10 个任务是**正交的**——只变一个维度，其他固定。组合起来可以定位泛化瓶颈在哪个维度。

### 2.2 LIBERO-PRO 扩展

LIBERO-PRO（2024）在 LIBERO 基础上增加了**鲁棒性测试**——对环境施加扰动，测量性能下降幅度：

| 扰动类型 | 说明 |
|----------|------|
| `env` | 物理参数扰动（摩擦、重力） |
| `object` | 物体外观扰动（纹理、颜色） |
| `lan` | 语言指令扰动（同义改写） |
| `task` | 任务结构扰动（目标位置偏移） |
| `swap` | 交叉评测（A suite 训练 → B suite 测试） |

LIBERO 的 success rate 回答"能不能做"，LIBERO-PRO 的 robustness gap 回答"扰动后还能不能做"。

---

## 3. 实战：Franka 评测流程

### 3.1 前提条件

- 华为云 ECS（V100 32GB）
- Docker 已安装
- SWR 镜像可访问

### 3.2 评测 LIBERO（真实流水线）

本教程的实测跑在**华为云 ECS（V100）**上，全流程由 GitHub Actions 编排（`.github/workflows/franka-eval.yml`），一键触发：

```bash
# 跑 libero_spatial：10 tasks × 10 eps = 100 eps（SmolVLA 官方协议）
gh workflow run franka-eval.yml \
  -f benchmarks="libero_spatial" \
  -f episodes_per_task="10"
```

流水线在容器内做三件事（全部可复现，见仓库 `scripts/`）：

1. `vla-eval serve` 启动 `lerobot/smolvla_libero` 模型服务（`configs/model_servers/smolvla_franka.yaml`）；
2. `vla-eval run` 原地执行 benchmark（容器内运行时 harness 跳过自身 docker 拉起）；
3. `vla-eval merge` + 诚实计分：直接读 `LIBEROBenchmark_*_aggregate.json` 统计成功/错误数——harness 即使 100 个 episode 全错也会 exit 0，**只能以 aggregate 为准**。

> 镜像：`swr.cn-north-4.myhuaweicloud.com/link-seek/so101-eval:latest`
> （实测 provenance：Actions run `33829389761`，2026-09-04，V100）。

### 3.3 评测 LIBERO-PRO（5 suites）

```bash
# 拉取 LIBERO-PRO 镜像
docker pull swr.cn-north-4.myhuaweicloud.com/link-seek/vla-eval-libero-pro:latest

# 评测 libero_pro_env（环境扰动）
docker run --gpus all \
  -e MUJOCO_GL=egl \
  swr.cn-north-4.myhuaweicloud.com/link-seek/vla-eval-libero-pro:latest \
  vla-eval run --config /workspace/configs/benchmarks/libero_pro_env.yaml
```

### 3.4 配置文件结构

每个 benchmark 的 YAML 配置：

```yaml
server:
  url: "ws://localhost:8000"
output_dir: "/data/eval/results/libero_spatial"
benchmarks:
  - benchmark: "vla_eval.benchmarks.libero.benchmark:LIBEROBenchmark"
    subname: libero_spatial
    episodes_per_task: 10
    params:
      suite: libero_spatial
      seed: 7
      num_steps_wait: 10
```

关键参数：
- `suite`：评测哪个 suite（spatial/object/goal）
- `episodes_per_task`：每个任务跑多少 episode（**10 是 SmolVLA 官方协议**，不是 50）
- `seed`：随机种子（可复现性）
- `num_steps_wait`：等待环境稳定的时间步

> 注意：上游 `vla-eval`（含 0.5.0）没有 `robot:` 参数，类名是 `LIBEROBenchmark`。换机器人不是改 YAML 参数，而是改文件（见 Ch7 §2 实测机制）。

---

## 4. 结果解读

### 4.1 LIBERO 标准指标

| 指标 | 计算方式 | 含义 |
|------|----------|------|
| `overall_success` | 所有任务成功率的均值 | 综合能力 |
| `per_task_success` | 每个任务的成功率 | 任务级表现 |
| `pc_success` | 按 episode 维度的成功率 | 统计精度 |

LIBERO 论文原版结果（参考值）：

| 模型 | libero_spatial | libero_object | libero_goal |
|------|---------------|---------------|-------------|
| GPT-4V + robotic pipeline | 0% | 0% | 0% |
| OpenVLA (7B) | 32% | 28% | 18% |
| π₀ (3B) | 78% | 72% | 65% |

### 4.2 LIBERO-PRO 指标

| 指标 | 计算方式 | 含义 |
|------|----------|------|
| `robustness_gap` | 原始成功率 − 扰动成功率 | 扰动敏感度 |
| `relative_gap` | gap / 原始成功率 | 相对退化比例 |

gap = 0 说明鲁棒，gap 大说明脆弱。一个策略可以 LIBERO 80% 但 LIBERO-PRO gap 40%，意味着泛化能力有但鲁棒性差。

### 4.3 实测基线（本教程真实跑分）

模型 `lerobot/smolvla_libero`，`libero_spatial` 10 tasks × 10 eps = **100 eps**，V100，2026-09-04（Actions run `33829389761`）：

| 指标 | 值 |
|------|-----|
| 成功率 | **47/100（47%）** |
| harness 错误数 | **0**（100 个 episode 全部正常 rollout，无异常） |
| 单 suite 耗时 | ~2h（V100） |

10 个任务全部 10/10 跑完，没有任何 `failure_reason: exception`——说明**评测管线本身是健康的**，47% 是策略的真实表现，不是框架 bug。

与官方 ~90% 的差距说明（诚实记录，未掩盖）：
- 官方数字的渲染/种子/解码配置与本流水线不完全一致（EGL 离屏渲染、seed 分布、`chunk_size=10`/`max_batch_size=1` 均为本工程选择）；
- 兼容层（见下）恢复了 API 调用，但随机数流与 1.4 时代不完全相同，任务初始分布有偏移；
- 本基线的价值是**可复现的起点分数**，不是 SOTA 复刻。后续工作可逐项对齐（渲染分辨率、温度/采样、seed 协议）。

> **开箱即用是个神话**：即使 Franka 是 LIBERO 默认机器人，2026 年的依赖组合（`vla-eval 0.4.0` + `robosuite 1.5.2` + LIBERO 快照）也跑不通，harness 全系（≤0.5.0）写的是 1.4 时代 API。本教程用**文件级兼容补丁**（不改 harness、不重建镜像）修了三处，全部挂载覆盖、断言防漂：
>
> | # | 现象 | 根因 | 补丁位置 |
> |---|------|------|----------|
> | 1 | `TypeError: 'NoneType' object is not callable`（100/100） | 1.5 把 `MujocoEnv.seed()` 方法改成了构造参数（实例属性 `None`），LIBERO 仍按方法调用 | `scripts/patch_env_wrapper.py` → `ControlEnv.seed` 改写为 `self.env.seed = seed` + 重建 `rng` |
> | 2 | `AttributeError: 'FixedBaseRobot' object has no attribute 'controller'` | 1.5 用 composite/part controllers 重构，删掉了 `robot.controller` | `scripts/patch_robosuite.py` → `FixedBaseRobot.controller` 兼容属性（`ee_pos`/`ee_ori_mat` 读 sim site 真值） |
> | 3 | `Server error: must override predict_batch() to use max_batch_size > 1` | 挂载的 lerobot server 脚本无 batch 实现 | `configs/model_servers/smolvla_franka.yaml`：`max_batch_size: 8 → 1`（同步评测不需要 batch） |
>
> 另有两个工程教训：`sitecustomize` 绝不能写 stdout（glfw 版本检查会 `eval()` 捕获到的 stdout，一行打印杀死全部 episode）；`vla-eval run` 即使全错也 exit 0，流水线必须以 aggregate 计分并在 errors>0 时判失败。

---

## 5. 这告诉我们什么

### 5.1 平台成熟度基线

Franka + LIBERO 的开箱即用体验展示了**评测平台的理想状态**：

| 维度 | Franka + LIBERO | SO101 + LIBERO |
|------|-----------------|----------------|
| 配置 | YAML 文件已存在 | 需要新建 |
| 镜像 | vla-eval-libero 已构建 | 需要修改镜像 |
| 模型 | OBS 上有预训练模型 | 只有自己的模型 |
| 结果 | 有社区基线可对比 | 无基线 |

### 5.2 从 Franka 到 SO101 的路径

Franka 评测验证了平台的**评测管线**是通的。问题出在**机器人适配**——SO101 不是 LIBERO 的原生机器人。

Ch7 将展示：RoboSuite 已有 12 种机器人，改配置就能换。Ch8 将展示：SO101 作为自定义机器人，需要改镜像才能集成。

### 5.3 评测规模参考（实测）

| 评测范围 | episodes | V100 实测耗时 |
|----------|----------|---------------|
| `libero_spatial` 10eps/task | 100 | ~2h（含 serve 启动 ~10min） |
| LIBERO 3 suites（同规模外推） | 300 | ~6h |
| 冒烟（1ep/task） | 10 | ~15min |

> 旧版表格中的 500eps/suite（~8h）是按 `episodes_per_task=50` 估算的；官方 SmolVLA 协议是 10eps/task，上表以实测为准。

---

## 思考题

1. **为什么 Franka 能开箱即用，SO101 不能？**  
   提示：LIBERO 基于 RoboSuite，Franka 是 RoboSuite 的原生机器人之一。SO101 不在 RoboSuite 中，需要自行集成。

2. **LIBERO 的 3 个 suite 为什么是正交的？**  
   提示：每次只变一个维度（位置/物体/目标），其他固定。组合起来可以定位泛化瓶颈在哪个维度。

3. **LIBERO-PRO 的 robustness gap 和 LIBERO 的 success rate 有什么互补性？**  
   提示：success rate 回答"能不能做"，gap 回答"扰动后还能不能做"。一个策略可以 LIBERO 80% 但 gap 40%。

4. **如果要在 SO101 上评测 LIBERO，需要解决什么问题？**  
   提示：SO101 不是 RoboSuite 原生机器人，需要添加机器人定义（XML + Python 类），适配 BDDL 任务文件，修改评测镜像。详见 Ch8。

---

> **上一章**：[Ch5 评测方法论](so101-tutorial-ch5-evaluation.md) | **下一章**：[Ch7 RoboSuite 机器人扩展](so101-tutorial-ch7-robosuite-extension.md)
