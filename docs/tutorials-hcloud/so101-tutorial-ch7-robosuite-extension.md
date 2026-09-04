# Ch7：RoboSuite 机器人扩展 — 改配置换机器人

> SO101 仿真评测教程 · 第七章
> 实战讨论：[Discussion #15 — Ch7: RoboSuite 机器人扩展 — 改配置换机器人](https://github.com/link-seek/so101-sim-pipeline/discussions/15)

---

## 1. 从 Franka 到一打机器人

Ch6 用 Franka 跑通了 LIBERO 评测。但 LIBERO 基于 RoboSuite，而 RoboSuite 内置了一打左右的机器人——不需要重写评测框架，换机器人只需要换配置+文件补丁。

### 1.1 RoboSuite 内置机器人（以本教程镜像实测为准）

本教程镜像的 `robosuite==1.5.2` 实测 `manipulators` 目录包含（版本不同组成会有出入，老版本如 1.4 另有 UR10e 等）：

| 机器人 | 类型 | 备注 |
|--------|------|------|
| Panda (Franka) | 7DoF 单臂 | LIBERO 默认，抓取/放置 |
| Sawyer | 7DoF 单臂 | Rethink Robotics，本章实测对象 |
| IIWA | 7DoF 单臂 | KUKA，高精度 |
| Jaco | 6DoF 单臂 | Kinova，协作机器人 |
| Kinova3 | 6DoF 单臂 | Kinova Gen3 |
| UR5e | 6DoF 单臂 | Universal Robots，工业标准 |
| Baxter | 7×2DoF 双臂 | Rethink 双臂协作 |
| XArm7 | 7DoF 单臂 | UFACTORY |
| GR1 | 人形上身 | Fourier，1.5 新增 |
| Tiago | 移动+单臂 | PAL Robotics |
| SpotArm | 四足+臂 | Boston Dynamics Spot 臂 |
| Humanoid | 人形 | 研究用 |

> 原稿表格里 Sawyer 重复列了两次，并混入了本镜像实测目录中不存在的 UR10e/WIDOWX/Albrrt，已按 `robosuite==1.5.2` 实测纠正。以镜像实测为准，不要背诵机器人名单。

### 1.2 为什么能改配置就换机器人

RoboSuite 的机器人定义是**声明式**的——通过 MuJoCo XML 文件和 Python 类定义，不需要修改评测框架代码：

```
RoboSuite 机器人定义
├── robot.xml          # MuJoCo 模型（关节、link、mesh）
├── robot_class.py     # Python 类（继承 ManipulatorModel）
└── default_config.py  # 默认配置（控制器、观测）
```

LIBERO 的 BDDL 任务定义也是声明式的——只描述初始状态和目标，不描述具体机器人。这意味着同一个 BDDL 可以在不同机器人上实例化（只要机器人支持所需谓词）。

---

## 2. 实战：换机器人评测（真实机制）

> **原稿勘误**：上游 `vla-eval`（含 0.5.0）的 LIBERO benchmark **没有 `robot:` 参数**，§2.1 旧版 YAML 里那个 `robot: sawyer` 是虚构的，照抄会静默跑回 Panda。本节是实测过的真实机制。

真相只有一处硬编码：LIBERO 的 `OffScreenRenderEnv(robots=["Panda"])` 默认构造 Franka。所以换机器人 = **文件级改写这个默认值 + 挂载覆盖镜像内文件**，不是改 YAML 参数：

1. `scripts/patch_env_wrapper.py` 把 `robots` 默认值改写为目标机器人（如 `["Sawyer"]`），改写行数断言 ≥1，漂移即失败；
2. `robot-switch.yml` 把改写后的 `env_wrapper.py`（及 robosuite 1.5 控制器兼容补丁 `fixed_base_robot.py`）挂载进容器；
3. YAML（`configs/benchmarks/libero_spatial_sawyer.yaml`）只负责改 `subname`/`output_dir` 做结果隔离，类名是 `LIBEROBenchmark`。

### 2.1 触发 Sawyer 演示

```bash
# Sawyer × libero_spatial：10 tasks × 1 ep = 10 eps（跑通即达标，见 §2.4）
gh workflow run robot-switch.yml \
  -f robot="Sawyer" \
  -f benchmarks="libero_spatial" \
  -f episodes_per_task="1"
```

### 2.2 运行评测

流水线与 Ch6 §3.2 同构（serve → run → merge + aggregate 诚实计分），镜像同为 `swr.cn-north-4.myhuaweicloud.com/link-seek/so101-eval:latest`。sawyer YAML 内 `docker.image` 字段仅作文档保留，容器内运行时 harness 跳过自身 docker 拉起。

### 2.3 批量评测多种机器人

```bash
# 3 种机器人 × 3 个 suite：每个组合触发一次 robot-switch.yml
for robot in Panda Sawyer UR5e; do
  for suite in libero_spatial libero_object libero_goal; do
    echo "Evaluating $robot on $suite..."
    gh workflow run robot-switch.yml \
      -f robot="$robot" \
      -f benchmarks="$suite" \
      -f episodes_per_task="1"
  done
done
```
> 每个组合需要配套 `configs/benchmarks/<suite>_<robotlower>.yaml`（本仓库实测仅有 `libero_spatial_sawyer.yaml`，其余按此模板复制改 `subname`/`output_dir` 即可）。

### 2.4 Sawyer 实测结果（本教程真实跑分）

`libero_spatial` 10 tasks × 1 ep = **10 eps**，V100，2026-09-04（Actions run `33836694327`）：

| 指标 | 值 |
|------|-----|
| 跑通 | **10/10 episodes 正常 rollout，harness 错误数 0** |
| 成功率 | **0/10（0%）** |
| 兼容补丁 | `[patch-check]` 双 True（`_rs15_seed_compat` + `_rs15_controller_compat`） |

0% 是**预期内**的：`lerobot/smolvla_libero` 是在 Panda 身体上训练的策略，把它的 Panda 空间动作直接作用到 Sawyer 身体上，理应做不成任务——但 10 个任务的环境**全部用 Sawyer 正确构造、全流程跑通**，这正是本章的达标线（1 task×5eps 跑通即达标，本次 10 tasks×1ep 超额完成）。它同时是 §3.2 那张兼容性表的活证据：Franka 模型 → Sawyer 评测 ❌（动作/观测空间不同），评测能跑通，但分数没有意义。

---

## 3. 限制与注意事项

### 3.1 机器人兼容性

不是所有机器人都能跑所有任务：

| 限制 | 说明 |
|------|------|
| DoF 不匹配 | 部分任务需要 7 自由度，5/6 DoF 机器人可能无法解 |
| 工作空间 | 不同机器人的可达范围不同，部分物体可能不可达 |
| 夹爪类型 | 不同夹爪的抓取能力不同，影响成功率 |

### 3.2 模型兼容性

**关键问题**：模型是在特定机器人上训练的，换机器人后模型可能不兼容。

| 场景 | 可行性 |
|------|--------|
| Franka 模型 → Franka 评测 | ✅ 直接可用 |
| Franka 模型 → Sawyer 评测 | ❌ 动作空间/观测空间不同 |
| SO101 模型 → Franka 评测 | ❌ 关节定义/动作语义不同 |

**改配置换机器人的前提**：模型本身支持目标机器人（在该机器人数据上训练过，或者模型具有跨机器人泛化能力）。

### 3.3 这一层能解决什么

| 问题 | 能解决 | 说明 |
|------|--------|------|
| 评测框架不支持新机器人 | ✅ | RoboSuite 内置的一打机器人 |
| 评测流程需要改代码 | ✅ | 文件补丁+挂载，不改 harness/镜像 |
| 模型不兼容新机器人 | ❌ | 需要重新训练或 fine-tune |
| 新机器人不在 RoboSuite 中 | ❌ | 需要自行集成（见 Ch8） |

---

## 4. 从配置级到镜像级

这一章展示了**配置级扩展**的边界：

| 扩展级别 | 方法 | 适用场景 |
|----------|------|----------|
| L0 零改动（Ch6） | 直接跑 | LIBERO 默认 Panda |
| L1 改文件补丁+挂载（本章） | robot-switch 机制（§2） | RoboSuite 内置机器人 |
| L2 改镜像（下一章） | 加机器人定义进镜像 | RoboSuite 没有的自定义机器人（如 SO101） |

Ch8 将展示：当机器人不在 RoboSuite 中时，如何修改镜像添加自定义机器人。

---

## 思考题

1. **RoboSuite 有一打内置机器人，为什么 LIBERO 只用 Franka？**  
   提示：LIBERO 的 BDDL 任务是针对 Franka 的工作空间和能力设计的。换机器人可能需要调整任务参数（放置范围、物体大小等）。

2. **改配置换机器人的前提是什么？**  
   提示：模型必须支持目标机器人——在该机器人数据上训练过，或者模型具有跨机器人泛化能力。否则评测结果没有意义。

3. **为什么 SO101 不能用配置级扩展？**  
   提示：SO101 不在 RoboSuite 的内置机器人中。需要自行添加机器人定义（XML + Python 类），这属于镜像级扩展。

4. **如果一个模型在 Franka 上训练，能在 Sawyer 上评测吗？**  
   提示：不能直接评测。Franka 和 Sawyer 的关节维度、动作空间、观测空间都不同。需要在 Sawyer 数据上重新训练或 fine-tune。

---

> **上一章**：[Ch6 Franka 评测能力盘点](so101-tutorial-ch6-franka-baseline.md) | **下一章**：[Ch8 自定义机器人扩展 + SO101 落地](so101-tutorial-ch8-custom-robot.md)
