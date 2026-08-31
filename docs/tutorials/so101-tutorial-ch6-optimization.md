# Ch6：LIBERO 落地方案

> SO101 仿真评测教程 · 第六章  
> 实战项目：[link-seek/so101-sim-pipeline](https://github.com/link-seek/so101-sim-pipeline)  
> 详细方案：[Discussion #9 — SO101 跑 LIBERO 评测：现状分析与落地方案](https://github.com/link-seek/so101-sim-pipeline/discussions/9)

---

## 1. 为什么这一章讲 LIBERO 落地

前五章我们完成了从基础设施到 PPO/VLA 训练到 Debug 到评测的完整闭环。但评测一直停留在**单任务**层面——Grid Sweep 只测一个 pick-and-place 任务的工作空间覆盖。

**LIBERO 是 VLA 领域的标准 benchmark**（CoRL 2023，2.2k stars），提供 3 个 suite × 10 tasks 的多任务泛化评测。如果一个 VLA 模型能在 LIBERO 上拿到高分，说明它具备跨任务泛化能力——这是衡量 VLA 质量的金标准。

但 Ch5 实战中我们跑出 0%——LIBERO 只有 Franka Panda 机器人，我们的模型是在 SO101 上训练的，关节定义、观测空间、动作语义全部不匹配。这一章讲**如何解决这个问题**。

---

## 2. 问题：模型-环境不兼容

| | 我们的 SmolVLA | LIBERO 环境 |
|--|----------------|-------------|
| 机器人 | SO101（6 DoF） | Franka Panda（7 DoF） |
| 训练数据 | SO101 真机/仿真演示 | LIBERO 仿真演示（RoboSuite） |
| 观测空间 | SO101 关节状态 + SO101 相机图像 | Franka 关节状态 + RoboSuite 相机图像 |
| 动作空间 | SO101 6 维（5 arm + 1 gripper） | Franka 7 维（7 arm + 1 gripper） |

我们的 SmolVLA 模型（`xieyucheng123/so101-smolvla`）在 SO101 数据上训练，**无法直接控制 LIBERO 的 Franka 机器人**——观测维度和动作语义都不匹配。

---

## 3. 方案：在 LIBERO 中添加 SO101 机器人

### 3.1 核心思路

**不是**在 LIBERO 数据上训练 Franka 模型（那训练的是控制 Franka 的模型，不是 SO101 模型），而是**把 SO101 作为自定义机器人添加到 LIBERO/RoboSuite 中**，然后用我们现有的 SO101 SmolVLA 模型直接评测。

```
旧方案（已废弃）: 在 LIBERO 数据上训练 Franka SmolVLA → 评测 Franka 模型
新方案:          在 LIBERO 中添加 SO101 机器人 → 评测我们自己的 SO101 SmolVLA
```

**新方案的核心优势**：
- 评测的是 **我们自己的 SO101 模型**（`xieyucheng123/so101-smolvla`），不是新训练的 Franka 模型
- 不需要新训练数据，可以直接 zero-shot 评测
- V100 能跑（LIBERO 用 RoboSuite/MuJoCo，不需要 Isaac Lab）

### 3.2 LIBERO 机器人定义机制

LIBERO 基于 [RoboSuite](https://github.com/ARISE-Initiative/robosuite)，机器人通过两个东西定义：

1. **MuJoCo XML 文件** — 运动链、关节、mesh 几何
2. **Python 类** — 继承 `ManipulatorModel`，指定默认配置

RoboSuite 已有 12 种机器人（baxter, panda, sawyer, ur5e 等），我们要添加第 13 种：SO101。

SO101 的 MJCF 已有——`robot_descriptions.so_arm101_mj_description` 提供完整 XML，`dyordan1/so101-mujoco` 也在用。需要适配 RoboSuite 的目录结构和接口规范。

> **详细实施方案**（代码示例、文件结构、参数配置）见 [Discussion #9](https://github.com/link-seek/so101-sim-pipeline/discussions/9) 第三节。

---

## 4. 路线图

```
Week 1: SO101 机器人集成
  ├── 导出 SO101 robot.xml → RoboSuite 格式
  ├── 编写 SO101 gripper (XML + Python)
  ├── 编写 MountedSO101 类 (继承 ManipulatorModel)
  ├── 配置 JOINT_POSITION 控制器
  ├── 适配 BDDL 任务文件（调整区域坐标）
  └── 集成测试
       ↓
Week 2: 评测与归档
  ├── Zero-shot 评测 LIBERO 3 suites (1,500 episodes, ~8h)
  ├── Zero-shot 评测 LIBERO-PRO 5 suites (2,500 episodes, ~14h)
  ├── 汇总结果 + 可视化
  └── CI 集成 + 结果归档到 OBS
```

**总工期**: ~2 周 | **V100 评测时间**: ~22h | **成本**: ~¥660

---

## 5. 技术风险

| 风险 | 缓解措施 |
|------|----------|
| SO101 工作空间 < Panda，部分 BDDL 任务物体不可达 | 缩小放置范围（×0.7） |
| SO101 6 DoF vs Panda 7 DoF，部分任务需 7 自由度 | 跳过不可解任务 |
| Zero-shot 性能极低 | 预期的——这正是泛化差距的量化 |

---

## 思考题

1. **为什么不能直接把 SO101 模型放到 LIBERO 的 Franka 环境里跑？**  
   提示：机器人不同——SO101（6 DoF）vs Franka（7 DoF），关节定义、观测空间、动作语义都不匹配。详见 [Discussion #9](https://github.com/link-seek/so101-sim-pipeline/discussions/9)。

2. **在 LIBERO 中添加 SO101 机器人后，评测的是哪个模型？**  
   提示：是我们现有的 `xieyucheng123/so101-smolvla`，在 SO101 数据上训练的 SO101 模型——不是新训练的 Franka 模型。

3. **Zero-shot 评测结果很低，有意义吗？**  
   提示：有。它量化了「单任务训练 → 多任务泛化」的差距，是后续优化的 baseline。0% 和 20% 的指导价值完全不同。

4. **为什么选择在 LIBERO 中添加 SO101，而不是自建 SO101 多任务 benchmark？**  
   提示：LIBERO 提供标准化的任务定义、评测协议、社区基线。自建 benchmark 缺乏可比性，且工作量更大。

---

> **上一章**：[Ch5 评测方法论](so101-tutorial-ch5-evaluation.md) | **下一章**：[附录](so101-tutorial-appendix.md)
