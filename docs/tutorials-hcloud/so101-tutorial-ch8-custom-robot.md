# Ch8：自定义机器人扩展 + SO101 落地

> SO101 仿真评测教程 · 第八章
> 详细方案：[Discussion #9 — SO101 跑 LIBERO 评测：现状分析与落地方案](https://github.com/link-seek/so101-sim-pipeline/discussions/9)

---

## 1. 镜像级扩展：当配置不够用

Ch7 展示了 L1 扩展——RoboSuite 内置机器人，文件补丁+挂载就能换（§2 真实机制，不是改 YAML 参数）。但 SO101 不在其中。

**镜像级扩展**适用于：目标机器人不在 RoboSuite 中，需要修改 Docker 镜像添加自定义机器人定义。

| 扩展级别 | 方法 | 工作量 | 适用场景 |
|----------|------|--------|----------|
| L0 零改动（Ch6） | 直接跑 | 0 | LIBERO 默认 Panda，实测 47%@100eps |
| L1 文件补丁+挂载（Ch7） | robot-switch 机制 | 小时级 | RoboSuite 内置机器人，Sawyer 实测跑通 |
| L2 改镜像（本章） | 加机器人定义进镜像 | 天级 | RoboSuite 没有的自定义机器人（如 SO101） |

> 本章是**方案级**（对应 Discussion #9）：SO101 集成尚未实施。它的价值在于：Ch6/Ch7 已经证明了"评测管线是通的"，剩下的只是机器人适配问题——本章把适配工作讲清楚，让它从未知变成待办。

---

## 2. 通用方法：如何添加自定义机器人到 RoboSuite

### 2.1 需要修改什么

```
添加自定义机器人到 RoboSuite
├── 1. MuJoCo XML 文件        # 机器人运动链、关节、mesh 几何
├── 2. Python 类              # 继承 ManipulatorModel，指定默认配置
├── 3. Gripper 定义           # 夹爪 XML + Python 类
├── 4. BDDL 任务适配          # 调整区域坐标、物体大小
└── 5. Docker 镜像修改        # 安装 RoboSuite + 自定义机器人
```

### 2.2 接口定义

被测模型需要提供：

| 必填项 | 说明 |
|--------|------|
| 模型路径 | OBS 路径或本地 checkpoint |
| 机器人类型 | 字符串标识（如 `so101`） |
| 评测环境/任务 | 在哪个 benchmark 上评 |

| 可选项 | 默认值 |
|--------|--------|
| 评测 episode 数 | 10（LIBERO/SmolVLA 官方协议，Ch6/Ch7 实测取值） |
| 随机种子 | 7（Ch6/Ch7 实测取值） |
| 额外配置 | 空 |

平台保证：

1. **环境搭建** — 拉取对应 Docker 镜像，安装依赖
2. **评测执行** — 运行评测脚本，收集指标
3. **结果归档** — 上传到 OBS（JSON 报告 + 视频）
4. **标准化输出** — 统一格式的评测报告

---

## 3. 案例：SO101 集成 LIBERO（方案级，尚未实施）

> 先说结论：SO101 集成**没有做**。本节是基于 Ch6/Ch7 实证的可执行方案，不是战报。证据链在 §3.4。

### 3.1 问题

| | 我们的 SmolVLA | LIBERO 环境 |
|--|----------------|-------------|
| 机器人 | SO101（6 DoF） | Franka Panda（7 DoF） |
| 训练数据 | SO101 真机/仿真演示 | LIBERO 仿真演示（RoboSuite） |
| 观测空间 | SO101 关节状态 + SO101 相机图像 | Franka 关节状态 + RoboSuite 相机图像 |
| 动作空间 | SO101 6 维（5 arm + 1 gripper） | Franka 7 维（7 arm + 1 gripper） |

我们的 SmolVLA 模型（`xieyucheng123/so101-smolvla`）在 SO101 数据上训练，**无法直接控制 LIBERO 的 Franka 机器人**（Ch7 §3.2 同款问题，Sawyer 0% 已演示：跨身体评测能跑通，但分数无意义）。

### 3.2 方案

**不是**在 LIBERO 数据上训练 Franka 模型，而是**把 SO101 作为自定义机器人添加到 LIBERO/RoboSuite 中**，然后用我们现有的 SO101 SmolVLA 模型直接评测。

```
旧方案: 在 LIBERO 数据上训练 Franka SmolVLA → 评测 Franka 模型
新方案: 在 LIBERO 中添加 SO101 机器人 → 评测我们自己的 SO101 SmolVLA
```

### 3.3 实施步骤（待办）

```
Week 1: SO101 机器人集成
  ├── 导出 SO101 robot.xml → RoboSuite 格式
  ├── 编写 SO101 gripper (XML + Python)
  ├── 编写 MountedSO101 类 (继承 ManipulatorModel)
  ├── 配置 JOINT_POSITION 控制器
  ├── 适配 BDDL 任务文件（调整区域坐标）
  └── 集成测试

Week 2: 评测与归档
  ├── Zero-shot 评测 LIBERO 3 suites（10eps/task × 30 tasks = 300 eps，约 6h，见 Ch6 §5.3 实测外推）
  ├── 汇总结果 + 可视化
  └── robot-switch 式流水线接入 + 结果归档到 OBS
```

**总工期**: ~2 周（集成工作量） | **V100 评测时间**: 以 Ch6 实测为准（100eps ~2h），不再沿用旧 50eps/task 估算

> **详细实施方案**（代码示例、文件结构、参数配置）见 [Discussion #9](https://github.com/link-seek/so101-sim-pipeline/discussions/9)。

### 3.4 为什么这个方案可信：Ch6/Ch7 已经证明的部分

SO101 集成没做，但方案依赖的每一层机制都已被实测验证：

| 方案依赖 | 实证 | 出处 |
|----------|------|------|
| 评测管线能跑 LIBERO（serve→run→merge+诚实计分） | 47%@100eps，0 harness 错误 | Ch6 §4.3（run `33829389761`） |
| 非默认机器人能构造+跑通全流程 | Sawyer 10/10 eps 跑通，0 错误 | Ch7 §2.4（run `33836694327`） |
| 版本漂移可用文件补丁消化（不改 harness/镜像） | seed/controller/batch 三补丁 | Ch6 §4.3 |
| 一键触发+OBS 归档 | `franka-eval.yml`/`robot-switch.yml` | 本章 §5 |

未知只剩 SO101 自己的 XML/类/控制器/BDDL 适配——这是纯机器人建模工作，不再涉及评测框架风险。

---

## 4. 技术风险（方案级评估）

| 风险 | 缓解措施 | 状态 |
|------|----------|------|
| SO101 工作空间 < Panda，部分 BDDL 任务物体不可达 | 缩小放置范围（×0.7） | 待实施时验证 |
| SO101 6 DoF vs Panda 7 DoF，部分任务需 7 自由度 | 跳过不可解任务 | 待实施时验证 |
| Zero-shot 性能极低 | 预期的——这正是泛化差距的量化（Ch7 Sawyer 0% 已演示"能跑通但无意义"） | 机制已验证 |

---

## 5. 现状与目标的差距（以实测为准）

### 5.1 当前状态（本教程结束时，已实现）

| 维度 | 现状 |
|------|------|
| 触发方式 | GitHub Actions 一键触发（`franka-eval.yml` / `robot-switch.yml`，workflow_dispatch） |
| 环境配置 | 镜像固定（`so101-eval:latest`）+ 文件补丁挂载，ECS 自托管 runner 执行 |
| 结果查看 | OBS 自动归档 + Actions artifacts（`console.log`/`eval_summary.json`/`server.log`），aggregate 诚实计分 |
| 评测实证 | Franka 47%@100eps（run `33829389761`）+ Sawyer 10eps 跑通（run `33836694327`） |
| 机器人支持 | Franka（L0）+ Sawyer（L1 实测）+ RoboSuite 内置（L1 机制就绪）；SO101（L2 未实施） |

### 5.2 目标状态（自动化评测平台，远期）

| 维度 | 目标 | 差距 |
|------|------|------|
| 触发方式 | Web UI / 接口调用 | 只有 Actions 手动触发，无 UI |
| 环境配置 | 自动匹配机器人类型 → 镜像 | 只有单镜像 + 手工补丁 |
| 结果查看 | 可视化报告 + 结果数据库 | 只有 JSON 日志 + 手工看表 |
| 多模型对比 | 批量评测 + 对比表 | 需逐次触发、手工汇总 |
| 机器人支持 | SO101 + Franka + RoboSuite 全家 | SO101 未集成 |

### 5.3 路线图

```
Phase 1: 原型验证（本教程 Ch1-Ch5）
  ├── ✅ Docker 镜像封装
  ├── ✅ 容器内跑通完整流程
  └── ✅ 结果归档到 OBS

Phase 2: 机器人扩展（Ch6-Ch8，本教程实测部分）
  ├── ✅ Franka 基线 47%@100eps（Ch6，run 33829389761）
  ├── ✅ Sawyer L1 机制+演示跑通（Ch7，run 33836694327）
  ├── ✅ 一键触发流水线（franka-eval.yml / robot-switch.yml）
  └── ⏳ SO101 自定义集成（Ch8，方案就绪、未实施）

Phase 3: 自动化（下一步）
  ├── 批量评测（多模型 × 多 suite 矩阵）
  ├── 自动环境匹配（robot → 镜像/补丁）
  └── 可视化报告

Phase 4: 平台化（远期）
  ├── Web UI
  ├── 多团队共享
  ├── 评测结果数据库
  └── 模型-环境兼容性自动检测
```

---

## 6. 总结

本教程从"为什么仿真训练"开始，经过基础设施搭建、PPO/VLA 训练、Debug 实战、评测方法论，最终到达机器人扩展和评测平台愿景。

**核心收获**：

| 章节 | 核心收获 |
|------|----------|
| Ch0 | 仿真训练的价值和 Sim-to-Real Gap |
| Ch1 | Docker 镜像 + 云服务器 = 可复现的训练环境 |
| Ch2 | PPO 验证环境可学性（100% 成功） |
| Ch3 | SmolVLA 验证视觉方案可行性（47% 成功） |
| Ch4 | Debug 方法论：先分层定位，再针对性修复 |
| Ch5 | 评测方法论：统计显著性 + 指标设计 + 踩坑驱动 |
| Ch6 | Franka 基线 47%@100eps + 三处 1.5 兼容补丁（实测） |
| Ch7 | L1 扩展真实机制（文件补丁+挂载）+ Sawyer 跑通（实测） |
| Ch8 | L2 方案就绪（SO101 未实施）+ 平台现状盘点（本章） |

**下一步**：实施 SO101 集成（§3.3 待办），向自动化评测平台演进（§5.2 差距表）。

---

## 思考题

1. **为什么不能直接把 SO101 模型放到 LIBERO 的 Franka 环境里跑？**  
   提示：机器人不同——SO101（6 DoF）vs Franka（7 DoF），关节定义、观测空间、动作语义都不匹配。详见 [Discussion #9](https://github.com/link-seek/so101-sim-pipeline/discussions/9)。

2. **在 LIBERO 中添加 SO101 机器人后，评测的是哪个模型？**  
   提示：是我们现有的 `xieyucheng123/so101-smolvla`，在 SO101 数据上训练的 SO101 模型——不是新训练的 Franka 模型。

3. **Zero-shot 评测结果很低，有意义吗？**  
   提示：有。它量化了「单任务训练 → 多任务泛化」的差距，是后续优化的 baseline。0% 和 20% 的指导价值完全不同。

4. **从本教程到自动化评测平台，最关键的下一步是什么？**  
   提示：流水线触发已经有了（`franka-eval.yml`/`robot-switch.yml` 一键触发 + OBS 归档）。下一步是批量评测矩阵（多模型 × 多 suite 自动跑 + 对比表）和可视化报告，见 §5.2 差距表。

---

> **上一章**：[Ch7 RoboSuite 机器人扩展](so101-tutorial-ch7-robosuite-extension.md) | **附录**：[环境速查](so101-tutorial-appendix.md)
