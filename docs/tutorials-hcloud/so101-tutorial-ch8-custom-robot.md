# Ch8：自定义机器人扩展 + SO101 落地

> SO101 仿真评测教程 · 第八章
> 详细方案：[Discussion #9 — SO101 跑 LIBERO 评测：现状分析与落地方案](https://github.com/link-seek/so101-sim-pipeline/discussions/9)

---

## 1. 镜像级扩展：当配置不够用

Ch7 展示了配置级扩展——RoboSuite 已有 12 种机器人，改 YAML 就能换。但 SO101 不在其中。

**镜像级扩展**适用于：目标机器人不在 RoboSuite 中，需要修改 Docker 镜像添加自定义机器人定义。

| 扩展级别 | 方法 | 工作量 | 适用场景 |
|----------|------|--------|----------|
| 配置级（Ch7） | 改 YAML 参数 | 几分钟 | RoboSuite 已有的 12 种机器人 |
| 镜像级（本章） | 改 Docker 镜像 | 数天 | RoboSuite 没有的自定义机器人 |

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
| 模型路径 | HF 仓库或本地 checkpoint |
| 机器人类型 | 字符串标识（如 `so101`） |
| 评测环境/任务 | 在哪个 benchmark 上评 |

| 可选项 | 默认值 |
|--------|--------|
| 评测 episode 数 | 50 |
| 随机种子 | 42 |
| 额外配置 | 空 |

平台保证：

1. **环境搭建** — 拉取对应 Docker 镜像，安装依赖
2. **评测执行** — 运行评测脚本，收集指标
3. **结果归档** — 上传到 OBS（JSON 报告 + 视频）
4. **标准化输出** — 统一格式的评测报告

---

## 3. 案例：SO101 集成 LIBERO

### 3.1 问题

| | 我们的 SmolVLA | LIBERO 环境 |
|--|----------------|-------------|
| 机器人 | SO101（6 DoF） | Franka Panda（7 DoF） |
| 训练数据 | SO101 真机/仿真演示 | LIBERO 仿真演示（RoboSuite） |
| 观测空间 | SO101 关节状态 + SO101 相机图像 | Franka 关节状态 + RoboSuite 相机图像 |
| 动作空间 | SO101 6 维（5 arm + 1 gripper） | Franka 7 维（7 arm + 1 gripper） |

我们的 SmolVLA 模型（`xieyucheng123/so101-smolvla`）在 SO101 数据上训练，**无法直接控制 LIBERO 的 Franka 机器人**。

### 3.2 方案

**不是**在 LIBERO 数据上训练 Franka 模型，而是**把 SO101 作为自定义机器人添加到 LIBERO/RoboSuite 中**，然后用我们现有的 SO101 SmolVLA 模型直接评测。

```
旧方案: 在 LIBERO 数据上训练 Franka SmolVLA → 评测 Franka 模型
新方案: 在 LIBERO 中添加 SO101 机器人 → 评测我们自己的 SO101 SmolVLA
```

### 3.3 实施步骤

```
Week 1: SO101 机器人集成
  ├── 导出 SO101 robot.xml → RoboSuite 格式
  ├── 编写 SO101 gripper (XML + Python)
  ├── 编写 MountedSO101 类 (继承 ManipulatorModel)
  ├── 配置 JOINT_POSITION 控制器
  ├── 适配 BDDL 任务文件（调整区域坐标）
  └── 集成测试

Week 2: 评测与归档
  ├── Zero-shot 评测 LIBERO 3 suites (1,500 episodes, ~8h)
  ├── Zero-shot 评测 LIBERO-PRO 5 suites (2,500 episodes, ~14h)
  ├── 汇总结果 + 可视化
  └── CI 集成 + 结果归档到 OBS
```

**总工期**: ~2 周 | **V100 评测时间**: ~22h | **成本**: ~¥660

> **详细实施方案**（代码示例、文件结构、参数配置）见 [Discussion #9](https://github.com/link-seek/so101-sim-pipeline/discussions/9)。

---

## 4. 技术风险

| 风险 | 缓解措施 |
|------|----------|
| SO101 工作空间 < Panda，部分 BDDL 任务物体不可达 | 缩小放置范围（×0.7） |
| SO101 6 DoF vs Panda 7 DoF，部分任务需 7 自由度 | 跳过不可解任务 |
| Zero-shot 性能极低 | 预期的——这正是泛化差距的量化 |

---

## 5. 现状与目标的差距

### 5.1 当前状态

| 维度 | 现状 |
|------|------|
| 触发方式 | 手动 `docker run` |
| 环境配置 | 手动指定镜像和参数 |
| 结果查看 | SSH 进容器看日志 |
| 多模型对比 | 手动跑多个，手动整理 |
| 机器人支持 | SO101（自定义）+ Franka（原生） |

### 5.2 目标状态

| 维度 | 目标 |
|------|------|
| 触发方式 | 一键触发（流水线 / Web UI） |
| 环境配置 | 自动匹配机器人类型 → 镜像 |
| 结果查看 | OBS 自动归档 + 可视化报告 |
| 多模型对比 | 批量评测 + 对比表 |
| 机器人支持 | SO101 + Franka + 12 RoboSuite 机器人 |

### 5.3 路线图

```
Phase 1: 原型验证（本教程 Ch1-Ch5）
  ├── ✅ Docker 镜像封装
  ├── ✅ 手动 docker run 跑通完整流程
  └── ✅ 结果归档到 OBS

Phase 2: 机器人扩展（Ch6-Ch8）
  ├── ✅ Franka 开箱即用（Ch6）
  ├── ✅ RoboSuite 12 机器人配置级扩展（Ch7）
  └── 🔧 SO101 自定义集成（Ch8，进行中）

Phase 3: 自动化（下一步）
  ├── 流水线集成（一键触发）
  ├── 自动环境匹配
  ├── 批量评测
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
| Ch6 | Franka 开箱即用 = 平台成熟度基线 |
| Ch7 | 配置级扩展 = RoboSuite 12 机器人 |
| Ch8 | 镜像级扩展 = 自定义机器人（SO101） |

**下一步**：完成 SO101 集成 LIBERO，向自动化评测平台演进。

---

## 思考题

1. **为什么不能直接把 SO101 模型放到 LIBERO 的 Franka 环境里跑？**  
   提示：机器人不同——SO101（6 DoF）vs Franka（7 DoF），关节定义、观测空间、动作语义都不匹配。详见 [Discussion #9](https://github.com/link-seek/so101-sim-pipeline/discussions/9)。

2. **在 LIBERO 中添加 SO101 机器人后，评测的是哪个模型？**  
   提示：是我们现有的 `xieyucheng123/so101-smolvla`，在 SO101 数据上训练的 SO101 模型——不是新训练的 Franka 模型。

3. **Zero-shot 评测结果很低，有意义吗？**  
   提示：有。它量化了「单任务训练 → 多任务泛化」的差距，是后续优化的 baseline。0% 和 20% 的指导价值完全不同。

4. **从本教程到自动化评测平台，最关键的一步是什么？**  
   提示：把手动 `docker run` 变成流水线触发。这需要定义清楚被测对象的接口（模型路径、机器人类型、评测任务），平台自动完成环境搭建、评测执行、结果归档。

---

> **上一章**：[Ch7 RoboSuite 机器人扩展](so101-tutorial-ch7-robosuite-extension.md) | **附录**：[环境速查](so101-tutorial-appendix.md)
