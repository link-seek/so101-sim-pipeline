# 序章：为什么要在仿真中训练和评测机器人

> SO101 仿真评测教程 · 序章  
> 实战项目：[link-seek/so101-sim-pipeline](https://github.com/link-seek/so101-sim-pipeline)

---

## 1. 真机训练的困境

如果你亲手操作过机器人，一定对这些场景不陌生：

- **硬件损耗**：SO-101 的伺服电机在连续遥操作 2 小时后发热严重，齿轮箱磨损导致关节精度下降
- **采集耗时**：录制 100 个 pick-and-place 演示需要一整天，还得人工筛选失败 episode
- **安全围栏**：机器人突然甩动可能打碎桌面物体、损坏自身，必须有人值守
- **不可复现**：同一个策略今天成功明天失败，光照、物体位置、电池电量都在变

这些问题的本质是：**真机训练的成本高、速度慢、不可控**。

NVIDIA 在 SO-101 Workshop 中给出过一组数据：

| 方式 | 采集 100 demos | 训练 20k steps | 评估 50 episodes |
|------|---------------|---------------|-----------------|
| 真机 | ~8 小时 | ~14 小时（需占用机器人） | ~2 小时 |
| 仿真 | ~30 分钟（脚本自动） | ~14 小时（GPU 并行） | ~30 秒（1024 envs 并行） |

仿真训练快了一个数量级，但代价是什么？

---

## 2. 仿真的核心矛盾：Sim-to-Real Gap

仿真训练快、便宜、可复现，但有一个致命问题：

> **仿真里学到的策略，放到真机上可能完全不能用。**

这就是 **Sim-to-Real Gap**（仿真到现实的鸿沟）。它的来源包括：

| 差异类型 | 仿真 | 真机 |
|----------|------|------|
| 光照 | 点光源 / 环境光 | 自然光 + 多光源 + 阴影 |
| 纹理 | 简单颜色 / 程序化纹理 | 真实材质（金属反光、塑料磨砂） |
| 背景 | 纯色地面 | 复杂桌面、杂物 |
| 物理 | 理想化接触模型 | 摩擦、变形、温度敏感 |
| 传感器 | 无噪声 / 固定延迟 | 噪声、延迟、丢包 |

在我们的项目中，这个矛盾以一个具体的形式出现：

> 训练数据来自真机照片，但评测在 MuJoCo 仿真中渲染。模型在真机图片上学到的视觉特征，应用到仿真画面时直接失效。

这就是我们在 Discussion #1 中发现的 **P1 Sim-to-Real Visual Gap**。

---

## 3. 两条路线：纯仿真验证 vs Sim-to-Sim 验证

面对 Sim-to-Real Gap，社区发展出了两条务实路线：

### 路线 A：纯仿真验证（PPO）

```
仿真环境自探索（RL）→ 仿真环境评估
```

- 不需要训练数据，策略通过 reward 信号自己探索
- 训练和评估在同一个仿真环境，不存在 domain gap
- **代价**：学到的策略是 MLP（非视觉），不能直接迁移到真机
- **价值**：验证环境的可学性，建立性能 baseline

### 路线 B：Sim-to-Sim 验证（VLA）

```
仿真数据采集 → 仿真训练 → 仿真评估
```

- 用仿真遥操作 / scripted expert 采集演示数据
- 训练 VLA 模型（图像 → 动作），在仿真中评估
- **关键**：训练数据和评估环境在同一个仿真，消除视觉域差异
- **价值**：验证 VLA 方案的可行性，为 sim-to-real 迈出第一步

### 我们项目的实践

| 路线 | 方法 | 成功率 | 状态 |
|------|------|--------|------|
| A（PPO） | WarpPickLift-v1, 30M steps | **98-100%** | ✅ 完成 |
| B（VLA） | pick-cube-so101-sim, 15k steps | **47%** | ✅ 首次成功 |

PPO 路线快速成功（86 分钟训练），验证了基础设施可用。VLA 路线经历了 5 轮失败迭代，最终通过 sim twin 数据集实现首次成功。

---

## 4. SO101 项目背景

### LeRobot 生态

[LeRobot](https://github.com/huggingface/lerobot) 是 Hugging Face 的机器人学习框架，提供：

- 统一的数据集格式（LeRobot v3.0）
- 预训练 VLA 模型（SmolVLA、ACT 等）
- 训练脚本（`lerobot-train`）
- SO-100 / SO-101 机器人支持

### so101_nexus

[so101_nexus](https://github.com/johnsutor/so101_nexus) 是 SO-101 的仿真环境包：

- MuJoCo 物理后端 + EGL 无头渲染
- Warp GPU 并行环境（1024 envs 同时 step）
- 内置 PickAndPlace / PickLift 任务
- 官方 PPO baseline（`ppo_warp.py`）

### SmolVLA

SmolVLA 是 LeRobot 的轻量 VLA 模型：

- 基于 SmolVLM（视觉-语言基础模型）
- Action Head 输出 6-DOF 关节角度
- Action Chunking：每次推理预测 50 个未来 action
- 预训练 base 模型可 fine-tune

### 我们的流水线

`so101-sim-pipeline` 把上述组件串成 CI/CD：

```
GitHub Actions 触发 → 华为云 V100 ECS 启动 → Docker 训练 → MuJoCo 评估 → OBS 归档 → ECS 停止
```

---

## 5. 本教程的承诺

这不是一篇理论综述，而是一个**真实项目的完整实战记录**。

你将看到：

1. **PPO 从 0 到 100%**：86 分钟训练，50 episodes 全部成功
2. **VLA 从 0% 到 47%**：5 轮失败迭代，3 个 bug 诊断，1 次方案终止，最终 sim twin 成功
3. **真实的踩坑过程**：相机不匹配、gripper 转换 bug、action chunking 误解、数据-环境不匹配
4. **可复现的代码**：所有 workflow、脚本、Dockerfile 都在公开仓库中

### 如何使用本教程

```bash
# 克隆实战项目
git clone https://github.com/link-seek/so101-sim-pipeline
cd so101-sim-pipeline

# 浏览 4 个 Discussion 了解项目演进
gh api graphql -f query='query { repository(owner: "link-seek", name: "so101-sim-pipeline") { discussions(first: 10) { nodes { number title } } } }'

# 查看流水线运行历史
gh run list --limit 20
```

### 教程结构

| 章节 | 主题 | 核心实战 |
|------|------|----------|
| 序章 | 为什么仿真训练 | （本文） |
| Ch1 | 仿真与评测技术框架 | 知识框架 + 流水线详解 |
| Ch2 | 基础设施搭建 | ECS + Docker + Runner |
| Ch3 | PPO 纯仿真 Baseline | 100% 成功的最简路径 |
| Ch4 | SmolVLA 仿真训练 | 数据集选择 + 训练 + 回放 |
| Ch5 | Debug 实战 | 从 0% 到 47% 的完整调试旅程 |
| Ch6 | 评测方法论 | Grid sweep + 指标设计 |
| Ch7 | 优化进阶 | 社区生态 + RL fine-tuning |
| 附录 | 环境速查 | 命令 + 配置 + 数据集对比 |

---

## 思考题

1. **如果你的最终目标是真机部署，纯仿真验证（PPO）有价值吗？**  
   提示：想想 baseline 的意义和环境可学性验证。

2. **Sim-to-Sim 验证消除了视觉域差异，但物理域差异呢？**  
   提示：MuJoCo 的接触模型和真实接触有什么不同？

3. **为什么我们的项目同时走 PPO 和 VLA 两条路线？**  
   提示：它们验证的东西不同，互为补充。

---

> **下一章**：[Ch1 仿真与评测技术框架](so101-tutorial-ch1-framework.md)
