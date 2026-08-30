# Ch7：算法选型与对比

> SO101 仿真评测教程 · 第七章
> 实战项目：[link-seek/so101-sim-pipeline](https://github.com/link-seek/so101-sim-pipeline)
> 关联 Discussion：[#3 算法选型讨论](https://github.com/link-seek/so101-sim-pipeline/discussions/3)

---

## 1. 本项目用的两个算法

### 1.1 PPO（Proximal Policy Optimization）

- **GitHub Workflow**: [`ppo-pipeline.yml`](https://github.com/link-seek/so101-sim-pipeline/blob/main/.github/workflows/ppo-pipeline.yml)
- **Docker 镜像**: [`ghcr.io/link-seek/so101-ppo:latest`](https://github.com/link-seek/so101-sim/pkgs/container/so101-ppo)
- **成功率**: 98-100%（在 pick-cube 任务上）
- **训练时间**: ~86 分钟
- **特点**: 不需要演示数据，通过自探索学习

```
PPO 训练流程:
环境交互 → 采集经验 → 计算优势函数 → 策略梯度更新 → 重复
```

### 1.2 SmolVLA（LeRobot VLA）

- **GitHub Workflow**: [`so101-mujoco-pipeline.yml`](https://github.com/link-seek/so101-sim-pipeline/blob/main/.github/workflows/so101-mujoco-pipeline.yml)
- **Docker 镜像**: [`ghcr.io/link-seek/so101-mujoco:latest`](https://github.com/link-seek/so101-sim/pkgs/container/so101-mujoco)
- **成功率**: 47%（目前，社区最高 86-90%）
- **训练时间**: ~10-14 小时
- **特点**: 需要演示数据，但有视觉能力

```
SmolVLA 训练流程:
采集演示数据 → 数据预处理 → 行为克隆训练 → 评估 → 迁移
```

---

## 2. 为什么选这两个算法

### 2.1 PPO 的优势

| 优势 | 说明 |
|------|------|
| **不需要数据** | 通过与环境交互自探索学习 |
| **训练快** | 86 分钟就能达到 98-100% |
| **简单易用** | 配置少，开箱即用 |
| **成功率高** | 在简单任务上接近 100% |

### 2.2 SmolVLA 的优势

| 优势 | 说明 |
|------|------|
| **有视觉** | 能看到物体、理解场景 |
| **可迁移** | 有迁移到真机的潜力 |
| **轻量级** | 450M 参数，V100 16GB 可运行 |
| **社区验证** | 多个项目达到 86-90% 成功率 |

### 2.3 为什么不用其他算法

| 算法 | 不选的原因 |
|------|------------|
| **RT-2** | 55B 参数，需要 TPU/H100，本项目硬件跑不了 |
| **OpenVLA** | 7B 参数，需要 A100 80GB，V100 16GB 装不下 |
| **Diffusion Policy** | LeRobot 支持，但不如 SmolVLA 的视觉能力强 |
| **ACT** | 有实验 notebook，但最终没用 |

---

## 3. PPO vs SmolVLA 详细对比

> VLA 模型内部对比（SmolVLA vs RT-2 vs OpenVLA）见 [Ch3 §0](so101-tutorial-ch3-vla-intro.md#0-vla-模型全景)。

### 3.1 核心维度对比

| 维度 | PPO | SmolVLA |
|------|-----|---------|
| **学习信号** | 奖励（环境） | 演示数据（行为克隆） |
| **输入** | 关节状态向量 | 图像 + 语言指令 |
| **输出** | 关节增量（向量） | 关节角度（向量） |
| **需要数据** | ❌ 不需要 | ✅ 需要演示数据 |
| **能迁移真机** | ❌ MLP 无视觉 | ✅ VLA 有视觉 |
| **训练时间** | ~86 分钟 | ~10-14 小时 |
| **成功率** | 98-100% | 47%（目前） |
| **学习范式** | 在线策略 RL（自探索） | 离线策略 BC（模仿） |

### 3.2 什么时候选 PPO

```
选 PPO 的条件:
├── 任务简单（pick-cube, push）
├── 有模拟环境
├── 不需要迁移到真机
├── 需要快速验证想法
└── 不想收集演示数据
```

### 3.3 什么时候选 SmolVLA

```
选 SmolVLA 的条件:
├── 任务复杂（多步操作）
├── 需要视觉理解
├── 计划迁移到真机
├── 有演示数据
└── 需要语言指令控制
```

---

## 4. 实战案例对比

### 4.1 PPO 训练 pick-cube

```yaml
# GitHub Actions Workflow: ppo-pipeline.yml
- name: 训练 PPO
  run: |
    python train_ppo.py \
      --task pick-cube \
      --max-steps 10000 \
      --learning-rate 3e-4

# 结果: 86 分钟, 成功率 98-100%
```

### 4.2 SmolVLA 训练 pick-cube

```yaml
# GitHub Actions Workflow: so101-mujoco-pipeline.yml
- name: 训练 SmolVLA
  run: |
    python train_smolvla.py \
      --dataset dobri420/pick-cube-so101-sim \
      --epochs 15000 \
      --batch-size 32

# 结果: 10-14 小时, 成功率 47%
```

### 4.3 成功率对比

| 算法 | pick-cube | 推箱子 | 多步操作 |
|------|-----------|--------|----------|
| **PPO** | 98-100% | 85% | 不适用 |
| **SmolVLA** | 47% | 60-80%（社区） | 需要更多数据 |

---

## 5. 选型决策树

```
开始选型
│
├── 任务是否需要视觉理解？
│   ├── 是 → SmolVLA
│   └── 否 → 继续
│
├── 是否有演示数据？
│   ├── 是 → SmolVLA（可训练）
│   └── 否 → PPO（自探索）
│
├── 是否计划迁移到真机？
│   ├── 是 → SmolVLA（有视觉）
│   └── 否 → PPO（更简单）
│
├── 训练时间限制？
│   ├── < 2 小时 → PPO
│   └── > 10 小时 → SmolVLA
│
└── 硬件限制？
    ├── V100 16GB → SmolVLA（唯一选择）
    └── A100 80GB → 可选 OpenVLA
```

---

## 6. 社区案例

### 6.1 成功案例

| 项目 | 算法 | 成功率 | 关键技术 |
|------|------|--------|----------|
| [Sa74ll/smolvla](https://github.com/Sa74ll/smolvla) | SmolVLA | 87.66% | 40 eps + 15k steps + hierarchical sampling |
| [MSSergeev/so101-lab](https://github.com/MSSergeev/so101-lab) | SmolVLA + IQL | 86-88% | 加权行为克隆 |
| [MSSergeev/so101-lab](https://github.com/MSSergeev/so101-lab) | SmolVLA + PPO | 90% | PPO 微调 |
| [ggand0/vla-so101](https://github.com/ggand0/vla-so101) | SmolVLA | 60-80% | 75 eps + 20k steps |

### 6.2 本项目现状

| 指标 | PPO | SmolVLA |
|------|-----|---------|
| **成功率** | 98-100% | 47% |
| **训练时间** | 86 分钟 | 10-14 小时 |
| **GitHub 运行次数** | 3 次 | 6 次 |
| **下一步** | 完成 | 优化训练数据 |

---

## 7. 总结

### 7.1 选型建议

| 场景 | 推荐算法 | 原因 |
|------|----------|------|
| **快速验证** | PPO | 86 分钟见结果 |
| **真机迁移** | SmolVLA | 有视觉能力 |
| **简单任务** | PPO | 成功率高 |
| **复杂任务** | SmolVLA | 视觉理解强 |
| **数据充足** | SmolVLA | 可充分利用数据 |
| **数据不足** | PPO | 不需要数据 |

### 7.2 本项目路线图

```
短期 (1-2 周):
├── 优化 SmolVLA 训练数据（目标: 60-70%）
├── 完成 PPO 训练（目标: 100%）
└── 修复 runner 离线问题

中期 (1-2 月):
├── SmolVLA 迁移到真机
├── 多任务训练
└── PPO + VLA 混合方案

长期 (3+ 月):
├── 真机部署
├── 多机器人协作
└── 开源社区贡献
```

---

## 8. 参考链接

- [PPO 原论文](https://arxiv.org/abs/1707.06347)
- [SmolVLA 论文](https://huggingface.co/papers/2501.14513)
- [LeRobot 文档](https://huggingface.co/docs/lerobot)
- [SO-101 机器人](https://github.com/TheRobotStudio/SO-101)
- [本项目 GitHub](https://github.com/link-seek/so101-sim-pipeline)
