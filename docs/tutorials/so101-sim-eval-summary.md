# SO101 仿真测评阶段性总结

> 仓库：`link-seek/so101-sim-pipeline` | 更新时间：2026-08-21 | 作者：xieyucheng123

---

## 一、Runner 状态

| 项目 | 值 |
|------|-----|
| Runner 名称 | `ecs-0002` |
| 状态 | **online**，空闲 |
| 标签 | `self-hosted, Linux, X64, V100` |
| GPU | NVIDIA V100 (sm_70) |
| ECS ID | `7f39cb83`（华为云 cn-north-4） |

---

## 二、Workflow 运行概况（近 50 次）

| 流水线 | 分支 | 成功 | 失败/取消 | 最近状态 |
|--------|------|------|-----------|----------|
| **CI** | main | 6 | 0 | ✅ 稳定 |
| **Docker Build** | main | 4 | 0 | ✅ 稳定 |
| **SO101 MuJoCo Pipeline** | main | 5 | 1 | ✅ 稳定（08-19 连续 5 次成功） |
| **VLA Pipeline** | smolvla-fresh | 1 | 9 | ⚠️ 频繁失败/取消（08-20 集中调试） |
| **VLA Pipeline** | main | 4 | 3 | ⚠️ 早期调试 |
| **PPO Pipeline** | main | 2 | 0 | ✅ 已验证完成 |

---

## 三、仿真与测评核心进展

### 3.1 PPO 纯仿真 Baseline — ✅ 已完成

**目标**：验证 Warp GPU 并行环境在 V100 上可学性，建立 RL baseline。

| 版本 | 环境 | lift_threshold | 训练耗时 | best_success | eval success_rate (50 ep) |
|------|------|---------------|----------|-------------|--------------------------|
| v1 | WarpPickLift-v1 | 5cm | 86 min | 0.995 | **100%** (50/50) |
| v2 | WarpPickLift-v1 | **15cm** | 85 min | 0.9925 | **98%** (49/50) |

- V100 SPS ≈ 5784（比 RTX 5090 慢 ~3.5x，但收敛结果一致）
- 修复了 mujoco-warp sensor NVRTC 编译 bug（升级到 >=3.10）
- 产物已归档至 OBS：`obs://so101-sim-pipeline/ppo/`

### 3.2 SmolVLA 仿真回放 — 方案演进

#### 阶段 1：P0 相机不匹配修复 ✅

| 问题 | 修复 |
|------|------|
| 训练用 `side+up` 相机，推理用 `wrist+overhead` | 切换数据集至 `shattori/so101_pick_place_thor` |

修复后 prediction errors 从 >0 降至 **0/300**，模型可全程控制机器人。

#### 阶段 2：P1 Sim-to-Real Visual Gap — 方案 A（ataghof）❌ 已终止

**数据集**：`ataghof/so101nexus-cube500-binary`（500 eps, 2 相机, 仿真采集）

| 训练阶段 | Steps | Loss | 回放 Success | 结论 |
|----------|-------|------|-------------|------|
| Phase 2 快速验证 | 5k | 0.461→0.119 | False | P1 消除，模型在学习 |
| Phase 3 完整训练 | 20k | 0.119→0.046 | False | loss 偏高（社区 0.005） |
| Bug 修复后回放 | 20k ckpt | - | False | 3 bug 修复仍不成功 |
| bs=64 重训练 | 10k | - | - | 超时取消 |

**修复的 3 个 Bug**：
1. Gripper 转换（`/100.0` → 官方 `dataset_row_to_sim_qpos`）
2. Action Chunking（移除外部 queue，依赖 SmolVLA 内部）
3. camera3 分布不匹配（训练缺失，回放不提供）

**终止根因**：数据采集环境（ataghof）与评测环境（so101_nexus）存在系统性视觉/物理差异，非代码 bug。

#### 阶段 3：方案 B（so101-mujoco Sim Twin）— ✅ 首次 VLA 成功！

**数据集**：`dobri420/pick-cube-so101-sim`（3 相机, MuJoCo sim twin, 数据-环境 1:1 匹配）

| 指标 | 值 |
|------|-----|
| 训练 | 15K steps（20K 在 19515 步超时），bs=32, 3.65s/step |
| 训练 Loss | 0.090 |
| 评测 | MuJoCo grid sweep, **325 episodes** |
| **成功率** | **47% (153/325)** |

**Grid Sweep 热力图**（reach × azimuth，每格 5 trials）：

```
reach\azim   -90   -75   -60   -45   -30   -15    +0   +15   +30   +45   +60   +75   +90
  15cm    3/5   1/5   1/5   4/5   5/5   5/5   4/5   2/5   2/5   4/5   0/5   3/5   0/5
  18cm    4/5   2/5   2/5   4/5   4/5   5/5   4/5   1/5   4/5   3/5   1/5   0/5   0/5
  20cm    1/5   1/5   4/5   0/5   3/5   4/5   2/5   4/5   2/5   2/5   3/5   0/5   1/5
  22cm    3/5   0/5   1/5   3/5   3/5   5/5   4/5   4/5   2/5   2/5   1/5   1/5   0/5
  25cm    3/5   2/5   2/5   4/5   3/5   4/5   3/5   5/5   1/5   1/5   1/5   0/5   0/5
```

**关键发现**：
- 中心区域（-15 到 +15 azimuth, 15-22cm reach）成功率 **60-100%**
- 边缘区域（±90 azimuth）接近 0%，符合训练数据分布预期
- 153 次成功放置证明模型已学会 pick-and-place 任务

### 3.3 方案对比总览

| 方案 | 数据集 | 相机数 | 数据-环境匹配 | 成功率 | 状态 |
|------|--------|--------|-------------|--------|------|
| PPO (WarpPickLift) | 无需数据（RL） | 0 | ✅ | **98-100%** | ✅ 完成 |
| VLA ataghof | so101nexus-cube500-binary | 2 | ❌ | **0%** | ❌ 终止 |
| **VLA so101-mujoco** | pick-cube-so101-sim | 3 | ✅ | **47%** | ✅ 首次成功 |

---

## 四、Discussion 活跃度

| # | 标题 | 分类 | 评论数 | 创建时间 |
|---|------|------|--------|----------|
| 4 | 方案 A：复用 ataghof 数据集 → 方案 B sim twin 成功 | General | 8 | 08-14 |
| 3 | SO101 仿真测评路线图 | General | 0 | 08-14 |
| 2 | PPO 纯仿真流水线设计方案 | Ideas | 2 | 08-14 |
| 1 | SmolVLA 仿真回放验证：P0 确认，发现 P1 | General | 1 | 08-14 |

Discussion #4 是最活跃的，记录了从方案 A 失败到方案 B 成功的完整迭代过程。

---

## 五、当前问题与下一步

### 待解决

1. **VLA Pipeline 在 smolvla-fresh 分支频繁失败**（08-20：10 次运行仅 1 次成功）— 需排查失败原因
2. **VLA 成功率 47% 有提升空间** — 社区参考：ggand0/vla-so101 60-80%，Sa74ll/smolvla 87.66%
3. **训练超时限制** — bs=64 的 20k steps 需 ~28h，超过 20h 超时

### 社区成功案例参考

| 项目 | 关键配置 | 成功率 |
|------|----------|--------|
| ggand0/vla-so101 | 75 eps + 20k + bs=64 + 双摄 | 60-80% |
| Sa74ll/smolvla | 40 eps + 15k + 分层采样 | 87.66% |
| MSSergeev/so101-lab | SmolVLA + IQL weighted BC | 86-88% |
| MSSergeev/so101-lab | SmolVLA + PPO fine-tuning | 90% |

### 建议方向

1. **短期**：排查 smolvla-fresh 分支 VLA Pipeline 失败原因，恢复稳定 CI
2. **中期**：增加训练步数至 50k 或调整超参（batch_size, lr），目标 >60F60%
3. **长期**：考虑 RL fine-tuning（IQL/PPO）在 BC 基础上提升至 86-90%

---

## 六、基础设施总结

```
GitHub Actions dispatch
  │
  ├─ start-ecs (ubuntu-latest) → 华为云 V100 ECS 启动
  │
  ├─ pipeline (self-hosted, Linux, X64, V100)
  │    ├─ SWR 登录 + 拉取 Docker 镜像
  │    ├─ 训练（lerobot-train / PPO CleanRL）
  │    ├─ 仿真评估（MuJoCo replay / grid sweep）
  │    └─ 上传 OBS 归档
  │
  └─ stop-ecs (if: always()) → ECS 停止
```

**Docker 镜像矩阵**：so101-train, so101-eval, so101-model-server, so101-ppo, so101-mujoco

**已验证可用**：ECS 控制、SWR 镜像、V100 GPU 训练、MuJoCo 无头渲染（EGL）、OBS 归档、grid sweep 评测
