# Ch6：优化进阶与社区生态

> SO101 仿真评测教程 · 第六章  
> 实战项目：[link-seek/so101-sim-pipeline](https://github.com/link-seek/so101-sim-pipeline)

---

## 1. 当前状态与提升空间

我们已实现 VLA 47% 成功率，但社区有更高的结果。差距在哪里？

| 方案 | 成功率 | 关键配置 |
|------|--------|----------|
| **我们** | **47%** | 15k steps, bs=32, 3 相机, sim twin |
| ggand0/vla-so101 | 60-80% | 20k steps, bs=64, 双摄, 75 eps |
| Sa74ll/smolvla | 87.66% | 15k steps, 40 eps, 分层采样 |
| MSSergeev/so101-lab (IQL) | 86-88% | BC + IQL advantage-weighted |
| MSSergeev/so101-lab (PPO) | 90% | BC + PPO fine-tuning |

---

## 2. 优化方向

### 2.1 训练超参数优化

| 参数 | 当前 | 目标 | 预期效果 | 挑战 |
|------|------|------|----------|------|
| batch_size | 32 | 64 | 更稳定梯度估计 | 显存 14GB，训练时间翻倍 |
| steps | 15k | 50k | 更充分学习 | 超时（需拆分或续训） |
| learning_rate | 1e-4 | 调参 | 更快收敛 | 需实验 |
| save_freq | 5000 | 2000 | 更细粒度 checkpoint | 存储空间 |

### 2.2 超时问题解决方案

batch_size=64 的 20k steps 需 ~28h，超过 GitHub Actions 限制：

**方案 A：拆分训练阶段**

```yaml
# 第一阶段：0-10k
gh workflow run so101-mujoco-pipeline.yml \
  -f steps1=10000 -f checkpoint_resume=""

# 第二阶段：10k-20k（从 checkpoint 续训）
gh workflow run so101-mujoco-pipeline.yml \
  -f steps2=10000 -f checkpoint_resume="obs://.../checkpoints/010000/"
```

**方案 B：nohup 后台运行**

```bash
# 在 ECS 上直接跑，不受 GitHub Actions 超时限制
nohup docker run --gpus all ... train_smolvla.py --steps=50000 &
```

**方案 C：减小 batch_size + 增加步数**

bs=32 的 50k steps 需 ~50h，但可以拆成 5 个 10k 阶段。

### 2.3 数据增强

#### DART 噪声注入

在执行演示动作时加关节抖动，但记录干净指令。模型学到恢复行为，提高鲁棒性。

```python
# DART: 执行时加噪声，记录干净 action
clean_action = dataset_action
noisy_action = clean_action + np.random.normal(0, 0.01, action.shape)
env.step(noisy_action)  # 执行带噪声
dataset.record(clean_action)  # 记录干净
```

#### 分层采样

Sa74ll/smolvla 用 40 episodes + 分层采样达到 87.66%。核心是确保每个训练 batch 覆盖不同阶段（接近、抓取、移动、放置）。

### 2.4 RL Fine-tuning

BC（行为克隆）的上限是演示数据质量。要突破这个上限，需要 RL fine-tuning：

```
BC 预训练 → SmolVLA 策略 → PPO/IQL 微调 → 更高成功率
```

| 方法 | 成功率 | 原理 |
|------|--------|------|
| 纯 BC | 47-76% | 模仿演示 |
| BC + IQL weighted BC | 86-88% | 用 advantage 加权好的演示 |
| BC + PPO | 90% | 在 BC 基础上在线 RL 探索 |

**实现思路**：

```python
# 1. BC 预训练（已完成）
policy = SmolVLA.from_pretrained("bc_checkpoint_015000")

# 2. PPO fine-tuning
for update in range(ppo_updates):
    # 用 SmolVLA 策略收集 rollout
    obs, actions, rewards = rollout(env, policy)
    
   )   # 计算 advantage
    advantages = compute_gae(rewards, ...)
    
    # 更新 SmolVLA（保持视觉特征，微调 action head）
    loss = ppo_loss(policy, obs, actions, advantages)
    loss.backward()
    optimizer.step()
```

**挑战**：SmolVLA 是大模型（~450M 参数），PPO 微调需要更多显存和时间。可以冻结 SmolVLM base，只微调 action head。

---

## 3. 社区生态

### 3.1 关键项目

| 项目 | 贡献 | 参考价值 |
|------|------|----------|
| [LeRobot](https://github.com/huggingface/lerobot) | 框架 + SmolVLA + 数据格式 | 基础设施 |
| [so101_nexus](https://github.com/johnsutor/so101_nexus) | MuJoCo 仿真环境 + PPO baseline | 环境和 RL |
| [dyordan1/so101-mujoco](https://github.com/dyordan1/so101-mujoco) | Sim twin 数据集 + 评测 | 数据-环境匹配 |
| [ggand0/vla-so101](https://github.com/ggand0/vla-so101) | VLA 训练最佳实践 | 超参数参考 |
| [MSSergeev/so101-lab](https://github.com/MSSergeev/so101-lab) | BC + RL fine-tuning | 性能提升 |
| [Sa74ll/smolvla_so101](https://github.com/Sa74ll/smolvla_so101_pickandplace) | 分层采样 + 少数据 | 数据效率 |

### 3.2 数据集生态

```
                    ┌─ johnsutor/MuJoCoPickAndPlace-v1 (10 eps, 仿真)
                    │
so101_nexus 生态 ───┼─ ataghof/so101nexus-cube500-binary (500 eps, 仿真)
                    │
                    └─ dobri420/pick-cube-so101-sim (sim twin, 3 相机)

真机数据 ─────── shattori/so101_pick_place_thor (100 eps, 真机)
```

### 3.3 技术路线图

```
当前: BC 47%
  ↓
短期: 超参优化 (bs=64, 50k steps) → 目标 60%
  ↓
中期: 数据增强 (DART, 分层采样) → 目标 75%
  ↓
长期: RL fine-tuning (IQL/PPO) → 目标 86-90%
  ↓
未来: Sim-to-Real 迁移 → 真机部署
```

---

## 4. 开放问题

### 4.1 VLA Pipeline 稳定性

smolvla-fresh 分支 08-20 有 10 次运行仅 1 次成功。需要排查失败原因：

```bash
# 查看失败 run 的日志
gh run view 32333190079 --log-failed
```

可能原因：
- Docker 镜像更新后依赖冲突
- 训练超时（batch_size=64 需 28h）
- ECS GPU 驱动问题

### 4.2 训练-评估一致性

sim twin 方案虽然数据-环境匹配，但仍有细节差异：
- 训练用 LeRobot v2 格式，评测用 MuJoCo 原生
- 相机渲染参数（FOV、位置）是否完全一致

### 4.3 Sim-to-Real 的下一步

当前所有验证都在仿真中。要迁移到真机，需要：
- 域随机化（domain randomization）：训练时随机化光照、纹理、物理参数
- 域适应（domain adaptation）：用少量真机数据微调
- 或直接用真机数据训练（但回到 P1 Visual Gap 问题）

---

## 思考题

1. **47% → 87% 需要改什么？数据、训练、还是方法？**  
   提示：三者都需要。数据（分层采样）、训练（bs=64, 50k）、方法（RL fine-tuning）。

2. **BC + PPO fine-tuning 和纯 PPO 有什么区别？**  
   提示：BC 提供好的初始策略（不从零开始），PPO 在此基础上微调。纯 PPO 从随机策略开始，需要更多探索。

3. **为什么冻结 SmolVLM base 只微调 action head？**  
   提示：SmolVLM base 是大规模预训练的视觉-语言理解能力，不需要改。只需调整 action 映射。

4. **如果最终目标是真机部署，47% 的仿真成功率够用吗？**  
   提示：可能不够。Sim-to-Real 通常有性能下降。需要更高仿真成功率（>90%）才能保证真机可用。

---

> **上一章**：[Ch5 评测方法论](so101-tutorial-ch5-evaluation.md) | **下一章**：[附录](so101-tutorial-appendix.md)
