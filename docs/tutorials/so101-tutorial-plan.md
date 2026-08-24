# SO101 仿真评测教程规划

> 以 `link-seek/so101-sim-pipeline` 为实战项目，结合仿真训练与评测全流程

---

## 教程定位

**目标读者**：有一定 Python / ML 基础，想入门机器人仿真学习（RL + VLA）的工程师  
**核心价值**：不是讲理论，而是讲"一个真实项目从 0 到 47% 成功率的完整踩坑过程"  
**输出形式**：系列教程文章 + 可复现的 GitHub Actions 流水线 + 视频回放

---

## 整体结构（6 章 + 序章 + 附录）

```
序章 → Ch1 基础设施 → Ch2 PPO Baseline → Ch3 VLA 入门 → Ch4 Debug 实战 → Ch5 评测方法论 → Ch6 优化进阶 → 附录
```

每章遵循：**概念（10%）→ 实战代码（60%）→ 踩坑复盘（20%）→ 思考题（10%）**

> **文件名映射**:
> | 章节号 | 文件名 |
> |--------|--------|
> | Ch1 基础设施 | `so101-tutorial-ch1-infrastructure.md` |
> | Ch2 PPO | `so101-tutorial-ch2-ppo-baseline.md` |
> | Ch3 VLA | `so101-tutorial-ch3-vla-intro.md` |
> | Ch4 Debug | `so101-tutorial-ch4-debug-journey.md` |
> | Ch5 评测 | `so101-tutorial-ch5-evaluation.md` |
> | Ch6 优化 | `so101-tutorial-ch6-optimization.md` |

---

## 序章：为什么要在仿真中训练和评测机器人

### 内容
- 真机训练的成本与危险（硬件损耗、采集耗时、安全围栏）
- 仿真的优势：可复现、低成本、并行加速
- 核心矛盾：**Sim-to-Real Gap** — 仿真训练 ≠ 真机可用
- **仿真环境基础**：MuJoCo 物理引擎、观测/动作空间、Reward、Episode
- 两条路线：纯仿真验证（PPO / RL）vs Sim-to-Sim 验证（VLA / BC），两种范式对比
- SO101 项目背景：LeRobot 生态、so101_nexus、SmolVLA

### 实战
- 克隆 `link-seek/so101-sim-pipeline`，浏览仓库结构
- 看 4 个 Discussion 了解项目演进脉络

### 产出
- 理解"为什么训练和评测必须在同一视觉域"

---

## Ch1：搭建仿真训练基础设施

### 概念
- GitHub Actions + Self-hosted Runner + 云 GPU ECS 的工作模式
- Docker 镜像隔离训练环境的意义
- 为什么选 V100（sm_70）+ CUDA 12.6 + MuJoCo
- **系统架构全景图**：控制层 / 计算层 / 存储层 / 数据层
- **数据流**：数据集 → 训练 → checkpoint → 评估 → 归档
- **8 个 Workflow + 5 个 Docker 镜像的职责矩阵**
- **流水线详解**：逐步拆解点击 Run 之后每一步在做什么
- **5 个关键设计决策的 Why**（GitHub Actions / Docker / self-hosted / 分镜像 / OBS）

### 实战
1. **ECS 生命周期管理**：`start-ecs` → `stop-ecs`，hcloud CLI 控制华为云服务器
2. **Docker 镜像构建**：5 个 Dockerfile 的职责划分
   - `Dockerfile.train`：lerobot + smolvla + so101_nexus
   - `Dockerfile.ppo`：so101_nexus[warp] + CleanRL
   - `Dockerfile.mujoco`：MuJoCo 评测环境
3. **Self-hosted Runner 配置**：V100 runner `ecs-0002` 的标签和调度
4. **OBS 归档**：checkpoint / eval_result / video 上传

### 踩坑复盘
- `mujoco-warp` sensor NVRTC 编译错误 → 升级到 >=3.10
- V100 sm_70 需要 `cu126`（含 sm_70 编译目标）
- PATHEXT 被覆盖导致 gh CLI 不可用

### 思考题
- 为什么不直接用 GitHub 托管 runner？
- Docker 镜像为什么不做一个大而全的？

---

## Ch2：PPO 纯仿真 Baseline（最简路径，100% 成功）

### 概念
- PPO 算法简介：Actor-Critic + Clipped Surrogate Objective
- Warp GPU 并行环境：1024 个环境同时 step
- CleanRL 风格：单文件实现，无框架抽象
- 为什么 PPO 不需要训练数据（RL 自探索）

### 实战
1. **阅读 `scripts/train_ppo.py`**：从官方 `ppo_warp.py` 适配的过程
2. **触发 `ppo-pipeline.yml`**：30M steps, 1024 envs, seed=1
3. **观察训练曲线**：best_success 从 0 → 0.995
4. **评估 `scripts/eval_ppo.py`**：MuJoCo 后端 50 episodes 确定性评估
5. **结果对比**：v1 (threshold=5cm, 100%) vs v2 (threshold=15cm, 98%)

### 踩坑复盘
- `lift_threshold=0.05` 太低 → 策略学会微抬 5cm 骗过 reward
- 提高到 0.15 后 success 仅降 2%，证明策略真正学会了抓取+抬起
- V100 SPS 5784 vs RTX 5090 ~20k，慢 3.5x 但结果一致

### 思考题
- 为什么 PPO success_rate 能到 100% 而 VLA 只有 47%？
- `episode_length=512` 固定 horizon 有什么意义？
- 如果换 PickAndPlace 任务，PPO 还能收敛吗？

---

## Ch3：SmolVLA 仿真训练入门（VLA 路线）

### 概念
- VLA（Vision-Language-Action）模型：图像 + 语言 → 动作
- SmolVLA 架构：SmolVLM base + action head + action chunking
- 行为克隆（BC）：从演示数据学习策略
- LeRobot 数据格式 v3.0：episodes, frames, cameras, action space
- `rename_map`：解决数据集相机名与模型期望不一致

### 实战
1. **数据集选择**：3 个数据集的对比
   - `shattori/so101_pick_place_thor`（真机, 100 eps, 2 相机）
   - `ataghof/so101nexus-cube500-binary`（仿真, 500 eps, 2 相机）
   - `dobri420/pick-cube-so101-sim`（仿真 sim twin, 3 相机）← 最终成功
2. **训练流程**：`lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id=...`
3. **仿真回放**：`replay_demo.py` 的观测→推理→执行循环
4. **触发 `so101-mujoco-pipeline.yml`**：端到端训练 + grid sweep 评测

### 踩坑复盘
- **P0 相机不匹配**：训练 side+up / 推理 wrist+overhead → prediction errors
- **P1 Sim-to-Real Visual Gap**：真机照片训练 → MuJoCo 渲染评估，视觉域不匹配
- 核心原则：**训练和评估必须在同一视觉域**

### 思考题
- 为什么 500 episodes 的 ataghof 失败了，但 sim twin 数据集成功了？
- Action Chunking（chunk_size=50）为什么比逐步执行好？

---

## Ch4：Debug 实战 — 从 0% 到 47% 的完整调试旅程

> 这是教程的核心章节，用 Discussion #4 的 8 条评论作为时间线

### 概念
- 系统性 Debug 方法论：假设 → 验证 → 修复 → 回归
- 如何区分"代码 bug"和"系统性问题"
- 社区调研驱动的 Debug：找成功案例，对比配置差异

### 实战（按时间线还原）

**Phase 1：P0 修复**
- 现象：prediction errors > 0, reward ≈ 0
- 假设：相机视角不匹配
- 验证：切换数据集后 errors → 0/300 ✅

**Phase 2：ataghof 方案迭代**
- 5k 训练：loss 0.461→0.119, Success=False
- 20k 训练：loss 0.119→0.046, Success=False
- 社区对比：成功案例 loss 到 0.005，我们 0.046 偏高

**Phase 3：3 个 Bug 诊断与修复**
1. **Gripper 转换 Bug**：自实现 `/100.0` vs 官方 `dataset_row_to_sim_qpos`
   - 数据集 gripper 范围 0-45，`/100.0` 导致最多闭合 45%
   - 修复：改用 so101_nexus 官方函数
2. **Action Chunking**：外部 queue 缓存 50 action
   - 发现 `select_action` 内部已处理，chunk shape (1,6)
   - 修复：移除外部 queue
3. **camera3 分布不匹配**：训练时缺失，回放时提供
   - 修复：回放时不提供 camera3

**Phase 4：方案终止决策**
- 3 bug 修复后回放仍 Success=False
- 根因：数据采集环境 ≠ 评测环境（系统性问题，非代码 bug）
- 决策：终止 ataghof 方案，转向 so101-mujoco sim twin

**Phase 5：Sim Twin 方案成功**
- 数据-环境 1:1 匹配
- 15K steps 训练, loss 0.090
- Grid sweep 325 episodes → **47% 成功率**

### 踩坑复盘
- 不要自己实现坐标转换，用官方函数
- "0 errors 但 Success=False" 说明不是代码 bug，是系统性问题
- 何时该坚持修 bug，何时该换方案

### 思考题
- 如果 gripper bug 修复后 Success=True，还会发现数据-环境不匹配问题吗？
- 如何在训练前就判断数据集和评测环境是否匹配？

---

## Ch5：仿真评测方法论

### 概念
- 为什么不能只看 training loss（loss 低 ≠ 性能好）
- 确定性评估 vs 随机评估
- Grid Sweep：参数空间扫描评测
- 评测指标：success_rate, avg_reward, avg_steps, prediction errors

### 实战
1. **PPO 评估**：MuJoCo 后端 50 episodes 确定性评估
   - `eval_ppo.py`：固定 seed+1000+ep，消除随机性
   - 输出 `eval_result.json` + `eval_video.mp4`
2. **VLA 评估**：MuJoCo grid sweep, 325 episodes
   - 5 reach × 13 azimuth × 5 trials
   - 热力图可视化：中心强、边缘弱
3. **回放验证**：`replay_demo.py` 300 steps
   - prediction errors / reward 曲线 / success 判定
4. **结果归档**：OBS 上传 checkpoint + eval result + video

### 踩坑复盘
- `lift_threshold` 影响 success 判定标准 → 评测指标要和任务语义对齐
- grid sweep 暴露了训练数据分布偏置（中心区域强，边缘弱）
- 47% 是 325 个不同初始条件的平均，不是单一条件

### 思考题
- 为什么 grid sweep 要用 5 trials 而不是 1？
- 如果训练数据只覆盖中心区域，边缘 0% 是 bug 还是预期？

---

## Ch6：优化进阶与社区生态

### 概念
- BC 的上限：演示数据质量决定性能天花板
- RL Fine-tuning：IQL advantage-weighted BC / PPO 在 BC 基础上提升
- 数据增强：DART 噪声注入、分层采样
- 多相机融合：2 相机 vs 3 相机的影响

### 实战
1. **社区成功案例对比**：
   | 项目 | 方法 | 成功率 |
   |------|------|--------|
   | ggand0/vla-so101 | 75 eps + 20k + bs=64 | 60-80% |
   | Sa74ll/smolvla | 40 eps + 15k + 分层采样 | 87.66% |
   | MSSergeev/so101-lab | SmolVLA + IQL | 86-88% |
   | MSSergeev/so101-lab | SmolVLA + PPO | 90% |
2. **优化方向规划**：
   - 增加 batch_size 32→64（需解决超时）
   - 增加训练步数 15k→50k
   - 分层采样数据增强
   - RL fine-tuning（IQL/PPO）
3. **超时问题解决**：拆分训练阶段、checkpoint 续训

### 思考题
- 47% → 87% 需要改什么？数据、训练、还是方法？
- BC + PPO fine-tuning 和纯 PPO 有什么区别？

---

## 附录

### A. 环境配置速查
- V100 ECS + CUDA 12.6 + Docker + self-hosted runner
- lerobot 0.6.1 + smolvla + so101_nexus[warp]
- MuJoCo 无头渲染 MUJOCO_GL=egl

### B. 常用命令
```bash
# 触发 PPO 流水线
gh workflow run ppo-pipeline.yml -f env_id=WarpPickLift-v1 -f total_timesteps=30000000

# 触发 MuJoCo VLA 流水线
gh workflow run so101-mujoco-pipeline.yml -f steps=20000 -f batch_size=32

# 查看训练日志
gh run view <run_id> --log

# 查看 runner 状态
gh api repos/link-seek/so101-sim-pipeline/actions/runners
```

### C. 数据集对比表
| 数据集 | 来源 | Episodes | 相机 | 视觉域 | 用途 |
|--------|------|----------|------|--------|------|
| shattori/so101_pick_place_thor | 真机 | 100 | 2 | 真机 | 已弃用 |
| ataghof/so101nexus-cube500-binary | 仿真 | 500 | 2 | 仿真≠评测 | 已终止 |
| dobri420/pick-cube-so101-sim | 仿真 twin | - | 3 | 仿真=评测 | ✅ 当前使用 |
| johnsutor/MuJoCoPickAndPlace-v1 | 仿真 | 10 | 2 | 仿真=仿真 | 备选（数据少） |

### D. Discussion 索引
- [#1](https://github.com/link-seek/so101-sim-pipeline/discussions/1) P0 修复 + P1 发现
- [#2](https://github.com/link-seek/so101-sim-pipeline/discussions/2) PPO 流水线设计 + 验证结果
- [#3](https://github.com/link-seek/so101-sim-pipeline/discussions/3) 路线图
- [#4](https://github.com/link-seek/so101-sim-pipeline/discussions/4) 方案 A→B 完整迭代（8 条评论，最活跃）

---

## 制作建议

| 形式 | 建议 |
|------|------|
| **文字教程** | 每章 3000-5000 字，配代码块和表格，发到博客/知乎/公众号 |
| **视频教程** | 每章 15-20 min，Ch4 Debug 实战可做 30min 长视频 |
| **代码仓库** | 新建 `so101-tutorial` 仓库，每章一个 branch/tag，可逐步 checkout |
| **交互式** | 用 Jupyter Notebook 嵌入训练曲线和回放视频 |

### 推荐发布顺序
1. **序章**（含仿真环境基础 + RL vs BC）→ 建立知识框架，理解全局
2. **Ch1**（基础设施 + 流水线详解）→ 动手搭建 ECS + Docker + Runner
3. **Ch2**（PPO 100%）→ 快速成就感，建立信心
4. **Ch3 + Ch4**（VLA Debug 旅程）→ 核心价值，最硬核内容
5. **Ch5**（评测方法论）→ 方法论提升
6. **Ch6**（优化进阶）→ 展望未来，引导社区参与
