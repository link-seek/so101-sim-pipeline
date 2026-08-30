# Ch1：搭建仿真训练基础设施

> SO101 仿真评测教程 · 第一章  
> 实战项目：[link-seek/so101-sim-pipeline](https://github.com/link-seek/so101-sim-pipeline)

---

## 1. 架构总览

我们的流水线不是在本地跑训练，而是把整个训练-评估流程自动化为 GitHub Actions CI/CD：

```
┌─────────────────────────────────────────────────────────────┐
│                    GitHub Actions (控制层)                    │
│  workflow dispatch → start-ecs → pipeline → stop-ecs         │
└──────────┬──────────────────────────────────┬───────────────┘
           │                                  │
           ▼                                  ▼
┌─────────────────────┐          ┌──────────────────────────┐
│  华为云 ECS (计算层)  │          │  华为云 OBS (存储层)       │
│  V100 32GB GPU       │          │  checkpoint/              │
│  self-hosted runner  │          │  eval_result.json         │
│  ┌─────────────────┐│          │  eval_video.mp4           │
│  │  Docker 容器     ││          └──────────────────────────┘
│  │  ┌───────────┐  ││                     ▲
│  │  │ 训练脚本   │  ││─── checkpoint ──────┘
│  │  │ 评估脚本   │  ││─── eval result ─────┘
│  │  └───────────┘  ││
│  └─────────────────┘│
└─────────────────────┘
           ▲
           │
┌──────────┴──────────────────────────────────────────────────┐
│              HuggingFace Hub (数据层)                         │
│  dobri420/pick-cube-so101-sim  (训练数据集)                  │
│  lerobot/smolvla_base          (预训练模型)                  │
└─────────────────────────────────────────────────────────────┘
```

### 1.1 数据流

训练和评估是两个阶段，通过 checkpoint 连接，在同一条流水线内顺序执行。PPO 和 MuJoCo 流水线均使用单镜像完成训练+评估：

```
阶段 1: 训练
  数据集 (HF Hub) ──→ Docker 容器 ──→ 训练脚本 ──→ checkpoint
  dobri420/pick-cube    so101-mujoco    train_smolvla_sim  /data/ckpt/20000

阶段 2: 评估
  checkpoint ──→ Docker 容器 ──→ 评估脚本 ──→ 评估结果
  /data/ckpt/     so101-mujoco    eval_mujoco_policy  eval_result.json
                                   grid_sweep          eval_video.mp4

阶段 3: 归档
  checkpoint + eval_result + eval_video ──→ OBS (长期存储)
```

### 1.2 8 个 Workflow 职责矩阵

| Workflow | 触发方式 | 做什么 | 用哪个镜像 |
|----------|---------|--------|-----------|
| `ci.yml` | push/PR | 代码检查 + 格式化 | 无 (ubuntu-latest) |
| `docker-build.yml` | push to main | 构建 5 个 Docker 镜像 | 无 (docker buildx) |
| `train.yml` | 手动 | 推送到 Kaggle 训练（非 Docker） | 无 (Kaggle) |
| `evaluate.yml` | 手动 | 启动 ECS + SSH 远程评估 | 无 (SSH) |
| `collect.yml` | 手动 | 采集 LIBERO 专家数据 | so101-eval |
| `ppo-pipeline.yml` | 手动 | **PPO 完整流水线** | so101-ppo |
| `so101-mujoco-pipeline.yml` | 手动 | **VLA MuJoCo 完整流水线** | so101-mujoco |
| `vla-pipeline.yml` | 手动 | VLA 训练+回放 (调试中) | so101-train + so101-eval + so101-model-server |

**活跃使用的流水线**：
- `so101-mujoco-pipeline.yml` — VLA 训练 + grid sweep 评估 + 归档（已验证，8 次运行）
- `ppo-pipeline.yml` — PPO 训练 + 评估 + 归档（已验证，3 次运行）
- `evaluate.yml` — 通用评估入口（最活跃，61 次运行）
- `collect.yml` — 采集专家数据（活跃，20 次运行）

### 1.3 设计决策

| 设计决策 | 理由 |
|----------|------|
| 云 GPU 而非本地 | V100 32GB 随用随开，不用时不计费 |
| Docker 隔离 | lerobot + so101_nexus 依赖复杂，容器化保证可复现 |
| GitHub Actions | 团队成员可一键触发，结果自动归档 |
| self-hosted runner | ECS 上直接跑，避免 GPU 任务排队 |

---

## 2. ECS 生命周期管理

### 2.1 为什么需要手动开关机

V100 ECS 按小时计费（约 ¥30/h），训练任务一天可能只跑 2-3 次，每次 1-2 小时。如果 7×24 常开，每月成本 ¥21000+。手动开关机后，成本降至 ¥1800/月。

### 2.2 start-ecs job

```yaml
# .github/workflows/ppo-pipeline.yml（简化）
jobs:
  start-ecs:
    runs-on: ubuntu-latest
    steps:
      - name: Start ECS
        run: |
          # 下载华为云 CLI
          curl -sSL https://hc.obs.cn-north-4.myhuaweicloud.com/hcloud/hcloud_linux.tar.gz | tar xz
          # 配置认证
          export OBS_AK=${{ secrets.OBS_AK }}
          export OBS_SK=${{ secrets.OBS_SK }}
          # 启动服务器
          ./hcloud ECS BatchStartServers \
            --region cn-north-4 \
            --server-id 7f39cb83-1a5c-4792-b65e-e578d7ddb88d
          # 等待 GPU 驱动就绪
          sleep 120
```

### 2.3 stop-ecs job

```yaml
  stop-ecs:
    needs: [start-ecs, pipeline]  # 等训练完成
    runs-on: ubuntu-latest
    if: always()  # 即使训练失败也关机
    steps:
      - name: Stop ECS
        run: |
          hcloud ECS BatchStopServers \
            --region cn-north-4 \
            --server-id 7f39cb83-1a5c-4792-b65e-e578d7ddb88d
```

**关键点**：`if: always()` 确保无论训练成功还是失败，ECS 都会被关闭。漏掉这一行，一次失败的训练就会让 ECS 开一整夜。

### 2.4 self-hosted runner

ECS 启动后，上面运行的 GitHub Actions runner 会自动注册到仓库：

```
Runner: ecs-0002
Labels: self-hosted, Linux, X64, V100
Status: online
```

流水线中通过标签选择 runner：

```yaml
  pipeline:
    needs: start-ecs
    runs-on: [self-hosted, Linux, X64, V100]
```

---

## 3. Docker 镜像矩阵

我们维护了 5 个 Docker 镜像，各有职责（为什么分镜像？见 §7.4）：

| 镜像 | Dockerfile | 关键依赖 | 用途 | 状态 |
|------|-----------|----------|------|------|
| `so101-eval` | Dockerfile.eval | vla-eval + mujoco + SQLite | LIBERO 评估 + 采集专家数据 | ✅ 活跃（collect/evaluate 频繁使用） |
| `so101-mujoco` | Dockerfile.mujoco | lerobot[training] + vla-eval + mujoco + grid sweep | Sim twin 训练 + 评估一体 | ✅ 已验证（8/19 跑通，grid sweep 任务成功率 47%，流水线成功率 83%） |
| `so101-ppo` | Dockerfile.ppo | so101_nexus[warp,train] + mujoco-warp | PPO 训练 + 评估一体 | ✅ 已验证（8/14 跑通，任务成功率 100%，流水线成功率 67%） |
| `so101-train` | Dockerfile.train | lerobot[smolvla] + so101_nexus + vla-eval | VLA 训练 + 回放（旧版） | 🔧 调试中（vla-pipeline 43 次运行多次失败，最近一次 8/20 成功） |
| `so101-model-server` | Dockerfile.model-server | 模型推理服务 | HTTP/ZMQ 推理 | ⏳ 未验证（无运行记录） |

### 3.1 各镜像职责详解

下面把每个镜像到底在跑什么、被谁调用、输入输出是什么逐一拆开，方便你点 Run 之前就知道屏幕上会发生什么。

#### `so101-train` — VLA 训练与回放（调试中）

> ⚠️ 此镜像当前仅被 `vla-pipeline.yml` 使用，该流水线在 smolvla-fresh 分支上调试中（43 次运行，多次失败/cancelled）。如果只是想跑 VLA 训练，建议用 `so101-mujoco` 镜像的 sim twin 方案。

- **职责**：在 LeRobot 格式数据集上微调 SmolVLA（行为克隆 / SFT），并负责训练后的「回放验证」——把模型加载进仿真跑一遍，确认动作能驱动机器人。
- **被谁调用**：`vla-pipeline.yml`（旧版训练+回放，调试中）。
- **关键脚本**：`train_smolvla.py`（训练）、`replay_demo.py`（回放验证）。
- **输入**：LeRobot 数据集（parquet）、SmolVLA base 权重、可选 `rename_map` 修正相机键名。
- **输出**：`/data/ckpt/<steps>/` 下的 checkpoint（上传 OBS）、回放视频 mp4。
- **典型命令**：

```bash
docker run --gpus all so101-train:latest \
  train_smolvla.py --dataset so101/so101_pick_cube_sim \
  --steps 20000 --batch_size 32
```

#### `so101-ppo` — 强化学习训练与评估一体

PPO 路线和 VLA 最大的结构差异是：**它一个镜像内就完成了「训练 + 评估」的闭环**，不需要单独的评估镜像，因为 RL 的评估就是在同一个仿真环境里再跑几百个 episode 算 success_rate。

- **职责**：用 Warp GPU 并行环境训练 PPO 策略（WarpPickLift 等），并立即在同一环境做独立评估。
- **被谁调用**：`ppo-pipeline.yml`。
- **关键脚本**：`train_ppo.py`（CleanRL 风格单文件实现）、`eval_ppo.py`（加载 `best_agent.pt` 跑评估）。
- **输入**：`env_id`（如 `WarpPickLift-v1`）、超参数（`total_timesteps`、`lr`、`gamma` 等）。
- **输出**：`best_agent.pt`（策略权重，约 591KB）、`eval_result.json`、`eval_video.mp4`，全部上传 OBS。
- **注意点**：镜像里打了 `mujoco-warp>=3.10.0.1,<3.12` 修复 NVRTC bug，并设置 `MUJOCO_GL=egl` 做无头渲染——这是 V100 上能跑的关键。

#### `so101-mujoco` — 仿真孪生训练与网格扫描评测

Sim twin 路线专属：它在 MuJoCo 里**训练一个 SO101 仿真孪生模型**（方案 B，即拿到 47% 成功率的那个），再对训练出的模型做 grid sweep 评测。

- **职责**：拉取仿真数据集 → 训练 SO101 sim twin（SmolVLA）→ 多初始条件 × 多 seed 的 grid sweep 评测，产出成功率热力图。
- **被谁调用**：`so101-mujoco-pipeline.yml`（流水线内依次调用以下三个脚本）。
- **关键脚本**：`download_sim_dataset.py`（拉取仿真数据集）、`train_smolvla_sim.py`（训练 sim twin）、`eval_mujoco_policy.py`（grid sweep 主程序）。
- **输入**：仿真数据集、SmolVLA base 权重、初始条件 grid 定义。
- **输出**：sim twin checkpoint、`success_rate` 矩阵、热力图（各区域成功率，中心 60-100% / 边缘 ~0% 就是这么来的）。
- **与 `so101-train` 的区别**：`train` 在原始 LeRobot 数据集上微调通用 SmolVLA；`mujoco` 在 MuJoCo 仿真数据上训练/评测 sim twin，两者数据来源与用途不同，所以分开成镜像。

#### `so101-eval` — LIBERO 标准化 benchmark 评估（已构建，已运行）

这是为「跨任务标准化评测」准备的镜像，基础设施已就绪，且已通过 `evaluate.yml` 实际跑过首次 LIBERO 评测（结果 0%，详见 Ch5 §4.8）。

- **职责**：通过 `vla-eval` harness 统一调度 LIBERO / LIBERO-PRO 等多个 benchmark，产出每任务 success_rate 与 SQLite 结果库。
- **被谁调用**：`evaluate.yml`（已运行，含 LIBERO 评测，首次跑出 0% 成功率，详见 Ch5 §4.8）。
- **关键脚本**：`eval_vla.py`。
- **输入**：模型 checkpoint、`configs/benchmarks/*.yaml`（8 个 LIBERO 配置）。
- **输出**：每任务 success_rate、SQLite 数据库。
- **当前状态**：首次实战已验证「模型-环境不兼容」——我们的 SO101 SmolVLA 无法驱动 LIBERO 的 Franka Panda（libero_goal 100 个 episode `steps=0` 初始化即失败，libero_spatial 20 个 episode 跑满 230 步仍 0% 成功）。根本解决仍需先在 LIBERO 中添加 SO101 机器人（Ch6 方案）。

#### `so101-model-server` — 模型推理部署

把训练好的策略包装成一个可被外部程序调用的推理服务，是「从训练环境走向真机/实时控制」的桥梁。

- **职责**：基于 `vla-eval serve` 子命令，把 SmolVLA SO101 模型托管为推理服务，对外接收观测、返回动作。Dockerfile 的启动命令即 `vla-eval serve --config configs/model_servers/smolvla_so101.yaml`。
- **接口**：由 `vla-eval` 提供的推理服务端点（HTTP）。
- **典型场景**：真机 SO101 实时控制、Web 可视化演示、或作为其他系统的策略后端。
- **与训练镜像的关系**：训练产出的 checkpoint 在这里被加载为常驻服务；它复用 `vla-eval` 推理栈（与 `so101-eval` 共享 lerobot[smolvla] + vla-eval 依赖），单独成镜像是为了把「服务部署」与「训练/评测」解耦。

> **一句话总结**：`train` 和 `ppo` 负责「造模型」，`mujoco` 和 `eval` 负责「测模型」，`model-server` 负责「用模型」。当前 PPO 和 MuJoCo 流水线都是单镜像完成训练+评估闭环；旧版 vla-pipeline 才分 train/eval 两镜像（依赖不同、镜像更小，见 §3.1）。

---

## 4. OBS 归档

训练产物上传到华为云 OBS（对象存储），用于后续分析和 checkpoint 续训：

```yaml
      - name: Upload to OBS
        run: |
          # 安装 obsutil
          wget https://obs-community.obs.cn-north-4.myhuaweicloud.com/obsutil/obsutil_linux.tar.gz
          tar xzf obsutil_linux.tar.gz
          
          # 上传 checkpoint
          ./obsutil cp /data/checkpoints/ppo/ \
            obs://so101-sim-pipeline/ppo/checkpoints/ -r -f
          
          # 上传评估结果
          ./obsutil cp /data/ppo/results/ \
            obs://so101-sim-pipeline/ppo/results/ -r -f
```

归档内容：

| 产物 | 路径 | 大小 | 用途 |
|------|------|------|------|
| best_agent.pt | obs://.../ppo/checkpoints/ | 591KB | PPO 策略权重 |
| eval_result.json | obs://.../ppo/results/ | ~1KB | 评估指标 |
| eval_video.mp4 | obs://.../ppo/results/ | ~5MB | 评估视频回放 |
| train_result.json | obs://.../ppo/results/ | ~1KB | 训练指标 |

---

## 5. 实战：触发你的第一次流水线

### 前置条件

- GitHub 仓库已配置 self-hosted runner
- Docker 镜像已推送到 SWR
- OBS bucket 已创建
- Secrets 已配置：`OBS_AK`, `OBS_SK`, `SWR_PASSWORD`

### 触发 PPO 流水线

```bash
# 方法 1：GitHub CLI
gh workflow run ppo-pipeline.yml \
  -f env_id=WarpPickLift-v1 \
  -f total_timesteps=30000000 \
  -f num_envs=1024 \
  -f seed=1

# 方法 2：GitHub Web UI
# Actions → PPO Pipeline → Run workflow → 填参数 → Run
```

### 监控运行

```bash
# 查看运行状态
gh run list --workflow=ppo-pipeline.yml --limit 5

# 查看实时日志
gh run watch

# 查看特定 step 日志
gh run view <run_id> --log | grep -E "(success|reward|SPS|Error)"
```

### 预期时间线

```
T+0min     start-ecs 开始
T+2min     start-ecs 完成，runner 上线
T+2min     pipeline 开始，拉取 Docker 镜像
T+5min     训练开始
T+90min    训练完成（30M steps）
T+90min    评估开始（50 episodes）
T+91min    评估完成
T+91min    OBS 上传
T+93min    stop-ecs 开始
T+94min    流水线完成
```

---

## 6. 流水线详解：点击 Run 之后发生了什么

上节展示了如何触发流水线，但这只是"点一下按钮"。现在逐步拆解流水线，让你知道每一步系统在做什么。

### 6.1 VLA 流水线 (`so101-mujoco-pipeline.yml`)

```
你点击 "Run workflow"
  │
  ├─ Step 1: start-ecs (ubuntu-latest, ~2min)
  │   ├─ 下载华为云 CLI (curl 二进制)
  │   ├─ 调用 hcloud API 启动 ECS 7f39cb83
  │   ├─ sleep 120s 等待 GPU 驱动就绪
  │   └─ ECS 启动, runner ecs-0002 自动注册到 GitHub
  │
  ├─ Step 2: pipeline (self-hosted V100, ~10h)
  │   ├─ 2a. 拉取 Docker 镜像 (~2min)
  │   │   docker pull swr.cn-north-4.myhuaweicloud.com/link-seek/so101-mujoco:latest
  │   │
  │   ├─ 2b. 训练阶段 (~7h)
  │   │   ├─ 启动训练容器: docker run --gpus all so101-mujoco ...
  │   │   ├─ 从 HF Hub 下载数据集 (dobri420/pick-cube-so101-sim)
  │   │   ├─ 加载预训练模型 (lerobot/smolvla_base)
  │   │   ├─ 训练循环 (20K steps):
  │   │   │   for step in range(20000):
  │   │   │       batch = sample_batch(dataset)        # 采样
  │   │   │       pred = policy(batch.obs, batch.lang)  # 前向
  │   │   │       loss = mse(pred, batch.action)        # 损失
  │   │   │       loss.backward(); optimizer.step()     # 反向
  │   │   │       if step % 5000 == 0: save_checkpoint()# 存盘
  │   │   └─ 输出: /data/checkpoints/20000/ (模型权重)
  │   │
  │   ├─ 2c. 评估阶段 (~30min)
  │   │   ├─ 同一容器内继续: so101-mujoco
  │   │   ├─ 加载 checkpoint, 创建 MuJoCo 评估环境
  │   │   ├─ Grid Sweep (325 episodes):
  │   │   │   for reach in [0.15, 0.18, 0.20, 0.22, 0.25]:    # 5 距离
  │   │   │     for azimuth in range(-90, 91, 15):             # 13 角度
  │   │   │       for trial in range(5):                        # 5 试验
  │   │   │         obs = env.reset(reach, azimuth, seed=trial)
  │   │   │         for step in range(300):
  │   │   │           action = policy.select_action(obs)
  │   │   │           obs, reward, done, info = env.step(action)
  │   │   │           if done: break
  │   │   │         record(success=info["success"])
  │   │   ├─ 统计: 153/325 = 47% success_rate
  │   │   └─ 输出: eval_result.json + eval_video.mp4 + heatmap.png
  │   │
  │   └─ 2d. OBS 归档 (~2min)
  │       ├─ 上传 checkpoint → obs://so101-sim-pipeline/mujoco-checkpoints/
  │       ├─ 上传 eval_result → obs://so101-sim-pipeline/mujoco-eval/
  │       └─ 上传 eval_video  → obs://so101-sim-pipeline/mujoco-eval/
  │
  └─ Step 3: stop-ecs (ubuntu-latest, if:always(), ~1min)
      └─ 调用 hcloud API 关闭 ECS (无论成功失败都关)
```

### 6.2 PPO 流水线 (`ppo-pipeline.yml`)

PPO 流水线结构相同，但训练和评估逻辑不同：

| 阶段 | VLA 流水线 | PPO 流水线 |
|------|-----------|-----------|
| 训练输入 | 数据集 (图像+语言) | 环境 (reward 信号) |
| 训练循环 | BC: 采样→前向→MSE→反向 | RL: rollout→GAE→clip→更新 |
| 训练步数 | 20K steps | 30M steps (1024 envs 并行) |
| 训练时间 | ~7h (20K steps) | ~86 min |
| 评估方式 | Grid sweep 325 episodes | 确定性 50 episodes |
| 评估时间 | ~30 min | ~14 sec |
| Docker 镜像 | so101-mujoco (单镜像) | so101-ppo (单镜像) |
| 总时间 | ~10h | ~90 min |

### 6.3 时间线总结

```
T+0min    ── 你点击 Run
T+2min    ── ECS 启动完成, runner 上线
T+4min    ── Docker 镜像拉取完成, 训练开始
T+424min  ── 训练结束 (~7h), checkpoint 保存
T+424min  ── 评估开始
T+454min  ── 评估结束, 47% success_rate
T+456min  ── OBS 归档完成
T+457min  ── ECS 关闭, 流水线完成
T+457min  ── 你收到 GitHub Actions 通知
```

---

## 7. 关键设计决策：为什么这样设计

理解"为什么"比记住"怎么用"更重要。

### 7.1 为什么用 GitHub Actions 而不是直接 SSH 跑脚本？

| 方案 | 直接 SSH | GitHub Actions |
|------|---------|----------------|
| 触发方式 | 手动 SSH 连服务器 | 一键 Web UI / CLI |
| 参数传递 | 命令行参数 | workflow input 参数 |
| 日志 | 自己管理 | 自动记录, 可回溯 |
| 失败通知 | 自己写脚本 | 自动 email/通知 |
| 团队协作 | 需要共享 SSH key | 仓库权限即可 |
| 成本控制 | 容易忘关机 | stop-ecs if:always() |

**核心价值**：把训练-评估流程**工程化**，不是一次性脚本，而是可复现、可协作的流水线。

### 7.2 为什么用 Docker 而不是 conda 环境？

| 方案 | conda | Docker |
|------|-------|--------|
| 依赖隔离 | 环境可能互相污染 | 完全隔离 |
| 可复现 | "works on my machine" | 镜像一致, 任何机器结果同 |
| GPU 支持 | 需要手动配 CUDA | nvidia/cuda 基础镜像 |
| CI 集成 | 需要在 runner 上配环境 | docker pull 即可 |

**核心价值**：lerobot + so101_nexus + MuJoCo 依赖链很复杂，Docker 保证所有人环境一致。

### 7.3 为什么用 self-hosted runner 而不是 GitHub 托管 runner？

| 方案 | GitHub 托管 GPU runner | self-hosted |
|------|----------------------|-------------|
| 成本 | $0.16/min ≈ ¥1.15/min | ¥30/h ≈ ¥0.50/min |
| 86min 训练 | $13.76 ≈ ¥99 | ¥43 |
| GPU 型号 | 不可选 | V100 32GB |
| 排队 | 可能排队 | 独占 |

**核心价值**：V100 按需开关机，成本只有 GitHub 托管 runner 的 1/2，且独占不排队。

### 7.4 为什么有 5 个 Docker 镜像？

```
so101-ppo        so101-mujoco          so101-train      so101-eval        so101-model-server
├ so101_nexus    ├ lerobot[training]   ├ lerobot[training]  ├ vla-eval    ├ HTTP/ZMQ 推理
├ mujoco-warp   ├ vla-eval            ├ so101_nexus       ├ mujoco       └ 模型加载
└ PPO 训练+评估  ├ mujoco + grid sweep └ 无 grid sweep     └ SQLite       ← 推理服务，不训练不评估
                └ 训练+评估一体        ← 仅训练（旧版）     ← 仅评估
```

**为什么不全合成一个镜像？** 训练依赖（lerobot[training]、so101_nexus[warp]）和评估依赖（vla-eval、mujoco、grid sweep）有差异，合在一起会冲突或镜像过大。PPO 和 MuJoCo 流水线各自把所需依赖打包成一个镜像完成训练+评估闭环；旧版 vla-pipeline 才拆成 train + eval 两个镜像。

### 7.5 为什么用 OBS 而不是 HF Hub 存评估结果？

| 方案 | HF Hub | OBS |
|------|--------|-----|
| 适用 | 模型/数据集 (结构化) | 任意文件 (视频/JSON) |
| 上传 | git LFS, 慢 | obsutil, 快 |
| 成本 | 免费但有限额 | 按存储量计费, 便宜 |

**核心价值**：评估产物是 JSON + MP4 + PNG，不是模型权重，OBS 更适合存这类非结构化文件。

---

## 踩坑复盘

### 坑 1：mujoco-warp sensor NVRTC 编译错误

**现象**：PPO 训练启动时报 `undefined _magnetometer_0, _cam_projection_0` 等符号错误。

**根因**：so101_nexus 0.5.1 限制 `mujoco-warp<3.10`，而 3.9.x 的 sensor.py 有代码生成 bug。

**修复**：在 Dockerfile.ppo 中追加：
```dockerfile
RUN pip install --no-cache-dir "mujoco-warp>=3.10.0.1,<3.12"
```

**教训**：依赖版本约束要看 changelog，`<3.10` 的限制可能是因为旧 bug，但新版已修复。

### 坑 2：V100 sm_70 需要 cu126

**现象**：PyTorch CUDA 12.1 的 wheel 不包含 sm_70 编译目标，V100 上运行报 `no kernel image is available`。

**修复**：必须用 `cu126`（CUDA 12.6 包含 sm_70）：
```dockerfile
RUN pip install torch --index-url https://download.pytorch.org/whl/cu126
```

**教训**：V100 是 2017 年的架构（sm_70），新 CUDA 版本可能不再默认包含。查 [PyTorch CUDA 兼容表](https://github.com/pytorch/pytorch/blob/main/torch/csrc/cuda/Module.cpp) 确认。

### 坑 3：ECS 忘关机

**现象**：一次训练失败后，ECS 开了一整夜，产生 ¥240 意外费用。

**根因**：stop-ecs job 缺少 `if: always()`，训练失败时 stop-ecs 被跳过。

**修复**：
```yaml
  stop-ecs:
    if: always()  # 关键！
```

**教训**：云资源生命周期管理必须有兜底机制，不能假设上游 job 一定成功。

---

## 思考题

1. **为什么不直接用 GitHub 托管的 GPU runner？**  
   提示：算算成本。GitHub GPU runner $0.16/min，86 分钟训练 = $13.76/次。

2. **Docker 镜像为什么 GHCR + SWR 双写？**  
   提示：GHCR 公开可访问，SWR 在华为云内网拉取更快。

3. **如果训练需要 28 小时（超过 GitHub Actions 6h 限制），怎么办？**  
   提示：考虑拆分训练阶段、checkpoint 续训、或用 nohup + 后台运行。

---

> **上一章**：[序章](so101-tutorial-ch0-prologue.md) | **下一章**：[Ch2 PPO 纯仿真 Baseline](so101-tutorial-ch2-ppo-baseline.md)
