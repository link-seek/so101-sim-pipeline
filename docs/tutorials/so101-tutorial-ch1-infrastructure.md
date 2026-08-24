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

训练和评估是两个独立阶段，通过 checkpoint 连接：

```
阶段 1: 训练
  数据集 (HF Hub) ──→ Docker 容器 ──→ 训练脚本 ──→ checkpoint
  dobri420/pick-cube    so101-train     train_smolvla    /data/ckpt/15000

阶段 2: 评估
  checkpoint ──→ Docker 容器 ──→ 评估脚本 ──→ 评估结果
  /data/ckpt/     so101-mujoco    eval_mujoco     eval_result.json
                                  grid_sweep      eval_video.mp4

阶段 3: 归档
  checkpoint + eval_result + eval_video ──→ OBS (长期存储)
```

### 1.2 8 个 Workflow 职责矩阵

| Workflow | 触发方式 | 做什么 | 用哪个镜像 |
|----------|---------|--------|-----------|
| `ci.yml` | push/PR | 代码检查 + 格式化 | 无 (ubuntu-latest) |
| `docker-build.yml` | push to main | 构建 5 个 Docker 镜像 | 无 (docker buildx) |
| `download-driver.yml` | 手动 | 下载 GPU 驱动到 ECS | 无 |
| `train.yml` | 手动 | 通用训练入口 | so101-train |
| `evaluate.yml` | 手动 | 通用评估入口 | so101-eval |
| `ppo-pipeline.yml` | 手动 | **PPO 完整流水线** | so101-ppo |
| `so101-mujoco-pipeline.yml` | 手动 | **VLA MuJoCo 完整流水线** | so101-train + so101-mujoco |
| `vla-pipeline.yml` | 手动 | VLA 训练+回放 (旧版) | so101-train |

**你最常用的是这两个**：
- `ppo-pipeline.yml` — PPO 训练 + 评估 + 归档
- `so101-mujoco-pipeline.yml` — VLA 训练 + grid sweep 评估 + 归档

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
          # 安装华为云 CLI
          pip install hcloud
          # 配置认证
          export HC_ACCESS_KEY=${{ secrets.HC_ACCESS_KEY }}
          export HC_SECRET_KEY=${{ secrets.HC_SECRET_KEY }}
          # 启动服务器
          hcloud ECS BatchStartServers \
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

我们维护了 5 个 Docker 镜像，各有职责：

| 镜像 | Dockerfile | 基础 | 关键依赖 | 用途 |
|------|-----------|------|----------|------|
| `so101-train` | Dockerfile.train | cuda:12.6.3 | torch cu126 + lerobot[smolvla] 0.6.1 + so101_nexus | VLA 训练 + 回放 |
| `so101-ppo` | Dockerfile.ppo | cuda:12.6.3 | torch cu126 + so101_nexus[warp,train] | PPO 训练 + 评估 |
| `so101-mujoco` | Dockerfile.mujoco | cuda:12.6.3 | robot_descriptions + mujoco_env | Sim twin 训练 + grid sweep |
| `so101-eval` | Dockerfile.eval | cuda:12.6.3 | vla-eval + SQLite | LIBERO benchmark 评估 |
| `so101-model-server` | Dockerfile.model-server | cuda:12.6.3 | 模型推理服务 | HTTP/ZMQ 推理 |

### 3.1 为什么不用一个大镜像

| 方案 | 大而全 | 分离镜像 |
|------|--------|----------|
| 镜像大小 | ~15GB | 每个 5-8GB |
| 拉取时间 | ~5min | ~2min |
| 依赖冲突 | so101_nexus vs robot_descriptions 可能冲突 | 隔离 |
| 构建时间 | 改一行全量重建 | 只重建受影响的 |

### 3.2 Dockerfile.ppo 示例

```dockerfile
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common git wget curl build-essential cmake \
    libegl1 libgles2 libgl1-mesa-glx libvulkan1 mesa-vulkan-drivers ffmpeg \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y python3.12 python3.12-venv python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# PyTorch with CUDA 12.6 (supports V100 sm_70)
RUN pip install --no-cache-dir torch==2.11.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

# so101_nexus with Warp GPU parallel env
RUN pip install --no-cache-dir "so101_nexus[warp,train]" tyro \
    "imageio[ffmpeg]" opencv-python

# Fix: mujoco-warp sensor NVRTC bug
RUN pip install --no-cache-dir "mujoco-warp>=3.10.0.1,<3.12"

ENV MUJOCO_GL=egl
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
WORKDIR /workspace
COPY scripts/ /workspace/scripts/
```

### 3.3 镜像构建流水线

```yaml
# .github/workflows/docker-build.yml
# push to main 自动触发，GHCR + SWR 双写
jobs:
  build-ppo:
    runs-on: ubuntu-latest
    steps:
      - uses: docker/build-push-action@v5
        with:
          context: .
          file: docker/Dockerfile.ppo
          push: true
          tags: |
            ghcr.io/link-seek/so101-ppo:latest
            swr.cn-north-4.myhuaweicloud.com/link-seek/so101-ppo:latest
```

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
- Secrets 已配置：`HC_ACCESS_KEY`, `HC_SECRET_KEY`, `SWR_PASSWORD`, `OBS_AK`, `OBS_SK`

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
  │   ├─ 安装华为云 CLI (pip install hcloud)
  │   ├─ 调用 hcloud API 启动 ECS 7f39cb83
  │   ├─ sleep 120s 等待 GPU 驱动就绪
  │   └─ ECS 启动, runner ecs-0002 自动注册到 GitHub
  │
  ├─ Step 2: pipeline (self-hosted V100, ~90min)
  │   ├─ 2a. 拉取 Docker 镜像 (~2min)
  │   │   docker pull swr.cn-north-4.myhuaweicloud.com/link-seek/so101-train:latest
  │   │   docker pull swr.cn-north-4.myhuaweicloud.com/link-seek/so101-mujoco:latest
  │   │
  │   ├─ 2b. 训练阶段 (~60min)
  │   │   ├─ 启动训练容器: docker run --gpus all so101-train ...
  │   │   ├─ 从 HF Hub 下载数据集 (dobri420/pick-cube-so101-sim)
  │   │   ├─ 加载预训练模型 (lerobot/smolvla_base)
  │   │   ├─ 训练循环 (15K steps):
  │   │   │   for step in range(15000):
  │   │   │       batch = sample_batch(dataset)        # 采样
  │   │   │       pred = policy(batch.obs, batch.lang)  # 前向
  │   │   │       loss = mse(pred, batch.action)        # 损失
  │   │   │       loss.backward(); optimizer.step()     # 反向
  │   │   │       if step % 5000 == 0: save_checkpoint()# 存盘
  │   │   └─ 输出: /data/checkpoints/15000/ (模型权重)
  │   │
  │   ├─ 2c. 评估阶段 (~30min)
  │   │   ├─ 启动评估容器: docker run --gpus all so101-mujoco ...
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
  │       ├─ 上传 checkpoint → obs://so101-sim-pipeline/vla/checkpoints/
  │       ├─ 上传 eval_result → obs://so101-sim-pipeline/vla/results/
  │       └─ 上传 eval_video  → obs://so101-sim-pipeline/vla/results/
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
| 训练步数 | 15K steps | 30M steps (1024 envs 并行) |
| 训练时间 | ~60 min | ~86 min |
| 评估方式 | Grid sweep 325 episodes | 确定性 50 episodes |
| 评估时间 | ~30 min | ~14 sec |
| Docker 镜像 | so101-train + so101-mujoco | so101-ppo (单镜像) |
| 总时间 | ~95 min | ~90 min |

### 6.3 时间线总结

```
T+0min    ── 你点击 Run
T+2min    ── ECS 启动完成, runner 上线
T+4min    ── Docker 镜像拉取完成, 训练开始
T+64min   ── 训练结束, checkpoint 保存
T+64min   ── 评估开始
T+94min   ── 评估结束, 47% success_rate
T+96min   ── OBS 归档完成
T+97min   ── ECS 关闭, 流水线完成
T+97min   ── 你收到 GitHub Actions 通知
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

### 7.4 为什么训练和评估用不同的 Docker 镜像？

```
训练镜像 (so101-train)              评估镜像 (so101-mujoco)
├── lerobot 0.6.1                   ├── robot_descriptions
├── smolvla                         ├── mujoco_env
├── so101_nexus                     ├── 无训练依赖
└── 无 grid sweep 评估              └── 有 grid sweep 评估
```

**原因**：训练需要 lerobot + smolvla（模型训练框架），评估需要 mujoco_env + grid sweep（评估工具链）。两者依赖不同，合在一起会冲突或镜像过大。

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
