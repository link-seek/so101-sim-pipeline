# Ch1：搭建仿真训练基础设施

> SO101 仿真评测教程 · 第一章  
> 实战项目：[link-seek/so101-sim-pipeline](https://github.com/link-seek/so101-sim-pipeline)

---

## 1. 架构总览

我们的流水线不是在本地跑训练，而是把整个训练-评估流程自动化为 GitHub Actions CI/CD：

```
GitHub Actions dispatch（手动触发）
  │
  ├─ start-ecs (ubuntu-latest)
  │    hcloud CLI → BatchStartServers → sleep 120s
  │    （启动华为云 V100 ECS，等待 GPU 就绪）
  │
  ├─ pipeline (self-hosted, Linux, X64, V100)
  │    ① SWR 登录 + 拉取 Docker 镜像
  │    ② 训练（lerobot-train / PPO CleanRL）
  │    ③ 仿真评估（MuJoCo replay / grid sweep）
  │    ④ 上传 OBS 归档（checkpoint + video + result）
  │
  └─ stop-ecs (ubuntu-latest, if: always())
       hcloud CLI → BatchStopServers
       （无论成功失败都关机省钱）
```

**为什么这样设计？**

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
