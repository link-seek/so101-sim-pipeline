# Ch1：搭建仿真训练基础设施

> SO101 仿真评测教程 · 第一章

---

## 1. 架构总览

整个训练-评估流程很简单：**一台云服务器 + 一个 Docker 镜像 + 一条命令**。

```
┌─────────────────────────────────────────────────┐
│          华为云 ECS（V100 32GB GPU）               │
│                                                   │
│   docker run --gpus all so101-mujoco:latest \     │
│     train_smolvla_sim.py --steps 20000            │
│                                                   │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│   │ OBS 数据集 │──→│ 训练脚本  │──→│ checkpoint│  │
│   └──────────┘    └──────────┘    └─────┬────┘  │
│                                         │        │
│                                    ┌────▼────┐  │
│                                    │ 评估脚本 │  │
│                                    └────┬────┘  │
│                                         │        │
│                              ┌──────────▼──────┐ │
│                              │ 评估结果 + 视频   │ │
│                              └──────────┬──────┘ │
└─────────────────────────────────────────┼────────┘
                                          │
                              ┌───────────▼────────┐
                              │  OBS / 本地磁盘      │
                              │  checkpoint + 结果   │
                              └────────────────────┘
```

### 1.1 数据流

```
数据集 (OBS) ──→ Docker 容器 ──→ 训练脚本 ──→ checkpoint ──→ 评估脚本 ──→ 评估结果
```

训练和评估在同一容器内顺序执行。一个镜像完成整个闭环。

---

## 2. Docker 镜像

仓库提供 4 个 Docker 镜像，各有职责：

| 镜像 | 用途 | 验证状态 |
|------|------|----------|
| `so101-ppo` | PPO 训练 + 评估（MuJoCo + Warp GPU） | ✅ 任务成功率 100% |
| `so101-mujoco` | SmolVLA 训练 + Grid Sweep 评测 | ✅ Grid Sweep 47% |
| `so101-eval` | LIBERO 评测 + 数据采集 | ✅ 已验证（0%，模型不兼容） |
| `so101-train` | SmolVLA 训练 + 回放验证 | 🔧 调试中 |

### 2.1 各镜像详解

#### `so101-ppo` — 强化学习训练与评估

一个镜像内完成「训练 + 评估」闭环。RL 的评估就是在同一仿真环境里跑几百个 episode 算 success_rate。

- **关键脚本**：`train_ppo.py`（CleanRL 风格）、`eval_ppo.py`
- **输入**：`env_id`（如 `WarpPickLift-v1`）、超参数
- **输出**：`best_agent.pt`、`eval_result.json`、`eval_video.mp4`

#### `so101-mujoco` — 仿真孪生训练与网格扫描

Sim twin 路线专属：在 MuJoCo 里训练 SO101 仿真孪生模型，再做 grid sweep 评测。

- **关键脚本**：`download_sim_dataset.py`、`train_smolvla_sim.py`、`eval_mujoco_policy.py`
- **输入**：仿真数据集、SmolVLA base 权重
- **输出**：checkpoint、成功率矩阵、热力图

#### `so101-eval` — LIBERO 标准化 benchmark

通过 `vla-eval` harness 调度 LIBERO / LIBERO-PRO 等多个 benchmark。

- **关键脚本**：`eval_vla.py`
- **输入**：模型 checkpoint、`configs/benchmarks/*.yaml`
- **输出**：每任务 success_rate、SQLite 数据库

#### `so101-train` — VLA 训练与回放

在 LeRobot 格式数据集上微调 SmolVLA（行为克隆 / SFT），训练后回放验证。

- **关键脚本**：`train_smolvla.py`、`replay_demo.py`
- **输入**：LeRobot 数据集、SmolVLA base 权重
- **输出**：checkpoint、回放视频 mp4

> **一句话总结**：`train` 和 `ppo` 造模型，`mujoco` 和 `eval` 测模型。

### 2.2 为什么有 4 个镜像？

```
so101-ppo        so101-mujoco          so101-train      so101-eval
├ so101_nexus    ├ lerobot[training]   ├ lerobot[training]  ├ vla-eval
├ mujoco-warp   ├ vla-eval            ├ so101_nexus       ├ mujoco
└ PPO 训练+评估  ├ mujoco + grid sweep └ 无 grid sweep     └ SQLite
                └ 训练+评估一体        ← 仅训练（旧版）     ← 仅评估
```

训练依赖和评估依赖有差异，合在一起会冲突或镜像过大。PPO 和 MuJoCo 各自打包成单镜像完成闭环。

---

## 3. 云服务器准备

### 3.1 选择服务器

| 配置 | 推荐 |
|------|------|
| GPU | V100 32GB（必须，SmolVLA 训练需要） |
| 系统 | Ubuntu 20.04+ |
| Docker | 安装 NVIDIA Container Toolkit |
| 网络 | 能访问 OBS (华为云对象存储) |

### 3.2 手动开关机

V100 ECS 按小时计费（约 ¥30/h）。不用时关机，用时开机。

- **开机**：华为云控制台 → ECS → 选择实例 → 启动
- **关机**：华为云控制台 → ECS → 选择实例 → 关闭

> **省钱提示**：训练任务一天只跑 2-3 次，每次 1-2 小时。常开每月 ¥21000+，手动开关机后 ¥1800/月。

### 3.3 SSH 连接

```bash
ssh root@<ECS公网IP>
```

---

## 4. 实战：用 Docker 跑你的第一次训练

### 4.1 前置条件

- 一台有 GPU 的云服务器（V100 32GB）
- Docker + NVIDIA Container Toolkit 已安装
- 能访问 OBS

### 4.2 拉取镜像

```bash
docker pull swr.cn-north-4.myhuaweicloud.com/link-seek/so101-mujoco:latest
```

### 4.3 运行训练

```bash
# PPO 训练（~90 分钟）
docker run --gpus all \
  -v /data:/data \
  swr.cn-north-4.myhuaweicloud.com/link-seek/so101-ppo:latest \
  python /workspace/scripts/train_ppo.py \
    --env-id WarpPickLift-v1 \
    --total-timesteps 30000000 \
    --num-envs 1024 \
    --seed 1
```

```bash
# SmolVLA 仿真训练 + Grid Sweep（训练 ~20 小时 + sweep ~10 小时，独占 GPU）
docker run --gpus all --shm-size=16g \
  -v /data:/data \
  -v /data/hf_cache:/root/.cache/huggingface \
  -e HF_HUB_DISABLE_XET=1 \
  swr.cn-north-4.myhuaweicloud.com/link-seek/so101-mujoco:latest \
  bash -c "python /workspace/scripts/train_smolvla_sim.py --steps 20000 && python /workspace/scripts/eval_mujoco_policy.py --checkpoint /data/checkpoints/smolvla-sim/checkpoints/015000/pretrained_model --mode sweep"
```

> 已验证（Discussion #20）：15K checkpoint sweep **153/325 = 47%**，与历史 run `32221378632` 一字不差。
> 三个坑：① `--shm-size` 必加（否则 DataLoader 起 worker 即崩）；② 国内网必备 `-e HF_HUB_DISABLE_XET=1` + HF 缓存挂载，否则数据集下载卡死；
> ③ **独占 GPU 跑**——并行训练任务会污染 PPO 收敛（实测 0.98→0.02）；④ sweep 用 `capture_output=True`，全程静默约 10h 属正常，结束才一次性输出。

### 4.4 查看日志

```bash
# 查看容器日志
docker logs <container_id> -f

# 查看训练进度
docker logs <container_id> 2>&1 | grep -E "(success|reward|SPS|Error)"
```

### 4.5 预期时间

| 任务 | 耗时 | 输出 |
|------|------|------|
| PPO 训练 + 评估 | ~90 min | `best_agent.pt`、`eval_result.json` |
| SmolVLA 训练 + Grid Sweep | 训练 ~20 h + sweep ~10 h | checkpoint、成功率矩阵、热力图 |

### 4.6 评估结果

```bash
# 评估完成后查看结果
cat /data/eval_result.json

# PPO 结果示例
{
  "success_rate": 1.0,
  "avg_reward": 1.440,
  "avg_steps": 50.0
}
```

---

## 5. OBS 归档

训练产物可以上传到华为云 OBS（对象存储）长期保存：

```bash
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

| 产物 | 大小 | 用途 |
|------|------|------|
| best_agent.pt | 591KB | PPO 策略权重 |
| eval_result.json | ~1KB | 评估指标 |
| eval_video.mp4 | ~5MB | 评估视频回放 |

---

## 6. 设计决策

### 6.1 为什么用 Docker 而不是 conda？

| 方案 | conda | Docker |
|------|-------|--------|
| 依赖隔离 | 可能互相污染 | 完全隔离 |
| 可复现 | "works on my machine" | 镜像一致，任何机器结果相同 |
| GPU 支持 | 手动配 CUDA | nvidia/cuda 基础镜像 |

**核心价值**：lerobot + so101_nexus + MuJoCo 依赖链复杂，Docker 保证所有人环境一致。

### 6.2 为什么用 OBS 存储？

| 内容 | 存储位置 |
|------|----------|
| 模型 + 数据集 | OBS（上传快、成本低、`obsutil` 一键上传） |
| 评估结果（JSON + 视频 + 截图） | OBS |

所有产物统一存放在 OBS，不依赖外部服务。

---

## 7. 踩坑复盘

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

**教训**：V100 是 2017 年的架构（sm_70），新 CUDA 版本可能不再默认包含。查 PyTorch CUDA 兼容表确认。

### 坑 3：云服务器忘关机

**现象**：一次训练失败后，服务器开了一整夜，产生意外费用。

**教训**：训练完成后**立即关机**。设置闹钟提醒，或者用脚本在训练结束后自动关机。

---
