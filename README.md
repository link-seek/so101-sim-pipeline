# SO101 SmolVLA 仿真流水线

端到端 VLA 训练流水线：GitHub Actions 触发 → 华为云 V100 ECS 训练 → 仿真回放 → OBS 归档。

## 架构

```
┌──────────────┐     ┌──────────────────────────────────────────┐     ┌──────────────┐
│  GitHub Push │────▶│          docker-build.yml                │     │  VLA Pipeline │
│  / dispatch  │     │  (ubuntu-latest, 并行构建 5 镜像)         │     │  (dispatch)   │
└──────────────┘     │  GHCR → SWR 镜像仓库                     │     └──────┬───────┘
                     └──────────────────────────────────────────┘            │
                                                                               ▼
                     ┌──────────────────────────────────────────┐     ┌──────────────┐
                     │              start-ecs                    │     │              │
                     │  (ubuntu-latest)                          │◀────│  触发流水线   │
                     │  hcloud CLI → BatchStartServers            │     │              │
                     │  等待 120s ECS 启动                       │     └──────────────┘
                     └──────────────────┬───────────────────────┘
                                        │ needs: start-ecs
                                        ▼
                     ┌──────────────────────────────────────────┐
                     │            vla-pipeline                   │
                     │  (self-hosted, Linux, X64, V100)          │
                     │                                          │
                     │  1. SWR 登录 + 拉取镜像                   │
                     │  2. Train SmolVLA (--rename_map)          │
                     │  3. 仿真回放 (replay_demo.py)             │
                     │  4. 上传结果到 OBS                        │
                     └──────────────────┬───────────────────────┘
                                        │ needs: vla-pipeline
                                        ▼
                     ┌──────────────────────────────────────────┐
                     │            stop-ecs                       │
                     │  (ubuntu-latest, if: always())            │
                     │  hcloud CLI → BatchStopServers            │
                     └──────────────────────────────────────────┘
```

## 流水线

| Pipeline | 文件 | 触发 | 运行环境 | 产出 |
|----------|------|------|----------|------|
| Docker Build | `docker-build.yml` | push to main / manual | GitHub Actions (ubuntu) | GHCR + SWR 镜像 |
| VLA Pipeline | `vla-pipeline.yml` | manual dispatch | self-hosted V100 ECS | 训练 checkpoint + 回放视频 → OBS |

## VLA Pipeline 流程

### Job 1: start-ecs (ubuntu-latest)
- 安装 hcloud CLI
- 调用 `BatchStartServers` 启动 V100 ECS
- 等待 120s 确保 ECS 就绪

### Job 2: vla-pipeline (self-hosted V100)
- SWR 登录 + 拉取 `so101-train` / `so101-eval` 镜像
- **训练**: `train_smolvla.py` — SmolVLA base → 20k steps 微调
  - `--rename_map` 映射数据集相机名 (side→camera1, up→camera2)
  - `--wandb.enable=false` (无 WANDB 密钥)
  - checkpoint 保存到 `/data/checkpoints/smolvla/checkpoints/`
- **仿真回放**: `replay_demo.py` — 用最新 checkpoint 在 so101_nexus MuJoCo 中执行任务并录像
  - 社区标准推理管线: `prepare_observation_for_inference` + `preprocess/postprocess`
  - 相机配置: WristCamera + OverheadCamera
  - 输出视频到 `/data/eval/results/replay_pickplace.mp4`
- **OBS 上传**: obsutil 上传 eval 结果 + checkpoints 到 `obs://so101-sim-pipeline/`

### Job 3: stop-ecs (ubuntu-latest, if: always())
- 调用 `BatchStopServers` 关闭 ECS
- `if: always()` 确保无论流水线成功/失败都会关机

## Docker 镜像

| 镜像 | Dockerfile | 用途 |
|------|-----------|------|
| `so101-train` | `Dockerfile.train` | 训练 + 回放 (lerobot + so101_nexus) |
| `so101-eval` | `Dockerfile.eval` | vla-eval 评估 (SQLite 3.45) |
| `so101-model-server` | `Dockerfile.model-server` | 模型推理服务 |

镜像构建在 GHCR，自动同步到华为云 SWR (`swr.cn-north-4.myhuaweicloud.com/link-seek/`)。

## ECS 配置

| 项目 | 值 |
|------|-----|
| Server ID | `7f39cb83-1a5c-4792-b65e-e578d7ddb88d` |
| Region | cn-north-4 |
| GPU | 2× Tesla V100-PCIE-32GB |
| Self-hosted runner | `/data/gh-runner/` (agent: ecs-0002) |
| Runner labels | `[self-hosted, Linux, X64, V100]` |
| 数据盘 | `/data` 789GB |

## Secrets

| Name | 用途 |
|------|------|
| `OBS_AK` | 华为云 AK (OBS + SWR + ECS 控制) |
| `OBS_SK` | 华为云 SK |
| `HF_TOKEN` | HuggingFace Token (模型下载/上传) |

## 使用

```bash
# 触发镜像构建 (push 到 main 自动触发，或手动)
gh workflow run docker-build.yml -R link-seek/so101-sim-pipeline

# 触发完整训练 (20k steps)
gh workflow run vla-pipeline.yml -R link-seek/so101-sim-pipeline

# 触发短训练验证 (100 steps)
gh workflow run vla-pipeline.yml -R link-seek/so101-sim-pipeline -f steps=100

# 跳过训练，仅回放 (需指定 checkpoint 路径)
gh workflow run vla-pipeline.yml -R link-seek/so101-sim-pipeline \
  -f skip_train=true \
  -f checkpoint=/data/checkpoints/smolvla/checkpoints/20000/pretrained_model
```

## OBS 产出

```
obs://so101-sim-pipeline/
├── eval/
│   ├── replay_pickplace.mp4      # 仿真回放视频
│   └── replay_pickplace.json     # 回放报告
└── checkpoints/
    ├── 05000/pretrained_model/   # 5k checkpoint
    ├── 10000/pretrained_model/   # 10k checkpoint
    ├── 15000/pretrained_model/   # 15k checkpoint
    └── 20000/pretrained_model/   # 20k checkpoint (最终)
```
