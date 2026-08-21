# SO101 仿真流水线

SO101 机械臂端到端机器人学习流水线：VLA 训练 / PPO 强化学习 / MuJoCo 仿真孪生 / Kaggle 训练 → 华为云 V100 ECS → 仿真回放 → OBS 归档。

## 架构总览

```
                     ┌─────────────────────────────────────────────┐
                     │            docker-build.yml                 │
  GitHub Push ──────▶│  并行构建 5 镜像 + 镜像 2 LIBERO benchmark  │
  / dispatch         │  GHCR → SWR 镜像仓库                       │
                     └─────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────┐
  │                        训练流水线 (3 条)                         │
  │                                                                  │
  │  VLA Pipeline          PPO Pipeline         MuJoCo Pipeline      │
  │  (vla-pipeline.yml)    (ppo-pipeline.yml)   (so101-mujoco-       │
  │                                            pipeline.yml)        │
  │  SmolVLA 微调          PPO 强化学习        Sim Twin 训练+评估    │
  │  真机数据集            Warp 仿真环境       MuJoCo 仿真数据集     │
  │  → 仿真回放            → 评估录像          → MuJoCo eval         │
  │  → OBS 归档            → OBS 归档          → OBS 归档            │
  └──────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────┐
  │                      Kaggle 训练 + 评估                          │
  │                                                                  │
  │  train.yml ──▶ Kaggle Notebook ──▶ evaluate.yml                 │
  │  (推送 notebook)    (GPU 训练)       (V100 SSH 评估 → OBS)       │
  └──────────────────────────────────────────────────────────────────┘

  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
  │  ci.yml          │  │ download-driver  │  │  space/          │
  │  Lint + Type     │  │ NVIDIA 驱动下载  │  │  HF Space        │
  │  Check + Test    │  │ → GitHub Release │  │  Webhook Relay   │
  └──────────────────┘  └──────────────────┘  └──────────────────┘
```

所有 ECS 流水线遵循相同模式：`start-ecs` → `训练/评估` → `stop-ecs (if: always())`。

## 流水线

| Pipeline | 文件 | 触发 | 运行环境 | 产出 |
|----------|------|------|----------|------|
| Docker Build | `docker-build.yml` | push to main / manual | GitHub Actions (ubuntu) | 5 镜像 → GHCR + SWR，2 LIBERO benchmark 镜像 → SWR |
| VLA Pipeline | `vla-pipeline.yml` | manual dispatch / `vla-trigger` | self-hosted V100 ECS | SmolVLA checkpoint + 回放视频 → OBS |
| PPO Pipeline | `ppo-pipeline.yml` | manual dispatch | self-hosted V100 ECS | PPO agent + 评估视频 → OBS |
| MuJoCo Pipeline | `so101-mujoco-pipeline.yml` | manual dispatch / `mujoco-trigger` | self-hosted V100 ECS | Sim SmolVLA checkpoint + MuJoCo eval → OBS |
| Train (Kaggle) | `train.yml` | manual dispatch / `train-trigger` | Kaggle GPU | 训练模型 → HF Hub，自动触发 Evaluate |
| Evaluate | `evaluate.yml` | `evaluate-trigger` | V100 ECS (SSH) | 评估报告 + 视频 → OBS + Artifacts |
| CI | `ci.yml` | push / PR to main | GitHub Actions (ubuntu) | Lint + Type check + Test + Docker build 验证 |
| Download Driver | `download-driver.yml` | manual dispatch | GitHub Actions (ubuntu) | NVIDIA 驱动 → GitHub Release |

## VLA Pipeline (`vla-pipeline.yml`)

SmolVLA 在真机数据集上微调，然后在 MuJoCo 中仿真回放。

### Job 1: start-ecs
- hcloud CLI → `BatchStartServers` 启动 V100 ECS，等待 120s

### Job 2: vla-pipeline (self-hosted V100)
- **预下载**: `predownload_dataset.py` 拉取数据集 + `smolvla_base` 基座模型 (hf-mirror)
- **训练**: `train_smolvla.py` — SmolVLA base → 微调
  - 数据集: `ataghof/so101nexus-cube500-binary`
  - `--rename_map` 映射相机名 (`cam0→camera2, cam1→camera1`)
  - `--batch_size=64`，`--env_eval_freq=2000`，`--save_freq=5000`
  - `--wandb.enable=false`，`--policy.push_to_hub=false`
  - checkpoint → `/data/checkpoints/smolvla/checkpoints/`
- **仿真回放**: `replay_demo.py` — 最新 checkpoint 在 so101_nexus MuJoCo 中执行任务并录像
  - `MUJOCO_GL=egl`，`--max-steps 300`
  - 输出 → `/data/eval/results/replay_pickplace.mp4`
- **OBS 上传**: eval 结果 + checkpoints → `obs://so101-sim-pipeline/`

### Job 3: stop-ecs (`if: always()`)

**输入参数**: `steps` (默认 20000), `dataset_fps` (默认 30), `skip_train`, `checkpoint`

## PPO Pipeline (`ppo-pipeline.yml`)

PPO 强化学习在 Warp 仿真环境中训练，支持大规模并行。

### 流程
- **PPO 训练**: `train_ppo.py` — Warp env，默认 30M timesteps，1024 并行环境
  - `--lift_threshold 0.15` (成功判定: 抬举高度)
  - checkpoint → `/data/checkpoints/ppo/best_agent.pt`
- **PPO 评估**: `eval_ppo.py` — 50 episodes，输出评估视频 + JSON 报告
- **OBS 上传**: 结果 + checkpoint → `obs://so101-sim-pipeline/ppo/`

**输入参数**: `env_id` (默认 `WarpPickLift-v1`), `total_timesteps` (默认 30M), `num_envs` (默认 1024), `seed`, `num_eval_episodes` (默认 50), `lift_threshold` (默认 0.15)

## MuJoCo Pipeline (`so101-mujoco-pipeline.yml`)

Sim Twin 路线：在仿真数据集上训练 SmolVLA，然后在 MuJoCo 中评估。

### 流程
- **下载数据**: `download_sim_dataset.py` 拉取 `dobri420/pick-cube-so101-sim` + `smolvla_base`
- **训练**: `train_smolvla_sim.py` — 在 sim twin 数据集上微调
  - `--save_freq=5000`，`--wandb.enable=false`
  - checkpoint → `/data/checkpoints/smolvla-sim/checkpoints/`
- **MuJoCo 评估**: `eval_mujoco_policy.py` — 支持 `sweep` / `record` 两种模式
- **OBS 上传**: eval 结果 → `mujoco-eval/`，checkpoints → `mujoco-checkpoints/`

**输入参数**: `steps` (默认 20000), `batch_size` (默认 32), `skip_train`, `checkpoint`, `mode` (默认 `sweep`)

## Kaggle 训练 (`train.yml`)

将训练 notebook 推送到 Kaggle GPU 执行，完成后自动触发评估。

### 流程
1. `prepare_kaggle_notebook.py` 注入参数到 notebook
2. `kaggle_push.py` 推送到 Kaggle (Bearer token API)
3. 设置 `HF_TOKEN` 为 Kaggle secret
4. 轮询 Kaggle kernel 状态 (最多 120 分钟)
5. 成功后触发 `evaluate-trigger` → `evaluate.yml`

**输入参数**: `dataset_repo` (默认 `xieyucheng123/so101-dataset`), `model_repo` (默认 `xieyucheng123/so101-act`), `policy_type` (默认 `act`), `training_steps` (默认 20000)

## Evaluate (`evaluate.yml`)

在 V100 ECS 上通过 SSH 执行评估脚本。

### 流程
- `ecs_control.py boot` 启动 ECS (c5b805bd...)
- 等待 SSH 就绪 (最多 10 分钟)
- SSH 执行 `run_eval.sh` 评估
- 下载结果 + 上传到 OBS (`robotwin-assets` bucket)
- `ecs_control.py shutdown` 关闭 ECS (`if: always()`)

## CI (`ci.yml`)

- **Lint**: `ruff check` + `ruff format --check` on `scripts/`
- **Type check**: `mypy` on `scripts/`
- **Test**: `pytest tests/`
- **Docker build**: 验证 `Dockerfile.eval` 可构建

## Docker 镜像

| 镜像 | Dockerfile | 用途 |
|------|-----------|------|
| `so101-train` | `Dockerfile.train` | VLA 训练 + 仿真回放 (lerobot + so101_nexus) |
| `so101-eval` | `Dockerfile.eval` | vla-eval 评估 (SQLite 3.45) |
| `so101-model-server` | `Dockerfile.model-server` | 模型推理服务 |
| `so101-ppo` | `Dockerfile.ppo` | PPO 强化学习 (Warp + Isaac) |
| `so101-mujoco` | `Dockerfile.mujoco` | MuJoCo sim twin 训练 + 评估 |

**镜像仓库**: GHCR (`ghcr.io/link-seek/so101-sim-pipeline/`)，自动同步到华为云 SWR (`swr.cn-north-4.myhuaweicloud.com/link-seek/`)。

**Benchmark 镜像** (从 AllenAI 镜像到 SWR):
- `vla-eval-libero` — LIBERO benchmark
- `vla-eval-libero-pro` — LIBERO-PRO benchmark

## ECS 配置

| 项目 | V100 ECS (训练) | 评估 ECS |
|------|-----------------|----------|
| Server ID | `7f39cb83-1a5c-4792-b65e-e578d7ddb88d` | `c5b805bd-5e8d-4ba5-a5ab-7523244da0fa` |
| IP | — | `1.94.192.234` |
| Region | cn-north-4 | cn-north-4 |
| GPU | 2× Tesla V100-PCIE-32GB | V100 |
| Runner | self-hosted `[Linux, X64, V100]` | SSH |
| 数据盘 | `/data` 789GB | `/data` |

## 配置文件

### `configs/benchmarks/`
LIBERO benchmark 配置 (8 个): `libero_goal`, `libero_object`, `libero_spatial`, `libero_pro_env`, `libero_pro_lan`, `libero_pro_object`, `libero_pro_swap`, `libero_pro_task`

### `configs/model_servers/smolvla_so101.yaml`
SmolVLA 模型推理服务配置: `policy_type: smolvla`，`checkpoint: xieyucheng123/so101-smolvla`，`chunk_size: 10`

### `kaggle-notebook/`
Kaggle 训练/评估 notebook: `train_act.ipynb`, `train_act_v3.ipynb`, `eval_act_v1.ipynb`

### `space/`
HuggingFace Space Webhook Relay (Docker, port 7860) — 接收外部 webhook 触发流水线

## 使用

```bash
# === Docker 镜像构建 ===
gh workflow run docker-build.yml -R link-seek/so101-sim-pipeline

# === VLA Pipeline (SmolVLA 微调 + 仿真回放) ===
gh workflow run vla-pipeline.yml -R link-seek/so101-sim-pipeline
gh workflow run vla-pipeline.yml -R link-seek/so101-sim-pipeline -f steps=100    # 短训练验证
gh workflow run vla-pipeline.yml -R link-seek/so101-sim-pipeline \
  -f skip_train=true \
  -f checkpoint=/data/checkpoints/smolvla/checkpoints/20000/pretrained_model      # 仅回放

# === PPO Pipeline (强化学习) ===
gh workflow run ppo-pipeline.yml -R link-seek/so101-sim-pipeline
gh workflow run ppo-pipeline.yml -R link-seek/so101-sim-pipeline \
  -f env_id=WarpPickLift-v1 -f total_timesteps=10000000 -f num_envs=512

# === MuJoCo Pipeline (Sim Twin) ===
gh workflow run so101-mujoco-pipeline.yml -R link-seek/so101-sim-pipeline
gh workflow run so101-mujoco-pipeline.yml -R link-seek/so101-sim-pipeline -f mode=record

# === Kaggle 训练 ===
gh workflow run train.yml -R link-seek/so101-sim-pipeline
gh workflow run train.yml -R link-seek/so101-sim-pipeline \
  -f policy_type=smolvla -f training_steps=30000

# === NVIDIA 驱动下载 ===
gh workflow run download-driver.yml -R link-seek/so101-sim-pipeline
```

## OBS 产出

```
obs://so101-sim-pipeline/
├── eval/                           # VLA Pipeline 回放
│   ├── replay_pickplace.mp4
│   └── replay_pickplace.json
├── checkpoints/                    # VLA Pipeline checkpoints
│   ├── 05000/pretrained_model/
│   ├── 10000/pretrained_model/
│   ├── 15000/pretrained_model/
│   └── 20000/pretrained_model/
├── ppo/                            # PPO Pipeline
│   ├── train_result.json
│   ├── eval_result.json
│   ├── eval_video.mp4
│   └── checkpoints/best_agent.pt
├── mujoco-eval/                    # MuJoCo Pipeline 评估结果
└── mujoco-checkpoints/             # MuJoCo Pipeline checkpoints
```

Kaggle 评估结果上传到 `obs://robotwin-assets/so101-eval/`。

## 脚本一览

| 脚本 | 用途 |
|------|------|
| `train_smolvla.py` | SmolVLA 微调训练 (真机数据集) |
| `train_smolvla_sim.py` | SmolVLA 微调训练 (sim twin 数据集) |
| `train_ppo.py` | PPO 强化学习训练 |
| `replay_demo.py` | 仿真回放录像 |
| `eval_mujoco_policy.py` | MuJoCo 策略评估 (sweep/record) |
| `eval_ppo.py` | PPO 策略评估 |
| `eval_vla.py` | VLA 评估 |
| `eval_so101_mujoco.py` | SO101 MuJoCo 评估 |
| `predownload_dataset.py` | 预下载数据集 |
| `download_sim_dataset.py` | 下载 sim twin 数据集 |
| `ecs_control.py` | ECS 启停控制 |
| `upload_obs.py` | OBS 上传 |
| `kaggle_push.py` | 推送 notebook 到 Kaggle |
| `prepare_kaggle_notebook.py` | 准备 Kaggle notebook |
| `collect_sim_data.py` | 采集仿真数据 |
| `verify_dataset.py` | 验证数据集 |
| `lerobot_train_patched.py` | LeRobot 训练 (patched) |
