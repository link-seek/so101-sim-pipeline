# 附录：环境配置与速查

> SO101 仿真评测教程 · 附录  
> 实战项目：[link-seek/so101-sim-pipeline](https://github.com/link-seek/so101-sim-pipeline)

---

## A. 环境配置速查

### A.1 ECS 配置

| 项目 | 值 |
|------|-----|
| 云服务商 | 华为云 cn-north-4 |
| ECS ID | `7f39cb83-1a5c-4792-b65e-e578d7ddb88d` |
| GPU | NVIDIA V100 32GB (sm_70) |
| OS | Ubuntu 22.04 |
| Runner | `ecs-0002` (self-hosted, Linux, X64, V100) |
| 计费 | 按小时 ~¥30/h，用完即关 |

### A.2 Docker 镜像

| 镜像 | SWR 地址 | 大小 |
|------|----------|------|
| so101-train | `swr.cn-north-4.myhuaweicloud.com/link-seek/so101-train:latest` | ~8GB |
| so101-ppo | `swr.cn-north-4.myhuaweicloud.com/link-seek/so101-ppo:latest` | ~6GB |
| so101-mujoco | `swr.cn-north-4.myhuaweicloud.com/link-seek/so101-mujoco:latest` | ~7GB |
| so101-eval | `swr.cn-north-4.myhuaweicloud.com/link-seek/so101-eval:latest` | ~6GB |
| so101-model-server | `swr.cn-north-4.myhuaweicloud.com/link-seek/so101-model-server:latest` | ~6GB |

### A.3 软件版本

| 软件 | 版本 | 备注 |
|------|------|------|
| CUDA | 12.6.3 | 包含 sm_70 (V100) |
| PyTorch | 2.11.0 | cu126 wheel |
| LeRobot | 0.6.1 | smolvla, training extras |
| so101_nexus | 0.5.1 | warp, train extras |
| mujoco-warp | >=3.10.0.1,<3.12 | 修复 sensor bug |
| MuJoCo | (随 so101_nexus) | MUJOCO_GL=egl |
| Python | 3.12 | venv 隔离 |

### A.4 环境变量

```bash
# GPU 渲染
export MUJOCO_GL=egl
export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# HuggingFace 镜像（国内加速）
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_LEROBOT_HOME=/data/datasets

# 华为云认证
export HC_ACCESS_KEY=...
export HC_SECRET_KEY=...
```

---

## B. 常用命令

### B.1 触发流水线

```bash
# PPO 流水线
gh workflow run ppo-pipeline.yml \
  -f env_id=WarpPickLift-v1 \
  -f total_timesteps=30000000 \
  -f num_envs=1024 \
  -f seed=1

# MuJoCo VLA 流水线
gh workflow run so101-mujoco-pipeline.yml \
  -f steps=20000 \
  -f batch_size=32

# VLA 流水线（ataghof，已终止）
gh workflow run vla-pipeline.yml \
  -f steps=20000 \
  -f batch_size=32

# Docker 构建
gh workflow run docker-build.yml
```

### B.2 监控运行

```bash
# 查看所有运行
gh run list --limit 20

# 查看特定流水线
gh run list --workflow=ppo-pipeline.yml --limit 5

# 实时监控
gh run watch

# 查看日志
gh run view <run_id> --log

# 查看失败日志
gh run view <run_id> --log-failed

# 搜索日志中的关键信息
gh run view <run_id> --log | grep -E "(success|reward|SPS|Error|loss)"
```

### B.3 Runner 管理

```bash
# 查看 runner 状态
gh api repos/link-seek/so101-sim-pipeline/actions/runners

# 查看 runner 详情
gh api repos/link-seek/so101-sim-pipeline/actions/runners --jq '.runners[] | {name, status, busy, labels: [.labels[].name]}'
```

### B.4 Discussion 管理

```bash
# 列出所有 Discussion
gh api graphql -f query='query { repository(owner: "link-seek", name: "so101-sim-pipeline") { discussions(first: 10) { nodes { number title comments { totalCount } } } } }'

# 查看特定 Discussion
gh api graphql -f query='query { repository(owner: "link-seek", name: "so101-sim-pipeline") { discussion(number: 4) { title body } } }'
```

### B.5 OBS 操作

```bash
# 安装 obsutil
wget https://obs-community.obs.cn-north-4.myhuaweicloud.com/obsutil/obsutil_linux.tar.gz
tar xzf obsutil_linux.tar.gz

# 配置
./obsutil config -i=$OBS_AK -k=$OBS_SK -e=obs.cn-north-4.myhuaweicloud.com

# 上传
./obsutil cp /data/checkpoints/ obs://so101-sim-pipeline/checkpoints/ -r -f

# 下载
./obsutil cp obs://so101-sim-pipeline/ppo/eval_video.mp4 . -f
```

---

## C. 数据集对比表

| 数据集 | 来源 | Episodes | 相机 | FPS | 视觉域 | 用途 | 状态 |
|--------|------|----------|------|-----|--------|------|------|
| `shattori/so101_pick_place_thor` | 真机遥操作 | 100 | wrist+overhead (2) | 30 | 真机 | P0 验证 | 已弃用 |
| `ataghof/so101nexus-cube500-binary` | 仿真 scripted expert | 500 | cam0+cam1 (2) | 33 | 仿真≠评测 | 方案 A | ❌ 终止 |
| `dobri420/pick-cube-so101-sim` | 仿真 sim twin | - | camera1/2/3 (3) | - | 仿真=评测 | 方案 B | ✅ 当前 |
| `johnsutor/MuJoCoPickAndPlace-v1` | 仿真遥操作 | 10 | wrist+overhead (2) | 30 | 仿真=仿真 | 备选 | 数据少 |

### rename_map 配置

```json
// ataghof: cam0/cam1 → overhead/wrist
{
  "observation.images.cam0": "observation.images.overhead",
  "observation.images.cam1": "observation.images.wrist"
}

// shattori: wrist/overhead → camera1/camera2
{
  "wrist": "camera1",
  "overhead": "camera2"
}

// dobri420: 无需 rename_map，3 相机原生匹配
```

---

## D. Discussion 索引

| # | 标题 | 分类 | 评论数 | 链接 |
|---|------|------|--------|------|
| 1 | SmolVLA 仿真回放验证：P0 确认，发现 P1 | General | 1 | [链接](https://github.com/link-seek/so101-sim-pipeline/discussions/1) |
| 2 | PPO 纯仿真流水线设计方案 | Ideas | 2 | [链接](https://github.com/link-seek/so101-sim-pipeline/discussions/2) |
| 3 | SO101 仿真测评路线图 | General | 0 | [链接](https://github.com/link-seek/so101-sim-pipeline/discussions/3) |
| 4 | 方案 A→B 完整迭代 | General | 8 | [链接](https://github.com/link-seek/so101-sim-pipeline/discussions/4) |

---

## E. 教程文件索引

| 文件 | 章节 |
|------|------|
| `so101-tutorial-ch0-prologue.md` | 序章：为什么仿真训练 |
| `so101-tutorial-ch1-infrastructure.md` | Ch1：基础设施搭建 |
| `so101-tutorial-ch2-ppo-baseline.md` | Ch2：PPO 纯仿真 Baseline |
| `so101-tutorial-ch3-vla-intro.md` | Ch3：SmolVLA 仿真训练入门 |
| `so101-tutorial-ch4-debug-journey.md` | Ch4：Debug 实战 |
| `so101-tutorial-ch5-evaluation.md` | Ch5：评测方法论 |
| `so101-tutorial-ch6-optimization.md` | Ch6：优化进阶 |
| `so101-tutorial-appendix.md` | 附录（本文） |
| `so101-tutorial-plan.md` | 教程规划 |
| `so101-sim-eval-summary.md` | 项目阶段性总结 |

---

## F. 关键 Commit 索引

| Commit | 内容 |
|--------|------|
| `f53dd6f` | Gripper 转换 bug 修复（用官方 dataset_row_to_sim_qpos） |
| `bcd3e30` | Action Chunking 修正（移除外部 queue） |
| `4efb51c` | camera3 分布不匹配修复 |
| `2cfb130` | PPO 流水线交付物（Dockerfile + scripts + workflow） |
| `8afd173` | mujoco-warp 版本修复 |
| `b09f762` | PPO v2: lift_threshold=0.15 |
| `8fb2c99` | so101-mujoco Docker 镜像 |

---

> **返回**：[序章](so101-tutorial-ch0-prologue.md) | [教程规划](so101-tutorial-plan.md) | [项目总结](so101-sim-eval-summary.md)
