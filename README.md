# SO101 LeRobot 仿真 CI/CD 流水线

## 架构

```
GitHub Push → CI (lint/test)
            → Train (Kaggle GPU) → 模型推 HF Hub
            → Evaluate (V100 ECS) → MuJoCo 渲染视频
```

## 流水线

| Pipeline | 触发 | 运行环境 | 产出 |
|----------|------|----------|------|
| CI | push/PR | GitHub Actions | lint/test 通过 |
| Train | manual/dispatch | Kaggle T4/P100 | HF model repo |
| Evaluate | Train完成 | V100 ECS | 仿真视频 |

## Secrets

| Name | Value |
|------|-------|
| HF_TOKEN | HuggingFace Write Token |
| KAGGLE_USERNAME | Kaggle 用户名 |
| KAGGLE_KEY | Kaggle API Key |
| HW_AK | 华为云 AK |
| HW_SK | 华为云 SK |

## 使用

```bash
# 手动触发训练
gh workflow run train.yml \
  -f dataset_repo=xieyucheng123/so101-dataset \
  -f model_repo=xieyucheng123/so101-act \
  -f policy_type=act \
  -f training_steps=20000

# 手动触发评测
gh workflow run evaluate.yml \
  -f model_repo=xieyucheng123/so101-act
```

## V100 ECS

- EIP: 1.94.192.234
- Server ID: c5b805bd-5e8d-4ba5-a5ab-7523244da0fa
- Region: cn-north-4
- GPU: V100 × 2
- OS: Ubuntu 22.04 + CUDA 11.4 (待升级至 12.8)
