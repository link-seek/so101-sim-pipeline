#!/bin/bash
set -e

export MUJOCO_GL=egl
export HF_ENDPOINT=https://hf-mirror.com

mkdir -p /data/pipeline/results

echo "=== SO101 Evaluation Pipeline ==="
echo "Model: $MODEL_REPO"
echo "Dataset: $DATASET_REPO"
echo "Episodes: $NUM_EPISODES"

python3 /data/pipeline/eval_so101_mujoco.py \
  --model-repo "$MODEL_REPO" \
  --dataset-repo "$DATASET_REPO" \
  --num-episodes "$NUM_EPISODES" \
  --hf-token "$HF_TOKEN"

echo "=== Evaluation Complete ==="
ls -la /data/pipeline/results/
