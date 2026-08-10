#!/bin/bash
set -e

export MUJOCO_GL=egl
export HF_ENDPOINT=https://hf-mirror.com

mkdir -p /data/eval/results

echo "=== VLA Evaluation Pipeline ==="
echo "Checkpoint: ${CHECKPOINT:-xieyucheng123/so101-smolvla}"
echo "Benchmarks: ${BENCHMARKS:-all}"

python3 /workspace/scripts/eval_vla.py \
  --checkpoint "${CHECKPOINT:-xieyucheng123/so101-smolvla}" \
  ${BENCHMARKS:+--benchmarks $BENCHMARKS}

echo "=== Evaluation Complete ==="
ls -la /data/eval/results/
