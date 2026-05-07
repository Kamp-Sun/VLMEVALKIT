#!/bin/bash

set -e

# Make conda activate work reliably in non-interactive shells
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate labshield

# Usage:
#   bash eval_without_fragile.sh MODEL_NAME                    # run all 34 samples without fragile label
#   bash eval_without_fragile.sh MODEL_NAME 10                 # run first 10 samples
#   bash eval_without_fragile.sh Qwen3-VL-30B-A3B-Instruct     # example

MODEL_NAME="${1:-o3}"
MAX_SAMPLES="${2:-0}"

# Fixed multi-view camera keys
CAMERA_KEYS="head_rgbd,torso_rgbd,left_wrist_rgbd,right_wrist_rgbd"

echo "=========================================="
echo "评测模型: ${MODEL_NAME}"
echo "数据集: dataset_without_fragile.json (34个没有透明物体标签的样本)"
echo "最大样本数: ${MAX_SAMPLES} (0表示全部)"
echo "=========================================="

python3 scripts/evaluate.py \
  --model "${MODEL_NAME}" \
  --dataset "data/dataset_without_fragile.json" \
  --max-samples "${MAX_SAMPLES}" \
  --camera-keys "${CAMERA_KEYS}" \
  --system-prompt-file scripts/syspem_prompt.json \
  # --verbose  # uncomment to see detailed API responses

echo ""
echo "=========================================="
echo "评测完成！结果保存在: results/${MODEL_NAME}/"
echo "=========================================="

