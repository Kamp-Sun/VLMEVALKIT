#!/bin/bash

set -e

# Make conda activate work reliably in non-interactive shells
source ~/miniconda3/etc/profile.d/conda.sh
conda activate robobrain2_5

# =============================================================================
# RoboBrain Evaluation Script
# =============================================================================
# Usage:
#   bash robobrain_eval.sh                                # 使用默认 robobrain2.5-8b, 运行所有样本
#   bash robobrain_eval.sh robobrain2.5-8b                # 指定模型, 运行所有样本
#   bash robobrain_eval.sh robobrain2.0-32b 10             # 指定模型, 运行前10个样本
#   bash robobrain_eval.sh robobrain2.0-3b 10 head_rgbd    # 指定模型, 样本数, 单视角
#
# 可用的 RoboBrain 模型:
#   - robobrain2.0-3b   (BAAI/RoboBrain2.0-3B)
#   - robobrain2.0-32b  (BAAI/RoboBrain2.0-32B)
#   - robobrain2.5-8b   (BAAI/RoboBrain2.5-8B-NV) - 默认, 推荐
# =============================================================================

# 参数解析
ROBOBRAIN_MODEL="${1:-robobrain2.5-8b}"
MAX_SAMPLES="${2:-0}"
CAMERA_KEYS="${3:-head_rgbd,torso_rgbd,left_wrist_rgbd,right_wrist_rgbd}"

echo "=========================================="
echo "RoboBrain 评估"
echo "模型: ${ROBOBRAIN_MODEL}"
echo "样本数: ${MAX_SAMPLES} (0=全部)"
echo "相机视角: ${CAMERA_KEYS}"
echo "=========================================="

python3 scripts/evaluate.py \
  --use-robobrain \
  --robobrain-model "${ROBOBRAIN_MODEL}" \
  --max-samples "${MAX_SAMPLES}" \
  --camera-keys "${CAMERA_KEYS}" \
  --system-prompt-file scripts/syspem_prompt.json
  --verbose

# 注意: Judge 模型仍然使用 GPT4o (需要设置 OPENAI_API_KEY 环境变量)