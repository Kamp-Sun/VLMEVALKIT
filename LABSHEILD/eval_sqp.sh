#! /bin/bash

set -e

# Make conda activate work reliably in non-interactive shells
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate labshield

# Usage:
#   bash eval_sqp.sh                      # run all samples with default model
#   bash eval_sqp.sh 10                   # run first 10 samples with default model
#   bash eval_sqp.sh 10 o3                # run first 10 samples with model=o3
#   bash eval_sqp.sh 0 gpt-5.2            # run all samples with model=gpt-5.2
#* - 6个 gpt-5-mini gpt-5.2 GPT4o gpt-5-nano o3 o4-mini
#* - GeminiFlashLite2-5, Gemini3ProPreview, Gemini3FlashPreview, GeminiFlash2-5
#* - Claude4V_Sonnet Claude4_Sonnet Claude4_Opus Claude3-Haiku
#* - Qwen3-VL-4B-Instruct Qwen3-VL-32B-Instruct 
#* - Qwen3-VL-4B-Thinking Qwen3-VL-32B-Thinking Qwen3-VL-30B-A3B-Instruct Qwen3-VL-32B-Instruct Qwen3-VL-30B-A3B-Thinking
MODEL_NAME="${1:-o3}"
MAX_SAMPLES="${2:-0}"
# Fixed multi-view camera keys (edit here if you want single-view for speed)
CAMERA_KEYS="head_rgbd,torso_rgbd,left_wrist_rgbd,right_wrist_rgbd"
echo "=========================================="
echo "现在正在评估的模型是: ${MODEL_NAME}"
echo "=========================================="

python3 scripts/evaluate.py \
  --model "${MODEL_NAME}" \
  --max-samples "${MAX_SAMPLES}" \
  --camera-keys "${CAMERA_KEYS}" \
  --system-prompt-file scripts/syspem_prompt.json \
  # --verbose  # uncomment to see detailed API responses

# conda activate labshield