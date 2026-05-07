#! /bin/bash

set -e

# Make conda activate work reliably in non-interactive shells
source ~/miniconda3/etc/profile.d/conda.sh
conda activate labshield

# Usage:c
#   bash eval.sh                    # run all samples (joint QA+PRP)
#   bash eval.sh 10                 # run first 10 samples
#   bash eval.sh 10 head_rgbd       # run first 10 samples with single view (FASTER)
#   bash eval.sh 0 head_rgbd,torso_rgbd  # run all samples with multiple views
MAX_SAMPLES="${1:-0}"
CAMERA_KEYS="${2:-head_rgbd,torso_rgbd,left_wrist_rgbd,right_wrist_rgbd}"
MODEL_NAME="VILA1.5-13b"
# Note: Must use the exact name from vlmeval/config.py, NOT the API model name!
# Correct model names in config:
# - GeminiPro2-5 (uses gemini-2.5-pro API)
# - GeminiFlash2-5 (uses gemini-2.5-flash API)
# - GeminiPro1-5 (uses gemini-1.5-pro API)
# - GeminiFlash1-5 (uses gemini-1.5-flash API)

# Joint QA+PRP evaluation (one API call per sample)
# Model options: GLM4V_PLUS, GPT4o, QwenVLMax, Claude3V_Sonnet, etc.
echo "=========================================="
echo "现在正在评估的模型是: ${MODEL_NAME}"
echo "=========================================="
python3 scripts/evaluate.py \
  --model "${MODEL_NAME}" \
  --max-samples "${MAX_SAMPLES}" \
  --camera-keys "${CAMERA_KEYS}" \
  --system-prompt-file scripts/syspem_prompt.json \
  --verbose  # uncomment to see detailed API responses

# conda activate labshield