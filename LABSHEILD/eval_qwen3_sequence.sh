#! /bin/bash

set -e

# Make conda activate work reliably in non-interactive shells
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate labshield

# 要评测的模型列表
MODELS=(
    "Qwen3-VL-32B-Instruct"
    "Qwen3-VL-30B-A3B-Instruct"
    "Qwen3-VL-30B-A3B-Thinking"
)

# 可选：从命令行参数获取最大样本数（默认运行所有样本）
MAX_SAMPLES="${1:-0}"

echo "=========================================="
echo "开始顺序评测 Qwen3-VL 模型"
echo "总共 ${#MODELS[@]} 个模型"
echo "最大样本数: ${MAX_SAMPLES} (0表示全部)"
echo "=========================================="

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_sqp.sh"

# 检查 eval_sqp.sh 是否存在
if [ ! -f "${EVAL_SCRIPT}" ]; then
    echo "错误: 找不到 ${EVAL_SCRIPT}"
    exit 1
fi

# 循环执行每个模型的评测
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_NUM=$((i + 1))
    
    echo ""
    echo "=========================================="
    echo "[${MODEL_NUM}/${#MODELS[@]}] 开始评测: ${MODEL}"
    echo "=========================================="
    echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    
    # 调用 eval_sqp.sh，传入模型名和最大样本数
    if [ "${MAX_SAMPLES}" = "0" ]; then
        # 运行所有样本
        bash "${EVAL_SCRIPT}" "${MODEL}" 0
    else
        # 运行指定数量的样本
        bash "${EVAL_SCRIPT}" "${MODEL}" "${MAX_SAMPLES}"
    fi
    
    EXIT_CODE=$?
    
    echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo "错误: ${MODEL} 评测失败，退出码: ${EXIT_CODE}"
        echo "是否继续下一个模型？(y/n)"
        read -r CONTINUE
        if [ "${CONTINUE}" != "y" ] && [ "${CONTINUE}" != "Y" ]; then
            echo "用户选择停止，退出脚本"
            exit $EXIT_CODE
        fi
    else
        echo "✓ ${MODEL} 评测完成"
    fi
    
    echo ""
done

echo "=========================================="
echo "所有模型评测完成！"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

