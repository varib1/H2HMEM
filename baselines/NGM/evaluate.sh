#!/bin/bash

# ======================================================
# NGM Memory Evaluator – Batch Processing (Linux/macOS)
# ======================================================

# --- 配置区（请根据实际情况修改）---
BASE_DIR=""                # 对话数据根目录
API_KEY="your-api-key"
VLM_MODEL=""
BASE_URL=""

# NGM 参数
ENCODER_METHOD="GMEEncoder"
ENCODER_PATH="Alibaba-NLP/gme-Qwen2-VL-7B-Instruct"
RETRIEVAL_TOPK=5
SIMILARITY_THRESHOLD=0.7
TRAVERSAL_STRATEGY="breadth_first"
MAX_DEPTH=3
MAX_NODES=5

# 并行设置（本脚本顺序执行，如需并行可自行修改）
# PYTHON_CMD="python3"
# ------------------------------------

PYTHON_CMD="python3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/NGM.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: $PYTHON_SCRIPT not found!"
    exit 1
fi

START_DIALOGUE=1
END_DIALOGUE=20

mkdir -p "$OUTPUT_BASE"

for i in $(seq $START_DIALOGUE $END_DIALOGUE); do
    CONV_DIR="${BASE_DIR}/dialogue${i}"
    
    if [ ! -d "$CONV_DIR" ]; then
        echo "Warning: $CONV_DIR not found, skipping"
        continue
    fi
    
    echo "========================================"
    echo "Processing dialogue${i} ..."
    echo "========================================"
    
    $PYTHON_CMD "$PYTHON_SCRIPT" \
        --conversations_dir "$CONV_DIR" \
        --api_key "$API_KEY" \
        --vlm_model "$VLM_MODEL" \
        --base_url "$BASE_URL" \
        --encoder_method "$ENCODER_METHOD" \
        --encoder_path "$ENCODER_PATH" \
        --retrieval_topk $RETRIEVAL_TOPK \
        --similarity_threshold $SIMILARITY_THRESHOLD \
        --traversal_strategy "$TRAVERSAL_STRATEGY" \
        --max_depth $MAX_DEPTH \
        --max_nodes $MAX_NODES \
        --verbose
    
    if [ $? -eq 0 ]; then
        echo "✓ dialogue${i} completed"
    else
        echo "✗ dialogue${i} failed"
    fi
    echo ""
done

echo "========================================"
echo "All dialogues processed.
echo "========================================"