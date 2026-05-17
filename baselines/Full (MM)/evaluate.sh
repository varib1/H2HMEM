#!/bin/bash

# ======================================================
# Multi‑Modal Memory Evaluator – Batch Processing Script
# ======================================================

# --- 配置区（请根据实际情况修改）---
BASE_DIR=""           # 对话数据根目录
API_KEY="your-api-key-here"
MODEL=""
BASE_URL=""

MAX_CONTEXT_TOKENS=4096
MAX_IMAGES=10
MAX_WORKERS=3
MAX_API_CONCURRENCY=6
START_DIALOGUE=1
END_DIALOGUE=20
# ------------------------------------

echo "======================================================"
echo "Multi-Modal Memory Evaluator - Batch Processing"
echo "Dialogues $START_DIALOGUE to $END_DIALOGUE"
echo "======================================================"

SCRIPT_DIR="" # Set this to the directory where FullMM.py is located if not the current directory
PYTHON_SCRIPT="$SCRIPT_DIR/FullMM.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: $PYTHON_SCRIPT not found!"
    exit 1
fi

SUCCESS=0
FAIL=0
SKIP=0

for i in $(seq $START_DIALOGUE $END_DIALOGUE); do
    CONV_DIR="${BASE_DIR}/dialogue${i}"
    
    echo "------------------------------------------------------"
    echo "Processing dialogue${i} ..."
    if [ ! -d "$CONV_DIR" ]; then
        echo "Warning: Directory $CONV_DIR not found, skipping"
        SKIP=$((SKIP+1))
        continue
    fi
    
    mkdir -p "$OUT_SUBDIR"
    
    python "$PYTHON_SCRIPT" \
        --conversations_dir "$CONV_DIR" \
        --api_key "$API_KEY" \
        --model "$MODEL" \
        --base_url "$BASE_URL" \
        --output_dir "$OUT_SUBDIR" \
        --max_context_tokens $MAX_CONTEXT_TOKENS \
        --max_images $MAX_IMAGES \
        --max_workers $MAX_WORKERS \
        --max_api_concurrency $MAX_API_CONCURRENCY \
        --verbose
    
    if [ $? -eq 0 ]; then
        echo "✓ dialogue${i} completed"
        SUCCESS=$((SUCCESS+1))
    else
        echo "✗ dialogue${i} failed"
        FAIL=$((FAIL+1))
    fi
done

echo "======================================================"
echo "Batch processing finished"
echo "Success: $SUCCESS, Fail: $FAIL, Skip: $SKIP"
echo "======================================================"