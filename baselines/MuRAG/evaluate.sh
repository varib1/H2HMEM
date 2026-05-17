#!/bin/bash

# ======================================================
# MURAG Memory Evaluator – Batch Processing (Linux/macOS)
# ======================================================

# --- 配置区（请修改）---
BASE_DIR=""          # 对话数据根目录
API_KEY="your-api-key"
MODEL=""
BASE_URL=""
BASE_OUTPUT_DIR=""

# RAG 参数
RETRIEVAL_TOPK=5
ENCODER_MODEL="Alibaba-NLP/gme-Qwen2-VL-7B-Instruct"
RETRIEVAL_MODE="cosine"
UTILIZATION_METHOD="MultiModalUtilization"
MAX_CONTEXT_TOKENS=4096
MAX_IMAGES=10

# 并行参数
MAX_WORKERS=3
MAX_API_CONCURRENCY=6

# 对话范围
START_DIALOGUE=1
END_DIALOGUE=20

PYTHON_SCRIPT="vlm_evaluator_murag.py"
# ------------------------------------

echo "==================================================="
echo "MURAG Memory Evaluator - Batch Processing"
echo "Dialogues $START_DIALOGUE to $END_DIALOGUE"
echo "==================================================="

mkdir -p "$BASE_OUTPUT_DIR"

for i in $(seq $START_DIALOGUE $END_DIALOGUE); do
    CONV_DIR="${BASE_DIR}/对话${i}"
    OUT_DIR="${BASE_OUTPUT_DIR}/对话${i}"
    
    if [ ! -d "$CONV_DIR" ]; then
        echo "Warning: $CONV_DIR not found, skipping"
        continue
    fi
    
    echo "Processing 对话${i} ..."
    mkdir -p "$OUT_DIR"
    
    python "$PYTHON_SCRIPT" \
        --conversations_dir "$CONV_DIR" \
        --api_key "$API_KEY" \
        --model "$MODEL" \
        --base_url "$BASE_URL" \
        --output_dir "$OUT_DIR" \
        --memory_type multimodal_rag \
        --encoder_model "$ENCODER_MODEL" \
        --retrieval_topk $RETRIEVAL_TOPK \
        --retrieval_mode "$RETRIEVAL_MODE" \
        --utilization_method "$UTILIZATION_METHOD" \
        --max_context_tokens $MAX_CONTEXT_TOKENS \
        --max_images $MAX_IMAGES \
        --max_workers $MAX_WORKERS \
        --max_api_concurrency $MAX_API_CONCURRENCY \
        --verbose
    
    if [ $? -eq 0 ]; then
        echo "✓ 对话${i} completed"
    else
        echo "✗ 对话${i} failed"
    fi
done

echo "==================================================="
echo "Batch processing finished. Results in $BASE_OUTPUT_DIR"
echo "==================================================="