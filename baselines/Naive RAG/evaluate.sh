#!/bin/bash

BASE_DIR=""           # 修改为你的数据根目录
API_KEY="your-api-key"
MODEL=""
BASE_URL=""


CHUNK_SIZE=1
TOP_K=5
MAX_WORKERS=3
MAX_API_CONCURRENCY=6
START=1
END=20

SCRIPT_DIR="" # Set this to the directory where Naive_RAG.py is located if not the current directory
PYTHON_SCRIPT="$SCRIPT_DIR/Naive_RAG.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: $PYTHON_SCRIPT not found!"
    exit 1
fi

for i in $(seq $START $END); do
    CONV_DIR="${BASE_DIR}/dialogue${i}"
    if [ ! -d "$CONV_DIR" ]; then
        echo "Warning: $CONV_DIR not found, skipping"
        continue
    fi
    echo "Processing dialogue${i} ..."

    python "$PYTHON_SCRIPT" \
        --conversations_dir "$CONV_DIR" \
        --api_key "$API_KEY" \
        --model "$MODEL" \
        --base_url "$BASE_URL" \
        --chunk_size $CHUNK_SIZE \
        --top_k $TOP_K \
        --max_workers $MAX_WORKERS \
        --max_api_concurrency $MAX_API_CONCURRENCY \
        --verbose
    echo "----------------------------------------"
done