#!/bin/bash
# Evaluate using stored memory notes

BASE_DIR=""
VLM_API_KEY="your-vlm-key"
VLM_MODEL=""
VLM_BASE_URL=""
RETRIEVE_K=5

START_DIALOGUE=1
END_DIALOGUE=20

SCRIPT_DIR="" # Set this to the directory where FullText.py is located if not the current directory
PYTHON_SCRIPT="$SCRIPT_DIR/A_MEM.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: $PYTHON_SCRIPT not found!"
    exit 1
fi

for i in $(seq $START_DIALOGUE $END_DIALOGUE); do
    dialogue="dialogue${i}"
    dialogue_path="${BASE_DIR}/${dialogue}"
    
    if [ ! -d "$dialogue_path" ]; then
        echo "警告: $dialogue_path 不存在，跳过"
        continue
    fi
    
    echo "Evaluating $dialogue ..."
    python -m "$PYTHON_SCRIPT" \
        --dialogue_path "$dialogue_path" \
        --model "$VLM_MODEL" \
        --api_key "$VLM_API_KEY" \
        --base_url "$VLM_BASE_URL" \
        --retrieve_k $RETRIEVE_K \
        --verbose
done