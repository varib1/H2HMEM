#!/bin/bash

# ========================================
# VLM Intra-Session Memory Evaluator
# Batch processing script for Unix-like systems
# ========================================

# Configuration – edit these lines
BASE_CONVERSATIONS_DIR=""          # directory containing        
API_KEY="your-api-key-here"
MODEL=""
BASE_URL=""

# Parallel settings
MAX_WORKERS=3
MAX_API_CONCURRENCY=6

# Context truncation
MAX_CONTEXT_TOKENS=4096
TRUNCATION_STRATEGY="head_only" # Options: head_only, head_tail

# Dialogue range
START_DIALOGUE=1
END_DIALOGUE=20

# ========================================

SCRIPT_DIR="" # Set this to the directory where FullText.py is located if not the current directory
PYTHON_SCRIPT="$SCRIPT_DIR/FullText.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: $PYTHON_SCRIPT not found!"
    exit 1
fi

SUCCESS=0
FAIL=0
TOTAL=0

for ((i=START_DIALOGUE; i<=END_DIALOGUE; i++)); do
    TOTAL=$((TOTAL+1))
    echo "========================================"
    echo "Processing dialogue$i ..."
    echo "========================================"
    
    CURRENT_CONV_DIR="$BASE_CONVERSATIONS_DIR/dialogue$i"
    
    if [ ! -d "$CURRENT_CONV_DIR" ]; then
        echo "Warning: Directory $CURRENT_CONV_DIR does not exist, skipping."
        FAIL=$((FAIL+1))
        continue
    fi
    
    python "$PYTHON_SCRIPT" \
        --conversations_dir "$CURRENT_CONV_DIR" \
        --api_key "$API_KEY" \
        --model "$MODEL" \
        --base_url "$BASE_URL" \
        --max_workers "$MAX_WORKERS" \
        --max_api_concurrency "$MAX_API_CONCURRENCY" \
        --max_context_tokens "$MAX_CONTEXT_TOKENS" \
        --truncation_strategy "$TRUNCATION_STRATEGY" \
        --verbose
    
    if [ $? -eq 0 ]; then
        echo "dialogue$i completed successfully."
        SUCCESS=$((SUCCESS+1))
    else
        echo "dialogue$i failed."
        FAIL=$((FAIL+1))
    fi
    echo ""
done

echo "========================================"
echo "Summary: Total=$TOTAL, Success=$SUCCESS, Fail=$FAIL"
echo "========================================"