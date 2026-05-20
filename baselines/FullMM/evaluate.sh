#!/bin/bash

# ======================================================
# Multi‑Modal Memory Evaluator – Batch Processing Script
# ======================================================

# ==================== Path & Dataset Configuration ====================
BASE_PATH=""          # Dataset directory (./dataset/dyadic or ./dataset/multi-party)
START_DIALOGUE=1      # Starting dialogue number (inclusive)
END_DIALOGUE=20       # Ending dialogue number (inclusive)
SCRIPT_DIR=""         # Directory containing FullMM.py
PYTHON_SCRIPT="$SCRIPT_DIR/FullMM.py"

# ==================== API Configuration ====================
MODEL=""              # Model for evaluation (e.g., gpt-4.1-nano)
API_KEY=""            # API key for authentication
BASE_URL=""           # Base URL for the API endpoint

# ==================== Context & Image Configuration ====================
MAX_CONTEXT_TOKENS=4096   # Maximum tokens per API call (adjust based on model limits)
MAX_IMAGES=10             # Maximum images per API call (adjust based on model limits)

# ==================== Concurrency Settings ====================
MAX_WORKERS=3             # Maximum concurrent worker threads
MAX_API_CONCURRENCY=6     # Maximum concurrent API calls
# ================================================================

echo "======================================================"
echo "Multi-Modal Memory Evaluator - Batch Processing"
echo "Dialogues $START_DIALOGUE to $END_DIALOGUE"
echo "======================================================"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: $PYTHON_SCRIPT not found!"
    exit 1
fi

SUCCESS=0  # Counter for successful dialogue processing
FAIL=0     # Counter for failed dialogue processing
SKIP=0     # Counter for skipped dialogues due to missing directories

for i in $(seq $START_DIALOGUE $END_DIALOGUE); do
    CONV_DIR="${BASE_PATH}/dialogue${i}"
    
    echo "------------------------------------------------------"
    echo "Processing dialogue${i} ..."
    
    if [ ! -d "$CONV_DIR" ]; then
        echo "Warning: Directory $CONV_DIR not found, skipping"
        SKIP=$((SKIP+1))
        continue
    fi
    
    python "$PYTHON_SCRIPT" \
        --conversations_dir "$CONV_DIR" \
        --api_key "$API_KEY" \
        --model "$MODEL" \
        --base_url "$BASE_URL" \
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