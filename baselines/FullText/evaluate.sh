#!/bin/bash
# ========================================
# VLM Memory Evaluator (FullText)
# Batch processing script for Unix-like systems
# ========================================

# ==================== Path & Dataset Configuration ====================
BASE_PATH=""          # Dataset directory (./dataset/dyadic or ./dataset/multi-party)
START_DIALOGUE=1      # Starting dialogue number (inclusive)
END_DIALOGUE=20       # Ending dialogue number (inclusive)
SCRIPT_DIR=""         # Directory containing FullText.py
PYTHON_SCRIPT="$SCRIPT_DIR/FullText.py"

# ==================== API Configuration ====================
API_KEY=""            # API key for authentication
BASE_URL=""           # Base URL for the API endpoint
MODEL=""              # Model for evaluation (e.g., gpt-4.1-nano)

# ==================== Concurrency Settings ====================
MAX_WORKERS=3         # Maximum concurrent worker threads (adjust based on system/API limits)
MAX_API_CONCURRENCY=6 # Maximum concurrent API calls (adjust based on rate limits)

# ==================== Context Configuration ====================
MAX_CONTEXT_TOKENS=4096    # Maximum tokens per API call (adjust based on model limits)
TRUNCATION_STRATEGY="head_only"  # Options: head_only, head_tail
# ================================================================

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: $PYTHON_SCRIPT not found!"
    exit 1
fi

SUCCESS=0  # Counter for successful dialogue processing
FAIL=0     # Counter for failed dialogue processing
TOTAL=0    # Counter for total dialogues processed

for ((i=START_DIALOGUE; i<=END_DIALOGUE; i++)); do
    TOTAL=$((TOTAL+1))
    echo "========================================"
    echo "Processing dialogue$i ..."
    echo "========================================"
    
    CURRENT_CONV_DIR="$BASE_PATH/dialogue$i"
    
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