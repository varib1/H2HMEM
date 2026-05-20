#!/bin/bash
# Naive RAG evaluation script for multiple dialogues

# ==================== Path & Dataset Configuration ====================
BASE_PATH=""          # Dataset directory (./dataset/dyadic or ./dataset/multi-party)
START=1               # Starting dialogue number (adjust based on the dataset)
END=20                # Ending dialogue number (adjust based on the dataset)
SCRIPT_DIR=""         # Directory containing Naive_RAG.py
PYTHON_SCRIPT="$SCRIPT_DIR/Naive_RAG.py"

# ==================== API Configuration ====================
API_KEY=""            # API key for authentication
BASE_URL=""           # Base URL for the API endpoint
MODEL=""              # Model for evaluation (e.g., gpt-4.1-nano)

# ==================== Retrieval Configuration ====================
CHUNK_SIZE=1          # Number of dialogue turns per chunk for retrieval
TOP_K=5               # Number of top retrieved chunks to include in context

# ==================== Performance Configuration ====================
MAX_WORKERS=3         # Maximum concurrent worker threads (adjust based on system/API limits)
MAX_API_CONCURRENCY=6 # Maximum concurrent API calls (adjust based on rate limits)
# ================================================================

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: $PYTHON_SCRIPT not found!"
    exit 1
fi

for i in $(seq $START $END); do
    CONV_DIR="${BASE_PATH}/dialogue${i}"
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