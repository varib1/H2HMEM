#!/bin/bash

# LLM-judge evaluator startup script

# ==================== Path Configuration ====================
BASE_DIR=""          # Dataset directory (./dataset/dyadic or ./dataset/multi-party)
OUTPUT_DIR=""        # Directory to save evaluation results
SCRIPT_DIR=""        # Directory containing LLM_as_judge.py
PYTHON_SCRIPT="$SCRIPT_DIR/LLM_as_judge.py"

# ==================== API Configuration ====================
API_KEY=""           # API key for LLM authentication
BASE_URL=""          # API base URL for LLM service
MODEL=""             # LLM model name for evaluation

# ==================== Evaluation Configuration ====================
PATTERN="results_*.json"    # Pattern to match result files
MAX_WORKERS=4               # Maximum number of parallel worker threads
# ================================================================

# Run evaluation
python3 "$PYTHON_SCRIPT" \
    --root_folder "$BASE_DIR" \
    --output_folder "$OUTPUT_DIR" \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL" \
    --model "$MODEL" \
    --pattern "$PATTERN" \
    --max_workers $MAX_WORKERS

echo "Evaluation completed. Results are saved in the specified output folder."