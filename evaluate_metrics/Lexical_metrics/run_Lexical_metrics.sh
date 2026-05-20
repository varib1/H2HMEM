#!/bin/bash
# Lexical metrics evaluation script (Precision, Recall, F1, BLEU-1)

# ==================== Path Configuration ====================
BASE_PATH=""         # Root directory containing dialogue folders
OUTPUT_DIR=""        # Directory to save lexical evaluation results
SCRIPT_DIR=""        # Directory containing Lexical_metrics.py
PYTHON_SCRIPT="$SCRIPT_DIR/Lexical_metrics.py"

# ==================== Evaluation Configuration ====================
PATTERN="results_*.json"    # Pattern to match result files
VERBOSE="--verbose"         # Enable verbose logging (remove to disable)
# ================================================================

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: $PYTHON_SCRIPT not found!"
    exit 1
fi

python "$PYTHON_SCRIPT" \
    --base_path "$BASE_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --pattern "$PATTERN" \
    $VERBOSE