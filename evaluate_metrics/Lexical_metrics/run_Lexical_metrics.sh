#!/bin/bash
BASE_PATH="./dyadic" # 
OUTPUT_DIR="./Lexical_metrics"
PATTERN="results_*.json"

python evaluate_metrics.py \
    --base_path "$BASE_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --pattern "$PATTERN" \
    --verbose