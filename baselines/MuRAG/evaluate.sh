#!/bin/bash

# ======================================================
# MURAG Memory Evaluator – Batch Processing (Linux/macOS)
# ======================================================

# ==================== Path & Dataset Configuration ====================
BASE_PATH=""          # Dataset directory (./dataset/dyadic or ./dataset/multi-party)
START_DIALOGUE=1      # Starting dialogue number (adjust based on the dataset)
END_DIALOGUE=20       # Ending dialogue number (adjust based on the dataset)
SCRIPT_DIR=""         # Directory containing MuRAG.py
PYTHON_SCRIPT="$SCRIPT_DIR/MuRAG.py"

# ==================== API Configuration ====================
API_KEY=""            # API key for authentication
BASE_URL=""           # Base URL for the API endpoint
MODEL=""              # Model for evaluation (e.g., gpt-4.1-nano)

# ==================== RAG Parameters ====================
RETRIEVAL_TOPK=5      # Number of top retrieved items to include in context
ENCODER_MODEL="Alibaba-NLP/gme-Qwen2-VL-7B-Instruct"  # Encoder model for retrieval
RETRIEVAL_MODE="cosine"  # Options: cosine, sparse, dense
UTILIZATION_METHOD="MultiModalUtilization"  # Method to utilize retrieved information
MAX_CONTEXT_TOKENS=4096  # Maximum tokens per API call (adjust based on model limits)
MAX_IMAGES=5          # Maximum images per API call (adjust based on model limits)

# ==================== Concurrency Settings ====================
MAX_WORKERS=3         # Maximum concurrent worker threads
MAX_API_CONCURRENCY=6 # Maximum concurrent API calls

# ==================== Runtime Settings ====================
export TOKENIZERS_PARALLELISM=false  # Disable tokenizer parallelism to avoid multiprocessing issues
# ================================================================

echo "==================================================="
echo "MURAG Memory Evaluator - Batch Processing"
echo "Dialogues $START_DIALOGUE to $END_DIALOGUE"
echo "==================================================="

for i in $(seq $START_DIALOGUE $END_DIALOGUE); do
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
        --memory_type multimodal_rag \
        --encoder_model "$ENCODER_MODEL" \
        --retrieval_topk $RETRIEVAL_TOPK \
        --retrieval_mode "$RETRIEVAL_MODE" \
        --utilization_method "$UTILIZATION_METHOD" \
        --max_context_tokens $MAX_CONTEXT_TOKENS \
        --max_images $MAX_IMAGES \
        --verbose
    
    if [ $? -eq 0 ]; then
        echo "✓ dialogue${i} completed"
    else
        echo "✗ dialogue${i} failed"
    fi
done

echo "==================================================="
echo "Batch processing finished."
echo "==================================================="