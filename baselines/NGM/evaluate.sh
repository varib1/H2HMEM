#!/bin/bash

# ======================================================
# NGM Memory Evaluator – Batch Processing (Linux/macOS)
# ======================================================

# ==================== Path & Dataset Configuration ====================
BASE_PATH=""          # Dataset directory (./dataset/dyadic or ./dataset/multi-party)
START_DIALOGUE=1      # Starting dialogue number (adjust based on the dataset)
END_DIALOGUE=20       # Ending dialogue number (adjust based on the dataset)
SCRIPT_DIR=""         # Directory containing NGM.py
PYTHON_SCRIPT="$SCRIPT_DIR/NGM.py"
PYTHON_CMD="python3"  # Python command (adjust for virtual environment)

# ==================== API Configuration ====================
API_KEY=""            # API key for authentication
BASE_URL=""           # Base URL for the API endpoint
MODEL=""              # VLM model for evaluation (e.g., gpt-4.1-nano)

# ==================== Encoder Configuration ====================
ENCODER_METHOD="GMEEncoder"                      # Encoder type: GMEEncoder or CLIPEncoder
ENCODER_Model="Alibaba-NLP/gme-Qwen2-VL-7B-Instruct"  # Path or name of the encoder model

# ==================== Graph Construction Parameters ====================
SIMILARITY_THRESHOLD=0.7    # Minimum similarity score for creating edges between memory nodes

# ==================== Graph Traversal Parameters ====================
RETRIEVAL_TOPK=5            # Number of top semantically similar memories to retrieve as starting nodes
TRAVERSAL_STRATEGY="breadth_first"  # Graph traversal method: 'breadth_first' or 'depth_first'
MAX_DEPTH=3                 # Maximum depth to traverse from starting nodes in the graph
MAX_NODES=5                 # Maximum number of memory nodes to return after graph traversal

# ==================== Runtime Settings ====================
export TOKENIZERS_PARALLELISM=false  # Disable tokenizer parallelism to avoid multiprocessing issues
# ================================================================

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: $PYTHON_SCRIPT not found!"
    exit 1
fi

for i in $(seq $START_DIALOGUE $END_DIALOGUE); do
    CONV_DIR="${BASE_PATH}/dialogue${i}"
    
    if [ ! -d "$CONV_DIR" ]; then
        echo "Warning: $CONV_DIR not found, skipping"
        continue
    fi
    
    echo "========================================"
    echo "Processing dialogue${i} ..."
    echo "========================================"
    
    $PYTHON_CMD "$PYTHON_SCRIPT" \
        --conversations_dir "$CONV_DIR" \
        --api_key "$API_KEY" \
        --vlm_model "$MODEL" \
        --base_url "$BASE_URL" \
        --encoder_method "$ENCODER_METHOD" \
        --encoder_model "$ENCODER_Model" \
        --retrieval_topk $RETRIEVAL_TOPK \
        --similarity_threshold $SIMILARITY_THRESHOLD \
        --traversal_strategy "$TRAVERSAL_STRATEGY" \
        --max_depth $MAX_DEPTH \
        --max_nodes $MAX_NODES \
        --verbose
    
    if [ $? -eq 0 ]; then
        echo "✓ dialogue${i} completed"
    else
        echo "✗ dialogue${i} failed"
    fi
    echo ""
done

echo "========================================"
echo "All dialogues processed."
echo "========================================"