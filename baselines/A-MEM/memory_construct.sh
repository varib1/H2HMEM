#!/bin/bash
# Process conversations using Hybrid retriever (A-MEM memory construction)

# ==================== Path & Dataset Configuration ====================
BASE_PATH=""         # Dataset directory (./dataset/dyadic or ./dataset/multi-party)
START_DIALOGUE=1     # Starting dialogue number (inclusive)
END_DIALOGUE=20      # Ending dialogue number (inclusive)
SCRIPT_DIR=""        # Directory containing process_conversations.py
PYTHON_SCRIPT="$SCRIPT_DIR/process_conversations.py"

# ==================== API Configuration ====================
API_KEY=""           # API key for LLM authentication
BASE_URL=""          # Base URL for the OpenAI API
Memoryconstruct_MODEL="gpt-4o-mini"   # LLM model for memory construction and evolution

# ==================== Retriever Configuration ====================
RETRIEVER_TYPE="hybrid"                # Options: 'simple' (semantic only) or 'hybrid' (BM25 + semantic)
HYBRID_ALPHA=0.5                       # Hybrid weight: 0=BM25 only, 1=semantic only, 0.5=balanced
EMBEDDING_MODEL_NAME="all-MiniLM-L6-v2"  # Sentence-BERT model for text embeddings
# ================================================================

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: $PYTHON_SCRIPT not found!"
    exit 1
fi

for i in $(seq $START_DIALOGUE $END_DIALOGUE); do
    dialogue="dialogue${i}"
    dialogue_path="${BASE_PATH}/${dialogue}"
    
    if [ ! -d "$dialogue_path" ]; then
        echo "Warning: $dialogue_path does not exist, skipping"
        continue
    fi
    
    echo "Processing $dialogue ..."
    python "$PYTHON_SCRIPT" \
        --base_dir "$BASE_PATH" \
        --dialogue "$dialogue" \
        --retriever_type "$RETRIEVER_TYPE" \
        --hybrid_alpha "$HYBRID_ALPHA" \
        --embedding_model_name "$EMBEDDING_MODEL_NAME" \
        --memoryconstruct_model "$Memoryconstruct_MODEL" \
        --api_key "$API_KEY" \
        --base_url "$BASE_URL" \
        --verbose
done