#!/bin/bash
# Process conversations using Hybrid retriever

BASE_DIR=""
API_KEY="your-openai-key"
BASE_URL=""
RETRIEVER_TYPE="hybrid"
HYBRID_ALPHA=0.5
LLM_MODEL="gpt-4o-mini"
EMBEDDING_MODEL_NAME="all-MiniLM-L6-v2"


START_DIALOGUE=1
END_DIALOGUE=20

SCRIPT_DIR="" # Set this to the directory where FullText.py is located if not the current directory
PYTHON_SCRIPT="$SCRIPT_DIR/process_conversations.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: $PYTHON_SCRIPT not found!"
    exit 1
fi


for i in $(seq $START_DIALOGUE $END_DIALOGUE); do
    dialogue="dialogue${i}"
    dialogue_path="${BASE_DIR}/${dialogue}"
    
    if [ ! -d "$dialogue_path" ]; then
        echo "警告: $dialogue_path 不存在，跳过"
        continue
    fi
    
    echo "Processing $dialogue ..."
    python -m "$PYTHON_SCRIPT" \
        --base_dir "$BASE_DIR" \
        --dialogue "$dialogue" \
        --retriever_type "$RETRIEVER_TYPE" \
        --hybrid_alpha "$HYBRID_ALPHA" \
        --embedding_model_name "$EMBEDDING_MODEL_NAME" \
        --llm_model "$LLM_MODEL" \
        --api_key "$API_KEY" \
        --verbose
done