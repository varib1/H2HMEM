#!/bin/bash
# Evaluate using stored memory notes (A-MEM memory system)

# ==================== Path & Dataset Configuration ====================
BASE_PATH=""              # Dataset directory containing dialogue folders
START_DIALOGUE=1          # First dialogue index to evaluate (inclusive)
END_DIALOGUE=20           # Last dialogue index to evaluate (inclusive)
SCRIPT_DIR=""             # Directory containing A_MEM.py
PYTHON_SCRIPT="$SCRIPT_DIR/A_MEM.py"

# ==================== API Configuration ====================
API_KEY=""                # API key for LLM authentication
BASE_URL=""               # API base URL for LLM service
BackBone_MODEL=""         # VLM model for answering questions (backbone)
Memoryconstruct_MODEL="gpt-4o-mini"  # Model for memory construction (only for loading, not used in evaluation)

# ==================== Retrieval Configuration ====================
RETRIEVE_K=5              # Number of relevant memories to retrieve per question
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
    
    echo "Evaluating $dialogue ..."
    python "$PYTHON_SCRIPT" \
        --dialogue_path "$dialogue_path" \
        --backbone_model "$BackBone_MODEL" \
        --memoryconstruct_model "$Memoryconstruct_MODEL" \
        --api_key "$API_KEY" \
        --base_url "$BASE_URL" \
        --retrieve_k "$RETRIEVE_K" 
done