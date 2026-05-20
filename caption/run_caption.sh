#!/bin/bash
# Batch image caption generation script
# Please modify the variables below according to your setup

# ==================== Path Configuration ====================
BASE_PATH=""                        # Dialogue root directory (./dataset/dyadic or ./dataset/multi-party)
SCRIPT_DIR=""                       # Directory containing gpt_4o_caption.py
PYTHON_SCRIPT="$SCRIPT_DIR/gpt_4o_caption.py"

# ==================== API Configuration ====================
API_KEY=""                          # Your API key
BASE_URL=""                         # Your API base URL 

# ==================== Caption Configuration ====================
CAPTION_MAX_TOKENS=256              # Maximum tokens for image description output
DIALOGUE_PATTERN="dialogue*"        # Pattern to match dialogue folders
# ================================================================

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: $PYTHON_SCRIPT not found!"
    exit 1
fi

python "$PYTHON_SCRIPT" \
    --base_path "$BASE_PATH" \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL" \
    --caption_max_tokens $CAPTION_MAX_TOKENS \
    --dialogue_pattern "$DIALOGUE_PATTERN" \
    --verbose