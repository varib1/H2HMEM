#!/bin/bash
# 批量生成图片描述示例脚本
# 请根据实际情况修改下面变量

BASE_PATH=""   # 对话根目录
API_KEY="your-openai-api-key"             # 你的 API 密钥
BASE_URL=""      # 如果需要代理，请修改
CAPTION_MAX_TOKENS=256                    # 描述最大 token 数
DIALOGUE_PATTERN="dialogue*"                  # 匹配对话文件夹的模式

python caption.py \
    --base_path "$BASE_PATH" \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL" \
    --caption_max_tokens $CAPTION_MAX_TOKENS \
    --dialogue_pattern "$DIALOGUE_PATTERN" \
    --verbose