#!/bin/bash

# LLM-judge 记忆系统评估器启动脚本 (Linux)
# 使用新版提示词 - 多线程批量评估模式

echo "========================================"
echo "  LLM-judge 记忆系统评估器启动脚本 (Linux)"
echo "  使用新版提示词 - 多线程批量评估模式"
echo "========================================"
echo ""

# 设置Python环境
PYTHON_CMD="python3"

# 检查Python是否可用


# ========== 路径配置 ==========
ROOT_FOLDER="/home/zhushiping/新对话信息"
SCRIPT_PATH="$ROOT_FOLDER/LLM_as_judge.py"
OUTPUT_FOLDER="$ROOT_FOLDER/LLM_judge/llm_judge_results"

# API配置
API_KEY="sk-B4e1UbpPGbNpxNDXY38bSznbUdayKcq67x7CYIW4YXJTS0j4"
BASE_URL="https://api.vectorengine.ai/v1"
MODEL="gpt-4o-mini"

PROMPT_FILE="$ROOT_FOLDER/llm_judge_prompt.txt"
BATCH_PROMPT_FILE="$ROOT_FOLDER/llm_judge_batch_prompt.txt"
PATTERN="results_NGM.json"

# ========== 多线程配置 ==========
MAX_WORKERS=4

MEMORY_TYPES=""

# 指定要评估的对话（多个用空格分隔，留空表示评估所有）
DIALOGUES=""

# 指定要评估的session（多个用空格分隔，留空表示评估所有）
SESSIONS=""

# ========== 显示配置信息 ==========
echo ""
echo "====== 配置信息 ======"
echo "根文件夹: $ROOT_FOLDER"
echo "输出文件夹: $OUTPUT_FOLDER"
echo "评估脚本: $SCRIPT_PATH"
echo "单条提示词文件: $PROMPT_FILE"
echo "批量提示词文件: $BATCH_PROMPT_FILE"
echo "API Base URL: $BASE_URL"
echo "模型: $MODEL"
echo "结果文件模式: $PATTERN"
echo "最大线程数: $MAX_WORKERS"
echo ""
echo "过滤选项:"
if [ -n "$MEMORY_TYPES" ]; then
    echo "  记忆系统类型: $MEMORY_TYPES"
else
    echo "  记忆系统类型: 全部"
fi
if [ -n "$DIALOGUES" ]; then
    echo "  对话: $DIALOGUES"
else
    echo "  对话: 全部"
fi
if [ -n "$SESSIONS" ]; then
    echo "  Session: $SESSIONS"
else
    echo "  Session: 全部"
fi
echo "====================="
echo ""

# 检查脚本文件是否存在
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "[错误] 评估脚本不存在: $SCRIPT_PATH"
    echo "请确保脚本文件在正确的位置"
    read -p "按回车键退出..."
    exit 1
fi

# 检查提示词文件是否存在，如果不存在则创建
if [ ! -f "$PROMPT_FILE" ]; then
    echo "[提示] 单条提示词文件不存在，将使用默认提示词"
fi

if [ ! -f "$BATCH_PROMPT_FILE" ]; then
    echo "[提示] 批量提示词文件不存在，将使用默认提示词"
    echo "正在创建批量提示词文件..."
    $PYTHON_CMD $SCRIPT_PATH --create_batch_prompt --batch_prompt_file "$BATCH_PROMPT_FILE"
fi

read -p "是否先列出找到的记忆系统类型? (Y/N): " LIST_TYPES
if [[ $LIST_TYPES == "Y" || $LIST_TYPES == "y" ]]; then
    echo ""
    echo "正在扫描记忆系统类型..."
    $PYTHON_CMD $SCRIPT_PATH --root_folder "$ROOT_FOLDER" --list_memory_types
    echo ""
    read -p "继续评估? (Y/N): " CONTINUE
    if [[ $CONTINUE != "Y" && $CONTINUE != "y" ]]; then
        echo "已取消"
        read -p "按回车键退出..."
        exit 0
    fi
fi

# 确认开始
echo ""
read -p "确认开始评估? (Y/N): " CONFIRM
if [[ $CONFIRM != "Y" && $CONFIRM != "y" ]]; then
    echo "已取消"
    read -p "按回车键退出..."
    exit 0
fi

# 构建命令
CMD="$PYTHON_CMD \"$SCRIPT_PATH\" \
    --root_folder \"$ROOT_FOLDER\" \
    --output_folder \"$OUTPUT_FOLDER\" \
    --prompt_file \"$PROMPT_FILE\" \
    --batch_prompt_file \"$BATCH_PROMPT_FILE\" \
    --api_key \"$API_KEY\" \
    --base_url \"$BASE_URL\" \
    --model \"$MODEL\" \
    --pattern \"$PATTERN\" \
    --max_workers $MAX_WORKERS \
    --delay 0.5"

if [ -n "$MEMORY_TYPES" ]; then
    CMD="$CMD --memory_types $MEMORY_TYPES"
fi

if [ -n "$DIALOGUES" ]; then
    CMD="$CMD --dialogues $DIALOGUES"
fi

if [ -n "$SESSIONS" ]; then
    CMD="$CMD --sessions $SESSIONS"
fi

# 添加详细输出选项（可选）
read -p "是否显示详细输出? (Y/N): " VERBOSE
if [[ $VERBOSE == "Y" || $VERBOSE == "y" ]]; then
    CMD="$CMD --verbose"
fi

# ========== 创建输出目录 ==========
if [ ! -d "$OUTPUT_FOLDER" ]; then
    mkdir -p "$OUTPUT_FOLDER"
fi

# ========== 执行评估 ==========
echo ""
echo "====== 执行命令 ======"
echo "$CMD"
echo "====================="
echo ""
echo "开始评估..."
echo "时间: $(date)"
echo ""

# 记录开始时间
START_TIME=$(date +%s)

# 执行命令
eval $CMD
ERROR_LEVEL=$?

# 记录结束时间
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "结束时间: $(date)"
echo "运行时长: $((ELAPSED / 60))分 $((ELAPSED % 60))秒"

if [ $ERROR_LEVEL -ne 0 ]; then
    echo ""
    echo "[错误] 评估过程中出现错误 (错误码: $ERROR_LEVEL)"
    echo "请检查日志获取详细信息"
else
    echo ""
    echo "[完成] 评估成功结束"
    echo ""
    echo "结果保存在: $OUTPUT_FOLDER"
    echo ""
    echo "生成的文件:"
    ls -1 "$OUTPUT_FOLDER"
fi

echo ""
read -p "按回车键退出..."