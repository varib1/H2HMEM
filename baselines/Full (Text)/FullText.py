import os
import json
import logging
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
from collections import defaultdict
from natsort import natsorted

# 在文件开头的导入部分添加
import concurrent.futures
import threading
from threading import Lock, Semaphore
import queue
from tqdm import tqdm  # 可选，需要安装：pip install tqdm

# 导入API相关库
import requests
from PIL import Image
import base64
from io import BytesIO

# 尝试导入tiktoken用于token计数，如果没有则使用简单估算
try:
    import tiktoken
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False
    logging.warning("tiktoken未安装，将使用简单的字符数估算token数")

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class QuestionAnswerPair:
    """问题-答案对 - 适配questions.json格式"""
    question_id: str
    session_id: str
    dialogue_name: str
    question_text: str
    question_image: str
    original_answer: str
    answer_source: str
    answer_session: List[str]
    question_type: Dict[str, str]
    difficulty: str
    supporting_evidence: List[Dict]
    image_context: Optional[List[str]] = None
    metadata: Optional[Dict] = None
    category: str = field(init=False)  # 从question_type派生
    
    def __post_init__(self):
        # 从question_type中提取category
        if self.question_type:
            subsub_type = self.question_type.get("subsub_type", "")
            if subsub_type:
                self.category = subsub_type
            else:
                sub_type = self.question_type.get("sub_type", "")
                self.category = sub_type or self.question_type.get("main_type", "general")
        else:
            self.category = "general"

@dataclass
class EvaluationResult:
    """评估结果"""
    sample_id: str
    session_id: str
    dialogue_name: str
    question_id: str
    question_text: str
    question_image: str
    system_answer: str
    original_answer: str
    answer_source: str
    question_type: Dict[str, str]
    category: str
    difficulty: str
    timestamp: str
    memory_type: str
    vlm_model: str
    processing_time: float
    confidence: Optional[float] = None
    supporting_evidence: Optional[List[Dict]] = None
    memory_context_summary: Optional[str] = None
    recall_method: str = "full_text"
    success: bool = True
    error_message: Optional[str] = None
    truncated: bool = False
    original_context_length: Optional[int] = None
    truncated_context_length: Optional[int] = None
    
    # 详细时间指标
    memory_load_time: float = 0.0  # 加载记忆时间
    memory_recall_time: float = 0.0  # 召回/构建记忆上下文时间
    llm_inference_time: float = 0.0  # LLM回答时间


class TokenCounter:
    """Token计数器"""
    
    def __init__(self, model_name: str = "cl100k_base"):
        """
        初始化token计数器
        
        Args:
            model_name: tokenizer模型名称（默认使用cl100k_base，适用于gpt-4, gpt-3.5-turbo）
        """
        self.model_name = model_name
        self.encoding = None
        
        if TOKENIZER_AVAILABLE:
            try:
                self.encoding = tiktoken.get_encoding(model_name)
                logger.info(f"成功加载tokenizer: {model_name}")
            except Exception as e:
                logger.warning(f"加载tokenizer失败: {e}，将使用估算方法")
    
    def count_tokens(self, text: str) -> int:
        """
        计算文本的token数量
        
        Args:
            text: 输入文本
        
        Returns:
            token数量
        """
        if not text:
            return 0
        
        if self.encoding:
            # 使用tiktoken精确计数
            return len(self.encoding.encode(text))
        else:
            # 简单估算：中文字符算2个token，英文单词算1.3个token
            chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
            other_chars = len(text) - chinese_chars
            
            # 估算：中文字符每个约2个token，其他字符每4个字符约1个token
            estimated_tokens = chinese_chars * 2 + other_chars * 0.25
            return int(estimated_tokens) + 1
    
    def truncate_text(self, text: str, max_tokens: int, preserve_ratio: float = 0.8) -> tuple:
        """
        截断文本到指定token数
        
        Args:
            text: 输入文本
            max_tokens: 最大token数
            preserve_ratio: 保留开头部分的比例（0.8表示保留前80%的token，后20%从末尾截断）
        
        Returns:
            (截断后的文本, 原始token数, 截断后token数)
        """
        if not text:
            return text, 0, 0
        
        original_tokens = self.count_tokens(text)
        
        if original_tokens <= max_tokens:
            return text, original_tokens, original_tokens
        
        if self.encoding:
            # 使用tiktoken精确截断
            tokens = self.encoding.encode(text)
            
            # 决定保留哪些部分
            keep_tokens = int(max_tokens * preserve_ratio)
            
            if keep_tokens >= max_tokens:
                keep_tokens = max_tokens
            
            # 保留开头部分
            truncated_tokens = tokens[:keep_tokens]
            
            # 如果需要，从末尾补充一些（这里简单处理，只保留开头）
            # 更复杂的策略可以根据需要实现
            
            truncated_text = self.encoding.decode(truncated_tokens)
            truncated_tokens_count = len(truncated_tokens)
            
            return truncated_text, original_tokens, truncated_tokens_count
        else:
            # 使用字符数估算截断
            # 估算每个token对应的字符数
            chars_per_token = len(text) / original_tokens
            
            # 需要保留的字符数
            keep_chars = int(max_tokens * chars_per_token * preserve_ratio)
            
            # 简单截断
            truncated_text = text[:keep_chars] + "... [内容已截断]"
            
            # 重新计算截断后的token数（估算）
            truncated_tokens = self.count_tokens(truncated_text)
            
            return truncated_text, original_tokens, truncated_tokens

class FullTextMemorySystem:
    """全文本记忆系统 - 存储整个对话的所有session内容"""
    def __init__(self, conversations_dir: str):
        self.conversations_dir = conversations_dir
        self.memory_storage = {}  # 存储所有session的内容，key为session_id
        self.all_dialogues = []   # 所有session的对话合并
        self.session_info = {}    # session额外信息
        
        # 添加存储时间统计
        self.store_times = []  # 记录每次存储的时间
        self.total_store_time = 0.0
        self.num_stores = 0
        self.session_load_times = {}  # 每个session的加载时间
        self.total_memory_load_time = 0.0  # 总记忆加载时间
        self.load_start_time = None  # 开始加载时间
        self.load_end_time = None    # 结束加载时间
    
    def load_all_conversations(self):
        """加载整个对话的所有session数据（带时间统计）"""
        self.load_start_time = time.time()
        
        scenes_dir = os.path.join(self.conversations_dir, "scenes")
        if not os.path.exists(scenes_dir):
            raise ValueError(f"找不到scenes目录: {scenes_dir}")
        
        session_dirs = natsorted([
            d for d in os.listdir(scenes_dir) 
            if os.path.isdir(os.path.join(scenes_dir, d))
        ])
        all_dialogues = []
        
        for session_dir_name in session_dirs:
            session_start = time.time()
            session_dir = os.path.join(scenes_dir, session_dir_name)
            session_data = self._load_single_session(session_dir_name, session_dir)
            
            if session_data:
                session_id = session_dir_name
                self.memory_storage[session_id] = session_data
                
                # 记录session加载时间
                session_load_time = time.time() - session_start
                self.session_load_times[session_id] = session_load_time
                self.total_memory_load_time += session_load_time
                self.num_stores += 1
                
                # 获取caption目录路径
                caption_dir = os.path.join(session_dir, "caption")
                caption_files_exist = os.path.exists(caption_dir)
                
                if not caption_files_exist:
                    logger.debug(f"session {session_id} 没有caption目录")
                
                # 提取对话内容并添加session信息
                dialogues = session_data.get("dialogue", [])
                processed_dialogues = []
                timeline_date = session_data.get("timeline_date", "")
                
                for i, dialogue in enumerate(dialogues, 1):
                    role = dialogue.get("role", "")
                    content = dialogue.get("content", {})
                    text = content.get("text", "")
                    image_filename = content.get("image", "")
                    
                    # 处理图片描述信息
                    image_description = ""
                    if image_filename and caption_files_exist:
                        # 提取文件名中的数字部分
                        caption_json = Path(image_filename).stem + ".json"  
                        if caption_json:
                            caption_file_path = os.path.join(caption_dir, caption_json)
                            
                            if os.path.exists(caption_file_path):
                                try:
                                    with open(caption_file_path, 'r', encoding='utf-8') as f:
                                        caption_data = json.load(f)
                                    
                                    # 提取description中的完整文字信息
                                    description = caption_data.get("description", {})
                                    
                                    # 提取所有文字信息
                                    description_texts = []
                                    
                                    # 1. 提取final_text
                                    final_text = description.get("final_text", "")
                                    if final_text:
                                        description_texts.append(final_text)
                                    
                                    # 合并所有描述文字
                                    if description_texts:
                                        image_description = "\n".join(description_texts)
                                        logger.debug(f"已加载图片 {image_filename} 的描述信息，字符数: {len(image_description)}")
                                    
                                except Exception as e:
                                    logger.error(f"加载图片描述文件 {caption_file_path} 失败: {e}")
                            else:
                                logger.debug(f"图片描述文件不存在: {caption_file_path}")
                        else:
                            logger.debug(f"无法从文件名 {image_filename} 中提取数字")
                    
                    # 创建包含图片描述的对话内容
                    content_with_description = content.copy()
                    content_with_description["text"] = timeline_date + ":" + content_with_description.get("text", "")
                    if image_description:
                        content_with_description["image_description"] = image_description
                    
                    dialogue_with_session = {
                        "session_id": session_id,
                        "session_title": session_data.get("session_title", ""),
                        "timeline_date": session_data.get("timeline_date", ""),
                        "session_dir_name": session_dir_name,
                        "dialogue_index": i,
                        "role": role,
                        "content": content_with_description
                    }
                    all_dialogues.append(dialogue_with_session)
                    processed_dialogues.append(dialogue_with_session)
                
                # 存储session信息
                self.session_info[session_id] = {
                    "session_dir_name": session_dir_name,
                    "session_title": session_data.get("session_title", ""),
                    "timeline_date": session_data.get("timeline_date", ""),
                    "generated_at": session_data.get("generated_at", ""),
                    "dialogue_count": len(dialogues),
                    "has_caption_dir": caption_files_exist,
                    "session_path": session_dir,
                    "load_time": session_load_time  # 添加加载时间
                }
                
                # 记录存储时间
                self.store_times.append(session_load_time)
                self.total_store_time += session_load_time
                
                logger.info(f"   加载 {session_id}: {session_load_time:.3f}秒, {len(dialogues)}轮对话")
        
        # 存储所有对话
        self.all_dialogues = all_dialogues
        
        self.load_end_time = time.time()
        total_load_time = self.load_end_time - self.load_start_time
        
        logger.info(f"已加载 {len(self.memory_storage)} 个session，共 {len(all_dialogues)} 轮对话")
        logger.info(f"总加载时间: {total_load_time:.2f}秒")
        
        # 输出存储时间统计
        if self.num_stores > 0:
            logger.info(f"存储时间统计:")
            logger.info(f"   平均每个session加载: {self.total_store_time/self.num_stores:.3f}秒")
            logger.info(f"   最快加载: {min(self.store_times):.3f}秒")
            logger.info(f"   最慢加载: {max(self.store_times):.3f}秒")
        
        # 统计图片描述信息
        dialogues_with_images = [d for d in all_dialogues if d["content"].get("image")]
        dialogues_with_description = [d for d in all_dialogues if d["content"].get("image_description")]
        
        logger.info(f"包含图片的对话: {len(dialogues_with_images)} 轮")
        logger.info(f"已加载图片描述的对话: {len(dialogues_with_description)} 轮")
        
        return total_load_time
    
    def _load_single_session(self, session_dir_name: str, session_dir: str) -> Optional[Dict]:
        """加载单个session的数据"""
        conversation_file = os.path.join(session_dir, "session.json")
        
        if not os.path.exists(conversation_file):
            logger.warning(f"未找到conversation.json文件: {conversation_file}")
            return None
        
        try:
            with open(conversation_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            logger.debug(f"成功加载 {session_dir_name} 的对话数据")
            return session_data
            
        except Exception as e:
            logger.error(f"加载 {conversation_file} 失败: {e}")
            return None
    
    
    
    def get_full_memory_context(self) -> Dict[str, Any]:
        """获取整个对话的完整记忆上下文"""
        return {
            "all_sessions": list(self.memory_storage.keys()),
            "session_info": self.session_info,
            "total_dialogues": len(self.all_dialogues),
            "memory_storage": self.memory_storage,
            "all_dialogues": self.all_dialogues
        }
    
    def get_session_context(self, target_session_id: str) -> Dict[str, Any]:
        """获取包含所有session内容的完整上下文，但标记目标session"""
        all_dialogues_with_context = []
        
        # 按session顺序组织所有对话
        for dialogue in self.all_dialogues:
            all_dialogues_with_context.append(dialogue)
        
        return {
            "target_session_id": target_session_id,
            "target_session_info": self.session_info.get(target_session_id, {}),
            "all_sessions": list(self.memory_storage.keys()),
            "session_info": self.session_info,
            "total_dialogues": len(self.all_dialogues),
            "dialogues_with_context": all_dialogues_with_context
        }

def create_memory_system(memory_type: str, conversations_dir: str):
    """创建记忆系统"""
    if memory_type == "full_text":
        return FullTextMemorySystem(conversations_dir)
    else:
        raise ValueError(f"不支持的记忆类型: {memory_type}")

class PromptTemplate:
        """Standardized prompt template for 9 question types"""
        
        # Instructions for each question type (only CD, AR, TTL include abbreviation)
        INSTRUCTIONS = {
            "Unimodal Precise Recall": "Accurately recall specific information from the conversation and answer directly.",
            "Cross-modal Related Retrieval": "Retrieve related information across different modalities (text and images) from the conversation.",
            "Knowledge Resolution": "Resolve and maintain knowledge consistency across the conversation.",
            "Temporal Reasoning": "Reason about temporal relationships and time-based information in the conversation.",
            "Multimodal Causal Reasoning": "Perform causal reasoning using both text and image information from the conversation.",
            "Reference & Evolution Tracking": "Track references and their evolution throughout the conversation.",
            "Test-Time Learning (TTL)": "Learn and adapt from the conversation context at test time to answer the question.",
            "Conflict Detection (CD)": "Check whether this information conflicts with the conversation.",
            "Answer Refusal (AR)": "Determine if the question can be answered based on the conversation."
        }
        
        # Response format requirements
        FORMAT_REQUIREMENTS = {
            "Conflict Detection (CD)": "Response format: Reply strictly with either 'Yes' or 'No' only.",
            "Answer Refusal (AR)": "Response format: If the information is present in the conversation, provide answer based on that information; if not present, reply with: 'Not mentioned.'",
            "default": "Response format: Provide clear and accurate answers based on the conversation memory."
        }
        
        # Base template
        TEMPLATE = """You are a memory testing system. {instruction}

    IMPORTANT: 
    1. Provide only the answer without any reasoning process. Give the answer directly in English.
    2. Keep your answer within 100 words. Short and concise answers are acceptable.
    3.  Answer in English. This is a strict requirement. Do not answer in any other language.
    {context_section}
    {context_note}

    Question: {question}

    {format_requirement}

    Examples:
    Question: What is the cat's name?
    Correct answer: Almond

    Incorrect answer example (DO NOT answer like this):
    We need answer: cat name is Almond because..."""

        def __init__(self, question_type: str, context: str, context_note: str, question: str):
            self.question_type = question_type
            self.context = context
            self.context_note = context_note
            self.question = question
        
        def build(self) -> str:
            """Build the complete prompt"""
            
            # Get instruction for question type
            instruction = self.INSTRUCTIONS.get(self.question_type, self.INSTRUCTIONS["Unimodal Precise Recall"])
            
            # Get format requirement
            if self.question_type in self.FORMAT_REQUIREMENTS:
                format_requirement = self.FORMAT_REQUIREMENTS[self.question_type]
            else:
                format_requirement = self.FORMAT_REQUIREMENTS["default"]
            
            # Build context section
            context_section = "Complete conversation memory (contains multiple sessions):\n" + self.context
            
            return self.TEMPLATE.format(
                instruction=instruction,
                context_section=context_section,
                context_note=self.context_note,
                question=self.question,
                format_requirement=format_requirement
            )


class VLMEvaluator:
    """VLM评估器 - 使用完整对话上下文"""
    
    def __init__(self, 
                memory_system: FullTextMemorySystem,
                api_key: str,
                model: str = "",
                base_url: str = "",
                verbose: bool = False,
                max_retries: int = 3,
                timeout: int = 60,
                max_context_tokens: Optional[int] = None,
                truncation_strategy: str = "head_only",
                max_workers: int = 3,
                max_api_concurrency: int = 2):
        """
        初始化VLM评估器（多线程版本）
        
        Args:
            memory_system: 记忆系统实例
            api_key: VLM API密钥
            model: VLM模型名称
            base_url: API基础URL
            verbose: 详细日志输出
            max_retries: 最大重试次数
            timeout: 请求超时时间（秒）
            max_context_tokens: 最大上下文token数
            truncation_strategy: 截断策略
            max_workers: 最大线程数（用于并行处理session）
            max_api_concurrency: 最大API并发数（用于并行处理问题）
        """
        self.memory_system = memory_system
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.verbose = verbose
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_context_tokens = max_context_tokens
        self.truncation_strategy = truncation_strategy
        
        # 新增：线程控制相关属性
        self.max_workers = max_workers
        self.max_api_concurrency = max_api_concurrency
        self.api_semaphore = Semaphore(max_api_concurrency)  # 控制API并发
        self.file_lock = Lock()  # 文件写入锁
        self.stats_lock = Lock()  # 统计信息更新锁
        self.results_queue = queue.Queue()  # 结果队列（可选）
        
        # 初始化token计数器
        self.token_counter = TokenCounter()
        
        # 存储每个session的统计信息（保留时间计算）
        self.session_statistics = defaultdict(lambda: {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "processing_time": 0.0,
            "by_category": defaultdict(int),
            "by_difficulty": defaultdict(int),
            "truncated_count": 0,
            # 时间指标
            "total_memory_load_time": 0.0,
            "total_memory_recall_time": 0.0,
            "total_llm_time": 0.0
        })
        
        # 记录整体开始和结束时间
        self.start_time = None
        self.end_time = None
        
        # 测试API连接
        self._test_api_connection()
    
    def _test_api_connection(self):
        """测试API连接"""
        try:
            test_url = f"{self.base_url}/models"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(test_url, headers=headers, timeout=10)
            if response.status_code == 200:
                logger.info(f"API连接测试成功，模型: {self.model}")
                logger.info(f"API端点: {self.base_url}")
                if self.max_context_tokens:
                    logger.info(f"上下文截断: 最大 {self.max_context_tokens} tokens，策略: {self.truncation_strategy}")
            else:
                logger.warning(f"API连接测试返回非200状态码: {response.status_code}")
        except Exception as e:
            logger.error(f"API连接测试失败: {e}")
            logger.warning("请检查API服务是否启动以及API密钥是否正确")
    
    def load_questions(self, conversations_dir: str) -> Dict[str, Dict]:
        """
        加载所有session的intra-session问题，按session分组
        
        Args:
            conversations_dir: 对话数据目录
        
        Returns:
            按session_id分组的字典：{session_id: {"questions": [], "session_path": str}}
        """
        sessions_questions = {}
        
        # 解析目录结构
        base_dir = Path(conversations_dir)
        
        # 检查是否是"dialogueX"这样的顶层目录
        if base_dir.name.startswith("dialogue"):
            dialogue_name = base_dir.name
            scenes_dir = base_dir / "scenes"
        else:
            # 尝试在目录下查找包含"dialogue"或"对话"的子目录
            dialogue_dirs = [d for d in base_dir.iterdir() if d.is_dir() and (d.name.startswith("dialogue"))]
            if not dialogue_dirs:
                raise ValueError(f"找不到对话目录: {base_dir}")
            
            dialogue_name = dialogue_dirs[0].name
            scenes_dir = dialogue_dirs[0] / "scenes"
        
        if not scenes_dir.exists():
            raise ValueError(f"找不到scenes目录: {scenes_dir}")
        
        logger.info(f"正在从 {scenes_dir} 加载问题文件...")
        
        #遍历所有session目录
        session_dirs = [d for d in scenes_dir.iterdir() if d.is_dir()]
        
        for session_dir in session_dirs:
            session_dir_name = session_dir.name
            question_file = session_dir / "questions.json"
            
            if question_file.exists():
                try:
                    # 首先读取session的conversation.json获取session_id
                    conversation_file = session_dir / "session.json"
                    session_id = session_dir_name  # 默认为目录名
                    
                    # 加载问题文件
                    with open(question_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    questions = data.get("questions", [])
                    
                    # 转换格式为QuestionAnswerPair列表
                    question_pairs = []
                    for q in questions:
                        qa_pair = QuestionAnswerPair(
                            question_id=q.get("question_id", f"unknown_{len(question_pairs)}"),
                            session_id=session_id,
                            dialogue_name=dialogue_name,
                            question_text=q.get("question", {}).get("text", ""),
                            question_image=q.get("question", {}).get("image", ""),
                            original_answer=q.get("original_answer", ""),
                            answer_source=q.get("answer_source", "unknown"),
                            answer_session=q.get("answer_session", []),
                            question_type=q.get("question_type", {}),
                            difficulty=q.get("difficulty", "medium"),
                            supporting_evidence=q.get("supporting_evidence", []),
                            metadata={
                                "timestamp_description": q.get("timestamp_description", ""),
                                "validation_notes": q.get("validation_notes", ""),
                                "validated": q.get("validated", False),
                                "generated_at": q.get("generated_at", "")
                            }
                        )
                        
                        # 添加图片上下文
                        if qa_pair.question_image:
                            if str(session_id) == "session0":
                                folder, filename = qa_pair.question_image.split("/", 1)
                                possible_paths = [
                                    scenes_dir / folder / "image" / filename
                                ]
                            else:
                                img_filename = qa_pair.question_image
                                # 尝试多种可能的图片路径
                                possible_paths = [
                                    session_dir / "image" / img_filename,
                                    session_dir / img_filename,
                                    session_dir / "images" / img_filename
                                ]
                                
                            for img_path in possible_paths:
                                if img_path.exists():
                                    qa_pair.image_context = [str(img_path)]
                                    break
                        
                        question_pairs.append(qa_pair)
                    
                    sessions_questions[session_id] = {
                        "questions": question_pairs,
                        "session_dir_name": session_dir_name,
                        "session_path": str(session_dir),
                        "question_file": str(question_file)
                    }
                    
                    logger.info(f"从 {session_id} ({session_dir_name}) 加载了 {len(question_pairs)} 个问题")
                    
                except Exception as e:
                    logger.error(f"加载问题文件失败 {question_file}: {e}")
            else:
                logger.warning(f"跳过 {session_dir_name}，未找到问题文件")
        
        logger.info(f"总共从 {len(sessions_questions)} 个session加载了问题")
        return sessions_questions
    
    def _format_memory_context_only(self, memory_context: Dict[str, Any]) -> str:
        """
        只格式化记忆上下文部分，不含任何提示词和问题
        
        Args:
            memory_context: 包含所有session的完整上下文
        
        Returns:
            纯记忆上下文字符串
        """
        if not memory_context:
            return "无可用记忆"
        
        context_parts = []
        
        # 添加总体信息
        all_sessions = memory_context.get("all_sessions", [])
        total_dialogues = memory_context.get("total_dialogues", 0)
        
        context_parts.append(f"总session数: {len(all_sessions)}")
        context_parts.append(f"总对话轮次: {total_dialogues}")
        
        # 显示每个session的信息
        context_parts.append("\n【各Session信息】")
        session_info = memory_context.get("session_info", {})
        
        for session_id in all_sessions:
            info = session_info.get(session_id, {})
            session_title = info.get("session_title", "")
            timeline_date = info.get("timeline_date", "")
            dialogue_count = info.get("dialogue_count", 0)
            
            if session_title:
                context_parts.append(f"Session {session_id}: ({timeline_date}) - {dialogue_count}轮对话")
            else:
                context_parts.append(f"Session {session_id}: {dialogue_count}轮对话")
        
        # 添加所有session的对话内容
        dialogues_with_context = memory_context.get("dialogues_with_context", [])
        if dialogues_with_context:
            context_parts.append("\n【完整对话内容】")
            
            current_session = None
            for dialogue in dialogues_with_context:
                session_id = dialogue.get("session_id", "未知session")
                session_title = dialogue.get("session_title", "")
                dialogue_index = dialogue.get("dialogue_index", 0)
                session_date = dialogue.get("timeline_date", "")
                
                # 显示session分隔
                if session_id != current_session:
                    current_session = session_id
                    context_parts.append(f"\nSession {session_id}: {session_date}")
                
                # 显示对话内容
                role = dialogue.get("role", "")
                content = dialogue.get("content", {})
                text = content.get("text", "")
                image = content.get("image", "")
                image_description = content.get("image_description", "")
            
                if image:
                    context_parts.append(f"  第{dialogue_index}轮 - {role}: [图片{image}: {image_description}] {text}")
                else:
                    context_parts.append(f"  第{dialogue_index}轮 - {role}: {text}")
        
        return "\n".join(context_parts)
        
    def _truncate_context(self, context_text: str) -> tuple:
        """
        根据设置截断上下文
        
        Args:
            context_text: 原始上下文文本
        
        Returns:
            (截断后的文本, 原始token数, 截断后token数, 是否被截断)
        """
        if not self.max_context_tokens:
            # 不截断
            token_count = self.token_counter.count_tokens(context_text)
            return context_text, token_count, token_count, False
        
        original_tokens = self.token_counter.count_tokens(context_text)
        
        if original_tokens <= self.max_context_tokens:
            # 不需要截断
            return context_text, original_tokens, original_tokens, False
        
        # 需要截断
        if self.truncation_strategy == "head_only":
            # 只保留开头部分
            truncated_text, original, truncated = self.token_counter.truncate_text(
                context_text, self.max_context_tokens, preserve_ratio=1.0
            )
        elif self.truncation_strategy == "head_tail":
            # 保留开头和结尾（这里简单实现，更复杂的实现需要分段处理）
            # 这里暂时使用相同的head_only策略，实际应该实现更复杂的截断
            truncated_text, original, truncated = self.token_counter.truncate_text(
                context_text, self.max_context_tokens, preserve_ratio=0.7
            )
            # 添加标记说明
            if truncated < original:
                truncated_text += "\n\n[中间部分内容已截断以节省token]"
        else:
            # 默认策略
            truncated_text, original, truncated = self.token_counter.truncate_text(
                context_text, self.max_context_tokens, preserve_ratio=0.8
            )
        
        return truncated_text, original_tokens, truncated, True
        
    def _prepare_image_for_api(self, image_path: str) -> str:
        """
        将图片准备为API可接受的格式（base64编码）
        
        Args:
            image_path: 图片路径
        
        Returns:
            base64编码的图片字符串
        """
        try:
            with Image.open(image_path) as img:
                # 调整图片大小以控制API负载（可选）
                max_size = (1024, 1024)
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # 转换为RGB模式（如果图片有alpha通道）
                if img.mode in ('RGBA', 'LA', 'P'):
                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    rgb_img.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = rgb_img
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 保存到内存缓冲区并编码为base64
                buffer = BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                return img_base64
                
        except Exception as e:
            logger.error(f"处理图片 {image_path} 失败: {e}")
            raise
    
    def _call_vlm_api(self, prompt: str, images: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        调用真实的VLM API（带并发控制）
        """
        start_time = time.time()
        
        # 信号量控制
        acquired = False
        if self.api_semaphore:
            self.api_semaphore.acquire()
            acquired = True
            if self.verbose:
                logger.debug(f"API信号量已获取，当前可用: {self.api_semaphore._value}")
        
        try:
            # 准备消息
            messages = []
            
            if images and len(images) > 0:
                try:
                    image_base64 = self._prepare_image_for_api(images[0])
                    
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    })
                    
                except Exception as e:
                    logger.error(f"处理图片失败，将仅使用文本: {e}")
                    messages.append({
                        "role": "user",
                        "content": prompt
                    })
            else:
                messages.append({
                    "role": "user",
                    "content": prompt
                })
            
            # 构建请求数据
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 1024,
                "temperature": 0.1,
                "top_p": 0.9,
                "stream": False
            }
            
            # 重试机制
            for attempt in range(self.max_retries):
                try:
                    api_url = f"{self.base_url}/chat/completions"
                    
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                    
                    logger.debug(f"调用API (尝试 {attempt + 1}/{self.max_retries})")
                    
                    response = requests.post(
                        api_url,
                        headers=headers,
                        json=payload,
                        timeout=self.timeout
                    )
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        
                        if "choices" in response_data and len(response_data["choices"]) > 0:
                            answer = response_data["choices"][0]["message"]["content"].strip()
                            processing_time = time.time() - start_time
                            
                            return {
                                "answer": answer,
                                "processing_time": processing_time,
                                "model": self.model,
                                "success": True,
                                "raw_response": response_data
                            }
                        else:
                            error_msg = "API响应中没有choices字段"
                            logger.error(f"{error_msg}: {response_data}")
                            
                    else:
                        error_msg = f"API返回错误状态码: {response.status_code}"
                        logger.error(f"{error_msg}: {response.text}")
                        
                        if response.status_code == 429 and attempt < self.max_retries - 1:
                            wait_time = 2 ** (attempt + 1)
                            logger.warning(f"速率限制，等待 {wait_time} 秒后重试...")
                            time.sleep(wait_time)
                            continue
                    
                    if attempt == self.max_retries - 1:
                        raise Exception(f"{error_msg}")
                    else:
                        time.sleep(1)
                        
                except requests.exceptions.Timeout:
                    logger.error(f"API请求超时 (尝试 {attempt + 1}/{self.max_retries})")
                    if attempt == self.max_retries - 1:
                        raise Exception("API请求超时")
                    time.sleep(2)
                    
                except requests.exceptions.ConnectionError:
                    logger.error(f"API连接错误 (尝试 {attempt + 1}/{self.max_retries})")
                    if attempt == self.max_retries - 1:
                        raise Exception("API连接错误")
                    time.sleep(3)
                    
                except Exception as e:
                    logger.error(f"API调用异常 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                    if attempt == self.max_retries - 1:
                        raise
                    time.sleep(1)
            
            # 所有重试都失败
            processing_time = time.time() - start_time
            return {
                "answer": f"[API调用失败: 所有{self.max_retries}次重试都失败]",
                "processing_time": processing_time,
                "model": self.model,
                "success": False,
                "error": "所有重试都失败"
            }
        finally:
            if acquired:
                self.api_semaphore.release()
                if self.verbose:
                    logger.debug(f"API信号量已释放，当前可用: {self.api_semaphore._value}")
        
    def _construct_prompt_with_truncated_context(self, 
                                           question_pair: QuestionAnswerPair,
                                           truncated_context: str,
                                           original_tokens: int,
                                           truncated_tokens: int,
                                           was_truncated: bool) -> str:
        """
        Build complete prompt using truncated context
        
        Args:
            question_pair: Question-answer pair
            truncated_context: Truncated context text
            original_tokens: Original token count
            truncated_tokens: Truncated token count
            was_truncated: Whether truncation occurred
        
        Returns:
            Complete prompt string
        """
        
        # Extract question components
        question_text = question_pair.question_text
        question_type = question_pair.question_type.get("subsub_type", "")
        
        # Build prompt using template system
        prompt = PromptTemplate(
            question_type=question_type,
            context=truncated_context,
            context_note=self._get_context_note(was_truncated, original_tokens, truncated_tokens),
            question=question_text
        )
        
        # Log debug info
        if self.verbose:
            logger.debug(f"Question type: {question_type}")
        
        return prompt.build()

    def _get_context_note(self, was_truncated: bool, original_tokens: int, truncated_tokens: int) -> str:
        """Get context truncation note"""
        if was_truncated:
            return f"[Note: Due to context length limitations, the original conversation memory has been truncated from approximately {original_tokens} tokens to {truncated_tokens} tokens.]"
        return ""

        
    def _calculate_confidence(self, prediction: str, reference: str, answer_source: str) -> float:
        """计算回答置信度"""
        if not prediction or prediction.startswith("[") or "错误" in prediction or "异常" in prediction:
            return 0.0
        
        prediction = prediction.strip()
        reference = str(reference).strip()
        
        # 对于AR问题，检查是否符合格式要求
        if "Not mentioned" in prediction or "信息未提及" in prediction or "未提及" in prediction:
            # AR问题的回答应该是"Not mentioned"或具体信息
            if "Not mentioned" in reference or "信息提及" in reference:
                # 如果参考答案也是"Not mentioned"，则完全匹配
                if "Not mentioned" in prediction:
                    return 1.0
                else:
                    return 0.3
            else:
                # 参考答案不是"Not mentioned"，预测是"Not mentioned"
                return 0.0
        
        # 对于CD问题，检查是否严格符合"Yes."或"No."格式
        if prediction in ["Yes.", "No.", "Yes。", "No。", "是", "否"]:
            # 标准化参考答案
            ref_normalized = reference.lower().replace(".", "").replace("。", "")
            pred_normalized = prediction.lower().replace(".", "").replace("。", "")
            
            # 映射可能的答案
            yes_aliases = ["yes", "是", "存在", "有"]
            no_aliases = ["no", "否", "不存在", "没有"]
            
            if (pred_normalized in yes_aliases and ref_normalized in yes_aliases) or \
            (pred_normalized in no_aliases and ref_normalized in no_aliases):
                return 1.0
            else:
                return 0.0
        
        # 检查是否包含不确定的表述
        uncertain_phrases = [
            "我不知道", "不清楚", "不确定", "不记得", "没有提到",
            "无法回答", "没有相关信息", "未提及", "可能", "也许",
            "大概", "似乎", "好像", "不确定是否", "不太清楚"
        ]
        
        pred_lower = prediction.lower()
        for phrase in uncertain_phrases:
            if phrase in pred_lower:
                return 0.3
        
        # 对于其他类型的问题，使用原有的置信度计算方法
        return 0.7
    
    def evaluate_single_question(self, 
                           question_pair: QuestionAnswerPair,
                           session_id: str) -> EvaluationResult:
        """评估单个问题 - 记录详细的时间指标"""
        start_time = time.time()
        memory_load_start = 0
        memory_recall_start = 0
        llm_start = 0
        
        memory_load_time = 0
        memory_recall_time = 0
        llm_inference_time = 0
        
        try:
            logger.debug(f"处理问题: {session_id} - {question_pair.question_id} ({question_pair.category})")
            
            # 1. 获取记忆上下文（记录加载时间）
            memory_load_start = time.time()
            full_context = self.memory_system.get_session_context(session_id)
            memory_load_time = time.time() - memory_load_start
            
            # 创建记忆上下文摘要
            total_sessions = len(full_context.get("all_sessions", []))
            total_dialogues = full_context.get("total_dialogues", 0)
            target_info = full_context.get("target_session_info", {})
            session_title = target_info.get("session_title", "")
            
            memory_context_summary = f"总session数: {total_sessions}, 总对话轮次: {total_dialogues}, 目标session: {session_id}"
            if session_title:
                memory_context_summary += f" ({session_title})"
            
            # 2. 格式化记忆上下文（记录召回时间）
            memory_recall_start = time.time()
            raw_context = self._format_memory_context_only(full_context)
            
            # 3. 对记忆上下文进行截断
            truncated_context, original_tokens, truncated_tokens, was_truncated = self._truncate_context(raw_context)
            
            # 4. 构建完整的提示词
            prompt = self._construct_prompt_with_truncated_context(
                question_pair=question_pair,
                truncated_context=truncated_context,
                original_tokens=original_tokens,
                truncated_tokens=truncated_tokens,
                was_truncated=was_truncated
            )
            memory_recall_time = time.time() - memory_recall_start
            
            # 5. 准备图片
            images = []
            if question_pair.question_image and question_pair.image_context:
                for img_path in question_pair.image_context:
                    if os.path.exists(img_path):
                        images.append(img_path)
            
            # 6. 调用VLM API（记录LLM回答时间）
            llm_start = time.time()
            vlm_response = self._call_vlm_api(prompt=prompt, images=images)
            llm_inference_time = vlm_response.get("processing_time", 0)
            
            system_answer = vlm_response.get("answer", "").strip()
            success = vlm_response.get("success", False)
            
            # 7. 计算置信度
            confidence = self._calculate_confidence(
                system_answer, 
                question_pair.original_answer,
                question_pair.answer_source
            )
            
            # 总处理时间
            total_processing_time = time.time() - start_time
            
            # 8. 创建评估结果
            result = EvaluationResult(
                sample_id=f"{session_id}_{question_pair.question_id}_{int(time.time())}",
                session_id=question_pair.session_id,
                dialogue_name=question_pair.dialogue_name,
                question_id=question_pair.question_id,
                question_text=question_pair.question_text,
                question_image=question_pair.question_image,
                system_answer=system_answer,
                original_answer=question_pair.original_answer,
                answer_source=question_pair.answer_source,
                question_type=question_pair.question_type,
                category=question_pair.category,
                difficulty=question_pair.difficulty,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                memory_type=type(self.memory_system).__name__,
                vlm_model=self.model,
                processing_time=total_processing_time,
                confidence=confidence,
                supporting_evidence=question_pair.supporting_evidence,
                memory_context_summary=memory_context_summary,
                success=success,
                error_message=None if success else vlm_response.get("error", ""),
                truncated=was_truncated,
                original_context_length=original_tokens,
                truncated_context_length=truncated_tokens,
                # 时间指标
                memory_load_time=memory_load_time,
                memory_recall_time=memory_recall_time,
                llm_inference_time=llm_inference_time
            )
            
            # 更新session统计
            with self.stats_lock:
                self.session_statistics[session_id]["successful"] += 1
                self.session_statistics[session_id]["processing_time"] += total_processing_time
                if result.truncated:
                    self.session_statistics[session_id]["truncated_count"] += 1
                
                # 累计时间统计
                self.session_statistics[session_id]["total_memory_load_time"] += memory_load_time
                self.session_statistics[session_id]["total_memory_recall_time"] += memory_recall_time
                self.session_statistics[session_id]["total_llm_time"] += llm_inference_time
            
            logger.info(f"✓ 成功处理: {session_id} - {question_pair.question_id} (总时间: {total_processing_time:.2f}秒, LLM: {llm_inference_time:.2f}秒)")
            if result.truncated:
                logger.info(f"  上下文已截断: {result.original_context_length} -> {result.truncated_context_length} tokens")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)[:200]
            logger.error(f"✗ 处理问题 {session_id} - {question_pair.question_id} 时出错: {error_msg}")
            
            # 创建错误结果
            result = EvaluationResult(
                sample_id=f"error_{question_pair.question_id}_{int(time.time())}",
                session_id=question_pair.session_id,
                dialogue_name=question_pair.dialogue_name,
                question_id=question_pair.question_id,
                question_text=question_pair.question_text,
                question_image=question_pair.question_image,
                system_answer=f"[处理错误: {error_msg}]",
                original_answer=question_pair.original_answer,
                answer_source=question_pair.answer_source,
                question_type=question_pair.question_type,
                category=question_pair.category,
                difficulty=question_pair.difficulty,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                memory_type=type(self.memory_system).__name__,
                vlm_model=self.model,
                processing_time=processing_time,
                confidence=0.0,
                supporting_evidence=question_pair.supporting_evidence,
                memory_context_summary="错误: 无法获取记忆上下文",
                success=False,
                error_message=error_msg,
                truncated=False,
                memory_load_time=memory_load_time,
                memory_recall_time=memory_recall_time,
                llm_inference_time=llm_inference_time
            )
            
            # 更新session统计
            self.session_statistics[session_id]["failed"] += 1
            
            return result
    
    def evaluate_session_questions(self,
                                 session_id: str,
                                 session_data: Dict,
                                 max_questions: Optional[int] = None) -> List[Dict[str, Any]]:
        """评估单个session的所有问题"""
        questions = session_data["questions"]
        session_path = Path(session_data["session_path"])
        session_dir_name = session_data.get("session_dir_name", session_id)
        
        if max_questions and max_questions < len(questions):
            questions = questions[:max_questions]
        
        total_questions = len(questions)
        logger.info(f"开始评估 {session_id} 的 {total_questions} 个问题")
        logger.info(f"使用完整对话上下文：包含 {len(self.memory_system.memory_storage)} 个session的内容")
        if self.max_context_tokens:
            logger.info(f"上下文截断限制: {self.max_context_tokens} tokens，策略: {self.truncation_strategy}")
        
        # 初始化session统计
        self.session_statistics[session_id]["total"] = total_questions
        for qa in questions:
            self.session_statistics[session_id]["by_category"][qa.category] += 1
            self.session_statistics[session_id]["by_difficulty"][qa.difficulty] += 1
        
        results = []
        
        for i, question_pair in enumerate(questions, 1):
            progress = f"[{i}/{total_questions}]"
            logger.info(f"{progress} 处理 {session_id} - {question_pair.question_id} ({question_pair.category})")
            
            # 评估单个问题
            result = self.evaluate_single_question(question_pair, session_id)
            result_dict = asdict(result)
            results.append(result_dict)
        
        # 保存最终结果
        self._save_session_results(session_id, session_dir_name, session_path, results, final=True)
        
        logger.info(f"完成评估 {session_id}: 成功 {self.session_statistics[session_id]['successful']}, 失败 {self.session_statistics[session_id]['failed']}, 截断 {self.session_statistics[session_id]['truncated_count']}")
        
        return results
    
    def _save_session_results(self, 
                            session_id: str,
                            session_dir_name: str,
                            session_path: Path, 
                            results: List[Dict[str, Any]], 
                            final: bool = False):
        """保存单个session的结果到对应session目录"""
        # 在session目录下创建结果目录
        session_results_dir = session_path / "evaluation_results"
        session_results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存JSON结果
        json_filename = f"results_Fulltext.json"
        json_file = session_results_dir / json_filename
        
        # 构建完整的结果数据结构
        full_results = {
            "metadata": {
                "session_id": session_id,
                "session_dir_name": session_dir_name,
                "session_path": str(session_path),
                "vlm_model": self.model,
                "memory_type": type(self.memory_system).__name__,
                "base_url": self.base_url,
                "context_type": "full_conversation",
                "max_context_tokens": self.max_context_tokens,
                "truncation_strategy": self.truncation_strategy
            },
            "results": results
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"已保存 {session_id} 的结果到: {json_file}")
    
    def evaluate_all_sessions(self,  
                        sessions_questions: Dict[str, Dict],
                        max_questions_per_session: Optional[int] = None):
        """
        并行评估所有session的问题（多线程版本）
        """
        self.start_time = time.time()
        total_sessions = len(sessions_questions)
        
        logger.info(f"开始并行评估 {total_sessions} 个session")
        logger.info(f"使用完整对话上下文：包含 {len(self.memory_system.memory_storage)} 个session的内容")
        logger.info(f"线程配置: max_workers={self.max_workers}, max_api_concurrency={self.max_api_concurrency}")
        if self.max_context_tokens:
            logger.info(f"上下文截断限制: {self.max_context_tokens} tokens，策略: {self.truncation_strategy}")
        
        # 使用线程池并行处理session
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有session任务
            future_to_session = {}
            for session_id, session_data in sessions_questions.items():
                future = executor.submit(
                    self._evaluate_session_parallel,
                    session_id,
                    session_data,
                    max_questions_per_session
                )
                future_to_session[future] = session_id
            
            # 收集结果（带进度显示）
            completed = 0
            for future in concurrent.futures.as_completed(future_to_session):
                session_id = future_to_session[future]
                completed += 1
                try:
                    results = future.result()
                    logger.info(f"[{completed}/{total_sessions}] Session {session_id} 处理完成，成功处理 {len(results)} 个问题")
                except Exception as e:
                    logger.error(f"Session {session_id} 处理失败: {e}")
        
        self.end_time = time.time()
        self._save_session_statistics()  # 保存session统计到文件
    
    def _evaluate_session_parallel(self, session_id: str, session_data: Dict, 
                             max_questions_per_session: Optional[int]) -> List[Dict]:
        """
        session评估的包装方法（用于线程池调用）
        """
        thread_name = threading.current_thread().name
        logger.info(f"线程 [{thread_name}] 开始处理 session: {session_id}")
        
        try:
            # 调用并行处理session内问题的方法
            results = self._evaluate_session_questions_parallel(
                session_id, session_data, max_questions_per_session
            )
            return results
        except Exception as e:
            logger.error(f"线程 [{thread_name}] 处理 session {session_id} 时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
        
    def _evaluate_session_questions_parallel(self, session_id: str, session_data: Dict,
                                       max_questions: Optional[int] = None) -> List[Dict]:
        """
        并行处理一个session内的所有问题
        """
        questions = session_data["questions"]
        session_path = Path(session_data["session_path"])
        session_dir_name = session_data.get("session_dir_name", session_id)
        
        if max_questions and max_questions < len(questions):
            questions = questions[:max_questions]
        
        total_questions = len(questions)
        logger.info(f"Session {session_id} 开始并行处理 {total_questions} 个问题 (API并发: {self.max_api_concurrency})")
        
        # 线程安全地初始化session统计
        with self.stats_lock:
            self.session_statistics[session_id]["total"] = total_questions
            for qa in questions:
                self.session_statistics[session_id]["by_category"][qa.category] += 1
                self.session_statistics[session_id]["by_difficulty"][qa.difficulty] += 1
        
        # 用于存储结果的线程安全列表
        results = []
        results_lock = Lock()  # 局部锁
        
        # 使用线程池并行处理问题
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_api_concurrency) as executor:
            # 提交所有问题任务
            future_to_question = {}
            for question_pair in questions:
                future = executor.submit(
                    self._evaluate_question_with_stats,
                    question_pair,
                    session_id
                )
                future_to_question[future] = question_pair
            
            # 使用tqdm显示进度（可选）
            from tqdm import tqdm
            with tqdm(total=total_questions, desc=f"Session {session_id}", unit="q") as pbar:
                for future in concurrent.futures.as_completed(future_to_question):
                    question_pair = future_to_question[future]
                    try:
                        result_dict = future.result()
                        
                        # 线程安全地添加结果
                        with results_lock:
                            results.append(result_dict)
                        
                        # 更新进度条
                        pbar.update(1)
                        if result_dict.get("success", False):
                            pbar.set_postfix({"成功": "✓", "ID": question_pair.question_id})
                        else:
                            pbar.set_postfix({"成功": "✗", "ID": question_pair.question_id})
                        
                    except Exception as e:
                        logger.error(f"问题 {question_pair.question_id} 处理失败: {e}")
                        
                        with results_lock:
                            results.append({
                                "sample_id": f"error_{question_pair.question_id}",
                                "session_id": session_id,
                                "question_id": question_pair.question_id,
                                "success": False,
                                "error_message": str(e)[:200]
                            })
                        pbar.update(1)
        
        # 最终保存结果（需要线程安全）
        with self.file_lock:
            self._save_session_results(
                session_id,
                session_dir_name,
                session_path,
                results,
                final=True
            )
        
        logger.info(f"Session {session_id} 并行处理完成: 成功 {len([r for r in results if r.get('success', False)])}/{total_questions}")
        
        return results
    
    def _evaluate_question_with_stats(self, question_pair: QuestionAnswerPair, 
                                    session_id: str) -> Dict[str, Any]:
        """
        评估单个问题并更新统计信息（线程安全）
        
        Args:
            question_pair: 问题-答案对
            session_id: session ID
        """
        try:
            # 调用原有的评估方法
            result = self.evaluate_single_question(question_pair, session_id)
            result_dict = asdict(result)
            
            # 注意：不再维护全局统计，只更新session统计已在evaluate_single_question中完成
            return result_dict
            
        except Exception as e:
            logger.error(f"评估问题 {question_pair.question_id} 时发生未捕获异常: {e}")
            
            # 返回错误结果
            error_result = EvaluationResult(
                sample_id=f"error_{question_pair.question_id}_{int(time.time())}",
                session_id=session_id,
                dialogue_name=question_pair.dialogue_name,
                question_id=question_pair.question_id,
                question_text=question_pair.question_text,
                question_image=question_pair.question_image,
                system_answer=f"[处理错误: {str(e)[:200]}]",
                original_answer=question_pair.original_answer,
                answer_source=question_pair.answer_source,
                question_type=question_pair.question_type,
                category=question_pair.category,
                difficulty=question_pair.difficulty,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                memory_type=type(self.memory_system).__name__,
                vlm_model=self.model,
                processing_time=0,
                confidence=0.0,
                success=False,
                error_message=str(e)[:200],
                truncated=False
            )
            
            # 更新失败统计
            with self.stats_lock:
                self.session_statistics[session_id]["failed"] += 1
            
            return asdict(error_result)
    

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="VLM intra-session memory evaluator (full context)")
    parser.add_argument("--conversations_dir", type=str, required=True,
                       help="对话数据目录（包含scenes子目录）")
    parser.add_argument("--api_key", type=str, required=True,
                       help="VLM API密钥")
    parser.add_argument("--model", type=str, required=True,
                       help="VLM模型名称")
    parser.add_argument("--base_url", type=str, required=True,
                       help="API基础URL")
    parser.add_argument("--memory_type", type=str, default="full_text",
                       choices=["full_text"],
                       help="记忆系统类型")
    parser.add_argument("--max_questions_per_session", type=int, default=None,
                       help="每个session最大处理问题数")
    parser.add_argument("--max_sessions", type=int, default=None,
                       help="最大处理session数")
    parser.add_argument("--verbose", action="store_true",
                       help="详细日志输出")
    parser.add_argument("--test_mode", action="store_true",
                       help="测试模式，每个session只处理前2个问题")
    parser.add_argument("--max_retries", type=int, default=3,
                       help="API调用最大重试次数")
    parser.add_argument("--timeout", type=int, default=60,
                       help="API调用超时时间（秒）")
    
    # 新增参数：上下文截断相关
    parser.add_argument("--max_context_tokens", type=int, default=None,
                       help="最大上下文token数，超过则截断（例如：8000）")
    parser.add_argument("--truncation_strategy", type=str, default="head_only",
                       choices=["head_only", "head_tail"],
                       help="截断策略：head_only（只保留开头），head_tail（保留开头和结尾）")
    # 并发处理
    parser.add_argument("--max_workers", type=int, default=3,
                       help="最大线程数")
    parser.add_argument("--max_api_concurrency", type=int, default=2,
                       help="最大API并发数")
    
    args = parser.parse_args()
    
    # 配置日志级别
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    print("=" * 70)
    print("VLM Intra-Session记忆能力评估器（使用完整对话上下文）")
    print(f"模型: {args.model}")
    print(f"API端点: {args.base_url}")
    if args.max_context_tokens:
        print(f"上下文截断: 最大 {args.max_context_tokens} tokens，策略: {args.truncation_strategy}")
    else:
        print("上下文截断: 无限制")
    print("=" * 70)
    
    # 测试模式设置
    if args.test_mode:
        args.max_questions_per_session = 2
        print("测试模式：每个session只处理前2个问题")
    
    # 1. 初始化记忆系统（加载所有session）
    print(f"\n[1] 初始化记忆系统 ({args.memory_type})...")
    print(f"   加载整个对话的所有session内容...")
    
    memory_load_start_total = time.time()
    memory_system = create_memory_system(args.memory_type, args.conversations_dir)
    total_load_time = memory_system.load_all_conversations()
    memory_load_total_time = time.time() - memory_load_start_total
    
    print(f"   已加载 {len(memory_system.memory_storage)} 个session，共 {len(memory_system.all_dialogues)} 轮对话")
    print(f"   总加载时间: {memory_load_total_time:.2f}秒")
    
    # 显示加载的session信息
    print(f"\n   已加载的session列表:")
    for session_id, info in memory_system.session_info.items():
        session_title = info.get("session_title", "<未命名>")
        load_time = info.get("load_time", 0)
        if session_title:
            print(f"     - {session_id}: 《{session_title}》 ({info.get('dialogue_count', 0)}轮) - {load_time:.3f}秒")
        else:
            print(f"     - {session_id}: {info.get('dialogue_count', 0)}轮对话 - {load_time:.3f}秒")
    
    # 2. 初始化VLM评估器
    print(f"\n[2] 初始化VLM评估器...")
    evaluator = VLMEvaluator(
        memory_system=memory_system,
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        verbose=args.verbose,
        max_retries=args.max_retries,
        timeout=args.timeout,
        max_context_tokens=args.max_context_tokens,
        truncation_strategy=args.truncation_strategy,
        max_workers=args.max_workers,
        max_api_concurrency=args.max_api_concurrency
    )
    
    # 3. 加载intra-session问题（按session分组）
    print(f"\n[3] 加载intra-session问题文件（按session分组）...")
    try:
        sessions_questions = evaluator.load_questions(args.conversations_dir)
    except Exception as e:
        print(f"   加载问题失败: {e}")
        return
    
    if not sessions_questions:
        print("   未找到任何session的问题文件")
        return
    
    print(f"   成功从 {len(sessions_questions)} 个session加载了问题")
    
    # 显示每个session的信息
    total_questions = sum(len(data["questions"]) for data in sessions_questions.values())
    print(f"   总问题数: {total_questions}")
    
    for session_id, session_data in sessions_questions.items():
        question_count = len(session_data["questions"])
        session_dir_name = session_data.get("session_dir_name", session_id)
        if session_dir_name != session_id:
            print(f"     - {session_id} ({session_dir_name}): {question_count} 个问题")
        else:
            print(f"     - {session_id}: {question_count} 个问题")
    
    # 限制处理的session数
    if args.max_sessions and args.max_sessions < len(sessions_questions):
        sessions_to_process = dict(list(sessions_questions.items())[:args.max_sessions])
        print(f"   限制处理前 {args.max_sessions} 个session")
    else:
        sessions_to_process = sessions_questions
    
    # 4. 执行评估（按session独立处理，但使用完整上下文）
    print(f"\n[4] 开始按session评估（使用完整对话上下文）...")
    print(f"   处理session数: {len(sessions_to_process)}")
    print(f"   总问题数: {total_questions}")
    print(f"   记忆系统中的session数: {len(memory_system.memory_storage)}")
    print("-" * 70)
    
    evaluator.evaluate_all_sessions(
        sessions_questions=sessions_to_process,
        max_questions_per_session=args.max_questions_per_session
    )
    
    # 5. 输出全局统计
    print(f"\n[5] 评估完成!")
    print("-" * 70)
    

if __name__ == "__main__":
    main()