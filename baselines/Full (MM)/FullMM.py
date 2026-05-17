import os
import json
import logging
import argparse
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict
from natsort import natsorted

# 多线程相关导入
import concurrent.futures
import threading
from threading import Lock, Semaphore
from functools import partial

# 图片处理和API相关
import requests
from PIL import Image
import base64
from io import BytesIO

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class QuestionAnswerPair:
    """问题-答案对"""
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
    category: str = field(init=False)
    
    def __post_init__(self):
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
    images_limited: bool = False
    original_image_count: Optional[int] = None
    limited_image_count: Optional[int] = None
    
    # 新增时间字段
    context_prepare_time: float = 0.0  # 准备对话上下文时间
    image_prepare_time: float = 0.0    # 准备图片时间
    prompt_build_time: float = 0.0     # 构建提示词时间
    api_call_time: float = 0.0         # API调用时间


class TokenCounter:
    """Token计数器 - 移除锁避免死锁"""
    
    _instance = None
    _lock = Lock()  # 只在创建单例时使用
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    
    def __init__(self, model_name: str = "cl100k_base"):
        if not hasattr(self, 'initialized'):
            self.model_name = model_name
            self.encoding = None
            # 移除实例锁，因为tiktoken是线程安全的
            
            try:
                import tiktoken
                self.encoding = tiktoken.get_encoding(model_name)
                logger.info(f"成功加载tokenizer: {model_name}")
            except ImportError:
                logger.warning("tiktoken未安装，将使用字符数估算token")
            except Exception as e:
                logger.warning(f"加载tokenizer失败: {e}，将使用估算方法")
            
            self.initialized = True
    
    def count_tokens(self, text: str) -> int:
        """计算token数量 - 无锁，tiktoken本身是线程安全的"""
        if not text:
            return 0
        
        # 移除锁，直接使用encoding
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # 估算：中文字符算2个token，其他字符每4个字符1个token
            chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
            other_chars = len(text) - chinese_chars
            estimated_tokens = chinese_chars * 2 + other_chars * 0.25
            return int(estimated_tokens) + 1
    
    def truncate_text(self, text: str, max_tokens: int) -> Tuple[str, int, int]:
        """截断文本到指定token数"""
        if not text:
            return text, 0, 0
        
        # 移除锁，直接调用count_tokens
        original_tokens = self.count_tokens(text)
        
        if original_tokens <= max_tokens:
            return text, original_tokens, original_tokens
        
        if self.encoding:
            tokens = self.encoding.encode(text)
            truncated_tokens = tokens[:max_tokens]
            truncated_text = self.encoding.decode(truncated_tokens)
            return truncated_text, original_tokens, len(truncated_tokens)
        else:
            # 估算截断
            chars_per_token = len(text) / original_tokens
            keep_chars = int(max_tokens * chars_per_token)
            truncated_text = text[:keep_chars] + "... [the conversation memory has been truncated due to token limit]"
            truncated_tokens = self.count_tokens(truncated_text)
            return truncated_text, original_tokens, truncated_tokens

class ImageProcessor:
    """图片处理器 - 线程安全"""
    
    def __init__(self, cache_enabled: bool = True, max_size: Tuple[int, int] = (1024, 1024), quality: int = 85):
        self.cache_enabled = cache_enabled
        self.max_size = max_size
        self.quality = quality
        self.image_cache = {}
        self.image_metadata = {}
        self._cache_lock = Lock()
        self._metadata_lock = Lock()
    
    def process_image(self, image_path: str, session_id: str = None, filename: str = None,
                      is_question_image: bool = False, question_id: str = None) -> Dict:
        """
        处理图片，返回包含Base64数据和元信息的字典
        
        Args:
            image_path: 图片路径
            session_id: 图片所属session
            filename: 图片文件名
            is_question_image: 是否为问题图片
            question_id: 问题ID（如果是问题图片）
        """
        # 检查缓存
        base64_data = None
        with self._cache_lock:
            if self.cache_enabled and image_path in self.image_cache:
                base64_data = self.image_cache[image_path]

        if base64_data is None:
            base64_data = self._image_to_base64(image_path)
            with self._cache_lock:
                if self.cache_enabled:
                    self.image_cache[image_path] = base64_data
        
        # 存储元数据
        if session_id and filename:
            with self._metadata_lock:
                self.image_metadata[image_path] = {
                    "session_id": session_id,
                    "filename": filename
                }
        
        # 构建返回信息
        image_info = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_data}"
            }
        }
        
        # 添加标记
        if is_question_image:
            image_info["is_question_image"] = True
            if question_id:
                image_info["question_id"] = question_id
        else:
            # 上下文图片保留元信息
            if session_id:
                image_info["session_id"] = session_id
            if filename:
                image_info["file_name"] = filename
                image_info["image_id"] = f"{session_id}_{filename}" if session_id else filename
        
        return image_info
    
    def _image_to_base64(self, image_path: str) -> str:
        """将图片转换为Base64编码"""
        try:
            with Image.open(image_path) as img:
                img.thumbnail(self.max_size, Image.Resampling.LANCZOS)
                
                if img.mode in ('RGBA', 'LA', 'P'):
                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    rgb_img.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = rgb_img
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                buffer = BytesIO()
                img.save(buffer, format='JPEG', quality=self.quality)
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"处理图片 {image_path} 失败: {e}")
            raise
    
    def clear_cache(self):
        with self._cache_lock:
            self.image_cache.clear()


class FullMMSystem:
    """多模态全上下文记忆系统 - 存储所有session的对话和图片"""
    
    def __init__(self, conversations_dir: str):
        self.conversations_dir = conversations_dir
        self.memory_storage = {}
        self.all_dialogues = []
        self.session_info = {}
        
        # 图片相关存储
        self.image_paths = {}  # {session_id: {filename: full_path}}
        self.image_session_map = {}  # {filename: session_id}
        self.dialogue_image_map = []  # 对话中的图片记录
        self.image_processor = ImageProcessor()
        
        # 新增：存储时间记录
        self.storage_time = 0.0      # 总存储时间
        self.loading_time = 0.0      # 数据加载时间
        self.image_scan_time = 0.0   # 图片扫描时间
        
    def load_all_conversations(self):
        """加载整个对话的所有session数据 - 添加时间记录"""
        overall_start = time.time()
        
        scenes_dir = os.path.join(self.conversations_dir, "scenes")
        if not os.path.exists(scenes_dir):
            raise ValueError(f"找不到scenes目录: {scenes_dir}")
        
        session_dirs = natsorted([
            d for d in os.listdir(scenes_dir) 
            if os.path.isdir(os.path.join(scenes_dir, d))
        ])
        all_dialogues = []
        
        # 1. 数据加载时间
        loading_start = time.time()
        
        for session_dir_name in session_dirs:
            session_dir = os.path.join(scenes_dir, session_dir_name)
            session_data = self._load_single_session(session_dir_name, session_dir)
    
            if session_data:
                session_id = session_dir_name
                self.memory_storage[session_id] = session_data
                
                # 扫描图片目录
                self._scan_image_directory(session_id, session_dir)
                
                dialogues = session_data.get("dialogue", [])
                processed_dialogues = []
                timeline_date = session_data.get("timeline_date", "")
                for i, dialogue in enumerate(dialogues, 1):
                    role = dialogue.get("role", "")
                    content = dialogue.get("content", {})
                    text = timeline_date + ":" + content.get("text", "")
                    image_filename = content.get("image", "")
                    
                    dialogue_entry = {
                        "session_id": session_id,
                        "session_title": session_data.get("session_title", ""),
                        "timeline_date": session_data.get("timeline_date", ""),
                        "session_dir_name": session_dir_name,
                        "dialogue_index": i,
                        "role": role,
                        "text": text,
                        "image": image_filename if image_filename else None
                    }
                    
                    if image_filename:
                        self.dialogue_image_map.append({
                            "session_id": session_id,
                            "dialogue_index": i,
                            "image_filename": image_filename,
                            "role": role
                        })
                    
                    all_dialogues.append(dialogue_entry)
                    processed_dialogues.append(dialogue_entry)
                
                self.session_info[session_id] = {
                    "session_dir_name": session_dir_name,
                    "session_title": session_data.get("session_title", ""),
                    "timeline_date": session_data.get("timeline_date", ""),
                    "generated_at": session_data.get("generated_at", ""),
                    "dialogue_count": len(dialogues),
                    "image_count": len(self.image_paths.get(session_id, {})),
                    "session_path": str(session_dir)
                }
        
        self.loading_time = time.time() - loading_start
        logger.info(f"数据加载耗时: {self.loading_time:.2f}秒")
        
        # 2. 图片扫描时间（已在加载过程中完成，这里记录）
        self.image_scan_time = 0.0  # 图片扫描已计入loading_time
        
        self.all_dialogues = all_dialogues
        
        # 总存储时间
        self.storage_time = time.time() - overall_start
        
        logger.info(f"已加载 {len(self.memory_storage)} 个session，共 {len(all_dialogues)} 轮对话")
        logger.info(f"图片总数: {len(self.image_session_map)}")
        logger.info(f"包含图片的对话: {len(self.dialogue_image_map)} 轮")
        logger.info(f"记忆存储总耗时: {self.storage_time:.2f}秒 (加载: {self.loading_time:.2f}s)")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆系统统计信息 - 添加存储时间"""
        return {
            "total_sessions": len(self.memory_storage),
            "total_dialogues": len(self.all_dialogues),
            "total_images": len(self.image_session_map),
            "dialogues_with_images": len(self.dialogue_image_map),
            "session_info": self.session_info,
            # 新增存储时间统计
            "storage_time": self.storage_time,
            "loading_time": self.loading_time,
            "image_scan_time": self.image_scan_time,
            "avg_time_per_dialogue": self.storage_time / len(self.all_dialogues) if self.all_dialogues else 0
        }
    
    def _scan_image_directory(self, session_id: str, session_dir: str):
        """扫描图片目录"""
        image_dir = os.path.join(session_dir, "image")
        if os.path.exists(image_dir):
            self.image_paths[session_id] = {}
            for img_file in os.listdir(image_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    full_path = os.path.join(image_dir, img_file)
                    self.image_paths[session_id][img_file] = full_path
                    self.image_session_map[img_file] = session_id
    
    def _load_single_session(self, session_dir_name: str, session_dir: str) -> Optional[Dict]:
        """加载单个session的数据"""
        conversation_file = os.path.join(session_dir, "session.json")
        if not os.path.exists(conversation_file):
            return None
        try:
            with open(conversation_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载 {conversation_file} 失败: {e}")
            return None
    
    def get_image_path(self, session_id: str, image_filename: str) -> Optional[str]:
        """获取图片路径"""
        if session_id in self.image_paths and image_filename in self.image_paths[session_id]:
            return self.image_paths[session_id][image_filename]
        return None
    
    def get_image_session(self, image_filename: str) -> Optional[str]:
        """获取图片所属session"""
        return self.image_session_map.get(image_filename)
    
    def get_image_for_api(self, image_filename: str, is_question_image: bool = False, 
                          question_id: str = None) -> Optional[Dict]:
        """获取处理好的图片数据"""
        session_id = question_id
        
        image_path = self.get_image_path(session_id, image_filename)
        if not image_path:
            print(image_path)
            print(session_id)
        if not image_path or not os.path.exists(image_path):
            logger.warning(f"图片文件不存在: {image_filename}")
            return None
        
        return self.image_processor.process_image(
            image_path=image_path,
            session_id=session_id if not is_question_image else None,
            filename=image_filename if not is_question_image else None,
            is_question_image=is_question_image,
            question_id=question_id
        )
    
    def format_dialogue_context(self, max_tokens: Optional[int] = None) -> Tuple[str, int, int, bool]:
        """格式化对话上下文，支持截断"""
        context_parts = []
        
        for dialogue in self.all_dialogues:
            session_id = dialogue["session_id"]
            role = dialogue["role"]
            text = dialogue["text"]
            image = dialogue["image"]
            
            if image:
                context_parts.append(f"[Session {session_id} 第{dialogue['dialogue_index']}轮 {role}]: [图片: {image}] {text}")
            else:
                context_parts.append(f"[Session {session_id} 第{dialogue['dialogue_index']}轮 {role}]: {text}")
        
        full_context = "\n".join(context_parts)
        
        token_counter = TokenCounter()
        original_tokens = token_counter.count_tokens(full_context)
        if max_tokens and original_tokens > max_tokens:
            truncated_text, _, truncated_tokens = token_counter.truncate_text(full_context, max_tokens)
            truncated_text += "\n\n[tips: the conversation memory has been truncated due to token limit]"
            return truncated_text, original_tokens, truncated_tokens, True
        
        return full_context, original_tokens, original_tokens, False
    
    def get_session_context(self, target_session_id: str) -> Dict[str, Any]:
        """获取完整上下文"""
        return {
            "target_session_id": target_session_id,
            "target_session_info": self.session_info.get(target_session_id, {}),
            "all_sessions": list(self.memory_storage.keys()),
            "total_dialogues": len(self.all_dialogues),
            "dialogues": self.all_dialogues
        }

class DialoguePromptTemplate:
    """Standardized prompt template for dialogue-based questions"""
    
    TEMPLATE = """You are a multimodal memory testing system. {instruction}

IMPORTANT: 
1. Provide only the answer without any reasoning process. Give the answer directly in English.
2. Keep your answer within 100 words. Short and concise answers are acceptable.
3. Answer in English. This is a strict requirement. Do not answer in any other language.
[Complete Conversation Memory]
{context}

[Question]
{question}
{image_note}

{format_requirement}

Please answer based on the above memory content:"""

    def __init__(self, instruction: str, context: str, question: str, 
                 format_requirement: str, has_question_images: bool = False):
        self.instruction = instruction
        self.context = context
        self.question = question
        self.format_requirement = format_requirement
        self.has_question_images = has_question_images
    
    def build(self) -> str:
        """Build the complete prompt"""
        
        # Add image note if applicable
        image_note = "\n[Note: The question contains an image. Please analyze it together with the question.]" if self.has_question_images else ""
        
        return self.TEMPLATE.format(
            instruction=self.instruction,
            context=self.context,
            question=self.question,
            image_note=image_note,
            format_requirement=self.format_requirement
        )


class VLMEvaluator:
    """VLM评估器 - 多线程版本"""
    
    def __init__(self,
                memory_system: FullMMSystem,
                api_key: str,
                model: str = "",
                base_url: str = "",
                verbose: bool = False,
                max_retries: int = 5,
                timeout: int = 60,
                max_context_tokens: Optional[int] = None,
                max_images: Optional[int] = None,
                max_workers: int = 3,
                max_api_concurrency: int = 2):
        
        self.memory_system = memory_system
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.verbose = verbose
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_context_tokens = max_context_tokens
        self.max_images = max_images
        self.max_workers = max_workers
        self.max_api_concurrency = max_api_concurrency
        
        self.token_counter = TokenCounter()
        
        
        # 线程同步工具
        self.api_semaphore = Semaphore(max_api_concurrency)
        self.stats_lock = Lock()
        self.file_lock = Lock()
        
        # 新增：记录失败的问题文件路径
        self.failed_json_files = set()
        self.failed_lock = Lock()
        
        # 统计信息 - 添加时间统计字段
        self.session_statistics = defaultdict(lambda: {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "processing_time": 0.0,
            "by_category": defaultdict(int),
            "by_difficulty": defaultdict(int),
            "truncated_count": 0,
            "images_limited_count": 0,
            # 新增时间统计字段
            "total_context_prepare_time": 0.0,
            "total_image_prepare_time": 0.0,
            "total_prompt_build_time": 0.0,
            "total_api_call_time": 0.0
        })
        
        self.global_statistics = {
            "total_sessions": 0,
            "total_questions": 0,
            "successful_questions": 0,
            "failed_questions": 0,
            "truncated_questions": 0,
            "images_limited_questions": 0,
            "start_time": None,
            "end_time": None,
            "max_context_tokens": max_context_tokens,
            "max_images": max_images,
            "max_workers": max_workers,
            "max_api_concurrency": max_api_concurrency,
            "memory_type": "FullMMSystem"
        }
        
        self._test_api_connection()
    
    def _test_api_connection(self):
        """测试API连接"""
        try:
            test_url = f"{self.base_url}/models"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(test_url, headers=headers, timeout=10)
            if response.status_code == 200:
                logger.info(f"API连接测试成功")
            else:
                logger.warning(f"API连接测试返回状态码: {response.status_code}")
        except Exception as e:
            logger.error(f"API连接测试失败: {e}")
    
    def load_questions(self, conversations_dir: str) -> Dict[str, Dict]:
        """加载所有session的问题"""
        sessions_questions = {}
        base_dir = Path(conversations_dir)
        
        if base_dir.name.startswith("dialogue"):
            dialogue_name = base_dir.name
            scenes_dir = base_dir / "scenes"
        else:
            dialogue_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("dialogue")]
            if not dialogue_dirs:
                raise ValueError(f"找不到对话目录: {base_dir}")
            dialogue_name = dialogue_dirs[0].name
            scenes_dir = dialogue_dirs[0] / "scenes"
        
        if not scenes_dir.exists():
            raise ValueError(f"找不到scenes目录: {scenes_dir}")
        
        logger.info(f"正在从 {scenes_dir} 加载问题文件...")
        
        for session_dir in scenes_dir.iterdir():
            if not session_dir.is_dir():
                continue
            
            session_dir_name = session_dir.name
            question_file = session_dir / "questions.json"
            
            if not question_file.exists():
                continue
            
            try:
                # 获取session_id
                conversation_file = session_dir / "session.json"
                session_id = session_dir_name
               
                
                # 加载问题
                with open(question_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                questions = data.get("questions", [])
                question_pairs = []
                
                for q in questions:
                    qa_pair = QuestionAnswerPair(
                        question_id=q.get("question_id", f"FullMM_{len(question_pairs)}"),
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
                        },
                    )
                    question_pairs.append(qa_pair)
                
                sessions_questions[session_id] = {
                    "questions": question_pairs,
                    "session_dir_name": session_dir_name,
                    "session_path": str(session_dir),
                    "question_file": str(question_file)
                }
                
                logger.info(f"从 {session_id} 加载了 {len(question_pairs)} 个问题")
                
            except Exception as e:
                logger.error(f"加载问题文件失败 {question_file}: {e}")
        
        logger.info(f"总共从 {len(sessions_questions)} 个session加载了问题")
        return sessions_questions
    
    def _prepare_images(self, question_pair: QuestionAnswerPair) -> Tuple[List[Dict], bool, int, int]:
        """准备所有图片（问题图片 + 上下文图片）"""
        all_images = []
        question_images_count = 0
        context_images_count = 0
        images_limited = False
        
        # 1. 准备问题图片
        if question_pair.question_image:
            if str(question_pair.session_id) == "session0":
                folder, img_file = question_pair.question_image.split('/', 1)
                img_info = self.memory_system.get_image_for_api(
                    image_filename=img_file,
                    is_question_image=True,
                    question_id=folder
                )
                if img_info:
                    all_images.append(img_info)
                    question_images_count += 1
                    logger.debug(f"添加问题图片: {img_file}")
                else:
                    logger.warning(f"无法找到问题图片: {img_file}")
            else:
                image_files = [f.strip() for f in question_pair.question_image.split(',') if f.strip()]
                
                for img_file in image_files:
                    img_info = self.memory_system.get_image_for_api(
                        img_file,
                        is_question_image=True,
                        question_id=question_pair.session_id
                    )
                    if img_info:
                        all_images.append(img_info)
                        question_images_count += 1
                        logger.debug(f"添加问题图片: {img_file}")
                    else:
                        logger.warning(f"无法找到问题图片: {img_file}")
        
        # 2. 准备上下文图片（最多max_images - 问题图片数）
        remaining_slots = self.max_images - question_images_count if self.max_images else None
        if remaining_slots is None or remaining_slots > 0:
            # 收集所有上下文图片
            context_images_set = set()
            for dialogue in self.memory_system.all_dialogues:
                if dialogue["image"]:
                    context_images_set.add((dialogue["image"], dialogue["session_id"]))
            # 限制数量
            context_images_list = list(context_images_set)
            if remaining_slots and len(context_images_list) > remaining_slots:
                context_images_list = context_images_list[:remaining_slots]
                images_limited = True
            
            # 处理上下文图片
            for img_file in context_images_list:
                img_info = self.memory_system.get_image_for_api(img_file[0], is_question_image=False, question_id=img_file[1])
                if img_info:
                    all_images.append(img_info)
                    context_images_count += 1
        
        original_count = question_images_count + len(context_images_set)
        limited_count = len(all_images)
        
        return all_images, images_limited, original_count, limited_count
    
    def _call_vlm_api(self, prompt: str, images: List[Dict]) -> Dict[str, Any]:
        """调用VLM API（带并发控制）"""
        start_time = time.time()
        # 获取信号量
        acquired = self.api_semaphore.acquire(timeout=30)
        if not acquired:
            logger.warning("等待API信号量超时")
            return {
                "answer": "[API调用失败: 等待信号量超时]",
                "processing_time": time.time() - start_time,
                "success": False,
                "error": "信号量获取超时"
            }
        
        try:
            # 构建消息
            content_list = [{"type": "text", "text": prompt}]
            content_list.extend(images)
            
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": content_list}],
                "max_tokens": 1024,
                "temperature": 0.1
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # 重试机制
            for attempt in range(self.max_retries):
                try:
                    response = requests.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=self.timeout
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        answer = data["choices"][0]["message"]["content"].strip()
                        return {
                            "answer": answer,
                            "processing_time": time.time() - start_time,
                            "success": True
                        }
                    elif response.status_code == 429 and attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        error_msg = f"API返回错误: {response.status_code}"
                        if attempt == self.max_retries - 1:
                            return {
                                "answer": f"[{error_msg}]",
                                "processing_time": time.time() - start_time,
                                "success": False,
                                "error": error_msg
                            }
                        time.sleep(1)
                        
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        return {
                            "answer": f"[API调用异常: {str(e)}]",
                            "processing_time": time.time() - start_time,
                            "success": False,
                            "error": str(e)
                        }
                    time.sleep(1)
            
            return {
                "answer": "[API调用失败]",
                "processing_time": time.time() - start_time,
                "success": False
            }
            
        finally:
            self.api_semaphore.release()
    
    def _get_instruction(self, question_pair: QuestionAnswerPair) -> str:
        """Get instruction for question type"""
        question_type = question_pair.question_type.get("subsub_type", "")
        
        # Instructions for 9 question types (only CD, AR, TTL include abbreviation)
        instructions = {
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
        
        # Return instruction based on question type
        if question_type in instructions:
            return instructions[question_type]
        else:
            return "Answer the question directly based on the conversation memory."


    def _build_prompt(self, question_pair: QuestionAnswerPair, 
                        dialogue_context: str, has_question_images: bool) -> str:
        """Build prompt for question"""
        
        question_type = question_pair.question_type.get("subsub_type", "")
        instruction = self._get_instruction(question_pair)
        
        # Get format requirement for special types
        format_requirement = self._get_format_requirement(question_type)
        
        # Build prompt using template
        prompt = DialoguePromptTemplate(
            instruction=instruction,
            context=dialogue_context,
            question=question_pair.question_text,
            format_requirement=format_requirement,
            has_question_images=has_question_images
        )
        
        return prompt.build()


    def _get_format_requirement(self, question_type: str) -> str:
        """Get format requirement based on question type"""
        
        format_requirements = {
            "Conflict Detection (CD)": "Response format: Reply strictly with either 'Yes' or 'No' only.",
            "Answer Refusal (AR)": "Response format: If the information is present in the conversation, provide answer based on that information; if not present, reply with: 'Not mentioned.'",
        }
        
        if question_type in format_requirements:
            return format_requirements[question_type]
        else:
            return "Response format: Provide clear and accurate answers based on the conversation memory."

    
    def _calculate_confidence(self, prediction: str, reference: str) -> float:
        """计算置信度"""
        if not prediction or prediction.startswith("["):
            return 0.0
        
        prediction = prediction.strip()
        reference = str(reference).strip()
        
        # 完全匹配
        if prediction == reference:
            return 1.0
        
        # 包含关系
        if reference in prediction or prediction in reference:
            return 0.8
        
        # 关键词匹配
        pred_words = set(prediction.lower().split())
        ref_words = set(reference.lower().split())
        if pred_words and ref_words:
            intersection = pred_words.intersection(ref_words)
            union = pred_words.union(ref_words)
            return len(intersection) / len(union)
        
        return 0.3
    
    def evaluate_single_question(self, question_pair: QuestionAnswerPair, 
                            question_file_path: str = None) -> EvaluationResult:
        """评估单个问题 - 添加详细时间计算和错误记录"""
        start_time = time.time()
        
        # 时间记录变量
        context_prepare_time = 0.0
        image_prepare_time = 0.0
        prompt_build_time = 0.0
        api_call_time = 0.0
        
        try:
            # 1. 准备对话上下文（记录时间）
            context_start = time.time()
            full_context = self.memory_system.get_session_context(question_pair.session_id)
            dialogue_context, orig_tokens, truncated_tokens, was_truncated = \
                self.memory_system.format_dialogue_context(self.max_context_tokens)
            context_prepare_time = time.time() - context_start
            
            # 2. 准备图片（记录时间）
            image_start = time.time()
            images, images_limited, orig_img_count, limited_img_count = self._prepare_images(question_pair)
            image_prepare_time = time.time() - image_start
            
            # 3. 构建提示词（记录时间）
            prompt_start = time.time()
            prompt = self._build_prompt(
                question_pair,
                dialogue_context,
                len([img for img in images if img.get("is_question_image")]) > 0
            )
            prompt_build_time = time.time() - prompt_start
            
            # 4. 调用API（记录时间）
            api_start = time.time()
            response = self._call_vlm_api(prompt, images)
            api_call_time = response.get("processing_time", time.time() - api_start)
            
            # 5. 计算置信度
            confidence = self._calculate_confidence(
                response.get("answer", ""),
                question_pair.original_answer
            )
            
            # 总处理时间
            total_processing_time = time.time() - start_time
            
            # 6. 创建结果
            result = EvaluationResult(
                sample_id=f"{question_pair.session_id}_{question_pair.question_id}_{int(time.time())}",
                session_id=question_pair.session_id,
                dialogue_name=question_pair.dialogue_name,
                question_id=question_pair.question_id,
                question_text=question_pair.question_text,
                question_image=question_pair.question_image,
                system_answer=response.get("answer", ""),
                original_answer=question_pair.original_answer,
                answer_source=question_pair.answer_source,
                question_type=question_pair.question_type,
                category=question_pair.category,
                difficulty=question_pair.difficulty,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                memory_type="FullMMSystem",
                vlm_model=self.model,
                processing_time=total_processing_time,
                confidence=confidence,
                supporting_evidence=question_pair.supporting_evidence,
                memory_context_summary=f"总对话轮次: {len(self.memory_system.all_dialogues)}",
                success=response.get("success", False),
                error_message=response.get("error"),
                truncated=was_truncated,
                original_context_length=orig_tokens if was_truncated else None,
                truncated_context_length=truncated_tokens if was_truncated else None,
                images_limited=images_limited,
                original_image_count=orig_img_count,
                limited_image_count=limited_img_count,
                # 新增时间字段
                context_prepare_time=context_prepare_time,
                image_prepare_time=image_prepare_time,
                prompt_build_time=prompt_build_time,
                api_call_time=api_call_time
            )
            
            # 输出详细日志
            logger.info(f"✓ {question_pair.session_id}/{question_pair.question_id} - "
                    f"总: {total_processing_time:.2f}s, "
                    f"上下文: {context_prepare_time:.3f}s, "
                    f"图片: {image_prepare_time:.3f}s, "
                    f"提示词: {prompt_build_time:.3f}s, "
                    f"API: {api_call_time:.2f}s")
            
            return result
            
        except Exception as e:
            total_processing_time = time.time() - start_time
            error_msg = str(e)[:200]
            logger.error(f"✗ 评估问题 {question_pair.question_id} 失败: {error_msg}")
            
            return EvaluationResult(
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
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                memory_type="FullMMSystem",
                vlm_model=self.model,
                processing_time=total_processing_time,
                confidence=0.0,
                supporting_evidence=question_pair.supporting_evidence,
                success=False,
                error_message=error_msg,
                context_prepare_time=context_prepare_time,
                image_prepare_time=image_prepare_time,
                prompt_build_time=prompt_build_time,
                api_call_time=api_call_time
            )
    
    def _evaluate_question_wrapper(self, question_pair: QuestionAnswerPair, 
                               question_file_path: str = None) -> Dict:
        """问题评估包装器（用于线程池）- 传递文件路径"""
        try:
            result = self.evaluate_single_question(question_pair, question_file_path)
            result_dict = asdict(result)
            
            # 更新统计
            with self.stats_lock:
                session_id = question_pair.session_id
                if result.success:
                    self.session_statistics[session_id]["successful"] += 1
                else:
                    self.session_statistics[session_id]["failed"] += 1
                
                self.session_statistics[session_id]["processing_time"] += result.processing_time
                
                # 累计时间统计
                self.session_statistics[session_id]["total_context_prepare_time"] += result.context_prepare_time
                self.session_statistics[session_id]["total_image_prepare_time"] += result.image_prepare_time
                self.session_statistics[session_id]["total_prompt_build_time"] += result.prompt_build_time
                self.session_statistics[session_id]["total_api_call_time"] += result.api_call_time
                
                if result.truncated:
                    self.session_statistics[session_id]["truncated_count"] += 1
                    self.global_statistics["truncated_questions"] += 1
                
                if result.images_limited:
                    self.session_statistics[session_id]["images_limited_count"] += 1
                    self.global_statistics["images_limited_questions"] += 1
            
            return result_dict
            
        except Exception as e:
            logger.error(f"包装器执行失败: {e}")
            # 记录失败的文件路径
            return {
                "sample_id": f"error_{question_pair.question_id}",
                "session_id": question_pair.session_id,
                "question_id": question_pair.question_id,
                "success": False,
                "error_message": str(e)[:200]
            }
    
    def evaluate_session(self, session_id: str, session_data: Dict,
                    max_questions: Optional[int] = None) -> List[Dict]:
        """并行评估一个session的所有问题 - 传递文件路径"""
        questions = session_data["questions"]
        session_path = Path(session_data["session_path"])
        question_file_path = session_data.get("question_file", "")  # 获取问题文件路径
        
        if max_questions and max_questions < len(questions):
            questions = questions[:max_questions]
        
        total = len(questions)
        logger.info(f"并行评估 session {session_id} 的 {total} 个问题 (API并发: {self.max_api_concurrency})")
        
        # 初始化统计
        with self.stats_lock:
            self.session_statistics[session_id]["total"] = total
            for q in questions:
                self.session_statistics[session_id]["by_category"][q.category] += 1
                self.session_statistics[session_id]["by_difficulty"][q.difficulty] += 1
        
        # 使用线程池并行处理问题
        results = []
        results_lock = Lock()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_api_concurrency) as executor:
            future_to_q = {
                executor.submit(self._evaluate_question_wrapper, q, question_file_path): q
                for q in questions
            }
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_q):
                try:
                    result = future.result(timeout=120)
                    with results_lock:
                        results.append(result)
                    
                    completed += 1
                    if completed % max(1, total // 10) == 0:
                        logger.info(f"[{session_id}] 进度: {completed}/{total}")
                        
                except Exception as e:
                    logger.error(f"获取结果失败: {e}")
        
        # 保存结果
        with self.file_lock:
            self._save_session_results(session_id, session_data["session_dir_name"], 
                                    session_path, results)
        
        # 更新全局统计
        with self.stats_lock:
            self.global_statistics["total_questions"] += total
            self.global_statistics["successful_questions"] += \
                self.session_statistics[session_id]["successful"]
            self.global_statistics["failed_questions"] += \
                self.session_statistics[session_id]["failed"]
        
        # 输出session时间统计
        session_stats = self.session_statistics[session_id]
        successful = session_stats["successful"]
        if successful > 0:
            logger.info(f"Session {session_id} 时间统计 - "
                    f"平均上下文: {session_stats['total_context_prepare_time']/successful:.3f}s, "
                    f"平均图片: {session_stats['total_image_prepare_time']/successful:.3f}s, "
                    f"平均提示词: {session_stats['total_prompt_build_time']/successful:.3f}s, "
                    f"平均API: {session_stats['total_api_call_time']/successful:.2f}s")
        
        logger.info(f"Session {session_id} 完成: 成功 {session_stats['successful']}/{total}")
        return results

    
    def _save_session_results(self, session_id: str, session_dir_name: str,
                            session_path: Path, results: List[Dict]):
        """保存session结果"""
        results_dir = session_path / "evaluation_results"
        results_dir.mkdir(exist_ok=True)
        
        output = {
            "metadata": {
                "session_id": session_id,
                "session_dir_name": session_dir_name,
                "vlm_model": self.model,
                "max_context_tokens": self.max_context_tokens,
                "max_images": self.max_images,
                "timestamp": datetime.now().isoformat()
            },
            "results": results
        }
        
        filepath = results_dir / "result_FullMM.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"结果已保存: {filepath}")
    
    def evaluate_all_sessions(self, sessions_questions: Dict[str, Dict],
                        max_questions_per_session: Optional[int] = None):
        """并行评估所有session - 添加失败记录保存"""
        self.global_statistics["start_time"] = time.time()
        self.global_statistics["total_sessions"] = len(sessions_questions)
        
        logger.info(f"开始并行评估 {len(sessions_questions)} 个session")
        logger.info(f"Session并行数: {self.max_workers}, API并发: {self.max_api_concurrency}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_session = {}
            for session_id, session_data in sessions_questions.items():
                future = executor.submit(
                    self.evaluate_session,
                    session_id,
                    session_data,
                    max_questions_per_session
                )
                future_to_session[future] = session_id
            
            for future in concurrent.futures.as_completed(future_to_session):
                session_id = future_to_session[future]
                try:
                    results = future.result()
                    logger.info(f"Session {session_id} 处理完成")
                except Exception as e:
                    logger.error(f"Session {session_id} 处理失败: {e}")
        self.global_statistics["end_time"] = time.time()
        


def main():
    parser = argparse.ArgumentParser(description="多模态记忆评估系统（多线程版）")
    parser.add_argument("--conversations_dir", type=str, required=True,
                       help="对话数据目录")
    parser.add_argument("--api_key", type=str, required=True,
                       help="VLM API密钥")
    parser.add_argument("--model", type=str, required=True,
                       help="VLM模型名称")
    parser.add_argument("--base_url", type=str, required=True,
                       help="API基础URL")
    parser.add_argument("--max_context_tokens", type=int, default=4096,
                       help="对话内容最大token数")
    parser.add_argument("--max_images", type=int, default=None,
                       help="最大图片数量")
    parser.add_argument("--max_workers", type=int, default=3,
                       help="Session级并行线程数")
    parser.add_argument("--max_api_concurrency", type=int, default=2,
                       help="API并发数")
    parser.add_argument("--max_questions_per_session", type=int, default=None,
                       help="每个session最大问题数")
    parser.add_argument("--max_sessions", type=int, default=None,
                       help="最大处理session数")
    parser.add_argument("--verbose", action="store_true",
                       help="详细日志")
    parser.add_argument("--test_mode", action="store_true",
                       help="测试模式（每个session处理2个问题）")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    print("=" * 70)
    print("多模态记忆评估系统（多线程版）")
    print(f"模型: {args.model}")
    print(f"Session并行: {args.max_workers}, API并发: {args.max_api_concurrency}")
    if args.max_context_tokens:
        print(f"对话截断: {args.max_context_tokens} tokens")
    if args.max_images:
        print(f"图片限制: {args.max_images} 张")
    print("=" * 70)
    
    if args.test_mode:
        args.max_questions_per_session = 2
        print("测试模式: 每个session处理2个问题")
    
    # 1. 初始化记忆系统
    print("\n[1] 初始化记忆系统...")
    memory_system = FullMMSystem(args.conversations_dir)
    memory_system.load_all_conversations()

    # 输出存储时间统计
    print(f"\n记忆存储时间统计:")
    print(f"   总存储时间: {memory_system.storage_time:.2f}秒")
    print(f"   数据加载: {memory_system.loading_time:.2f}秒")
    print(f"   总对话轮次: {len(memory_system.all_dialogues)}")
    if len(memory_system.all_dialogues) > 0:
        print(f"   平均每轮对话: {memory_system.storage_time / len(memory_system.all_dialogues):.3f}秒")
    print(f"   总图片数: {len(memory_system.image_session_map)}")
    
    # 2. 初始化评估器
    print("\n[2] 初始化评估器...")
    evaluator = VLMEvaluator(
        memory_system=memory_system,
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        verbose=args.verbose,
        max_context_tokens=args.max_context_tokens,
        max_images=args.max_images,
        max_workers=args.max_workers,
        max_api_concurrency=args.max_api_concurrency
    )
    
    # 3. 加载问题
    print("\n[3] 加载问题...")
    try:
        sessions_questions = evaluator.load_questions(args.conversations_dir)
    except Exception as e:
        print(f"加载问题失败: {e}")
        return
    
    if not sessions_questions:
        print("未找到问题文件")
        return
    
    total = sum(len(d["questions"]) for d in sessions_questions.values())
    print(f"从 {len(sessions_questions)} 个session加载了 {total} 个问题")
    
    # 限制session数
    if args.max_sessions and args.max_sessions < len(sessions_questions):
        sessions_to_process = dict(list(sessions_questions.items())[:args.max_sessions])
        print(f"限制处理前 {args.max_sessions} 个session")
    else:
        sessions_to_process = sessions_questions
    
    # 4. 执行评估
    print("\n[4] 开始并行评估...")
    evaluator.evaluate_all_sessions(
        sessions_questions=sessions_to_process,
        max_questions_per_session=args.max_questions_per_session
    )
    
    # 5. 输出结果
    print("\n[5] 评估完成!")


if __name__ == "__main__":
    main()