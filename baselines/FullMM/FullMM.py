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

# Multi-threading imports
import concurrent.futures
import threading
from threading import Lock, Semaphore
from functools import partial

# Image processing and API imports
import requests
from PIL import Image
import base64
from io import BytesIO

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class QuestionAnswerPair:
    """Question-Answer Pair"""
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
    image_context: Optional[List[str]] = None
    metadata: Optional[Dict] = None
    category: str = field(init=False)
    
    def __post_init__(self):
        if self.question_type:
            sub_type = self.question_type.get("sub_type", "")
            if sub_type:
                self.category = sub_type
            else:
                sub_type = self.question_type.get("sub_type", "")
                self.category = sub_type or self.question_type.get("main_type", "general")
        else:
            self.category = "general"


@dataclass
class EvaluationResult:
    """Evaluation result"""
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
    
    # Timing fields
    context_prepare_time: float = 0.0
    image_prepare_time: float = 0.0
    prompt_build_time: float = 0.0
    api_call_time: float = 0.0


class TokenCounter:
    """Token counter - lock-free to avoid deadlocks"""
    
    _instance = None
    _lock = Lock()  # Only used for singleton creation
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    
    def __init__(self, model_name: str = "cl100k_base"):
        if not hasattr(self, 'initialized'):
            self.model_name = model_name
            self.encoding = None
            
            try:
                import tiktoken
                self.encoding = tiktoken.get_encoding(model_name)
                logger.info(f"Successfully loaded tokenizer: {model_name}")
            except ImportError:
                logger.warning("tiktoken not installed, will estimate tokens using character count")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}, will use estimation")
            
            self.initialized = True
    
    def count_tokens(self, text: str) -> int:
        """Count tokens - lock-free, tiktoken is thread-safe"""
        if not text:
            return 0
        
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Estimate: Chinese characters count as 2 tokens, other chars as 0.25 tokens per character
            chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
            other_chars = len(text) - chinese_chars
            estimated_tokens = chinese_chars * 2 + other_chars * 0.25
            return int(estimated_tokens) + 1
    
    def truncate_text(self, text: str, max_tokens: int) -> Tuple[str, int, int]:
        """Truncate text to specified token limit"""
        if not text:
            return text, 0, 0
        
        original_tokens = self.count_tokens(text)
        
        if original_tokens <= max_tokens:
            return text, original_tokens, original_tokens
        
        if self.encoding:
            tokens = self.encoding.encode(text)
            truncated_tokens = tokens[:max_tokens]
            truncated_text = self.encoding.decode(truncated_tokens)
            return truncated_text, original_tokens, len(truncated_tokens)
        else:
            # Estimate truncation
            chars_per_token = len(text) / original_tokens
            keep_chars = int(max_tokens * chars_per_token)
            truncated_text = text[:keep_chars] + "... [the conversation memory has been truncated due to token limit]"
            truncated_tokens = self.count_tokens(truncated_text)
            return truncated_text, original_tokens, truncated_tokens


class ImageProcessor:
    """Image processor - thread-safe"""
    
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
        Process image, return dict with Base64 data and metadata
        
        Args:
            image_path: Path to image
            session_id: Session the image belongs to
            filename: Image filename
            is_question_image: Whether this is a question image
            question_id: Question ID (if it's a question image)
        """
        # Check cache
        base64_data = None
        with self._cache_lock:
            if self.cache_enabled and image_path in self.image_cache:
                base64_data = self.image_cache[image_path]

        if base64_data is None:
            base64_data = self._image_to_base64(image_path)
            with self._cache_lock:
                if self.cache_enabled:
                    self.image_cache[image_path] = base64_data
        
        # Store metadata
        if session_id and filename:
            with self._metadata_lock:
                self.image_metadata[image_path] = {
                    "session_id": session_id,
                    "filename": filename
                }
        
        # Build return info
        image_info = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_data}"
            }
        }
        
        # Add markers
        if is_question_image:
            image_info["is_question_image"] = True
            if question_id:
                image_info["question_id"] = question_id
        else:
            # Keep metadata for context images
            if session_id:
                image_info["session_id"] = session_id
            if filename:
                image_info["file_name"] = filename
                image_info["image_id"] = f"{session_id}_{filename}" if session_id else filename
        
        return image_info
    
    def _image_to_base64(self, image_path: str) -> str:
        """Convert image to Base64 encoding"""
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
            logger.error(f"Failed to process image {image_path}: {e}")
            raise
    
    def clear_cache(self):
        with self._cache_lock:
            self.image_cache.clear()


class FullMMSystem:
    """Multimodal full-context memory system - stores all session dialogues and images"""
    
    def __init__(self, conversations_dir: str):
        self.conversations_dir = conversations_dir
        self.memory_storage = {}
        self.all_dialogues = []
        self.session_info = {}
        
        # Image storage
        self.image_paths = {}  # {session_id: {filename: full_path}}
        self.image_session_map = {}  # {filename: session_id}
        self.dialogue_image_map = []  # Image records in dialogues
        self.image_processor = ImageProcessor()
        
        # Time recording fields
        self.storage_time = 0.0
        self.loading_time = 0.0
        self.image_scan_time = 0.0
        
    def load_all_conversations(self):
        """Load all session data from the conversation - with timing"""
        overall_start = time.time()
        
        scenes_dir = os.path.join(self.conversations_dir, "scenes")
        if not os.path.exists(scenes_dir):
            raise ValueError(f"Scenes directory not found: {scenes_dir}")
        
        session_dirs = natsorted([
            d for d in os.listdir(scenes_dir) 
            if os.path.isdir(os.path.join(scenes_dir, d))
        ])
        all_dialogues = []
        
        # 1. Data loading time
        loading_start = time.time()
        
        for session_dir_name in session_dirs:
            session_dir = os.path.join(scenes_dir, session_dir_name)
            session_data = self._load_single_session(session_dir_name, session_dir)
    
            if session_data:
                session_id = session_dir_name
                self.memory_storage[session_id] = session_data
                
                # Scan image directory
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
        logger.info(f"Data loading time: {self.loading_time:.2f} seconds")
        
        # 2. Image scan time (already included in loading_time)
        self.image_scan_time = 0.0
        
        self.all_dialogues = all_dialogues
        
        # Total storage time
        self.storage_time = time.time() - overall_start
        
        logger.info(f"Loaded {len(self.memory_storage)} sessions, {len(all_dialogues)} dialogue turns")
        logger.info(f"Total images: {len(self.image_session_map)}")
        logger.info(f"Dialogues containing images: {len(self.dialogue_image_map)} turns")
        logger.info(f"Memory storage total time: {self.storage_time:.2f}s (Loading: {self.loading_time:.2f}s)")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics - with storage time"""
        return {
            "total_sessions": len(self.memory_storage),
            "total_dialogues": len(self.all_dialogues),
            "total_images": len(self.image_session_map),
            "dialogues_with_images": len(self.dialogue_image_map),
            "session_info": self.session_info,
            "storage_time": self.storage_time,
            "loading_time": self.loading_time,
            "image_scan_time": self.image_scan_time,
            "avg_time_per_dialogue": self.storage_time / len(self.all_dialogues) if self.all_dialogues else 0
        }
    
    def _scan_image_directory(self, session_id: str, session_dir: str):
        """Scan image directory"""
        image_dir = os.path.join(session_dir, "image")
        if os.path.exists(image_dir):
            self.image_paths[session_id] = {}
            for img_file in os.listdir(image_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    full_path = os.path.join(image_dir, img_file)
                    self.image_paths[session_id][img_file] = full_path
                    self.image_session_map[img_file] = session_id
    
    def _load_single_session(self, session_dir_name: str, session_dir: str) -> Optional[Dict]:
        """Load single session data"""
        conversation_file = os.path.join(session_dir, "session.json")
        if not os.path.exists(conversation_file):
            return None
        try:
            with open(conversation_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {conversation_file}: {e}")
            return None
    
    def get_image_path(self, session_id: str, image_filename: str) -> Optional[str]:
        """Get image path"""
        if session_id in self.image_paths and image_filename in self.image_paths[session_id]:
            return self.image_paths[session_id][image_filename]
        return None
    
    def get_image_session(self, image_filename: str) -> Optional[str]:
        """Get session that the image belongs to"""
        return self.image_session_map.get(image_filename)
    
    def get_image_for_api(self, image_filename: str, is_question_image: bool = False, 
                          question_id: str = None) -> Optional[Dict]:
        """Get processed image data for API"""
        session_id = question_id
        
        image_path = self.get_image_path(session_id, image_filename)
        if not image_path:
            print(image_path)
            print(session_id)
        if not image_path or not os.path.exists(image_path):
            logger.warning(f"Image file does not exist: {image_filename}")
            return None
        
        return self.image_processor.process_image(
            image_path=image_path,
            session_id=session_id if not is_question_image else None,
            filename=image_filename if not is_question_image else None,
            is_question_image=is_question_image,
            question_id=question_id
        )
    
    def format_dialogue_context(self, max_tokens: Optional[int] = None) -> Tuple[str, int, int, bool]:
        """Format dialogue context, with truncation support"""
        context_parts = []
        
        for dialogue in self.all_dialogues:
            session_id = dialogue["session_id"]
            role = dialogue["role"]
            text = dialogue["text"]
            image = dialogue["image"]
            
            if image:
                context_parts.append(f"[Session {session_id} Turn {dialogue['dialogue_index']} {role}]: [Image: {image}] {text}")
            else:
                context_parts.append(f"[Session {session_id} Turn {dialogue['dialogue_index']} {role}]: {text}")
        
        full_context = "\n".join(context_parts)
        
        token_counter = TokenCounter()
        original_tokens = token_counter.count_tokens(full_context)
        if max_tokens and original_tokens > max_tokens:
            truncated_text, _, truncated_tokens = token_counter.truncate_text(full_context, max_tokens)
            truncated_text += "\n\n[tips: the conversation memory has been truncated due to token limit]"
            return truncated_text, original_tokens, truncated_tokens, True
        
        return full_context, original_tokens, original_tokens, False
    
    def get_session_context(self, target_session_id: str) -> Dict[str, Any]:
        """Get complete context"""
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
3. Answer in English. This is a strict requirement. Do not answer in any other language. If the picture contains other languages, it still needs to be translated into English to answer.
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
    """VLM Evaluator - Multi-threaded version"""
    
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
        self.memory_type = "FullMMSystem"
        self.token_counter = TokenCounter()
        
        # Thread synchronization tools
        self.api_semaphore = Semaphore(max_api_concurrency)
        self.stats_lock = Lock()
        self.file_lock = Lock()
    
        # Statistics - with timing fields
        self.session_statistics = defaultdict(lambda: {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "processing_time": 0.0,
            "by_category": defaultdict(int),
            "by_difficulty": defaultdict(int),
            "truncated_count": 0,
            "images_limited_count": 0,
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
        """Test API connection"""
        try:
            test_url = f"{self.base_url}/models"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(test_url, headers=headers, timeout=10)
            if response.status_code == 200:
                logger.info(f"API connection test successful")
            else:
                logger.warning(f"API connection test returned status code: {response.status_code}")
        except Exception as e:
            logger.error(f"API connection test failed: {e}")
    
    def load_questions(self, conversations_dir: str) -> Dict[str, Dict]:
        """Load questions from all sessions"""
        sessions_questions = {}
        base_dir = Path(conversations_dir)
        
        if base_dir.name.startswith("dialogue"):
            dialogue_name = base_dir.name
            scenes_dir = base_dir / "scenes"
        else:
            dialogue_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("dialogue")]
            if not dialogue_dirs:
                raise ValueError(f"Dialogue directory not found: {base_dir}")
            dialogue_name = dialogue_dirs[0].name
            scenes_dir = dialogue_dirs[0] / "scenes"
        
        if not scenes_dir.exists():
            raise ValueError(f"Scenes directory not found: {scenes_dir}")
        
        logger.info(f"Loading question files from {scenes_dir}...")
        
        for session_dir in scenes_dir.iterdir():
            if not session_dir.is_dir():
                continue
            
            session_dir_name = session_dir.name
            question_file = session_dir / "questions.json"
            
            if not question_file.exists():
                continue
            
            try:
                # Get session_id
                conversation_file = session_dir / "session.json"
                session_id = session_dir_name
               
                # Load questions
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
                
                logger.info(f"Loaded {len(question_pairs)} questions from {session_id}")
                
            except Exception as e:
                logger.error(f"Failed to load question file {question_file}: {e}")
        
        logger.info(f"Loaded questions from {len(sessions_questions)} sessions total")
        return sessions_questions
    
    def _prepare_images(self, question_pair: QuestionAnswerPair) -> Tuple[List[Dict], bool, int, int]:
        """Prepare all images (question images + context images)"""
        all_images = []
        question_images_count = 0
        context_images_count = 0
        images_limited = False
        
        # 1. Prepare question images
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
                    logger.debug(f"Added question image: {img_file}")
                else:
                    logger.warning(f"Cannot find question image: {img_file}")
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
                        logger.debug(f"Added question image: {img_file}")
                    else:
                        logger.warning(f"Cannot find question image: {img_file}")
        
        # 2. Prepare context images (up to max_images - number of question images)
        remaining_slots = self.max_images - question_images_count if self.max_images else None
        if remaining_slots is None or remaining_slots > 0:
            # Collect all context images
            context_images_set = set()
            for dialogue in self.memory_system.all_dialogues:
                if dialogue["image"]:
                    context_images_set.add((dialogue["image"], dialogue["session_id"]))
            # Limit count
            context_images_list = list(context_images_set)
            if remaining_slots and len(context_images_list) > remaining_slots:
                context_images_list = context_images_list[:remaining_slots]
                images_limited = True
            
            # Process context images
            for img_file in context_images_list:
                img_info = self.memory_system.get_image_for_api(img_file[0], is_question_image=False, question_id=img_file[1])
                if img_info:
                    all_images.append(img_info)
                    context_images_count += 1
        
        original_count = question_images_count + len(context_images_set)
        limited_count = len(all_images)
        
        return all_images, images_limited, original_count, limited_count
    
    def _call_vlm_api(self, prompt: str, images: List[Dict]) -> Dict[str, Any]:
        """Call VLM API (with concurrency control)"""
        start_time = time.time()
        # Acquire semaphore
        acquired = self.api_semaphore.acquire(timeout=30)
        if not acquired:
            logger.warning("Timeout waiting for API semaphore")
            return {
                "answer": "[API call failed: Semaphore acquisition timeout]",
                "processing_time": time.time() - start_time,
                "success": False,
                "error": "Semaphore acquisition timeout"
            }
        
        try:
            # Build message
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
            
            # Retry mechanism
            for attempt in range(self.max_retries):
                try:
                    response = requests.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=self.timeout,
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
                        error_msg = f"API returned error: {response.status_code}"
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
                            "answer": f"[API call exception: {str(e)}]",
                            "processing_time": time.time() - start_time,
                            "success": False,
                            "error": str(e)
                        }
                    time.sleep(1)
            
            return {
                "answer": "[API call failed]",
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
            "Test-Time Learning": "Learn and adapt from the conversation context at test time to answer the question.",
            "Conflict Detection": "Check whether this information conflicts with the conversation.",
            "Answer Refusal": "Determine if the question can be answered based on the conversation."
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
            "Conflict Detection": "Response format: Reply strictly with either 'Yes' or 'No' only.",
            "Answer Refusal": "Response format: If the information is present in the conversation, provide answer based on that information; if not present, reply with: 'Not mentioned.'",
        }
        
        if question_type in format_requirements:
            return format_requirements[question_type]
        else:
            return "Response format: Provide clear and accurate answers based on the conversation memory."

    
    def _calculate_confidence(self, prediction: str, reference: str) -> float:
        """Calculate confidence score"""
        if not prediction or prediction.startswith("["):
            return 0.0
        
        prediction = prediction.strip()
        reference = str(reference).strip()
        
        # Exact match
        if prediction == reference:
            return 1.0
        
        # Contains relation
        if reference in prediction or prediction in reference:
            return 0.8
        
        # Keyword matching
        pred_words = set(prediction.lower().split())
        ref_words = set(reference.lower().split())
        if pred_words and ref_words:
            intersection = pred_words.intersection(ref_words)
            union = pred_words.union(ref_words)
            return len(intersection) / len(union)
        
        return 0.3
    
    def evaluate_single_question(self, question_pair: QuestionAnswerPair, 
                            question_file_path: str = None) -> EvaluationResult:
        """Evaluate a single question - with detailed timing and error recording"""
        start_time = time.time()
        
        # Timing variables
        context_prepare_time = 0.0
        image_prepare_time = 0.0
        prompt_build_time = 0.0
        api_call_time = 0.0
        
        try:
            # 1. Prepare dialogue context (record time)
            context_start = time.time()
            full_context = self.memory_system.get_session_context(question_pair.session_id)
            dialogue_context, orig_tokens, truncated_tokens, was_truncated = \
                self.memory_system.format_dialogue_context(self.max_context_tokens)
            context_prepare_time = time.time() - context_start
            
            # 2. Prepare images (record time)
            image_start = time.time()
            images, images_limited, orig_img_count, limited_img_count = self._prepare_images(question_pair)
            image_prepare_time = time.time() - image_start
            
            # 3. Build prompt (record time)
            prompt_start = time.time()
            prompt = self._build_prompt(
                question_pair,
                dialogue_context,
                len([img for img in images if img.get("is_question_image")]) > 0
            )
            prompt_build_time = time.time() - prompt_start
            
            # 4. Call API (record time)
            api_start = time.time()
            response = self._call_vlm_api(prompt, images)
            api_call_time = response.get("processing_time", time.time() - api_start)
            
            # 5. Calculate confidence
            confidence = self._calculate_confidence(
                response.get("answer", ""),
                question_pair.original_answer
            )
            
            # Total processing time
            total_processing_time = time.time() - start_time
            
            # 6. Create result
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
                memory_context_summary=f"Total dialogue turns: {len(self.memory_system.all_dialogues)}",
                success=response.get("success", False),
                error_message=response.get("error"),
                truncated=was_truncated,
                original_context_length=orig_tokens if was_truncated else None,
                truncated_context_length=truncated_tokens if was_truncated else None,
                images_limited=images_limited,
                original_image_count=orig_img_count,
                limited_image_count=limited_img_count,
                # New timing fields
                context_prepare_time=context_prepare_time,
                image_prepare_time=image_prepare_time,
                prompt_build_time=prompt_build_time,
                api_call_time=api_call_time
            )
            
            # Output detailed log
            logger.info(f"✓ {question_pair.session_id}/{question_pair.question_id} - "
                    f"Total: {total_processing_time:.2f}s, "
                    f"Context: {context_prepare_time:.3f}s, "
                    f"Image: {image_prepare_time:.3f}s, "
                    f"Prompt: {prompt_build_time:.3f}s, "
                    f"API: {api_call_time:.2f}s")
            
            return result
            
        except Exception as e:
            total_processing_time = time.time() - start_time
            error_msg = str(e)[:200]
            logger.error(f"✗ Failed to evaluate question {question_pair.question_id}: {error_msg}")
            
            return EvaluationResult(
                sample_id=f"error_{question_pair.question_id}_{int(time.time())}",
                session_id=question_pair.session_id,
                dialogue_name=question_pair.dialogue_name,
                question_id=question_pair.question_id,
                question_text=question_pair.question_text,
                question_image=question_pair.question_image,
                system_answer=f"[Processing error: {error_msg}]",
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
                success=False,
                error_message=error_msg,
                context_prepare_time=context_prepare_time,
                image_prepare_time=image_prepare_time,
                prompt_build_time=prompt_build_time,
                api_call_time=api_call_time
            )
    
    def _evaluate_question_wrapper(self, question_pair: QuestionAnswerPair, 
                               question_file_path: str = None) -> Dict:
        """Question evaluation wrapper (for thread pool) - pass file path"""
        try:
            result = self.evaluate_single_question(question_pair, question_file_path)
            result_dict = asdict(result)
            
            # Update statistics
            with self.stats_lock:
                session_id = question_pair.session_id
                if result.success:
                    self.session_statistics[session_id]["successful"] += 1
                else:
                    self.session_statistics[session_id]["failed"] += 1
                
                self.session_statistics[session_id]["processing_time"] += result.processing_time
                
                # Accumulate timing statistics
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
            logger.error(f"Wrapper execution failed: {e}")
            # Record failed file path
            return {
                "sample_id": f"error_{question_pair.question_id}",
                "session_id": question_pair.session_id,
                "question_id": question_pair.question_id,
                "success": False,
                "error_message": str(e)[:200]
            }
    
    def evaluate_session(self, session_id: str, session_data: Dict,
                    max_questions: Optional[int] = None) -> List[Dict]:
        """Parallel evaluation of all questions in a session - pass file path"""
        questions = session_data["questions"]
        session_path = Path(session_data["session_path"])
        question_file_path = session_data.get("question_file", "")
        
        if max_questions and max_questions < len(questions):
            questions = questions[:max_questions]
        
        total = len(questions)
        logger.info(f"Parallel evaluation of {total} questions in session {session_id} (API concurrency: {self.max_api_concurrency})")
        
        # Initialize statistics
        with self.stats_lock:
            self.session_statistics[session_id]["total"] = total
            for q in questions:
                self.session_statistics[session_id]["by_category"][q.category] += 1
                self.session_statistics[session_id]["by_difficulty"][q.difficulty] += 1
        
        # Use thread pool for parallel question processing
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
                        logger.info(f"[{session_id}] Progress: {completed}/{total}")
                        
                except Exception as e:
                    logger.error(f"Failed to get result: {e}")
        
        # Save results
        with self.file_lock:
            self._save_session_results(session_id, session_data["session_dir_name"], 
                                    session_path, results)
        
        # Update global statistics
        with self.stats_lock:
            self.global_statistics["total_questions"] += total
            self.global_statistics["successful_questions"] += \
                self.session_statistics[session_id]["successful"]
            self.global_statistics["failed_questions"] += \
                self.session_statistics[session_id]["failed"]
        
        # Output session timing statistics
        session_stats = self.session_statistics[session_id]
        successful = session_stats["successful"]
        if successful > 0:
            logger.info(f"Session {session_id} timing statistics - "
                    f"Avg context: {session_stats['total_context_prepare_time']/successful:.3f}s, "
                    f"Avg image: {session_stats['total_image_prepare_time']/successful:.3f}s, "
                    f"Avg prompt: {session_stats['total_prompt_build_time']/successful:.3f}s, "
                    f"Avg API: {session_stats['total_api_call_time']/successful:.2f}s")
        
        logger.info(f"Session {session_id} completed: Successful {session_stats['successful']}/{total}")
        return results

    
    def _save_session_results(self, session_id: str, session_dir_name: str,
                        session_path: Path, results: List[Dict]):
        """Save session results"""
        results_dir = session_path / "evaluation_results"
        results_dir.mkdir(exist_ok=True)
        
        output = {
            "metadata": {
                "session_id": session_id,
                "session_dir_name": session_dir_name,
                "vlm_model": self.model,
                "memory_type": self.memory_type,
                "max_context_tokens": self.max_context_tokens,
                "max_images": self.max_images,
                "timestamp": datetime.now().isoformat()
            },
            "results": results
        }
        
        filepath = results_dir / "results_FullMM.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"Results saved: {filepath}")
    
    def evaluate_all_sessions(self, sessions_questions: Dict[str, Dict]):
        self.global_statistics["start_time"] = time.time()
        self.global_statistics["total_sessions"] = len(sessions_questions)
        self.global_statistics["memory_type"] = "FullMMSystem"
        
        logger.info(f"Starting parallel evaluation of {len(sessions_questions)} sessions")
        logger.info(f"Session parallelism: {self.max_workers}, API concurrency: {self.max_api_concurrency}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_session = {}
            for session_id, session_data in sessions_questions.items():
                future = executor.submit(
                    self.evaluate_session,
                    session_id,
                    session_data
                )
                future_to_session[future] = session_id
            
            for future in concurrent.futures.as_completed(future_to_session):
                session_id = future_to_session[future]
                try:
                    results = future.result()
                    logger.info(f"Session {session_id} processing completed")
                except Exception as e:
                    logger.error(f"Session {session_id} processing failed: {e}")
        self.global_statistics["end_time"] = time.time()
        


def main():
    parser = argparse.ArgumentParser(description="Multimodal Memory Evaluation System (Multi-threaded Version)")
    parser.add_argument("--conversations_dir", type=str, required=True,
                       help="Conversation data directory")
    parser.add_argument("--api_key", type=str, required=True,
                       help="VLM API key")
    parser.add_argument("--model", type=str, required=True,
                       help="VLM model name")
    parser.add_argument("--base_url", type=str, required=True,
                       help="API base URL")
    parser.add_argument("--max_context_tokens", type=int, default=4096,
                       help="Maximum tokens for dialogue context")
    parser.add_argument("--max_images", type=int, default=None,
                       help="Maximum number of images")
    parser.add_argument("--max_workers", type=int, default=3,
                       help="Number of session-level parallel threads")
    parser.add_argument("--max_api_concurrency", type=int, default=2,
                       help="API concurrency count")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    print("=" * 70)
    print("Multimodal Memory Evaluation System (Multi-threaded Version)")
    print(f"Memory Type: FullMMSystem") 
    print(f"Model: {args.model}")
    print(f"Session Parallelism: {args.max_workers}, API Concurrency: {args.max_api_concurrency}")
    if args.max_context_tokens:
        print(f"Dialogue Truncation: {args.max_context_tokens} tokens")
    if args.max_images:
        print(f"Image Limit: {args.max_images} images")
    print("=" * 70)
    
    # 1. Initialize memory system
    print("\n[1] Initializing memory system...")
    memory_system = FullMMSystem(args.conversations_dir)
    memory_system.load_all_conversations()

    # Output storage time statistics
    print(f"\nMemory Storage Time Statistics:")
    print(f"   Total storage time: {memory_system.storage_time:.2f} seconds")
    print(f"   Data loading: {memory_system.loading_time:.2f} seconds")
    print(f"   Total dialogue turns: {len(memory_system.all_dialogues)}")
    if len(memory_system.all_dialogues) > 0:
        print(f"   Average per dialogue: {memory_system.storage_time / len(memory_system.all_dialogues):.3f} seconds")
    print(f"   Total images: {len(memory_system.image_session_map)}")
    
    # 2. Initialize evaluator
    print("\n[2] Initializing evaluator...")
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
    
    # 3. Load questions
    print("\n[3] Loading questions...")
    try:
        sessions_questions = evaluator.load_questions(args.conversations_dir)
    except Exception as e:
        print(f"Failed to load questions: {e}")
        return
    
    if not sessions_questions:
        print("No question files found")
        return
    
    total = sum(len(d["questions"]) for d in sessions_questions.values())
    print(f"Loaded {total} questions from {len(sessions_questions)} sessions")
    sessions_to_process = sessions_questions
    
    # 4. Execute evaluation
    print("\n[4] Starting parallel evaluation...")
    evaluator.evaluate_all_sessions(
        sessions_questions=sessions_to_process
    )
    
    # 5. Output results
    print("\n[5] Evaluation complete!")


if __name__ == "__main__":
    main()