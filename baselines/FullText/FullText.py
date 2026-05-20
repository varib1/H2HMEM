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

# Add imports at the beginning of the file
import concurrent.futures
import threading
from threading import Lock, Semaphore
import queue
from tqdm import tqdm  # Optional, requires: pip install tqdm

# Import API related libraries
import requests
from PIL import Image
import base64
from io import BytesIO

# Try to import tiktoken for token counting, use simple estimation if not available
try:
    import tiktoken
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False
    logging.warning("tiktoken not installed, using simple character-based token estimation")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class QuestionAnswerPair:
    """Question-Answer Pair - adapted to questions.json format"""
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
    category: str = field(init=False)  # Derived from question_type
    
    def __post_init__(self):
        # Extract category from question_type
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
    
    # Detailed timing metrics
    memory_load_time: float = 0.0  # Memory loading time
    memory_recall_time: float = 0.0  # Memory recall/context building time
    llm_inference_time: float = 0.0  # LLM response time


class TokenCounter:
    """Token counter"""
    
    def __init__(self, model_name: str = "cl100k_base"):
        """
        Initialize token counter
        
        Args:
            model_name: Tokenizer model name (default uses cl100k_base, suitable for gpt-4, gpt-3.5-turbo)
        """
        self.model_name = model_name
        self.encoding = None
        
        if TOKENIZER_AVAILABLE:
            try:
                self.encoding = tiktoken.get_encoding(model_name)
                logger.info(f"Successfully loaded tokenizer: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}, using estimation method")
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in text
        
        Args:
            text: Input text
        
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        
        if self.encoding:
            # Use tiktoken for precise counting
            return len(self.encoding.encode(text))
        else:
            # Simple estimation: Chinese characters count as 2 tokens, English words as 1.3 tokens
            chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
            other_chars = len(text) - chinese_chars
            
            # Estimation: each Chinese character ~2 tokens, every 4 other chars ~1 token
            estimated_tokens = chinese_chars * 2 + other_chars * 0.25
            return int(estimated_tokens) + 1
    
    def truncate_text(self, text: str, max_tokens: int, preserve_ratio: float = 0.8) -> tuple:
        """
        Truncate text to specified token count
        
        Args:
            text: Input text
            max_tokens: Maximum number of tokens
            preserve_ratio: Ratio to preserve from the beginning (0.8 means keep first 80% of tokens, the remaining 20% truncated from the end)
        
        Returns:
            (Truncated text, original token count, truncated token count)
        """
        if not text:
            return text, 0, 0
        
        original_tokens = self.count_tokens(text)
        
        if original_tokens <= max_tokens:
            return text, original_tokens, original_tokens
        
        if self.encoding:
            # Use tiktoken for precise truncation
            tokens = self.encoding.encode(text)
            
            # Decide which parts to keep
            keep_tokens = int(max_tokens * preserve_ratio)
            
            if keep_tokens >= max_tokens:
                keep_tokens = max_tokens
            
            # Keep the beginning part
            truncated_tokens = tokens[:keep_tokens]
            
            # Simple handling: keep only the beginning
            # More complex strategies can be implemented as needed
            
            truncated_text = self.encoding.decode(truncated_tokens)
            truncated_tokens_count = len(truncated_tokens)
            
            return truncated_text, original_tokens, truncated_tokens_count
        else:
            # Estimate truncation using character count
            # Estimate characters per token
            chars_per_token = len(text) / original_tokens
            
            # Number of characters to keep
            keep_chars = int(max_tokens * chars_per_token * preserve_ratio)
            
            # Simple truncation
            truncated_text = text[:keep_chars] + "... [content truncated]"
            
            # Recalculate truncated token count (estimate)
            truncated_tokens = self.count_tokens(truncated_text)
            
            return truncated_text, original_tokens, truncated_tokens


class FullTextMemorySystem:
    """Full-text memory system - stores entire dialogue content from all sessions"""
    def __init__(self, conversations_dir: str):
        self.conversations_dir = conversations_dir
        self.memory_storage = {}  # Stores content for all sessions, keyed by session_id
        self.all_dialogues = []   # Merged dialogues from all sessions
        self.session_info = {}    # Session additional information
        
        # Add storage time statistics
        self.store_times = []  # Record time for each storage operation
        self.total_store_time = 0.0
        self.num_stores = 0
        self.session_load_times = {}  # Load time for each session
        self.total_memory_load_time = 0.0  # Total memory load time
        self.load_start_time = None  # Load start time
        self.load_end_time = None    # Load end time
    
    def load_all_conversations(self):
        """Load all session data from the entire conversation (with timing statistics)"""
        self.load_start_time = time.time()
        
        scenes_dir = os.path.join(self.conversations_dir, "scenes")
        if not os.path.exists(scenes_dir):
            raise ValueError(f"Scenes directory not found: {scenes_dir}")
        
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
                
                # Record session load time
                session_load_time = time.time() - session_start
                self.session_load_times[session_id] = session_load_time
                self.total_memory_load_time += session_load_time
                self.num_stores += 1
                
                # Get caption directory path
                caption_dir = os.path.join(session_dir, "caption")
                caption_files_exist = os.path.exists(caption_dir)
                
                if not caption_files_exist:
                    logger.debug(f"Session {session_id} has no caption directory")
                
                # Extract dialogue content and add session information
                dialogues = session_data.get("dialogue", [])
                processed_dialogues = []
                timeline_date = session_data.get("timeline_date", "")
                
                for i, dialogue in enumerate(dialogues, 1):
                    role = dialogue.get("role", "")
                    content = dialogue.get("content", {})
                    text = content.get("text", "")
                    image_filename = content.get("image", "")
                    
                    # Process image description information
                    image_description = ""
                    if image_filename and caption_files_exist:
                        # Extract numeric part from filename
                        caption_json = Path(image_filename).stem + ".json"  
                        if caption_json:
                            caption_file_path = os.path.join(caption_dir, caption_json)
                            
                            if os.path.exists(caption_file_path):
                                try:
                                    with open(caption_file_path, 'r', encoding='utf-8') as f:
                                        caption_data = json.load(f)
                                    
                                    # Extract complete text information from description
                                    description = caption_data.get("description", {})
                                    
                                    # Extract all text information
                                    description_texts = []
                                    
                                    # 1. Extract final_text
                                    final_text = description.get("final_text", "")
                                    if final_text:
                                        description_texts.append(final_text)
                                    
                                    # Combine all description text
                                    if description_texts:
                                        image_description = "\n".join(description_texts)
                                        logger.debug(f"Loaded description for image {image_filename}, character count: {len(image_description)}")
                                    
                                except Exception as e:
                                    logger.error(f"Failed to load image description file {caption_file_path}: {e}")
                            else:
                                logger.debug(f"Image description file does not exist: {caption_file_path}")
                        else:
                            logger.debug(f"Cannot extract number from filename {image_filename}")
                    
                    # Create dialogue content with image description
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
                
                # Store session information
                self.session_info[session_id] = {
                    "session_dir_name": session_dir_name,
                    "session_title": session_data.get("session_title", ""),
                    "timeline_date": session_data.get("timeline_date", ""),
                    "generated_at": session_data.get("generated_at", ""),
                    "dialogue_count": len(dialogues),
                    "has_caption_dir": caption_files_exist,
                    "session_path": session_dir,
                    "load_time": session_load_time  # Add load time
                }
                
                # Record storage time
                self.store_times.append(session_load_time)
                self.total_store_time += session_load_time
                
                logger.info(f"   Loaded {session_id}: {session_load_time:.3f}s, {len(dialogues)} dialogue turns")
        
        # Store all dialogues
        self.all_dialogues = all_dialogues
        
        self.load_end_time = time.time()
        total_load_time = self.load_end_time - self.load_start_time
        
        logger.info(f"Loaded {len(self.memory_storage)} sessions, total {len(all_dialogues)} dialogue turns")
        logger.info(f"Total load time: {total_load_time:.2f}s")
        
        # Output storage time statistics
        if self.num_stores > 0:
            logger.info(f"Storage time statistics:")
            logger.info(f"   Average load per session: {self.total_store_time/self.num_stores:.3f}s")
            logger.info(f"   Fastest load: {min(self.store_times):.3f}s")
            logger.info(f"   Slowest load: {max(self.store_times):.3f}s")
        
        # Count image description information
        dialogues_with_images = [d for d in all_dialogues if d["content"].get("image")]
        dialogues_with_description = [d for d in all_dialogues if d["content"].get("image_description")]
        
        logger.info(f"Dialogues containing images: {len(dialogues_with_images)} turns")
        logger.info(f"Dialogues with loaded image descriptions: {len(dialogues_with_description)} turns")
        
        return total_load_time
    
    def _load_single_session(self, session_dir_name: str, session_dir: str) -> Optional[Dict]:
        """Load data from a single session"""
        conversation_file = os.path.join(session_dir, "session.json")
        
        if not os.path.exists(conversation_file):
            logger.warning(f"Conversation.json file not found: {conversation_file}")
            return None
        
        try:
            with open(conversation_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            logger.debug(f"Successfully loaded dialogue data from {session_dir_name}")
            return session_data
            
        except Exception as e:
            logger.error(f"Failed to load {conversation_file}: {e}")
            return None
    
    
    
    def get_full_memory_context(self) -> Dict[str, Any]:
        """Get complete memory context of the entire conversation"""
        return {
            "all_sessions": list(self.memory_storage.keys()),
            "session_info": self.session_info,
            "total_dialogues": len(self.all_dialogues),
            "memory_storage": self.memory_storage,
            "all_dialogues": self.all_dialogues
        }
    
    def get_session_context(self, target_session_id: str) -> Dict[str, Any]:
        """Get complete context containing all sessions, but mark the target session"""
        all_dialogues_with_context = []
        
        # Organize all dialogues by session
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
    """Create memory system"""
    if memory_type == "full_text":
        return FullTextMemorySystem(conversations_dir)
    else:
        raise ValueError(f"Unsupported memory type: {memory_type}")


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
        "Test-Time Learning": "Learn and adapt from the conversation context at test time to answer the question.",
        "Conflict Detection": "Check whether this information conflicts with the conversation.",
        "Answer Refusal": "Determine if the question can be answered based on the conversation."
    }
    
    # Response format requirements
    FORMAT_REQUIREMENTS = {
        "Conflict Detection": "Response format: Reply strictly with either 'Yes' or 'No' only.",
        "Answer Refusal": "Response format: If the information is present in the conversation, provide answer based on that information; if not present, reply with: 'Not mentioned.'",
        "default": "Response format: Provide clear and accurate answers based on the conversation memory."
    }
    
    # Base template
    TEMPLATE = """You are a memory testing system. {instruction}

IMPORTANT: 
1. Provide only the answer without any reasoning process. Give the answer directly in English.
2. Keep your answer within 100 words. Short and concise answers are acceptable.
3. Answer in English. This is a strict requirement. Do not answer in any other language.
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
    """VLM Evaluator - uses complete dialogue context"""
    
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
        Initialize VLM evaluator (multi-threaded version)
        
        Args:
            memory_system: Memory system instance
            api_key: VLM API key
            model: VLM model name
            base_url: API base URL
            verbose: Verbose logging output
            max_retries: Maximum number of retries
            timeout: Request timeout (seconds)
            max_context_tokens: Maximum context tokens
            truncation_strategy: Truncation strategy
            max_workers: Maximum number of threads (for parallel session processing)
            max_api_concurrency: Maximum API concurrency (for parallel question processing)
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
        
        # Thread control attributes
        self.max_workers = max_workers
        self.max_api_concurrency = max_api_concurrency
        self.api_semaphore = Semaphore(max_api_concurrency)  # Control API concurrency
        self.file_lock = Lock()  # File write lock
        self.stats_lock = Lock()  # Statistics update lock
        self.results_queue = queue.Queue()  # Results queue (optional)
        
        # Initialize token counter
        self.token_counter = TokenCounter()
        
        # Store statistics for each session (preserve timing calculations)
        self.session_statistics = defaultdict(lambda: {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "processing_time": 0.0,
            "by_category": defaultdict(int),
            "by_difficulty": defaultdict(int),
            "truncated_count": 0,
            # Timing metrics
            "total_memory_load_time": 0.0,
            "total_memory_recall_time": 0.0,
            "total_llm_time": 0.0
        })
        
        # Record overall start and end times
        self.start_time = None
        self.end_time = None
        
        # Test API connection
        self._test_api_connection()
    
    def _test_api_connection(self):
        """Test API connection"""
        try:
            test_url = f"{self.base_url}/models"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(test_url, headers=headers, timeout=10)
            if response.status_code == 200:
                logger.info(f"API connection test successful, model: {self.model}")
                logger.info(f"API endpoint: {self.base_url}")
                if self.max_context_tokens:
                    logger.info(f"Context truncation: max {self.max_context_tokens} tokens, strategy: {self.truncation_strategy}")
            else:
                logger.warning(f"API connection test returned non-200 status code: {response.status_code}")
        except Exception as e:
            logger.error(f"API connection test failed: {e}")
            logger.warning("Please check if the API service is running and the API key is correct")
    
    def load_questions(self, conversations_dir: str) -> Dict[str, Dict]:
        """
        Load questions from all sessions, grouped by session
        
        Args:
            conversations_dir: Conversation data directory
        
        Returns:
            Dictionary grouped by session_id: {session_id: {"questions": [], "session_path": str}}
        """
        sessions_questions = {}
        
        # Parse directory structure
        base_dir = Path(conversations_dir)
        
        # Check if it's a top-level directory like "dialogueX"
        if base_dir.name.startswith("dialogue"):
            dialogue_name = base_dir.name
            scenes_dir = base_dir / "scenes"
        else:
            # Try to find subdirectories containing "dialogue" in the directory
            dialogue_dirs = [d for d in base_dir.iterdir() if d.is_dir() and (d.name.startswith("dialogue"))]
            if not dialogue_dirs:
                raise ValueError(f"Cannot find dialogue directory: {base_dir}")
            
            dialogue_name = dialogue_dirs[0].name
            scenes_dir = dialogue_dirs[0] / "scenes"
        
        if not scenes_dir.exists():
            raise ValueError(f"Cannot find scenes directory: {scenes_dir}")
        
        logger.info(f"Loading question files from {scenes_dir}...")
        
        # Iterate through all session directories
        session_dirs = [d for d in scenes_dir.iterdir() if d.is_dir()]
        
        for session_dir in session_dirs:
            session_dir_name = session_dir.name
            question_file = session_dir / "questions.json"
            
            if question_file.exists():
                try:
                    # First read session.json from the session to get session_id
                    conversation_file = session_dir / "session.json"
                    session_id = session_dir_name  # Default to directory name
                    
                    # Load question file
                    with open(question_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    questions = data.get("questions", [])
                    
                    # Convert format to list of QuestionAnswerPair
                    question_pairs = []
                    for q in questions:
                        qa_pair = QuestionAnswerPair(
                            question_id=q.get("question_id", f"FullText_{len(question_pairs)}"),
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
                            }
                        )
                        
                        # Add image context
                        if qa_pair.question_image:
                            if str(session_id) == "session0":
                                folder, filename = qa_pair.question_image.split("/", 1)
                                possible_paths = [
                                    scenes_dir / folder / "image" / filename
                                ]
                            else:
                                img_filename = qa_pair.question_image
                                # Try multiple possible image paths
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
                    
                    logger.info(f"Loaded {len(question_pairs)} questions from {session_id} ({session_dir_name})")
                    
                except Exception as e:
                    logger.error(f"Failed to load question file {question_file}: {e}")
            else:
                logger.warning(f"Skipping {session_dir_name}, no question file found")
        
        logger.info(f"Loaded questions from {len(sessions_questions)} sessions total")
        return sessions_questions
    
    def _format_memory_context_only(self, memory_context: Dict[str, Any]) -> str:
        """
        Format only the memory context part, without any prompts or questions
        
        Args:
            memory_context: Complete context containing all sessions
        
        Returns:
            Pure memory context string
        """
        if not memory_context:
            return "No available memory"
        
        context_parts = []
        
        # Add overall information
        all_sessions = memory_context.get("all_sessions", [])
        total_dialogues = memory_context.get("total_dialogues", 0)
        
        context_parts.append(f"Total sessions: {len(all_sessions)}")
        context_parts.append(f"Total dialogue turns: {total_dialogues}")
        
        # Display information for each session
        context_parts.append("\n[Session Information]")
        session_info = memory_context.get("session_info", {})
        
        for session_id in all_sessions:
            info = session_info.get(session_id, {})
            session_title = info.get("session_title", "")
            timeline_date = info.get("timeline_date", "")
            dialogue_count = info.get("dialogue_count", 0)
            
            if session_title:
                context_parts.append(f"Session {session_id}: ({timeline_date}) - {dialogue_count} dialogue turns")
            else:
                context_parts.append(f"Session {session_id}: {dialogue_count} dialogue turns")
        
        # Add dialogue content from all sessions
        dialogues_with_context = memory_context.get("dialogues_with_context", [])
        if dialogues_with_context:
            context_parts.append("\n[Complete Dialogue Content]")
            
            current_session = None
            for dialogue in dialogues_with_context:
                session_id = dialogue.get("session_id", "unknown session")
                session_title = dialogue.get("session_title", "")
                dialogue_index = dialogue.get("dialogue_index", 0)
                session_date = dialogue.get("timeline_date", "")
                
                # Show session separator
                if session_id != current_session:
                    current_session = session_id
                    context_parts.append(f"\nSession {session_id}: {session_date}")
                
                # Show dialogue content
                role = dialogue.get("role", "")
                content = dialogue.get("content", {})
                text = content.get("text", "")
                image = content.get("image", "")
                image_description = content.get("image_description", "")
            
                if image:
                    context_parts.append(f"  Turn {dialogue_index} - {role}: [Image: {image_description}] {text}")
                else:
                    context_parts.append(f"  Turn {dialogue_index} - {role}: {text}")
        
        return "\n".join(context_parts)
        
    def _truncate_context(self, context_text: str) -> tuple:
        """
        Truncate context according to settings
        
        Args:
            context_text: Original context text
        
        Returns:
            (Truncated text, original token count, truncated token count, whether truncated)
        """
        if not self.max_context_tokens:
            # No truncation
            token_count = self.token_counter.count_tokens(context_text)
            return context_text, token_count, token_count, False
        
        original_tokens = self.token_counter.count_tokens(context_text)
        
        if original_tokens <= self.max_context_tokens:
            # No truncation needed
            return context_text, original_tokens, original_tokens, False
        
        # Need truncation
        if self.truncation_strategy == "head_only":
            # Keep only the beginning
            truncated_text, original, truncated = self.token_counter.truncate_text(
                context_text, self.max_context_tokens, preserve_ratio=1.0
            )
        elif self.truncation_strategy == "head_tail":
            # Keep beginning and end (simple implementation, more complex implementation would require segmented processing)
            # Temporarily use same head_only strategy, actual implementation should use more complex truncation
            truncated_text, original, truncated = self.token_counter.truncate_text(
                context_text, self.max_context_tokens, preserve_ratio=0.7
            )
            # Add note indicating truncation
            if truncated < original:
                truncated_text += "\n\n[Middle part of content truncated to save tokens]"
        else:
            # Default strategy
            truncated_text, original, truncated = self.token_counter.truncate_text(
                context_text, self.max_context_tokens, preserve_ratio=0.8
            )
        
        return truncated_text, original_tokens, truncated, True
        
    def _prepare_image_for_api(self, image_path: str) -> str:
        """
        Prepare image in API-acceptable format (base64 encoded)
        
        Args:
            image_path: Path to image
        
        Returns:
            Base64 encoded image string
        """
        try:
            with Image.open(image_path) as img:
                # Resize image to control API payload (optional)
                max_size = (1024, 1024)
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # Convert to RGB mode (if image has alpha channel)
                if img.mode in ('RGBA', 'LA', 'P'):
                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    rgb_img.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = rgb_img
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save to memory buffer and encode as base64
                buffer = BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                return img_base64
                
        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            raise
    
    def _call_vlm_api(self, prompt: str, images: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Call real VLM API (with concurrency control)
        """
        start_time = time.time()
        
        # Semaphore control
        acquired = False
        if self.api_semaphore:
            self.api_semaphore.acquire()
            acquired = True
            if self.verbose:
                logger.debug(f"API semaphore acquired, available: {self.api_semaphore._value}")
        
        try:
            # Prepare message
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
                    logger.error(f"Failed to process image, will use text only: {e}")
                    messages.append({
                        "role": "user",
                        "content": prompt
                    })
            else:
                messages.append({
                    "role": "user",
                    "content": prompt
                })
            
            # Build request payload
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 1024,
                "temperature": 0.1,
                "top_p": 0.9,
                "stream": False
            }
            
            # Retry mechanism
            for attempt in range(self.max_retries):
                try:
                    api_url = f"{self.base_url}/chat/completions"
                    
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                    
                    logger.debug(f"Calling API (attempt {attempt + 1}/{self.max_retries})")
                    
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
                            error_msg = "API response missing 'choices' field"
                            logger.error(f"{error_msg}: {response_data}")
                            
                    else:
                        error_msg = f"API returned error status code: {response.status_code}"
                        logger.error(f"{error_msg}: {response.text}")
                        
                        if response.status_code == 429 and attempt < self.max_retries - 1:
                            wait_time = 2 ** (attempt + 1)
                            logger.warning(f"Rate limited, waiting {wait_time} seconds before retry...")
                            time.sleep(wait_time)
                            continue
                    
                    if attempt == self.max_retries - 1:
                        raise Exception(f"{error_msg}")
                    else:
                        time.sleep(1)
                        
                except requests.exceptions.Timeout:
                    logger.error(f"API request timeout (attempt {attempt + 1}/{self.max_retries})")
                    if attempt == self.max_retries - 1:
                        raise Exception("API request timeout")
                    time.sleep(2)
                    
                except requests.exceptions.ConnectionError:
                    logger.error(f"API connection error (attempt {attempt + 1}/{self.max_retries})")
                    if attempt == self.max_retries - 1:
                        raise Exception("API connection error")
                    time.sleep(3)
                    
                except Exception as e:
                    logger.error(f"API call exception (attempt {attempt + 1}/{self.max_retries}): {e}")
                    if attempt == self.max_retries - 1:
                        raise
                    time.sleep(1)
            
            # All retries failed
            processing_time = time.time() - start_time
            return {
                "answer": f"[API call failed: All {self.max_retries} retries failed]",
                "processing_time": processing_time,
                "model": self.model,
                "success": False,
                "error": "All retries failed"
            }
        finally:
            if acquired:
                self.api_semaphore.release()
                if self.verbose:
                    logger.debug(f"API semaphore released, available: {self.api_semaphore._value}")
        
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
        question_type = question_pair.question_type.get("sub_type", "")
        
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
        """Calculate answer confidence score"""
        if not prediction or prediction.startswith("[") or "error" in prediction.lower() or "exception" in prediction.lower():
            return 0.0
        
        prediction = prediction.strip()
        reference = str(reference).strip()
        
        # For AR questions, check if format requirements are met
        if "Not mentioned" in prediction or "information not mentioned" in prediction.lower():
            # AR question answer should be "Not mentioned" or specific information
            if "Not mentioned" in reference:
                # If reference answer is also "Not mentioned", exact match
                if "Not mentioned" in prediction:
                    return 1.0
                else:
                    return 0.3
            else:
                # Reference answer is not "Not mentioned", prediction is "Not mentioned"
                return 0.0
        
        # For CD questions, check if strictly following "Yes." or "No." format
        if prediction in ["Yes.", "No.", "Yes", "No"]:
            # Normalize reference answer
            ref_normalized = reference.lower().replace(".", "").strip()
            pred_normalized = prediction.lower().replace(".", "").strip()
            
            # Map possible answers
            yes_aliases = ["yes", "是"]
            no_aliases = ["no", "否"]
            
            if (pred_normalized in yes_aliases and ref_normalized in yes_aliases) or \
               (pred_normalized in no_aliases and ref_normalized in no_aliases):
                return 1.0
            else:
                return 0.0
        
        # Check for uncertain expressions
        uncertain_phrases = [
            "I don't know", "not sure", "uncertain", "don't remember", "not mentioned",
            "cannot answer", "no relevant information", "not mentioned", "maybe", "perhaps",
            "probably", "seems", "appears", "not sure if", "not clear"
        ]
        
        pred_lower = prediction.lower()
        for phrase in uncertain_phrases:
            if phrase in pred_lower:
                return 0.3
        
        # For other question types, use original confidence calculation method
        return 0.7
    
    def evaluate_single_question(self, 
                           question_pair: QuestionAnswerPair,
                           session_id: str) -> EvaluationResult:
        """Evaluate a single question - record detailed timing metrics"""
        start_time = time.time()
        memory_load_start = 0
        memory_recall_start = 0
        llm_start = 0
        
        memory_load_time = 0
        memory_recall_time = 0
        llm_inference_time = 0
        
        try:
            logger.debug(f"Processing question: {session_id} - {question_pair.question_id} ({question_pair.category})")
            
            # 1. Get memory context (record loading time)
            memory_load_start = time.time()
            full_context = self.memory_system.get_session_context(session_id)
            memory_load_time = time.time() - memory_load_start
            
            # Create memory context summary
            total_sessions = len(full_context.get("all_sessions", []))
            total_dialogues = full_context.get("total_dialogues", 0)
            target_info = full_context.get("target_session_info", {})
            session_title = target_info.get("session_title", "")
            
            memory_context_summary = f"Total sessions: {total_sessions}, Total dialogue turns: {total_dialogues}, Target session: {session_id}"
            if session_title:
                memory_context_summary += f" ({session_title})"
            
            # 2. Format memory context (record recall time)
            memory_recall_start = time.time()
            raw_context = self._format_memory_context_only(full_context)
            
            # 3. Truncate memory context if needed
            truncated_context, original_tokens, truncated_tokens, was_truncated = self._truncate_context(raw_context)
            
            # 4. Build complete prompt
            prompt = self._construct_prompt_with_truncated_context(
                question_pair=question_pair,
                truncated_context=truncated_context,
                original_tokens=original_tokens,
                truncated_tokens=truncated_tokens,
                was_truncated=was_truncated
            )
            memory_recall_time = time.time() - memory_recall_start
            
            # 5. Prepare images
            images = []
            if question_pair.question_image and question_pair.image_context:
                for img_path in question_pair.image_context:
                    if os.path.exists(img_path):
                        images.append(img_path)
            
            # 6. Call VLM API (record LLM response time)
            llm_start = time.time()
            vlm_response = self._call_vlm_api(prompt=prompt, images=images)
            llm_inference_time = vlm_response.get("processing_time", 0)
            
            system_answer = vlm_response.get("answer", "").strip()
            success = vlm_response.get("success", False)
            
            # 7. Calculate confidence
            confidence = self._calculate_confidence(
                system_answer, 
                question_pair.original_answer,
                question_pair.answer_source
            )
            
            # Total processing time
            total_processing_time = time.time() - start_time
            
            # 8. Create evaluation result
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
                memory_context_summary=memory_context_summary,
                success=success,
                error_message=None if success else vlm_response.get("error", ""),
                truncated=was_truncated,
                original_context_length=original_tokens,
                truncated_context_length=truncated_tokens,
                # Timing metrics
                memory_load_time=memory_load_time,
                memory_recall_time=memory_recall_time,
                llm_inference_time=llm_inference_time
            )
            
            # Update session statistics
            with self.stats_lock:
                self.session_statistics[session_id]["successful"] += 1
                self.session_statistics[session_id]["processing_time"] += total_processing_time
                if result.truncated:
                    self.session_statistics[session_id]["truncated_count"] += 1
                
                # Accumulate timing statistics
                self.session_statistics[session_id]["total_memory_load_time"] += memory_load_time
                self.session_statistics[session_id]["total_memory_recall_time"] += memory_recall_time
                self.session_statistics[session_id]["total_llm_time"] += llm_inference_time
            
            logger.info(f"✓ Successfully processed: {session_id} - {question_pair.question_id} (Total time: {total_processing_time:.2f}s, LLM: {llm_inference_time:.2f}s)")
            if result.truncated:
                logger.info(f"  Context truncated: {result.original_context_length} -> {result.truncated_context_length} tokens")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)[:200]
            logger.error(f"✗ Error processing question {session_id} - {question_pair.question_id}: {error_msg}")
            
            # Create error result
            result = EvaluationResult(
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
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                memory_type=type(self.memory_system).__name__,
                vlm_model=self.model,
                processing_time=processing_time,
                confidence=0.0,
                memory_context_summary="Error: Unable to retrieve memory context",
                success=False,
                error_message=error_msg,
                truncated=False,
                memory_load_time=memory_load_time,
                memory_recall_time=memory_recall_time,
                llm_inference_time=llm_inference_time
            )
            
            # Update session statistics
            self.session_statistics[session_id]["failed"] += 1
            
            return result
    
    def evaluate_session_questions(self,
                                 session_id: str,
                                 session_data: Dict,
                                 max_questions: Optional[int] = None) -> List[Dict[str, Any]]:
        """Evaluate all questions in a single session"""
        questions = session_data["questions"]
        session_path = Path(session_data["session_path"])
        session_dir_name = session_data.get("session_dir_name", session_id)
        
        if max_questions and max_questions < len(questions):
            questions = questions[:max_questions]
        
        total_questions = len(questions)
        logger.info(f"Starting evaluation of {total_questions} questions for {session_id}")
        logger.info(f"Using complete dialogue context: contains content from {len(self.memory_system.memory_storage)} sessions")
        if self.max_context_tokens:
            logger.info(f"Context truncation limit: {self.max_context_tokens} tokens, strategy: {self.truncation_strategy}")
        
        # Initialize session statistics
        self.session_statistics[session_id]["total"] = total_questions
        for qa in questions:
            self.session_statistics[session_id]["by_category"][qa.category] += 1
            self.session_statistics[session_id]["by_difficulty"][qa.difficulty] += 1
        
        results = []
        
        for i, question_pair in enumerate(questions, 1):
            progress = f"[{i}/{total_questions}]"
            logger.info(f"{progress} Processing {session_id} - {question_pair.question_id} ({question_pair.category})")
            
            # Evaluate single question
            result = self.evaluate_single_question(question_pair, session_id)
            result_dict = asdict(result)
            results.append(result_dict)
        
        # Save final results
        self._save_session_results(session_id, session_dir_name, session_path, results, final=True)
        
        logger.info(f"Completed evaluation for {session_id}: Successful {self.session_statistics[session_id]['successful']}, Failed {self.session_statistics[session_id]['failed']}, Truncated {self.session_statistics[session_id]['truncated_count']}")
        
        return results
    
    def _save_session_results(self, 
                            session_id: str,
                            session_dir_name: str,
                            session_path: Path, 
                            results: List[Dict[str, Any]], 
                            final: bool = False):
        """Save results for a single session to the corresponding session directory"""
        # Create results directory within session directory
        session_results_dir = session_path / "evaluation_results"
        session_results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_filename = f"results_Fulltext.json"
        json_file = session_results_dir / json_filename
        
        # Build complete result data structure
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
        
        logger.debug(f"Saved results for {session_id} to: {json_file}")
    
    def evaluate_all_sessions(self,  
                        sessions_questions: Dict[str, Dict],
                        max_questions_per_session: Optional[int] = None):
        """
        Evaluate all sessions in parallel (multi-threaded version)
        """
        self.start_time = time.time()
        total_sessions = len(sessions_questions)
        
        logger.info(f"Starting parallel evaluation of {total_sessions} sessions")
        logger.info(f"Using complete dialogue context: contains content from {len(self.memory_system.memory_storage)} sessions")
        logger.info(f"Thread configuration: max_workers={self.max_workers}, max_api_concurrency={self.max_api_concurrency}")
        if self.max_context_tokens:
            logger.info(f"Context truncation limit: {self.max_context_tokens} tokens, strategy: {self.truncation_strategy}")
        
        # Use thread pool to process sessions in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all session tasks
            future_to_session = {}
            for session_id, session_data in sessions_questions.items():
                future = executor.submit(
                    self._evaluate_session_parallel,
                    session_id,
                    session_data,
                    max_questions_per_session
                )
                future_to_session[future] = session_id
            
            # Collect results (with progress display)
            completed = 0
            for future in concurrent.futures.as_completed(future_to_session):
                session_id = future_to_session[future]
                completed += 1
                try:
                    results = future.result()
                    logger.info(f"[{completed}/{total_sessions}] Session {session_id} processing completed, successfully processed {len(results)} questions")
                except Exception as e:
                    logger.error(f"Session {session_id} processing failed: {e}")
        
        self.end_time = time.time()
        self._save_session_statistics()  # Save session statistics to file
    
    def _evaluate_session_parallel(self, session_id: str, session_data: Dict, 
                             max_questions_per_session: Optional[int]) -> List[Dict]:
        """
        Wrapper method for session evaluation (for thread pool calls)
        """
        thread_name = threading.current_thread().name
        logger.info(f"Thread [{thread_name}] starting to process session: {session_id}")
        
        try:
            # Call method to process questions within session in parallel
            results = self._evaluate_session_questions_parallel(
                session_id, session_data, max_questions_per_session
            )
            return results
        except Exception as e:
            logger.error(f"Thread [{thread_name}] error processing session {session_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
        
    def _evaluate_session_questions_parallel(self, session_id: str, session_data: Dict,
                                       max_questions: Optional[int] = None) -> List[Dict]:
        """
        Process all questions within a session in parallel
        """
        questions = session_data["questions"]
        session_path = Path(session_data["session_path"])
        session_dir_name = session_data.get("session_dir_name", session_id)
        
        if max_questions and max_questions < len(questions):
            questions = questions[:max_questions]
        
        total_questions = len(questions)
        logger.info(f"Session {session_id} starting parallel processing of {total_questions} questions (API concurrency: {self.max_api_concurrency})")
        
        # Thread-safe initialization of session statistics
        with self.stats_lock:
            self.session_statistics[session_id]["total"] = total_questions
            for qa in questions:
                self.session_statistics[session_id]["by_category"][qa.category] += 1
                self.session_statistics[session_id]["by_difficulty"][qa.difficulty] += 1
        
        # Thread-safe list for storing results
        results = []
        results_lock = Lock()  # Local lock
        
        # Use thread pool to process questions in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_api_concurrency) as executor:
            # Submit all question tasks
            future_to_question = {}
            for question_pair in questions:
                future = executor.submit(
                    self._evaluate_question_with_stats,
                    question_pair,
                    session_id
                )
                future_to_question[future] = question_pair
            
            # Use tqdm to show progress (optional)
            from tqdm import tqdm
            with tqdm(total=total_questions, desc=f"Session {session_id}", unit="q") as pbar:
                for future in concurrent.futures.as_completed(future_to_question):
                    question_pair = future_to_question[future]
                    try:
                        result_dict = future.result()
                        
                        # Thread-safe addition of results
                        with results_lock:
                            results.append(result_dict)
                        
                        # Update progress bar
                        pbar.update(1)
                        if result_dict.get("success", False):
                            pbar.set_postfix({"Success": "✓", "ID": question_pair.question_id})
                        else:
                            pbar.set_postfix({"Success": "✗", "ID": question_pair.question_id})
                        
                    except Exception as e:
                        logger.error(f"Question {question_pair.question_id} processing failed: {e}")
                        
                        with results_lock:
                            results.append({
                                "sample_id": f"error_{question_pair.question_id}",
                                "session_id": session_id,
                                "question_id": question_pair.question_id,
                                "success": False,
                                "error_message": str(e)[:200]
                            })
                        pbar.update(1)
        
        # Final save of results (needs thread safety)
        with self.file_lock:
            self._save_session_results(
                session_id,
                session_dir_name,
                session_path,
                results,
                final=True
            )
        
        logger.info(f"Session {session_id} parallel processing completed: Successfully processed {len([r for r in results if r.get('success', False)])}/{total_questions}")
        
        return results
    
    def _evaluate_question_with_stats(self, question_pair: QuestionAnswerPair, 
                                    session_id: str) -> Dict[str, Any]:
        """
        Evaluate a single question and update statistics (thread-safe)
        
        Args:
            question_pair: Question-answer pair
            session_id: Session ID
        """
        try:
            # Call the original evaluation method
            result = self.evaluate_single_question(question_pair, session_id)
            result_dict = asdict(result)
            
            return result_dict
            
        except Exception as e:
            logger.error(f"Uncaught exception while evaluating question {question_pair.question_id}: {e}")
            
            # Return error result
            error_result = EvaluationResult(
                sample_id=f"error_{question_pair.question_id}_{int(time.time())}",
                session_id=session_id,
                dialogue_name=question_pair.dialogue_name,
                question_id=question_pair.question_id,
                question_text=question_pair.question_text,
                question_image=question_pair.question_image,
                system_answer=f"[Processing error: {str(e)[:200]}]",
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
            
            # Update failure statistics
            with self.stats_lock:
                self.session_statistics[session_id]["failed"] += 1
            
            return asdict(error_result)
    
    def _save_session_statistics(self):
        """Save session statistics to file"""
        # Create statistics directory
        stats_dir = Path("evaluation_statistics")
        stats_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_file = stats_dir / f"session_statistics_{self.model}_{timestamp}.json"
        
        # Convert defaultdict to regular dict for JSON serialization
        stats_dict = {}
        for session_id, stats in self.session_statistics.items():
            stats_dict[session_id] = dict(stats)
            # Convert nested defaultdicts
            stats_dict[session_id]["by_category"] = dict(stats["by_category"])
            stats_dict[session_id]["by_difficulty"] = dict(stats["by_difficulty"])
        
        # Calculate overall statistics
        total_questions = sum(s["total"] for s in self.session_statistics.values())
        total_successful = sum(s["successful"] for s in self.session_statistics.values())
        total_failed = sum(s["failed"] for s in self.session_statistics.values())
        total_truncated = sum(s["truncated_count"] for s in self.session_statistics.values())
        total_processing_time = sum(s["processing_time"] for s in self.session_statistics.values())
        
        # Calculate timing statistics
        total_memory_load_time = sum(s["total_memory_load_time"] for s in self.session_statistics.values())
        total_memory_recall_time = sum(s["total_memory_recall_time"] for s in self.session_statistics.values())
        total_llm_time = sum(s["total_llm_time"] for s in self.session_statistics.values())
        
        overall_stats = {
            "total_sessions": len(self.session_statistics),
            "total_questions": total_questions,
            "total_successful": total_successful,
            "total_failed": total_failed,
            "total_truncated": total_truncated,
            "success_rate": total_successful / total_questions if total_questions > 0 else 0,
            "total_processing_time": total_processing_time,
            "avg_processing_time_per_question": total_processing_time / total_questions if total_questions > 0 else 0,
            # Timing metrics
            "total_memory_load_time": total_memory_load_time,
            "total_memory_recall_time": total_memory_recall_time,
            "total_llm_time": total_llm_time,
            "avg_memory_load_time": total_memory_load_time / total_questions if total_questions > 0 else 0,
            "avg_memory_recall_time": total_memory_recall_time / total_questions if total_questions > 0 else 0,
            "avg_llm_time": total_llm_time / total_questions if total_questions > 0 else 0,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_time)) if self.start_time else None,
            "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.end_time)) if self.end_time else None,
            "total_duration": self.end_time - self.start_time if self.start_time and self.end_time else 0,
            "model": self.model,
            "memory_type": type(self.memory_system).__name__,
            "max_context_tokens": self.max_context_tokens,
            "truncation_strategy": self.truncation_strategy
        }
        
        output_data = {
            "overall_statistics": overall_stats,
            "per_session_statistics": stats_dict
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Session statistics saved to: {stats_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="VLM memory evaluator (full context)")
    parser.add_argument("--conversations_dir", type=str, required=True,
                       help="Conversation data directory (contains scenes subdirectory)")
    parser.add_argument("--api_key", type=str, required=True,
                       help="VLM API key")
    parser.add_argument("--model", type=str, required=True,
                       help="VLM model name")
    parser.add_argument("--base_url", type=str, required=True,
                       help="API base URL")
    parser.add_argument("--memory_type", type=str, default="full_text",
                       choices=["full_text"],
                       help="Memory system type")
    parser.add_argument("--max_questions_per_session", type=int, default=None,
                       help="Maximum number of questions to process per session")
    parser.add_argument("--max_sessions", type=int, default=None,
                       help="Maximum number of sessions to process")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose logging output")
    parser.add_argument("--test_mode", action="store_true",
                       help="Test mode, process only first 2 questions per session")
    parser.add_argument("--max_retries", type=int, default=3,
                       help="Maximum number of API call retries")
    parser.add_argument("--timeout", type=int, default=60,
                       help="API call timeout (seconds)")
    
    # New parameters: context truncation related
    parser.add_argument("--max_context_tokens", type=int, default=None,
                       help="Maximum context tokens, truncate if exceeded (e.g., 8000)")
    parser.add_argument("--truncation_strategy", type=str, default="head_only",
                       choices=["head_only", "head_tail"],
                       help="Truncation strategy: head_only (keep only beginning), head_tail (keep beginning and end)")
    # Concurrency processing
    parser.add_argument("--max_workers", type=int, default=3,
                       help="Maximum number of threads")
    parser.add_argument("--max_api_concurrency", type=int, default=2,
                       help="Maximum API concurrency")
    
    args = parser.parse_args()
    
    # Configure log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    print("=" * 70)
    print("VLM Memory Capability Evaluator (Using Complete Dialogue Context)")
    print(f"Model: {args.model}")
    print(f"API Endpoint: {args.base_url}")
    if args.max_context_tokens:
        print(f"Context Truncation: max {args.max_context_tokens} tokens, strategy: {args.truncation_strategy}")
    else:
        print("Context Truncation: No limit")
    print("=" * 70)
    
    # Test mode setting
    if args.test_mode:
        args.max_questions_per_session = 2
        print("Test mode: Processing only first 2 questions per session")
    
    # 1. Initialize memory system (load all sessions)
    print(f"\n[1] Initializing memory system ({args.memory_type})...")
    print(f"   Loading entire conversation from all sessions...")
    
    memory_load_start_total = time.time()
    memory_system = create_memory_system(args.memory_type, args.conversations_dir)
    total_load_time = memory_system.load_all_conversations()
    memory_load_total_time = time.time() - memory_load_start_total
    
    print(f"   Loaded {len(memory_system.memory_storage)} sessions, total {len(memory_system.all_dialogues)} dialogue turns")
    print(f"   Total load time: {memory_load_total_time:.2f}s")
    
    # Display loaded session information
    print(f"\n   Loaded session list:")
    for session_id, info in memory_system.session_info.items():
        session_title = info.get("session_title", "<Unnamed>")
        load_time = info.get("load_time", 0)
        if session_title:
            print(f"     - {session_id}: \"{session_title}\" ({info.get('dialogue_count', 0)} turns) - {load_time:.3f}s")
        else:
            print(f"     - {session_id}: {info.get('dialogue_count', 0)} dialogue turns - {load_time:.3f}s")
    
    # 2. Initialize VLM evaluator
    print(f"\n[2] Initializing VLM evaluator...")
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
    
    # 3. Load questions (grouped by session)
    print(f"\n[3] Loading question files (grouped by session)...")
    try:
        sessions_questions = evaluator.load_questions(args.conversations_dir)
    except Exception as e:
        print(f"   Failed to load questions: {e}")
        return
    
    if not sessions_questions:
        print("   No question files found for any session")
        return
    
    print(f"   Successfully loaded questions from {len(sessions_questions)} sessions")
    
    # Display information for each session
    total_questions = sum(len(data["questions"]) for data in sessions_questions.values())
    print(f"   Total questions: {total_questions}")
    
    for session_id, session_data in sessions_questions.items():
        question_count = len(session_data["questions"])
        session_dir_name = session_data.get("session_dir_name", session_id)
        if session_dir_name != session_id:
            print(f"     - {session_id} ({session_dir_name}): {question_count} questions")
        else:
            print(f"     - {session_id}: {question_count} questions")
    
    # Limit the number of sessions to process
    if args.max_sessions and args.max_sessions < len(sessions_questions):
        sessions_to_process = dict(list(sessions_questions.items())[:args.max_sessions])
        print(f"   Limiting to first {args.max_sessions} sessions")
    else:
        sessions_to_process = sessions_questions
    
    # 4. Execute evaluation (process by session, but use complete context)
    print(f"\n[4] Starting session-by-session evaluation (using complete dialogue context)...")
    print(f"   Processing {len(sessions_to_process)} sessions")
    print(f"   Total questions: {total_questions}")
    print(f"   Sessions in memory system: {len(memory_system.memory_storage)}")
    print("-" * 70)
    
    evaluator.evaluate_all_sessions(
        sessions_questions=sessions_to_process,
        max_questions_per_session=args.max_questions_per_session
    )
    
    # 5. Output global statistics
    print(f"\n[5] Evaluation complete!")
    print("-" * 70)
    

if __name__ == "__main__":
    main()