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

# Add at the beginning of the imports section
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Add at the beginning of the imports section
import concurrent.futures
import threading
from threading import Lock, Semaphore
from tqdm import tqdm  # Optional, for progress bars, requires: pip install tqdm

# Import API related libraries
import requests
from PIL import Image
import base64
from io import BytesIO

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
    supporting_evidence: List[Dict]
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
    supporting_evidence: Optional[List[Dict]] = None
    memory_context_summary: Optional[str] = None
    recall_method: str = "naive_rag"
    retrieved_chunks: Optional[List[Dict]] = None
    success: bool = True
    error_message: Optional[str] = None
    reasoning_process: Optional[str] = None
    
    # New timing fields
    memory_retrieve_time: float = 0.0    # Memory retrieval time
    prompt_build_time: float = 0.0       # Prompt building time
    api_call_time: float = 0.0           # API call time
    retrieval_timing: Dict = field(default_factory=dict)  # Retrieval timing details


class NaiveRAGMemorySystem:
    """
    Naive RAG based memory system
    - Splits dialogues into chunks
    - Uses keyword retrieval
    - Only retrieves parts relevant to the question
    """
    
    def __init__(self, conversations_dir: str, chunk_size: int = 1, top_k: int = 3, 
             embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize Naive RAG memory system (using BERT embedding)
        
        Args:
            conversations_dir: Conversation data directory
            chunk_size: Number of dialogue turns per chunk
            top_k: Number of chunks to retrieve
            embedding_model: Name of the embedding model to use
        """
        self.conversations_dir = conversations_dir
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.embedding_model_name = embedding_model
        
        self.memory_storage = {}  # Store content for all sessions
        self.all_chunks = []      # All dialogue chunks
        self.chunk_metadata = []  # Metadata for each chunk
        self.chunk_embeddings = None  # Embeddings for all chunks
        self.session_info = {}     # Session additional information
        
        # Storage time recording
        self.storage_time = 0.0      # Total storage time
        self.loading_time = 0.0      # Data loading time
        self.chunking_time = 0.0     # Chunking time
        self.embedding_time = 0.0    # Vectorization time
        
        # Load embedding model
        self._load_embedding_model()
        
    def _load_embedding_model(self):
        """Load BERT embedding model"""
        try:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            # Check if GPU is available
            if torch.cuda.is_available():
                self.embedding_model = self.embedding_model.to('cuda')
                logger.info("Using GPU for embedding computation")
            else:
                logger.info("Using CPU for embedding computation")
                
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            logger.warning("Will use fallback keyword matching method")
            self.embedding_model = None
        
    def load_all_conversations(self):
        """Load all session data from the entire conversation, chunk them, and compute embeddings - with timing"""
        overall_start = time.time()
        
        scenes_dir = os.path.join(self.conversations_dir, "scenes")
        if not os.path.exists(scenes_dir):
            raise ValueError(f"Scenes directory not found: {scenes_dir}")
        
        session_dirs = natsorted([
            d for d in os.listdir(scenes_dir) 
            if os.path.isdir(os.path.join(scenes_dir, d))
        ])
        
        all_chunks = []
        chunk_metadata = []
        
        # 1. Data loading time
        loading_start = time.time()
        
        for session_dir_name in session_dirs:
            session_dir = os.path.join(scenes_dir, session_dir_name)
            session_data = self._load_single_session(session_dir_name, session_dir)
            
            if session_data:
                session_id = session_dir_name
                self.memory_storage[session_id] = session_data
                
                # Get caption directory path
                caption_dir = os.path.join(session_dir, "caption")
                caption_files_exist = os.path.exists(caption_dir)
                
                # Extract dialogue content and add session information
                dialogues = session_data.get("dialogue", [])
                processed_dialogues = []
                timeline_date = session_data.get("timeline_date", "")
                for i, dialogue in enumerate(dialogues, 1):
                    role = dialogue.get("role", "")
                    content = dialogue.get("content", {})
                    text = timeline_date + ":" + content.get("text", "")
                    image_filename = content.get("image", "")
                    
                    # Process image description information
                    image_description = self._load_image_description(image_filename, caption_dir, caption_files_exist)
                    
                    # Create dialogue content with image description
                    content_with_description = content.copy()
                    if image_description:
                        content_with_description["image_description"] = image_description
                    
                    dialogue_with_session = {
                        "session_id": session_id,
                        "session_title": session_data.get("session_title", ""),
                        "timeline_date": session_data.get("timeline_date", ""),
                        "session_dir_name": session_dir_name,
                        "dialogue_index": i,
                        "role": role,
                        "content": content_with_description,
                        "text": text,
                        "has_image": bool(image_filename),
                        "image_filename": image_filename,
                        "image_description": image_description,
                        "combined_text": self._combine_dialogue_text(role, text, image_description)
                    }
                    processed_dialogues.append(dialogue_with_session)
                
                # Store session information
                self.session_info[session_id] = {
                    "session_dir_name": session_dir_name,
                    "session_title": session_data.get("session_title", ""),
                    "timeline_date": session_data.get("timeline_date", ""),
                    "generated_at": session_data.get("generated_at", ""),
                    "dialogue_count": len(dialogues),
                    "has_caption_dir": caption_files_exist,
                    "session_path": session_dir
                }
                
                # Chunk the current session's dialogues (chunking time counted)
                chunk_start = time.time()
                session_chunks, session_metadata = self._chunk_dialogues(processed_dialogues, session_id)
                all_chunks.extend(session_chunks)
                chunk_metadata.extend(session_metadata)
                self.chunking_time += time.time() - chunk_start
        
        self.loading_time = time.time() - loading_start
        logger.info(f"Data loading time: {self.loading_time:.2f} seconds")
        
        # 2. Store chunks and metadata
        self.all_chunks = all_chunks
        self.chunk_metadata = chunk_metadata
        
        logger.info(f"Chunking time: {self.chunking_time:.2f} seconds")
        
        # 3. Vectorization time
        embedding_start = time.time()
        self._compute_chunk_embeddings()
        self.embedding_time = time.time() - embedding_start
        logger.info(f"Vectorization time: {self.embedding_time:.2f} seconds")
        
        # Total storage time
        self.storage_time = time.time() - overall_start
        
        logger.info(f"Loaded {len(self.memory_storage)} sessions")
        logger.info(f"Dialogue chunking complete: {len(self.all_chunks)} chunks, max {self.chunk_size} turns per chunk")
        logger.info(f"Retrieval config: top_k={self.top_k}, embedding_model={self.embedding_model_name}")
        logger.info(f"Memory storage total time: {self.storage_time:.2f}s (Loading: {self.loading_time:.2f}s, Chunking: {self.chunking_time:.2f}s, Vectorization: {self.embedding_time:.2f}s)")
        
        # Statistics
        chunks_with_images = sum(1 for meta in self.chunk_metadata if meta["has_image"])
        logger.info(f"Chunks containing images: {chunks_with_images}")

    def _combine_dialogue_text(self, role: str, text: str, image_description: str) -> str:
        """Combine dialogue text for embedding calculation"""
        combined = []
        if role:
            combined.append(f"{role}:")
        if text:
            combined.append(text)
        if image_description:
            combined.append(f"[Image description: {image_description}]")
        return " ".join(combined)

    def _compute_chunk_embeddings(self):
        """Compute embeddings for all chunks"""
        if self.embedding_model is None:
            logger.warning("No embedding model available, skipping embedding computation")
            self.chunk_embeddings = None
            return
        
        logger.info(f"Computing embeddings for {len(self.all_chunks)} chunks...")
        
        # Prepare text list
        texts = []
        for i, chunk in enumerate(self.all_chunks):
            # Extract plain text from chunk text (remove markers)
            text = self._extract_text_from_chunk(chunk)
            texts.append(text)
        
        # Batch compute embeddings
        try:
            embeddings = self.embedding_model.encode(
                texts, 
                show_progress_bar=True,
                batch_size=32,
                convert_to_numpy=True
            )
            self.chunk_embeddings = embeddings
            logger.info(f"Embedding computation complete, dimension: {embeddings.shape}")
        except Exception as e:
            logger.error(f"Failed to compute embeddings: {e}")
            self.chunk_embeddings = None

    def _extract_text_from_chunk(self, chunk: str) -> str:
        """Extract plain text from chunk text (remove markers)"""
        lines = chunk.split('\n')
        text_lines = []
        
        for line in lines:
            # Skip session marker lines
            if line.startswith('[Session') or line.startswith('---') or not line.strip():
                continue
            
            # Extract dialogue content
            # Format: "Turn X role: text"
            match = re.match(r'Turn\s+(\d+)\s+(\w+):\s*(.*)', line)
            if match:
                role, content = match.group(2), match.group(3) if len(match.groups()) > 2 else ""
                text_lines.append(content)
            else:
                # If not in standard format, keep original line
                text_lines.append(line)
        
        return " ".join(text_lines)
        
    def _load_image_description(self, image_filename: str, caption_dir: str, caption_files_exist: bool) -> str:
        """Load image description information"""
        if not image_filename or not caption_files_exist:
            return ""
        
        image_description = ""
        # Extract numeric part from filename
        caption_json = Path(image_filename).stem + ".json"  # Get filename without extension and add .json
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
                        image_description = " | ".join(description_texts)
                        
                except Exception as e:
                    logger.error(f"Failed to load image description file {caption_file_path}: {e}")
        
        return image_description
    
    def _load_single_session(self, session_dir_name: str, session_dir: str) -> Optional[Dict]:
        """Load data from a single session"""
        conversation_file = os.path.join(session_dir, "session.json")
        
        if not os.path.exists(conversation_file):
            logger.warning(f"Session.json file not found: {conversation_file}")
            return None
        
        try:
            with open(conversation_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            logger.debug(f"Successfully loaded dialogue data from {session_dir_name}")
            return session_data
            
        except Exception as e:
            logger.error(f"Failed to load {conversation_file}: {e}")
            return None
    
    def _chunk_dialogues(self, dialogues: List[Dict], session_id: str) -> Tuple[List[str], List[Dict]]:
        """
        Chunk dialogues
        
        Args:
            dialogues: List of dialogues
            session_id: Session identifier
        
        Returns:
            chunks: List of text chunks
            metadata: Metadata for each chunk
        """
        chunks = []
        metadata = []
        
        for i in range(0, len(dialogues), self.chunk_size):
            chunk_dialogues = dialogues[i:i + self.chunk_size]
            
            # Build chunk text (for display)
            chunk_text = f"[Session {session_id} - Dialogue chunk {i//self.chunk_size + 1}]\n"
            
            # Build combined text for embedding
            combined_texts = []
            
            # Record chunk information
            has_image = False
            image_info = []
            dialogue_indices = []
            
            for dia in chunk_dialogues:
                dialogue_index = dia.get("dialogue_index", 0)
                dialogue_indices.append(dialogue_index)
                
                role = dia.get("role", "")
                text = dia.get("text", "")
                image_filename = dia.get("image_filename", "")
                image_description = dia.get("image_description", "")
                combined_text = dia.get("combined_text", "")
                
                if image_filename:
                    has_image = True
                    image_info.append({
                        "filename": image_filename,
                        "description": image_description if image_description else ""
                    })
                    chunk_text += f"Turn {dialogue_index} {role}: [Image: {image_filename}] {image_description} {text}\n"
                else:
                    chunk_text += f"Turn {dialogue_index} {role}: {text}\n"
                
                # Collect text for embedding
                combined_texts.append(combined_text)
            
            chunks.append(chunk_text)
            metadata.append({
                "session_id": session_id,
                "chunk_index": i // self.chunk_size + 1,
                "start_index": i + 1,
                "end_index": min(i + self.chunk_size, len(dialogues)),
                "dialogue_indices": dialogue_indices,
                "has_image": has_image,
                "image_info": image_info,
                "text_length": len(chunk_text),
                "combined_text": " ".join(combined_texts),  # Save combined text for embedding
                "embedding": None  # Will be calculated later
            })
        
        return chunks, metadata
        
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text
        
        Args:
            text: Input text
        
        Returns:
            List of keywords
        """
        # Simple word segmentation and filtering
        words = re.findall(r'[\u4e00-\u9fff\w]+', text)
        
        # Filter stop words (can be extended)
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        
        # Filter words that are too short
        keywords = [w for w in words if len(w) > 1 and w.lower() not in stopwords]
        
        return keywords[:10]  # Return at most 10 keywords
    
    def _calculate_similarity(self, question_embedding: np.ndarray, chunk_embedding: np.ndarray) -> float:
        """
        Calculate cosine similarity between question and chunk
        
        Args:
            question_embedding: Embedding vector of the question
            chunk_embedding: Embedding vector of the chunk
        
        Returns:
            Cosine similarity score (between 0 and 1)
        """
        if question_embedding is None or chunk_embedding is None:
            return 0.0
        
        # Calculate cosine similarity
        similarity = cosine_similarity(
            question_embedding.reshape(1, -1),
            chunk_embedding.reshape(1, -1)
        )[0][0]
        
        return float(similarity)

    # Modified version of retrieve_relevant_context method in NaiveRAGMemorySystem class
    def retrieve_relevant_context(self, question_text: str, target_session_id: str) -> Dict[str, Any]:
        """
        Retrieve context relevant to the question (using BERT embedding) - with timing
        """
        start_time = time.time()
        
        # 1. Compute embedding for the question
        embedding_start = time.time()
        question_embedding = self._get_embedding(question_text)
        embedding_time = time.time() - embedding_start
        
        retrieval_time = 0
        similarity_time = 0
        
        if question_embedding is None or self.chunk_embeddings is None:
            logger.warning("Cannot compute embedding, using fallback keyword matching method")
            return self._retrieve_by_keywords_fallback(question_text, target_session_id)
        
        # 2. Calculate similarity with all chunks
        similarity_start = time.time()
        similarities = []
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            similarity = self._calculate_similarity(question_embedding, chunk_embedding)
            similarities.append({
                "chunk_index": i,
                "chunk_text": self.all_chunks[i],
                "metadata": self.chunk_metadata[i],
                "similarity": similarity
            })
        similarity_time = time.time() - similarity_start
        
        # 3. Sort by similarity and select top_k
        retrieval_start = time.time()
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        top_chunks = similarities[:self.top_k]
        
        # 4. Build retrieved context
        retrieved_context = []
        context_summary = []
        
        for chunk in top_chunks:
            if chunk["similarity"] > 0.1:
                retrieved_context.append(chunk["chunk_text"])
                context_summary.append({
                    "session_id": chunk["metadata"]["session_id"],
                    "chunk_index": chunk["metadata"]["chunk_index"],
                    "dialogue_indices": chunk["metadata"]["dialogue_indices"],
                    "similarity": chunk["similarity"],
                    "has_image": chunk["metadata"]["has_image"],
                    "embedding_similarity": True
                })
        
        # 5. If no relevant chunks retrieved, use recent chunks from target session as fallback
        if not retrieved_context:
            logger.warning(f"No relevant content retrieved, using recent chunks from target session {target_session_id} as fallback")
            return self._get_fallback_context(target_session_id)
        
        retrieval_time = time.time() - retrieval_start
        total_time = time.time() - start_time
        
        # 6. Build complete context information
        context_info = {
            "retrieved_chunks": retrieved_context,
            "retrieved_metadata": context_summary,
            "total_chunks": len(self.all_chunks),
            "top_k": self.top_k,
            "similarity_method": "bert_embedding_cosine",
            "timing": {
                "embedding_time": embedding_time,
                "similarity_time": similarity_time,
                "retrieval_time": retrieval_time,
                "total_time": total_time
            }
        }
        
        return context_info

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding vector for text
        
        Args:
            text: Input text
        
        Returns:
            Embedding vector
        """
        if self.embedding_model is None:
            return None
        
        try:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Failed to compute embedding: {e}")
            return None
    
    def get_session_context(self, target_session_id: str, question_text: str) -> Dict[str, Any]:
        """
        Get retrieval-augmented context for the specific question
        
        Args:
            target_session_id: Target session ID
            question_text: Question text
        
        Returns:
            Retrieval-augmented context
        """
        # Execute retrieval
        retrieval_result = self.retrieve_relevant_context(question_text, target_session_id)
        
        # Build return context format
        context = {
            "target_session_id": target_session_id,
            "target_session_info": self.session_info.get(target_session_id, {}),
            "all_sessions": list(self.memory_storage.keys()),
            "total_chunks": len(self.all_chunks),
            "retrieval_info": retrieval_result,
            "retrieved_context": "\n\n---\n\n".join(retrieval_result["retrieved_chunks"])
        }
        
        return context
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics - with storage time"""
        return {
            "total_sessions": len(self.memory_storage),
            "total_chunks": len(self.all_chunks),
            "chunk_size": self.chunk_size,
            "top_k": self.top_k,
            "embedding_model": self.embedding_model_name,
            "embedding_available": self.embedding_model is not None,
            "embedding_dimension": self.chunk_embeddings.shape[1] if self.chunk_embeddings is not None else 0,
            "session_info": self.session_info,
            "chunks_with_images": sum(1 for meta in self.chunk_metadata if meta["has_image"]),
            # Storage time statistics
            "storage_time": self.storage_time,
            "loading_time": self.loading_time,
            "chunking_time": self.chunking_time,
            "embedding_time": self.embedding_time,
            "avg_time_per_chunk": self.storage_time / len(self.all_chunks) if self.all_chunks else 0,
            "avg_embedding_time_per_chunk": self.embedding_time / len(self.all_chunks) if self.all_chunks else 0
        }


class NaiveRAGPromptTemplate:
    """Standardized prompt template for Naive RAG with 9 question types"""
    
    # Instructions for each question type (only CD, AR, TTL include abbreviation)
    INSTRUCTIONS = {
        "Unimodal Precise Recall": "Accurately recall specific information from the retrieved conversation chunks and answer directly. Note: You may not see the complete conversation history, only the retrieved relevant chunks.",
        "Cross-modal Related Retrieval": "Retrieve related information across different modalities (text and images) from the retrieved conversation chunks.",
        "Knowledge Resolution": "Resolve and maintain knowledge consistency based on the retrieved conversation chunks.",
        "Temporal Reasoning": "Reason about temporal relationships and time-based information from the retrieved conversation chunks.",
        "Multimodal Causal Reasoning": "Perform causal reasoning using both text and image information from the retrieved conversation chunks.",
        "Reference & Evolution Tracking": "Track references and their evolution from the retrieved conversation chunks.",
        "Test-Time Learning": "Learn and adapt from the retrieved conversation context at test time to answer the question.",
        "Conflict Detection": "Check whether this information conflicts with the retrieved conversation chunks.",
        "Answer Refusal": "Determine if the question can be answered based on the retrieved conversation chunks."
    }
    
    # Response format requirements
    FORMAT_REQUIREMENTS = {
        "Conflict Detection": "Response format: Reply strictly with either 'Yes' or 'No' only.",
        "Answer Refusal": "Response format: If the information is present in the retrieved conversation chunks, provide answer based on that information; if not present, reply with: 'Not mentioned.'",
        "default": "Response format: Provide clear and accurate answers based on the retrieved conversation chunks."
    }
    
    # Base template
    TEMPLATE = """You are a memory testing system using Naive RAG to retrieve relevant conversation chunks. {instruction}

IMPORTANT: 
1. Provide your answer based on the retrieved conversation chunks only. You may not have the complete conversation history.
2. Keep your answer within 100 words. Short and concise answers are acceptable.
3. Answer in English. This is a strict requirement. Do not answer in any other language.

Retrieved relevant conversation chunks (may be incomplete):
{context}

Question: {question}

{format_requirement}

Please output in the following JSON format:
{{
    "reasoning_process": "Your reasoning process explaining how you arrived at the answer from the retrieved conversation",
    "system_answer": "Your final answer"
}}

Examples:
Question: What is the cat's name?
Correct output:
{{
    "reasoning_process": "The retrieved conversation mentions 'The cat's name is Almond' in session 2.",
    "system_answer": "Almond"
}}

Incorrect output example (DO NOT answer like this):
{{
    "reasoning_process": "I think the cat's name might be...",
    "system_answer": "We need answer: cat name is Almond because..."
}}"""

    def __init__(self, question_type: str, context: str, question: str):
        self.question_type = question_type
        self.context = context
        self.question = question
    
    def build(self) -> str:
        """Build the complete prompt"""
        
        # Get instruction for question type
        instruction = self.INSTRUCTIONS.get(
            self.question_type, 
            self.INSTRUCTIONS["Unimodal Precise Recall"]
        )
        
        # Get format requirement
        if self.question_type in self.FORMAT_REQUIREMENTS:
            format_requirement = self.FORMAT_REQUIREMENTS[self.question_type]
        else:
            format_requirement = self.FORMAT_REQUIREMENTS["default"]
        
        return self.TEMPLATE.format(
            instruction=instruction,
            context=self.context,
            question=self.question,
            format_requirement=format_requirement
        )
    
    def _calculate_confidence(self) -> float:
        return 0.7


class VLMEvaluator:
    def __init__(self, 
             memory_system: NaiveRAGMemorySystem,
             api_key: str,
             model: str = "",
             base_url: str = "",
             verbose: bool = False,
             max_retries: int = 3,
             timeout: int = 60,
             max_workers: int = 3,  # New: maximum worker threads
             max_api_concurrency: int = 2):  # New: maximum API concurrency
        """
        Initialize VLM evaluator (multi-threaded version)
        
        Args:
            memory_system: Memory system instance (Naive RAG)
            api_key: VLM API key
            model: VLM model name
            base_url: API base URL
            verbose: Verbose logging output
            max_retries: Maximum number of retries
            timeout: Request timeout (seconds)
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
        
        # New: thread control related attributes
        self.max_workers = max_workers
        self.max_api_concurrency = max_api_concurrency
        self.api_semaphore = Semaphore(max_api_concurrency)  # Control API concurrency
        self.file_lock = Lock()  # File write lock
        self.stats_lock = Lock()  # Statistics update lock
        
        # Store statistics
        self.session_statistics = defaultdict(lambda: {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "processing_time": 0.0,
            "by_category": defaultdict(int),
            "by_difficulty": defaultdict(int)
        })
        
        self.global_statistics = {
            "total_sessions": 0,
            "total_questions": 0,
            "successful_questions": 0,
            "failed_questions": 0,
            "start_time": None,
            "end_time": None,
            "max_workers": max_workers,  # New
            "max_api_concurrency": max_api_concurrency  # New
        }
        
        # Test API connection
        self._test_api_connection()

        # New: record failed question file paths (using set for automatic deduplication)
        self.failed_json_files = set()  # Using set for automatic deduplication
        self.failed_lock = Lock()  # Thread-safe lock

        
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
            # Try to find subdirectories containing "dialogue"
            dialogue_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("dialogue")]
            if not dialogue_dirs:
                raise ValueError(f"Dialogue directory not found: {base_dir}")
            
            dialogue_name = dialogue_dirs[0].name
            scenes_dir = dialogue_dirs[0] / "scenes"
        
        if not scenes_dir.exists():
            raise ValueError(f"Scenes directory not found: {scenes_dir}")
        
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
                    
                    if conversation_file.exists():
                        with open(conversation_file, 'r', encoding='utf-8') as f:
                            conv_data = json.load(f)
                            session_id = conv_data.get("session_id", session_dir_name)
                    
                    # Load question file
                    with open(question_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    questions = data.get("questions", [])
                    
                    # Convert format to list of QuestionAnswerPair
                    question_pairs = []
                    for q in questions:
                        qa_pair = QuestionAnswerPair(
                            question_id=q.get("question_id", f"Naive_rag_{len(question_pairs)}"),
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
                        
                        # Add image context
                        if qa_pair.question_image:
                            if str(session_id) == "session0":
                                print("Processing session0 image path", qa_pair.question_image)
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
    
    def _format_retrieved_context(self, context: Dict[str, Any]) -> str:
        """
        Format retrieved context as text
        
        Args:
            context: Context containing retrieval results
        
        Returns:
            Formatted text
        """
        if not context:
            return "No available memory"
        
        context_parts = []
        
        # Add retrieval information
        retrieval_info = context.get("retrieval_info", {})
        retrieved_metadata = retrieval_info.get("retrieved_metadata", [])
        similarity_method = retrieval_info.get("similarity_method", "unknown")
        
        context_parts.append("[Naive RAG Retrieval Results (BERT Embedding)]")
        context_parts.append(f"Similarity calculation method: {similarity_method}")
        context_parts.append(f"Retrieved {len(retrieved_metadata)} relevant chunks (top_k={self.memory_system.top_k})")
        
        # Display retrieved chunk information
        context_parts.append("\n[Retrieved Relevant Chunks]")
        for i, meta in enumerate(retrieved_metadata, 1):
            session_id = meta.get("session_id", "unknown")
            chunk_idx = meta.get("chunk_index", 0)
            dial_indices = meta.get("dialogue_indices", [])
            similarity = meta.get("similarity", 0)
            has_image = meta.get("has_image", False)
            is_fallback = meta.get("fallback", False)
            method = meta.get("method", "embedding")
            
            if is_fallback:
                context_parts.append(f"\nChunk {i} [Session {session_id} chunk {chunk_idx} - Fallback] (dialogue turns: {dial_indices[0]}-{dial_indices[-1]})")
            else:
                context_parts.append(f"\nChunk {i} [Session {session_id} chunk {chunk_idx}] (similarity: {similarity:.4f}, method: {method}, dialogue turns: {dial_indices[0]}-{dial_indices[-1]})")
            
            if has_image:
                context_parts.append(f"  (contains image)")
        
        # Add retrieved content
        retrieved_chunks = retrieval_info.get("retrieved_chunks", [])
        if retrieved_chunks:
            context_parts.append("\n[Retrieved Dialogue Content]")
            for i, chunk in enumerate(retrieved_chunks, 1):
                context_parts.append(f"\n--- Chunk {i} ---")
                context_parts.append(chunk)
        
        return "\n".join(context_parts)
    
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
                print("Successfully processed")
                return img_base64
                
        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            raise
    
    def _call_vlm_api(self, prompt: str, images: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Call real VLM API (with concurrency control)
        
        Args:
            prompt: Prompt text
            images: List of image paths
        
        Returns:
            API response result
        """
        start_time = time.time()
        # New: use semaphore to control API concurrency
        acquired = False
        if self.api_semaphore:
            self.api_semaphore.acquire()
            acquired = True
            if self.verbose:
                logger.debug(f"API semaphore acquired, available: {self.api_semaphore._value}")
        
        try:
            # Prepare message
            messages = []
            
            # If there are images, include them as part of the message
            if images and len(images) > 0:
                # Process the first image
                try:
                    image_base64 = self._prepare_image_for_api(images[0])
                    # Build message (supports image format)
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
                # Text only
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
                    
                    logger.debug(f"Calling API (attempt {attempt + 1}/{self.max_retries}): {api_url}")
                    
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
                            
                            logger.debug(f"API call successful, response time: {processing_time:.2f} seconds")
                            
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
            
            processing_time = time.time() - start_time
            return {
                "answer": f"[API call failed: All {self.max_retries} retries failed]",
                "processing_time": processing_time,
                "model": self.model,
                "success": False,
                "error": "All retries failed"
            }
        finally:
            # New: release semaphore
            if acquired:
                self.api_semaphore.release()
                if self.verbose:
                    logger.debug(f"API semaphore released, available: {self.api_semaphore._value}")
    
    def _construct_prompt_for_question(self, 
                                  question_pair: QuestionAnswerPair,
                                  memory_context: Dict[str, Any]) -> str:
        """Build prompt for specific question using Naive RAG retrieved context"""
        
        # Format retrieved context
        context_str = self._format_retrieved_context(memory_context)
        
        # Extract question components
        question_text = question_pair.question_text
        question_type = question_pair.question_type.get("sub_type", "")
        
        # Build prompt using template system
        prompt = NaiveRAGPromptTemplate(
            question_type=question_type,
            context=context_str,
            question=question_text
        )
        
        # Log debug info
        if self.verbose:
            logger.debug(f"Question type: {question_type}")
        
        return prompt.build()
    
    def evaluate_single_question(self, 
                            question_pair: QuestionAnswerPair,
                            session_id: str,
                            question_file_path: str = None) -> EvaluationResult:
        """Evaluate a single question - with detailed timing and error recording"""
        start_time = time.time()
        
        # Timing variables
        memory_retrieve_time = 0
        prompt_build_time = 0
        api_call_time = 0
        
        try:
            logger.debug(f"Processing question: {session_id} - {question_pair.question_id} ({question_pair.category})")
            
            # 1. Prepare query text (including image caption information)
            query_text = question_pair.question_text
            
            # If there are images, add image caption information
            if question_pair.question_image and question_pair.image_context:
                captions = []
                for img_path in question_pair.image_context:
                    # Generate corresponding caption file path from image path
                    img_dir = os.path.dirname(img_path)
                    img_filename = os.path.basename(img_path)
                    img_name_without_ext = os.path.splitext(img_filename)[0]
                    
                    # Build caption path
                    caption_dir = img_dir.replace('\\image\\', '\\caption\\')
                    caption_path = os.path.join(caption_dir, f"{img_name_without_ext}.json")
                    
                    # If the above path doesn't exist, try other possible path structures
                    if not os.path.exists(caption_path):
                        possible_paths = [
                            caption_path,
                            img_path.replace('\\image\\', '\\caption\\').replace(os.path.splitext(img_filename)[1], '.json'),
                            os.path.join(os.path.dirname(img_dir.replace('\\image\\', '\\caption\\')), f"{img_name_without_ext}.json"),
                        ]
                        
                        for possible_path in possible_paths:
                            if os.path.exists(possible_path):
                                caption_path = possible_path
                                break
                    
                    if os.path.exists(caption_path):
                        try:
                            with open(caption_path, 'r', encoding='utf-8') as f:
                                caption_data = json.load(f)
                                if caption_data.get('success') and 'description' in caption_data:
                                    caption_text = caption_data['description'].get('final_text') or caption_data['description'].get('full_text', '')
                                    if caption_text:
                                        captions.append(caption_text)
                                        logger.debug(f"Successfully read caption: {caption_path}")
                        except Exception as e:
                            logger.warning(f"Failed to read caption file {caption_path}: {e}")
                
                # Add image caption information to query text
                if captions:
                    captions_text = ' '.join([f"[image{i+1}description]: {cap}" for i, cap in enumerate(captions)])
                    query_text = f"[question image description]: {captions_text}\n[user question]: {question_pair.question_text}"
                    logger.debug(f"Enhanced query text length: {len(query_text)}")
            
            # 2. Get retrieval-augmented context (record retrieval time)
            memory_retrieve_start = time.time()
            rag_context = self.memory_system.get_session_context(session_id, query_text)
            memory_retrieve_time = time.time() - memory_retrieve_start
            
            # Get retrieval timing details
            retrieval_timing = rag_context.get("retrieval_info", {}).get("timing", {})
            
            # Create memory context summary
            retrieval_info = rag_context.get("retrieval_info", {})
            retrieved_metadata = retrieval_info.get("retrieved_metadata", [])
            
            memory_context_summary = f"Retrieved {len(retrieved_metadata)} relevant chunks"
            if retrieval_timing:
                memory_context_summary += f" (retrieval time: {retrieval_timing.get('total_time', 0):.3f} seconds)"
            
            # 3. Build prompt (record building time)
            prompt_build_start = time.time()
            prompt = self._construct_prompt_for_question(question_pair, rag_context)
            prompt_build_time = time.time() - prompt_build_start
            
            # 4. Prepare images (for VLM visual input)
            images = []
            if question_pair.question_image and question_pair.image_context:
                for img_path in question_pair.image_context:
                    if os.path.exists(img_path):
                        images.append(img_path)
                    else:
                        logger.warning(f"Image file does not exist: {img_path}")
            
            # 5. Call VLM API (record API call time)
            api_call_start = time.time()
            vlm_response = self._call_vlm_api(prompt=prompt, images=images)
            api_call_time = vlm_response.get("processing_time", time.time() - api_call_start)
            
            response_text = vlm_response.get("answer", "").strip()
            success = vlm_response.get("success", False)
            
            # 6. Parse JSON response to extract reasoning process and answer
            reasoning_process = None
            system_answer = response_text
            
            if success:
                try:
                    # Find JSON content (may be wrapped in markdown code blocks)
                    json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        # Try to directly find JSON object
                        json_match = re.search(r'(\{.*?\})', response_text, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(1)
                        else:
                            json_str = response_text
                    
                    # Parse JSON
                    parsed_response = json.loads(json_str)
                    
                    # Extract reasoning process and answer
                    reasoning_process = parsed_response.get("reasoning_process", "")
                    system_answer = parsed_response.get("system_answer", response_text)
                    
                    logger.debug(f"Successfully parsed JSON response - reasoning process length: {len(reasoning_process) if reasoning_process else 0}")
                    
                except json.JSONDecodeError as e:
                    # If unable to parse as JSON, use original response as answer
                    logger.warning(f"Unable to parse API response as JSON: {e}")
                    logger.debug(f"Original response: {response_text[:200]}...")
                    system_answer = response_text
                    reasoning_process = None
            
            # 7. Calculate confidence
            confidence = 0.7
            
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
                supporting_evidence=question_pair.supporting_evidence,
                memory_context_summary=memory_context_summary,
                recall_method="naive_rag",
                retrieved_chunks=retrieved_metadata,
                success=success,
                error_message=None if success else vlm_response.get("error", ""),
                reasoning_process=reasoning_process,
                # New timing fields
                memory_retrieve_time=memory_retrieve_time,
                prompt_build_time=prompt_build_time,
                api_call_time=api_call_time,
                retrieval_timing=retrieval_timing
            )
            
            # Update session statistics
            with self.stats_lock:
                self.session_statistics[session_id]["successful"] += 1
                self.session_statistics[session_id]["processing_time"] += total_processing_time
                
                # Accumulate timing statistics
                if "total_memory_retrieve_time" not in self.session_statistics[session_id]:
                    self.session_statistics[session_id]["total_memory_retrieve_time"] = 0
                    self.session_statistics[session_id]["total_prompt_build_time"] = 0
                    self.session_statistics[session_id]["total_api_call_time"] = 0
                
                self.session_statistics[session_id]["total_memory_retrieve_time"] += memory_retrieve_time
                self.session_statistics[session_id]["total_prompt_build_time"] += prompt_build_time
                self.session_statistics[session_id]["total_api_call_time"] += api_call_time
            
            logger.info(f"✓ Successfully processed: {session_id} - {question_pair.question_id} (Total: {total_processing_time:.2f}s, Retrieval: {memory_retrieve_time:.3f}s, API: {api_call_time:.2f}s)")
            
            return result
            
        except Exception as e:
            total_processing_time = time.time() - start_time
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
                processing_time=total_processing_time,
                confidence=0.0,
                supporting_evidence=question_pair.supporting_evidence,
                memory_context_summary="Error: Unable to retrieve context",
                recall_method="naive_rag",
                retrieved_chunks=[],
                success=False,
                error_message=error_msg,
                reasoning_process=None,
                memory_retrieve_time=memory_retrieve_time,
                prompt_build_time=prompt_build_time,
                api_call_time=api_call_time,
                retrieval_timing={}
            )
            
            # Update session statistics
            with self.stats_lock:
                self.session_statistics[session_id]["failed"] += 1
            
            return result
    
    def evaluate_session_questions(self,
                                session_id: str,
                                session_data: Dict,
                                max_questions: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Evaluate all questions in a session in parallel
        """
        questions = session_data["questions"]
        session_path = Path(session_data["session_path"])
        session_dir_name = session_data.get("session_dir_name", session_id)
        question_file_path = session_data.get("question_file", "")  # Get question file path
        
        if max_questions and max_questions < len(questions):
            questions = questions[:max_questions]
        
        total_questions = len(questions)
        logger.info(f"Starting parallel evaluation of {total_questions} questions for {session_id}")
        logger.info(f"Using Naive RAG method: chunk_size={self.memory_system.chunk_size}, top_k={self.memory_system.top_k}")
        logger.info(f"API concurrency: {self.max_api_concurrency}")
        
        # Thread-safe initialization of session statistics
        with self.stats_lock:
            self.session_statistics[session_id]["total"] = total_questions
            for qa in questions:
                self.session_statistics[session_id]["by_category"][qa.category] += 1
                self.session_statistics[session_id]["by_difficulty"][qa.difficulty] += 1
            
            # Initialize timing fields
            if "total_memory_retrieve_time" not in self.session_statistics[session_id]:
                self.session_statistics[session_id]["total_memory_retrieve_time"] = 0
                self.session_statistics[session_id]["total_prompt_build_time"] = 0
                self.session_statistics[session_id]["total_api_call_time"] = 0
        
        # Thread-safe list for storing results
        results = []
        results_lock = Lock()
        processed_count = 0
        processed_lock = Lock()
        
        # Use thread pool to process questions in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_api_concurrency) as executor:
            # Submit all question tasks
            future_to_question = {}
            for question_pair in questions:
                future = executor.submit(
                    self._evaluate_question_with_stats,
                    question_pair,
                    session_id,
                    question_file_path  # Pass file path
                )
                future_to_question[future] = question_pair
            
            # Use tqdm to show progress (optional)
            try:
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
                                pbar.set_postfix({"Success": "✓", "ID": question_pair.question_id[:20]})
                            else:
                                pbar.set_postfix({"Success": "✗", "ID": question_pair.question_id[:20]})
                            
                        except Exception as e:
                            logger.error(f"Question {question_pair.question_id} processing failed: {e}")
                            
                            error_result = {
                                "sample_id": f"error_{question_pair.question_id}",
                                "session_id": session_id,
                                "question_id": question_pair.question_id,
                                "success": False,
                                "error_message": str(e)[:200]
                            }
                            with results_lock:
                                results.append(error_result)
                            pbar.update(1)
            except ImportError:
                # If tqdm is not available, use simple progress display
                for future in concurrent.futures.as_completed(future_to_question):
                    question_pair = future_to_question[future]
                    try:
                        result_dict = future.result()
                        
                        with results_lock:
                            results.append(result_dict)
                        
                        with processed_lock:
                            processed_count += 1
                            if processed_count % 5 == 0 or processed_count == total_questions:
                                logger.info(f"[{session_id}] Progress: {processed_count}/{total_questions}")
                        
                    except Exception as e:
                        logger.error(f"Question {question_pair.question_id} processing failed: {e}")
                        
                        error_result = {
                            "sample_id": f"error_{question_pair.question_id}",
                            "session_id": session_id,
                            "question_id": question_pair.question_id,
                            "success": False,
                            "error_message": str(e)[:200]
                        }
                        with results_lock:
                            results.append(error_result)
                        
                        with processed_lock:
                            processed_count += 1
        
        # Final save of results (needs thread safety)
        with self.file_lock:
            self._save_session_results(
                session_id,
                session_dir_name,
                session_path,
                results,
                final=True
            )
        
        successful = len([r for r in results if r.get('success', False)])
        logger.info(f"Session {session_id} parallel processing completed: Successful {successful}/{total_questions}")
        
        # Output session timing statistics
        with self.stats_lock:
            session_stats = self.session_statistics[session_id]
            if successful > 0:
                logger.info(f"  Session timing statistics - Avg retrieval: {session_stats.get('total_memory_retrieve_time', 0)/successful:.3f}s, "
                        f"Avg prompt: {session_stats.get('total_prompt_build_time', 0)/successful:.3f}s, "
                        f"Avg API: {session_stats.get('total_api_call_time', 0)/successful:.2f}s")
        
        # Update global statistics
        with self.stats_lock:
            self.global_statistics["total_questions"] += total_questions
            self.global_statistics["successful_questions"] += successful
            self.global_statistics["failed_questions"] += (total_questions - successful)
        
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
        json_filename = f"results_naive_rag.json"
        json_file = session_results_dir / json_filename
        
        # Build complete result data structure
        full_results = {
            "metadata": {
                "session_id": session_id,
                "session_dir_name": session_dir_name,
                "session_path": str(session_path),
                "evaluation_time": timestamp,
                "vlm_model": self.model,
                "memory_type": type(self.memory_system).__name__,
                "base_url": self.base_url,
                "context_type": "naive_rag",
                "chunk_size": self.memory_system.chunk_size,
                "top_k": self.memory_system.top_k
            },
            "results": results
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"Saved Naive RAG results for {session_id} to: {json_file}")
        
    
    def evaluate_all_sessions(self,
                        sessions_questions: Dict[str, Dict],
                        max_questions_per_session: Optional[int] = None):
        """
        Evaluate all sessions in parallel (multi-threaded version)
        """
        self.global_statistics["start_time"] = time.time()
        self.global_statistics["total_sessions"] = len(sessions_questions)
        
        memory_stats = self.memory_system.get_memory_stats()
        logger.info(f"Starting parallel evaluation of {len(sessions_questions)} sessions")
        logger.info(f"Memory system: {type(self.memory_system).__name__}")
        logger.info(f"  Total chunks: {memory_stats['total_chunks']}")
        logger.info(f"  Chunk size: {memory_stats['chunk_size']}")
        logger.info(f"  Retrieval top_k: {memory_stats['top_k']}")
        logger.info(f"Thread configuration: max_workers={self.max_workers}, max_api_concurrency={self.max_api_concurrency}")
        
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
            successful_sessions = 0
            failed_sessions = 0
            
            # Use tqdm to show session progress
            try:
                from tqdm import tqdm
                with tqdm(total=len(sessions_questions), desc="Overall Progress", unit="session") as pbar:
                    for future in concurrent.futures.as_completed(future_to_session):
                        session_id = future_to_session[future]
                        completed += 1
                        try:
                            results = future.result()
                            successful_sessions += 1
                            pbar.set_postfix({"Success": "✓", "session": session_id[:10]})
                            logger.info(f"[{completed}/{len(sessions_questions)}] Session {session_id} processing completed, successfully processed {len(results)} questions")
                        except Exception as e:
                            failed_sessions += 1
                            logger.error(f"Session {session_id} processing failed: {e}")
                            pbar.set_postfix({"Success": "✗", "session": session_id[:10]})
                        pbar.update(1)
            except ImportError:
                # If tqdm is not available, use simple progress display
                for future in concurrent.futures.as_completed(future_to_session):
                    session_id = future_to_session[future]
                    completed += 1
                    try:
                        results = future.result()
                        successful_sessions += 1
                        logger.info(f"[{completed}/{len(sessions_questions)}] Session {session_id} processing completed, successfully processed {len(results)} questions")
                    except Exception as e:
                        failed_sessions += 1
                        logger.error(f"Session {session_id} processing failed: {e}")
        
        self.global_statistics["end_time"] = time.time()
        self.global_statistics["successful_sessions"] = successful_sessions
        self.global_statistics["failed_sessions"] = failed_sessions
        
        # Output overall statistics
        total_time = self.global_statistics["end_time"] - self.global_statistics["start_time"]
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluation Complete Statistics:")
        logger.info(f"  - Total Sessions: {len(sessions_questions)}")
        logger.info(f"  - Successful Sessions: {successful_sessions}")
        logger.info(f"  - Failed Sessions: {failed_sessions}")
        logger.info(f"  - Total Questions: {self.global_statistics['total_questions']}")
        logger.info(f"  - Successful Questions: {self.global_statistics['successful_questions']}")
        logger.info(f"  - Failed Questions: {self.global_statistics['failed_questions']}")
        logger.info(f"  - Total Time: {total_time:.2f} seconds")
        if self.global_statistics['total_questions'] > 0:
            logger.info(f"  - Average per Question: {total_time/self.global_statistics['total_questions']:.2f} seconds")
        logger.info(f"{'='*60}")
        
    def _evaluate_session_parallel(self, session_id: str, session_data: Dict, 
                             max_questions_per_session: Optional[int]) -> List[Dict]:
        """
        Wrapper method for session evaluation (for thread pool calls)
        """
        import threading
        thread_name = threading.current_thread().name
        logger.info(f"Thread [{thread_name}] starting to process session: {session_id}")
        
        try:
            # Call method to process questions within session in parallel
            results = self.evaluate_session_questions(
                session_id, session_data, max_questions_per_session
            )
            return results
        except Exception as e:
            logger.error(f"Thread [{thread_name}] error processing session {session_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def _evaluate_question_with_stats(self, question_pair: QuestionAnswerPair, 
                                 session_id: str,
                                 question_file_path: str = None) -> Dict[str, Any]:
        """
        Evaluate a single question and update statistics (thread-safe)
        
        Args:
            question_pair: Question-answer pair
            session_id: Session ID
            question_file_path: Path to the question file
        """
        try:
            # Call the original evaluation method, passing file path
            result = self.evaluate_single_question(question_pair, session_id, question_file_path)
            result_dict = asdict(result)
            
            # Thread-safe update of statistics
            with self.stats_lock:
                if result.success:
                    self.session_statistics[session_id]["successful"] += 1
                else:
                    self.session_statistics[session_id]["failed"] += 1
                
                # Accumulate timing statistics
                self.session_statistics[session_id]["total_memory_retrieve_time"] += getattr(result, 'memory_retrieve_time', 0)
                self.session_statistics[session_id]["total_prompt_build_time"] += getattr(result, 'prompt_build_time', 0)
                self.session_statistics[session_id]["total_api_call_time"] += getattr(result, 'api_call_time', 0)
                self.session_statistics[session_id]["processing_time"] += result.processing_time
            
            return result_dict
            
        except Exception as e:
            logger.error(f"Uncaught exception while evaluating question {question_pair.question_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
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
                supporting_evidence=question_pair.supporting_evidence,
                memory_context_summary="Error: Processing exception",
                recall_method="naive_rag",
                retrieved_chunks=[],
                success=False,
                error_message=str(e)[:200],
                reasoning_process=None,
                memory_retrieve_time=0,
                prompt_build_time=0,
                api_call_time=0,
                retrieval_timing={}
            )
            
            # Update failure statistics
            with self.stats_lock:
                self.session_statistics[session_id]["failed"] += 1
            
            return asdict(error_result)


def create_memory_system(memory_type: str, conversations_dir: str, **kwargs):
    """Create memory system"""

    if memory_type == "naive_rag":
        chunk_size = kwargs.get("chunk_size", 5)
        top_k = kwargs.get("top_k", 3)
        return NaiveRAGMemorySystem(conversations_dir, chunk_size=chunk_size, top_k=top_k)
    else:
        raise ValueError(f"Unsupported memory type: {memory_type}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="VLM Memory Capability Evaluator (using Naive RAG)")
    parser.add_argument("--conversations_dir", required=True, help="Path to dialogue folder")
    parser.add_argument("--api_key", required=True, help="VLM API key")
    parser.add_argument("--model", required=True, help="VLM model name")
    parser.add_argument("--base_url", required=True, help="API base URL")
    parser.add_argument("--memory_type", default="naive_rag", choices=["naive_rag"], help="Memory type (only naive_rag supported)")
    parser.add_argument("--chunk_size", type=int, default=1, help="Number of dialogue turns per chunk")
    parser.add_argument("--top_k", type=int, default=3, help="Number of chunks to retrieve")
    parser.add_argument("--max_questions_per_session", type=int, default=None, help="Limit questions per session")
    parser.add_argument("--max_sessions", type=int, default=None, help="Limit number of sessions to process")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--test_mode", action="store_true", help="process only first 2 questions per session (for testing)")
    parser.add_argument("--max_retries", type=int, default=3, help="API retries")
    parser.add_argument("--timeout", type=int, default=60, help="API timeout in seconds")
    
    # Parallel processing
    parser.add_argument("--max_workers", type=int, default=3, help="Parallel sessions")
    parser.add_argument("--max_api_concurrency", type=int, default=2, help="Parallel questions per session")
    
    parser.add_argument("--embedding_model", default="all-MiniLM-L6-v2", help="Sentence-Transformer model name")
    
    args = parser.parse_args()
    
    # Configure log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    print("=" * 70)
    print("VLM Intra-Session Memory Capability Evaluator (using Naive RAG)")
    print(f"Model: {args.model}")
    print(f"API Endpoint: {args.base_url}")
    print(f"Memory system: {args.memory_type}")
    if args.memory_type == "naive_rag":
        print(f"  Chunk size: {args.chunk_size}")
        print(f"  Retrieval top_k: {args.top_k}")
    print("=" * 70)
    
    # Test mode setting
    if args.test_mode:
        args.max_questions_per_session = 2
        print("Test mode: Processing only first 2 questions per session")
    
    # 1. Initialize memory system
    print(f"\n[1] Initializing memory system ({args.memory_type})...")
    print(f"   Loading entire conversation from all sessions...")
    
    # When creating memory system
    if args.memory_type == "naive_rag":
        memory_system = NaiveRAGMemorySystem(
            args.conversations_dir, 
            chunk_size=args.chunk_size,
            top_k=args.top_k,
            embedding_model=args.embedding_model  # New
        )
    
    memory_system.load_all_conversations()
    
    # Display memory system statistics
    if args.memory_type == "naive_rag":
        stats = memory_system.get_memory_stats()
        print(f"   Loaded {stats['total_sessions']} sessions")
        print(f"   Chunking complete: {stats['total_chunks']} chunks")
        print(f"   Chunks containing images: {stats['chunks_with_images']}")
    
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
        max_workers=args.max_workers,  # New
        max_api_concurrency=args.max_api_concurrency  # New
    )
    
    # 3. Load questions
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
    
    total_questions = sum(len(data["questions"]) for data in sessions_questions.values())
    print(f"   Total questions: {total_questions}")
    
    # Limit number of sessions to process
    if args.max_sessions and args.max_sessions < len(sessions_questions):
        sessions_to_process = dict(list(sessions_questions.items())[:args.max_sessions])
        print(f"   Limiting to first {args.max_sessions} sessions")
    else:
        sessions_to_process = sessions_questions
    
    # 4. Execute evaluation
    print(f"\n[4] Starting session-by-session evaluation (using Naive RAG retrieval)...")
    print(f"   Processing {len(sessions_to_process)} sessions")
    print(f"   Total questions: {total_questions}")
    print(f"   Retrieval configuration: chunk_size={args.chunk_size}, top_k={args.top_k}")
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