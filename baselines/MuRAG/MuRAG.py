import os
import warnings
import json
import logging
import argparse
import time
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict
from natsort import natsorted
from abc import ABC, abstractmethod

# Import image processing libraries
from PIL import Image
import base64
from io import BytesIO
import requests

from transformers import AutoModel
import logging

# Import multimodal model libraries
from transformers import CLIPModel, CLIPProcessor

# Try to import tiktoken for token counting
try:
    import tiktoken
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False
    logging.warning("tiktoken not installed, using simple character-based token estimation")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==================== Data Class Definitions ====================

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
class MemoryElement:
    """Memory unit - stores content of a single dialogue turn"""
    memory_id: int
    session_id: str
    dialogue_index: int
    role: str
    text: str
    image_filename: Optional[str] = None
    image_path: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: Optional[Dict] = None
    image_id: Optional[str] = None  # Unique image ID
    
    def __post_init__(self):
        # Generate image ID
        if self.image_filename and self.session_id and self.dialogue_index:
            self.image_id = f"{self.image_filename}"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            'memory_id': self.memory_id,
            'session_id': self.session_id,
            'dialogue_index': self.dialogue_index,
            'role': self.role,
            'text': self.text,
            'image': self.image_filename,
            'image_path': self.image_path,
            'image_id': self.image_id,
            'timestamp': self.timestamp
        }
    
    def to_observation(self) -> Union[str, Dict]:
        """
        Convert to observation format suitable for encoding
        Adds role information to text to distinguish speakers during encoding
        """
        # Add role information to text
        role_text = f"[{self.role}] {self.text}"
        
        if self.image_filename and self.image_path:
            return {
                'text': role_text,  # Use text with role
                'image': {'path': self.image_path}
            }
        else:
            return role_text  # Return text with role
    
    def get_image_info(self) -> Optional[Dict]:
        """Get image information (for API calls)"""
        if not self.image_path:
            return None
        return {
            'image_path': self.image_path,
            'session_id': self.session_id,
            'dialogue_index': self.dialogue_index,
            'role': self.role,
            'filename': self.image_filename,
            'image_id': self.image_id,
            'dialogue_text': self.text
        }


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
    recall_method: str = "multimodal_rag"
    success: bool = True
    error_message: Optional[str] = None
    truncated: bool = False
    original_context_length: Optional[int] = None
    truncated_context_length: Optional[int] = None
    retrieved_memory_ids: List[int] = field(default_factory=list)
    retrieval_scores: List[float] = field(default_factory=list)
    images_limited: bool = False
    original_image_count: Optional[int] = None
    limited_image_count: Optional[int] = None
    retrieved_chunks: List[Dict] = field(default_factory=list)
    
    # New timing fields
    retrieval_time: float = 0.0      # Memory retrieval time
    images_prepare_time: float = 0.0 # Image preparation time
    prompt_build_time: float = 0.0   # Prompt building time
    api_call_time: float = 0.0       # API call time


# ==================== Token Counter ====================

class TokenCounter:
    """Token counter"""
    
    def __init__(self, model_name: str = "cl100k_base"):
        self.model_name = model_name
        self.encoding = None
        
        if TOKENIZER_AVAILABLE:
            try:
                self.encoding = tiktoken.get_encoding(model_name)
                logger.info(f"Successfully loaded tokenizer: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}, using estimation method")
    
    def count_tokens(self, text: str) -> int:
        """Count number of tokens in text"""
        if not text:
            return 0
        
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
            other_chars = len(text) - chinese_chars
            estimated_tokens = chinese_chars * 2 + other_chars * 0.25
            return int(estimated_tokens) + 1
    
    def truncate_text(self, text: str, max_tokens: int) -> Tuple[str, int, int]:
        """Truncate text to specified token count"""
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
            chars_per_token = len(text) / original_tokens
            keep_chars = int(max_tokens * chars_per_token)
            truncated_text = text[:keep_chars] + "... [content truncated]"
            truncated_tokens = self.count_tokens(truncated_text)
            return truncated_text, original_tokens, truncated_tokens


# ==================== Image Processor ====================

class ImageProcessor:
    """Image processor - thread-safe"""
    
    def __init__(self, cache_enabled: bool = True, max_size: Tuple[int, int] = (1024, 1024), quality: int = 85):
        self.cache_enabled = cache_enabled
        self.max_size = max_size
        self.quality = quality
        self.image_cache = {}
        self.image_metadata = {}
    
    def process_image(self, image_path: str, session_id: str = None, filename: str = None,
                      is_question_image: bool = False, question_id: str = None,
                      dialogue_index: int = None, role: str = None,
                      dialogue_text: str = None) -> Dict:
        """
        Process image, return dictionary with Base64 data and metadata
        
        Args:
            image_path: Path to image
            session_id: Session the image belongs to
            filename: Image filename
            is_question_image: Whether this is a question image
            question_id: Question ID (if it's a question image)
            dialogue_index: Dialogue turn index
            role: Speaker role
            dialogue_text: Corresponding dialogue text
        """
        # Check cache
        base64_data = None
        if self.cache_enabled and image_path in self.image_cache:
            base64_data = self.image_cache[image_path]
            logger.debug(f"Using cached image: {filename}")
        
        if base64_data is None:
            base64_data = self._image_to_base64(image_path)
            if self.cache_enabled:
                self.image_cache[image_path] = base64_data
        
        # Store metadata - direct assignment
        if session_id and filename:
            self.image_metadata[image_path] = {
                "session_id": session_id,
                "filename": filename
            }
        
        # Build return information
        image_info = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_data}"
            }
        }
        
        # Add markers
        if is_question_image:
            image_info["is_question_image"] = True
            image_info["question_id"] = question_id
            image_info["marker"] = f"【question_image-{question_id}】"
        else:
            # Keep complete dialogue context markers for context images
            image_info["session_id"] = session_id
            image_info["file_name"] = filename
            image_info["dialogue_index"] = dialogue_index
            image_info["role"] = role
            image_info["image_id"] = f"{filename}"
            image_info["marker"] = f"【memory_image-{session_id}-{dialogue_index}-{role}】"
            if dialogue_text:
                image_info["dialogue_text_preview"] = dialogue_text
        
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
        self.image_cache.clear()
        logger.info("Image cache cleared")



# ==================== Multimodal Encoder ====================

class BaseMultiModalEncoder(ABC):
    """Base class for multimodal encoder"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def reset(self):
        pass
    
    @abstractmethod
    def encode_text(self, text, return_type='numpy'):
        pass
    
    @abstractmethod
    def encode_image(self, image_path_or_url, return_type='numpy'):
        pass
    
    @abstractmethod
    def encode_multimodal(self, text=None, image=None, return_type='numpy'):
        pass


class CLIPEncoder(BaseMultiModalEncoder):
    """CLIP-based multimodal encoder"""
    def __init__(self, config):
        super().__init__(config)
        
        model_name = getattr(config, 'model_name', 'openai/clip-vit-base-patch32')
        logger.info(f"Loading CLIP model: {model_name}")
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        
        logger.info(f"CLIP model loaded successfully on {self.device}")
    
    def _load_image(self, image_path_or_url):
        """Load image from local path or URL."""
        try:
            if image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://'):
                response = requests.get(image_path_or_url, timeout=10)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                if not os.path.exists(image_path_or_url):
                    raise FileNotFoundError(f"Image file not found: {image_path_or_url}")
                image = Image.open(image_path_or_url).convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path_or_url}: {e}")
            return Image.new('RGB', (224, 224), color='white')
    
    def encode_text(self, text, return_type='numpy'):
        """Encode text into embeddings."""
        if not text or text.strip() == '':
            text = " "
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        if return_type == 'numpy':
            return text_features.cpu().numpy()
        elif return_type == 'tensor':
            return text_features
        else:
            raise ValueError(f"Unrecognized return type: {return_type}")
    
    def encode_image(self, image_path_or_url, return_type='numpy'):
        """Encode image into embeddings."""
        image = self._load_image(image_path_or_url)
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        if return_type == 'numpy':
            return image_features.cpu().numpy()
        elif return_type == 'tensor':
            return image_features
        else:
            raise ValueError(f"Unrecognized return type: {return_type}")
    
    def encode_multimodal(self, text=None, image=None, return_type='numpy'):
        """
        Encode multimodal data.
        If both are provided, average the embeddings.
        """
        embeddings = []
        
        if text is not None and text.strip() != '':
            text_emb = self.encode_text(text, return_type='tensor')
            embeddings.append(text_emb)
        
        if image is not None:
            if isinstance(image, dict) and 'path' in image:
                image_emb = self.encode_image(image['path'], return_type='tensor')
                embeddings.append(image_emb)
            elif isinstance(image, str):
                image_emb = self.encode_image(image, return_type='tensor')
                embeddings.append(image_emb)
        
        if not embeddings:
            return self.encode_text(" ", return_type=return_type)
        
        if len(embeddings) > 1:
            combined = torch.mean(torch.stack(embeddings), dim=0)
        else:
            combined = embeddings[0]
        
        combined = combined / combined.norm(dim=-1, keepdim=True)
        
        if return_type == 'numpy':
            return combined.cpu().numpy()
        elif return_type == 'tensor':
            return combined
        else:
            raise ValueError(f"Unrecognized return type: {return_type}")
    
    def __call__(self, obj, return_type='numpy'):
        """Main entry point"""
        if isinstance(obj, str):
            return self.encode_text(obj, return_type)
        elif isinstance(obj, dict):
            text = obj.get('text', '')
            image = obj.get('image', None)
            return self.encode_multimodal(text, image, return_type)
        else:
            raise ValueError(f"Unsupported input type: {type(obj)}")


class GMEEncoder(BaseMultiModalEncoder):
    """
    GME (General Multimodal Embedding) Qwen2-VL-based encoder for text and images.
    Supports unified multimodal representations for Any2Any Search.
    """
    def __init__(self, config):
        super().__init__(config)
        
        model_name = getattr(config, 'model_name', 'Alibaba-NLP/gme-Qwen2-VL-7B-Instruct')
        # Also supports specifying via 'path' parameter
        if hasattr(config, 'path'):
            model_name = config.path
            
        logger.info(f"Loading GME model: {model_name}")
        
        # Load GME model with trust_remote_code=True
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
            device_map='auto',
            trust_remote_code=True
        )
        self.model.eval()  # Set to evaluation mode
        
        # Get model dimension
        self.dimension = self.model.config.hidden_size
        logger.info(f"GME model loaded successfully on {self.device}")
        logger.info(f"Model dimension: {self.dimension}")
    
    def _load_image(self, image_path_or_url):
        """Load image from local path or URL."""
        # If already absolute path or URL, use directly
        if os.path.isabs(image_path_or_url) or image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://'):
            final_path = image_path_or_url
        else:
            # Only add prefix for relative paths
            # You may need to set the default image root directory
            final_path = os.path.join("/path/to/your/images", image_path_or_url)  # Modify to your image path
        
        try:
            if final_path.startswith('http://') or final_path.startswith('https://'):
                # Load from URL
                response = requests.get(final_path, timeout=10)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                # Load from local path
                if not os.path.exists(final_path):
                    raise FileNotFoundError(f"Image file not found: {final_path}")
                image = Image.open(final_path).convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path_or_url} (tried {final_path}): {e}")
            # Return a blank image as fallback
            return Image.new('RGB', (224, 224), color='white')
    
    def encode_text(self, text, return_type='numpy'):
        """Encode text into embeddings."""
        if not text or text.strip() == '':
            text = " "
        
        with torch.no_grad():
            text_emb = self.model.get_text_embeddings(texts=[text])
            # Normalize
            text_emb = text_emb / torch.norm(text_emb, dim=-1, keepdim=True)
        
        if return_type == 'numpy':
            return text_emb.cpu().numpy()
        elif return_type == 'tensor':
            return text_emb
        else:
            raise ValueError(f"Unrecognized return type: {return_type}")
    
    def encode_image(self, image_path_or_url, return_type='numpy'):
        """Encode image into embeddings."""
        image = self._load_image(image_path_or_url)
        
        with torch.no_grad():
            # GME API: get_image_embeddings expects images as list of PIL Image
            image_emb = self.model.get_image_embeddings(images=[image])
            # Normalize
            image_emb = image_emb / torch.norm(image_emb, dim=-1, keepdim=True)
        
        if return_type == 'numpy':
            return image_emb.cpu().numpy()
        elif return_type == 'tensor':
            return image_emb
        else:
            raise ValueError(f"Unrecognized return type: {return_type}")
    
    def encode_multimodal(self, text=None, image=None, return_type='numpy'):
        """
        Encode multimodal data (text and/or image).
        GME supports single-modal (text/image) and fused-modal embeddings.
        If both are provided, we use fused-modal embedding for better representation.
        """
        # Determine which encoding method to use
        has_text = text is not None and text.strip() != ''
        has_image = image is not None
        if has_text and has_image:
            # Fused-modal embedding: best for multimodal representation
            if isinstance(image, dict) and 'path' in image:
                image_path = image['path']
            else:
                image_path = image
            
            loaded_image = self._load_image(image_path)
            with torch.no_grad():
                fused_emb = self.model.get_fused_embeddings(texts=[text], images=[loaded_image])
                fused_emb = fused_emb / torch.norm(fused_emb, dim=-1, keepdim=True)
            
            if return_type == 'numpy':
                return fused_emb.cpu().numpy()
            elif return_type == 'tensor':
                return fused_emb
            else:
                raise ValueError(f"Unrecognized return type: {return_type}")
        
        elif has_text:
            # Single-modal text embedding
            return self.encode_text(text, return_type)
        
        elif has_image:
            # Single-modal image embedding
            if isinstance(image, dict) and 'path' in image:
                return self.encode_image(image['path'], return_type)
            else:
                return self.encode_image(image, return_type)
        
        else:
            # If both are empty, encode empty text
            return self.encode_text(" ", return_type)
    
    def __call__(self, obj, return_type='numpy'):
        """
        Main entry point. obj can be:
        - str: treated as text
        - dict with 'text' and/or 'image' keys
        """
        if isinstance(obj, str):
            return self.encode_text(obj, return_type)
        elif isinstance(obj, dict):
            text = obj.get('text', '')
            image = obj.get('image', None)
            return self.encode_multimodal(text, image, return_type)
        else:
            raise ValueError(f"Unsupported input type: {type(obj)}")


# ==================== Configuration Classes ====================

class Config:
    """Simple configuration class"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class EncoderConfig:
    """Encoder configuration"""
    def __init__(self, method='GMEEncoder', model_name='BAAI/bge-m3'):  # Default changed to GME
        self.method = method
        self.model_name = model_name


class RetrievalConfig:
    """Retrieval configuration"""
    def __init__(self, mode='cosine', topk=10):
        self.mode = mode
        self.topk = topk
        self.encoder = EncoderConfig()


class UtilizationConfig:
    """Utilization strategy configuration"""
    def __init__(self, method='ConcateUtilization', max_tokens=4000):
        self.method = method
        self.max_tokens = max_tokens


class TruncationConfig:
    """Truncation configuration"""
    def __init__(self, method='SimpleTruncation', max_tokens=4000):
        self.method = method
        self.max_tokens = max_tokens


# ==================== Storage Classes ====================

class BaseStore:
    """Base storage class"""
    def __init__(self, config):
        self.config = config
    
    def reset(self):
        pass


class SimpleStorage(BaseStore):
    """Simple in-memory storage"""
    def __init__(self, config):
        super().__init__(config)
        self.memories = []  # Store MemoryElement objects
        self.memory_counter = 0
    
    def reset(self):
        self.memories = []
        self.memory_counter = 0
    
    def is_empty(self) -> bool:
        return len(self.memories) == 0
    
    def add(self, observation, metadata=None) -> int:
        """
        Add memory
        
        Args:
            observation: MemoryElement object or dict
            metadata: Additional metadata
        
        Returns:
            Memory ID
        """
        memory_id = self.memory_counter
        self.memory_counter += 1
        
        if isinstance(observation, MemoryElement):
            memory = observation
            memory.memory_id = memory_id
        else:
            # Create MemoryElement
            memory = MemoryElement(
                memory_id=memory_id,
                session_id=metadata.get('session_id', 'unknown') if metadata else 'unknown',
                dialogue_index=metadata.get('dialogue_index', 0) if metadata else 0,
                role=metadata.get('role', 'unknown') if metadata else 'unknown',
                text=observation if isinstance(observation, str) else observation.get('text', ''),
                image_filename=metadata.get('image_filename') if metadata else None,
                image_path=metadata.get('image_path') if metadata else None,
                metadata=metadata
            )
        self.memories.append(memory)
        return memory_id
    
    def get_memory_by_id(self, memory_id: int) -> Optional[MemoryElement]:
        """Get memory by ID"""
        if 0 <= memory_id < len(self.memories):
            return self.memories[memory_id]
        return None
    
    def get_memory_element_by_mid(self, mid: int) -> Optional[Dict]:
        """Compatibility interface: get memory by mid (returns dictionary format)"""
        memory = self.get_memory_by_id(mid)
        if memory:
            return memory.to_dict()
        return None
    
    def get_all_memories(self) -> List[MemoryElement]:
        return self.memories
    
    def __len__(self):
        return len(self.memories)


# ==================== Multimodal Retriever ====================

class MultiModalRetrieval:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize encoder
        encoder_method = getattr(config.encoder, 'method', 'GMEEncoder')  # Default to GMEEncoder
        if encoder_method == 'CLIPEncoder':
            self.encoder = CLIPEncoder(config.encoder)
        elif encoder_method == 'GMEEncoder':  # Add GMEEncoder support
            self.encoder = GMEEncoder(config.encoder)
        else:
            raise ValueError(f"Unsupported encoder method: {encoder_method}")
        
        # Get encoder output dimension
        self.encoder_dim = getattr(self.encoder, 'dimension', 1024)
        logger.info(f"Encoder output dimension: {self.encoder_dim}")
        
        # Store embedding vectors for all memories
        self.tensorstore = None
        
        # Store metadata for each memory
        self.memory_metadata = []
        
        # Store mapping from memory ID to index
        self.id_to_index = {}
        self.index_to_id = []
        
        # Last retrieval scores and indices
        self.last_scores = None
        self.last_indices = None
    
    def reset(self):
        self.tensorstore = None
        self.memory_metadata = []
        self.id_to_index = {}
        self.index_to_id = []
        self.last_scores = None
        self.last_indices = None
    
    def __normalize__(self, embedding):
        return torch.nn.functional.normalize(embedding, dim=-1)
    
    def add(self, obj, memory_id: int = None):
        """
        Add memory embedding to retriever
        
        Args:
            obj: str or dict {'text': ..., 'image': ...}
            memory_id: Memory ID (if None, use current index)
        
        Returns:
            Embedding vector
        """
        embedding = self.encoder(obj, return_type='tensor')
        
        if self.config.mode == 'cosine':
            embedding = self.__normalize__(embedding)
        
        # Determine index
        if self.tensorstore is None:
            self.tensorstore = embedding
            index = 0
        else:
            self.tensorstore = torch.cat([self.tensorstore, embedding], dim=0)
            index = self.tensorstore.size(0) - 1
        
        # Store metadata
        metadata = {
            'has_text': isinstance(obj, str) or (isinstance(obj, dict) and 'text' in obj and obj['text']),
            'has_image': isinstance(obj, dict) and 'image' in obj and obj['image']
        }
        self.memory_metadata.append(metadata)
        
        # Record ID mapping
        if memory_id is not None:
            self.id_to_index[memory_id] = index
            # Ensure index_to_id is long enough
            while len(self.index_to_id) <= index:
                self.index_to_id.append(None)
            self.index_to_id[index] = memory_id
        else:
            memory_id = index
            self.id_to_index[memory_id] = index
            while len(self.index_to_id) <= index:
                self.index_to_id.append(None)
            self.index_to_id[index] = memory_id
        
        return embedding
    
    def __calculate_scores__(self, query):
        """
        Calculate similarity scores between query and all stored memories
        """
        query_embedding = self.encoder(query, return_type='tensor')
        
        if self.config.mode == 'cosine':
            query_embedding = self.__normalize__(query_embedding)
        
        if self.config.mode in ['cosine', 'dot']:
            scores = torch.matmul(self.tensorstore, query_embedding.squeeze())
        elif self.config.mode == 'L2':
            scores = - torch.norm(
                self.tensorstore - query_embedding.squeeze(), 
                p=2, 
                dim=1
            )
        else:
            raise ValueError(f"Unrecognized mode: {self.config.mode}")
        
        return scores
    
    def __call__(self, query, topk='config', with_score=False, sort=True, return_ids=True):
        """
        Search for memories most similar to the query
        
        Args:
            query: Query (string or dictionary)
            topk: Number of results to return
            with_score: Whether to return scores
            sort: Whether to sort
            return_ids: Whether to return memory IDs (otherwise return indices)
        
        Returns:
            List of memory IDs or (scores, indices) tuple
        """
        if self.tensorstore is None or self.tensorstore.size(0) == 0:
            return torch.tensor([]) if not with_score else (torch.tensor([]), torch.tensor([]))
        
        scores = self.__calculate_scores__(query)
        self.last_scores = scores
        
        if sort:
            scores, indices = torch.sort(scores, descending=True)
            self.last_indices = indices
        else:
            indices = torch.arange(self.tensorstore.size(0))
            self.last_indices = indices
        
        # Determine number of results to return
        if topk is False:
            pass
        elif topk == 'config':
            k = min(self.config.topk, self.tensorstore.size(0))
            scores = scores[:k]
            indices = indices[:k]
        elif isinstance(topk, int):
            k = min(topk, self.tensorstore.size(0))
            scores = scores[:k]
            indices = indices[:k]
        
        if return_ids:
            # Convert indices to memory IDs
            ids = []
            for idx in indices:
                idx_item = idx.item() if torch.is_tensor(idx) else idx
                if idx_item < len(self.index_to_id) and self.index_to_id[idx_item] is not None:
                    ids.append(self.index_to_id[idx_item])
                else:
                    ids.append(idx_item)
            indices = torch.tensor(ids)
        
        if with_score:
            return scores, indices
        else:
            return indices
    
    def get_retrieved_chunks_info(self, query, topk=None) -> List[Dict]:
        """
        Get detailed information about retrieved chunks, including similarity scores and metadata
        """
        if self.tensorstore is None or self.tensorstore.size(0) == 0:
            return []
        
        # Get retrieval results with scores
        if topk is None:
            topk = self.config.topk
        
        scores, indices = self.__call__(query, topk=topk, with_score=True, sort=True, return_ids=True)
        
        retrieved_chunks = []
        for i, (score, idx) in enumerate(zip(scores, indices)):
            memory_id = idx.item() if torch.is_tensor(idx) else idx
            chunk_info = {
                "chunk_index": i + 1,  # Sequence number in retrieval results
                "similarity": float(score),
                "embedding_similarity": True
            }
            
            # Get detailed memory information
            if hasattr(self, 'storage') and self.storage:
                mem = self.storage.get_memory_by_id(memory_id)
                if mem:
                    chunk_info["session_id"] = mem.session_id
                    chunk_info["dialogue_indices"] = [mem.dialogue_index]
                    chunk_info["has_image"] = mem.image_path is not None
            
            retrieved_chunks.append(chunk_info)
        
        return retrieved_chunks
    
    def update(self, index, obj, memory_id=None):
        """Update embedding at specified position"""
        embedding = self.encoder(obj, return_type='tensor')
        
        if self.config.mode == 'cosine':
            embedding = self.__normalize__(embedding)
        
        self.tensorstore[index] = embedding.squeeze()
        
        if index < len(self.memory_metadata):
            self.memory_metadata[index] = {
                'has_text': isinstance(obj, str) or (isinstance(obj, dict) and 'text' in obj and obj['text']),
                'has_image': isinstance(obj, dict) and 'image' in obj and obj['image']
            }
        
        if memory_id is not None and index < len(self.index_to_id):
            self.index_to_id[index] = memory_id
            self.id_to_index[memory_id] = index
    
    def delete(self, index):
        """Delete embedding at specified position"""
        if index >= self.tensorstore.size(0):
            return
        
        memory_id = self.index_to_id[index] if index < len(self.index_to_id) else None
        
        self.tensorstore = torch.cat([
            self.tensorstore[:index], 
            self.tensorstore[index+1:]
        ])
        
        if index < len(self.memory_metadata):
            self.memory_metadata.pop(index)
        
        if memory_id is not None and memory_id in self.id_to_index:
            del self.id_to_index[memory_id]
        
        self.index_to_id.pop(index)
        for i in range(index, len(self.index_to_id)):
            if self.index_to_id[i] is not None:
                self.id_to_index[self.index_to_id[i]] = i
    
    def get_tensor_by_ids(self, id_list):
        """Get embedding tensor by list of IDs"""
        indices = []
        for mid in id_list:
            if mid in self.id_to_index:
                indices.append(self.id_to_index[mid])
        if indices:
            return self.tensorstore[indices]
        return None


# ==================== Utilization Strategies ====================

class BaseUtilization:
    """Base class for utilization strategy"""
    def __init__(self, config):
        self.config = config
    
    def reset(self):
        pass
    
    def __call__(self, memories):
        raise NotImplementedError


class ConcateUtilization(BaseUtilization):
    """Simple concatenation utilization strategy - returns text string"""
    def __init__(self, config):
        super().__init__(config)
        self.token_counter = TokenCounter()
    
    def __call__(self, memories):
        """
        Concatenate list of memories into text
        
        Args:
            memories: List of MemoryElement objects or dictionaries
        
        Returns:
            Concatenated text string
        """
        if not memories:
            return "No relevant memories"
        
        context_parts = []
        
        for i, mem in enumerate(memories):
            if isinstance(mem, dict):
                session_id = mem.get('session_id', 'unknown')
                role = mem.get('role', 'unknown')
                text = mem.get('text', '')
                image = mem.get('image', '')
                dialogue_index = mem.get('dialogue_index', i+1)
                image_id = mem.get('image_id', '')
                
                prefix = f"[Memory{i+1}] Session {session_id} Turn {dialogue_index} - {role}: "
                if image:
                    prefix += f"[Image: {image_id}] "
                context_parts.append(prefix + text)
            elif hasattr(mem, 'to_dict'):
                mem_dict = mem.to_dict()
                session_id = mem_dict.get('session_id', 'unknown')
                role = mem_dict.get('role', 'unknown')
                text = mem_dict.get('text', '')
                image = mem_dict.get('image', '')
                image_id = mem_dict.get('image_id', '')
                dialogue_index = mem_dict.get('dialogue_index', i+1)
                
                prefix = f"[Memory{i+1}] Session {session_id} Turn {dialogue_index} - {role}: "
                if image:
                    prefix += f"[Image: {image_id}] "
                context_parts.append(prefix + text)
            else:
                context_parts.append(str(mem))
        
        full_text = "\n".join(context_parts)
        token_count = self.token_counter.count_tokens(full_text)
        
        if token_count > getattr(self.config, 'max_tokens', 4000):
            logger.debug(f"Memory text exceeds token limit ({token_count} > {self.config.max_tokens}), truncating")
            truncated_text, _, _ = self.token_counter.truncate_text(
                full_text, 
                getattr(self.config, 'max_tokens', 4000)
            )
            return truncated_text
        
        return full_text


class MultiModalUtilization(BaseUtilization):
    """Multimodal utilization strategy - returns dictionary containing text and images"""
    def __init__(self, config):
        super().__init__(config)
        self.token_counter = TokenCounter()
    
    def __call__(self, memories):
        """
        Organize list of memories into multimodal format
        
        Args:
            memories: List of MemoryElement objects or dictionaries
        
        Returns:
            Dictionary: {'text': str, 'images': list of image paths, 'memory_objects': list}
        """
        if not memories:
            return {'text': 'No relevant memories', 'images': [], 'memory_objects': []}
        
        text_parts = []
        image_paths = []
        memory_objects = []
        
        for i, mem in enumerate(memories):
            if isinstance(mem, dict):
                session_id = mem.get('session_id', 'unknown')
                role = mem.get('role', 'unknown')
                text = mem.get('text', '')
                image = mem.get('image', '')
                image_id = mem.get('image_id', '')
                dialogue_index = mem.get('dialogue_index', i+1)
                
                prefix = f"[Memory{i+1}] Session {session_id} Turn {dialogue_index} - {role}: "
                if image:
                    prefix += f"[Image: {image_id}] "
                text_parts.append(prefix + text)
                
                if mem.get('image_path') and os.path.exists(mem['image_path']):
                    image_paths.append(mem['image_path'])
                memory_objects.append(mem)
            elif hasattr(mem, 'to_dict'):
                mem_dict = mem.to_dict()
                session_id = mem_dict.get('session_id', 'unknown')
                role = mem_dict.get('role', 'unknown')
                text = mem_dict.get('text', '')
                image = mem_dict.get('image', '')
                image_id = mem_dict.get('image_id', '')
                dialogue_index = mem_dict.get('dialogue_index', i+1)
                
                prefix = f"[Memory{i+1}] Session {session_id} Turn {dialogue_index} - {role}: "
                if image:
                    prefix += f"[Image: {image_id}] "
                text_parts.append(prefix + text)
                if mem.image_path and os.path.exists(mem.image_path):
                    image_paths.append(mem.image_path)
                memory_objects.append(mem)
        
        full_text = "\n".join(text_parts)
        
        token_count = self.token_counter.count_tokens(full_text)
        if token_count > getattr(self.config, 'max_tokens', 4000):
            logger.debug(f"Memory text exceeds token limit ({token_count} > {self.config.max_tokens}), truncating")
            full_text, _, _ = self.token_counter.truncate_text(
                full_text, 
                getattr(self.config, 'max_tokens', 4000)
            )
        
        return {
            'text': full_text,
            'images': image_paths,
            'memory_objects': memory_objects
        }


# ==================== Truncation Strategies ====================

class BaseTruncation:
    """Base class for truncation strategy"""
    def __init__(self, config):
        self.config = config
    
    def reset(self):
        pass
    
    def __call__(self, text):
        raise NotImplementedError


class SimpleTruncation(BaseTruncation):
    """Simple truncation strategy"""
    def __init__(self, config):
        super().__init__(config)
        self.token_counter = TokenCounter()
    
    def __call__(self, text):
        """Truncate text to specified length"""
        if not text:
            return text
        
        max_tokens = getattr(self.config, 'max_tokens', 4000)
        truncated_text, _, _ = self.token_counter.truncate_text(text, max_tokens)
        return truncated_text


# ==================== Multimodal Memory Recall ====================

class MMMemoryRecall:
    """Multimodal memory recall"""
    def __init__(self, config, **kwargs):
        self.config = config
        
        self.storage = kwargs['storage']
        
        # Initialize truncation strategy
        truncation_method = getattr(config.truncation, 'method', 'SimpleTruncation')
        if truncation_method == 'SimpleTruncation':
            self.truncation = SimpleTruncation(config.truncation)
        else:
            self.truncation = SimpleTruncation(config.truncation)
        
        # Initialize utilization strategy
        utilization_method = getattr(config.utilization, 'method', 'ConcateUtilization')
        if utilization_method == 'ConcateUtilization':
            self.utilization = ConcateUtilization(config.utilization)
        elif utilization_method == 'MultiModalUtilization':
            self.utilization = MultiModalUtilization(config.utilization)
        else:
            self.utilization = ConcateUtilization(config.utilization)
        
        # Initialize multimodal retriever
        self.multimodal_retrieval = kwargs['multimodal_retrieval']
        
        # Add storage reference to retriever for detailed information
        self.multimodal_retrieval.storage = self.storage
        
        # Record last retrieved IDs and details
        self.last_retrieved_ids = []
        self.last_retrieved_chunks = []
    
    def reset(self):
        self.last_retrieved_ids = []
        self.last_retrieved_chunks = []
    
    def __call__(self, query, topk=None) -> Dict:
        """
        Retrieve memories based on multimodal query
        
        Returns:
            Dictionary containing:
            - 'text': Formatted text context
            - 'images': List of image paths
            - 'memory_objects': List of complete memory objects
            - 'retrieved_chunks': Detailed information about retrieved chunks
        """
        if self.storage.is_empty():
            self.last_retrieved_ids = []
            self.last_retrieved_chunks = []
            return {
                'text': self.utilization([]),
                'images': [],
                'memory_objects': [],
                'retrieved_chunks': []
            }
        
        # Get detailed information about retrieved chunks
        retrieved_chunks = self.multimodal_retrieval.get_retrieved_chunks_info(query, topk)
        self.last_retrieved_chunks = retrieved_chunks
        
        # Extract memory IDs
        ranking_ids = [chunk.get('memory_id') for chunk in retrieved_chunks if 'memory_id' in chunk]
        if not ranking_ids:
            # If no memory_id, try alternative method
            if topk is None:
                ranking_ids = self.multimodal_retrieval(query, topk='config')
            else:
                ranking_ids = self.multimodal_retrieval(query, topk=topk)
            
            if torch.is_tensor(ranking_ids):
                ranking_ids = ranking_ids.tolist()
        
        if len(ranking_ids) == 0:
            self.last_retrieved_ids = []
            return {
                'text': self.utilization([]),
                'images': [],
                'memory_objects': [],
                'retrieved_chunks': retrieved_chunks
            }
        
        # Collect memory objects
        memories = []
        memory_objects = []
        retrieved_ids = []
        image_paths = []
        
        for i, mid in enumerate(ranking_ids):
            mem = self.storage.get_memory_by_id(int(mid))
            if mem:
                memories.append(mem)
                memory_objects.append(mem)
                retrieved_ids.append(mem.dialogue_index)
                if mem.image_path:
                    image_paths.append(mem.image_path)
                
                # If retrieved_chunks doesn't have memory_id yet, add it
                if i < len(retrieved_chunks) and 'memory_id' not in retrieved_chunks[i]:
                    retrieved_chunks[i]['memory_id'] = mem.memory_id
                    retrieved_chunks[i]['session_id'] = mem.session_id
                    retrieved_chunks[i]['dialogue_indices'] = [mem.dialogue_index]
                    retrieved_chunks[i]['has_image'] = mem.image_path is not None
        
        # Record retrieved IDs
        self.last_retrieved_ids = retrieved_ids
        
        # Format results using utilization strategy
        result = self.utilization(memories)
        
        # Ensure return format contains complete information
        if isinstance(result, str):
            return {
                'text': self.truncation(result),
                'images': image_paths,
                'memory_objects': memory_objects,
                'retrieved_chunks': retrieved_chunks
            }
        elif isinstance(result, dict):
            result['text'] = self.truncation(result.get('text', ''))
            result['images'] = result.get('images', image_paths)
            result['memory_objects'] = memory_objects
            result['retrieved_chunks'] = retrieved_chunks
            return result
        else:
            return {
                'text': str(result),
                'images': image_paths,
                'memory_objects': memory_objects,
                'retrieved_chunks': retrieved_chunks
            }


# ==================== Multimodal RAG Memory System ====================

class MultiModalRAGMemorySystem:
    """
    Multimodal RAG Memory System
    - Uses CLIP encoder to encode dialogue content (text + images) into vectors
    - Stores vectors in retriever
    - Retrieves most relevant memories for queries
    - Supports multimodal querying and recall
    """
    
    def __init__(self, conversations_dir: str, config: Dict = None):
        self.conversations_dir = conversations_dir
        self.all_dialogues = []  # Raw data for all dialogues
        self.session_info = {}    # Session additional information
        self.image_paths = {}      # Image path mapping {session_id: {image_filename: full_path}}
        
        # Image processor
        self.image_processor = ImageProcessor()
        
        # Storage time recording
        self.storage_time = 0.0      # Total storage time
        self.loading_time = 0.0      # Data loading time
        self.encoding_time = 0.0     # Vector encoding time
        
        # Default configuration
        default_config = {
            'encoder': {
                'method': 'GMEEncoder',
                'model_name': 'BAAI/bge-m3'
            },
            'retrieval': {
                'mode': 'cosine',
                'topk': 10
            },
            'utilization': {
                'method': 'MultiModalUtilization',
                'max_tokens': 4096
            },
            'truncation': {
                'method': 'SimpleTruncation',
                'max_tokens': 4096
            }
        }
        
        if config:
            # Merge configuration
            for key, value in config.items():
                if key in default_config and isinstance(value, dict):
                    default_config[key].update(value)
                else:
                    default_config[key] = value
        
        # Create configuration objects
        self.encoder_config = EncoderConfig(**default_config['encoder'])
        self.retrieval_config = RetrievalConfig(**default_config['retrieval'])
        self.retrieval_config.encoder = self.encoder_config
        
        self.utilization_config = UtilizationConfig(**default_config['utilization'])
        self.truncation_config = TruncationConfig(**default_config['truncation'])
        
        # Initialize components
        self._init_components()
        
        logger.info(f"Multimodal RAG memory system initialization complete")
        logger.info(f"  Encoder: {self.encoder_config.method} ({self.encoder_config.model_name})")
        logger.info(f"  Retrieval mode: {self.retrieval_config.mode}, top-k: {self.retrieval_config.topk}")
        logger.info(f"  Utilization strategy: {self.utilization_config.method}")
    
    def _init_components(self):
        """Initialize all RAG components"""
        # Create storage
        self.storage = SimpleStorage(Config())
        
        # Create multimodal retriever
        self.multimodal_retrieval = MultiModalRetrieval(self.retrieval_config)
        
        # Create recaller
        recall_kwargs = {
            'storage': self.storage,
            'multimodal_retrieval': self.multimodal_retrieval
        }
        self.recall = MMMemoryRecall(
            Config(
                truncation=self.truncation_config,
                utilization=self.utilization_config
            ),
            **recall_kwargs
        )
    
    def load_all_conversations(self):
        """Load all conversations and encode for storage - with timing"""
        overall_start = time.time()
        
        scenes_dir = os.path.join(self.conversations_dir, "scenes")
        if not os.path.exists(scenes_dir):
            raise ValueError(f"Scenes directory not found: {scenes_dir}")
        
        session_dirs = natsorted([
            d for d in os.listdir(scenes_dir) 
            if os.path.isdir(os.path.join(scenes_dir, d))
        ])
        
        total_memories = 0
        
        # 1. Data loading time
        loading_start = time.time()
        
        for session_dir_name in session_dirs:
            session_dir = os.path.join(scenes_dir, session_dir_name)
            
            # Scan image directory
            image_dir = os.path.join(session_dir, "image")
            if os.path.exists(image_dir):
                self.image_paths[session_dir_name] = {}
                for img_file in os.listdir(image_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                        self.image_paths[session_dir_name][img_file] = os.path.join(image_dir, img_file)
                logger.debug(f"Session {session_dir_name} scanned {len(self.image_paths[session_dir_name])} images")
            
            # Load session data
            session_memories = self._load_and_encode_session(session_dir_name, session_dir)
            if session_memories > 0:
                total_memories += session_memories
        
        self.loading_time = time.time() - loading_start
        logger.info(f"Data loading time: {self.loading_time:.2f} seconds")
        
        # 2. Encoding time (already included in _load_and_encode_session)
        # Record total storage time here
        self.storage_time = time.time() - overall_start
        
        logger.info(f"Loaded {len(self.session_info)} sessions, total {len(self.all_dialogues)} dialogue turns")
        logger.info(f"Encoded and stored {total_memories} memory vectors")
        logger.info(f"Memory storage total time: {self.storage_time:.2f}s (Loading: {self.loading_time:.2f}s)")
        
        # Count memories containing images
        memories_with_images = sum(1 for mem in self.storage.get_all_memories() if mem.image_path)
        logger.info(f"Memories containing images: {memories_with_images}")
    
    def _load_and_encode_session(self, session_dir_name: str, session_dir: str) -> int:
        """Load a single session and encode for storage - with encoding time recording"""
        conversation_file = os.path.join(session_dir, "session.json")
        if not os.path.exists(conversation_file):   
            logger.warning(f"session.json file not found: {conversation_file}")
            return 0
        
        try:
            with open(conversation_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            session_id = session_dir_name
            
            # Store session information
            self.session_info[session_id] = {
                "session_dir_name": session_dir_name,
                "session_title": session_data.get("session_title", ""),
                "timeline_date": session_data.get("timeline_date", ""),
                "generated_at": session_data.get("generated_at", ""),
                "dialogue_count": 0,
                "has_image_dir": os.path.exists(os.path.join(session_dir, "image")),
                "session_path": str(session_dir)
            }
            timeline_date = session_data.get("timeline_date", "")
            # Process dialogues
            dialogues = session_data.get("dialogue", [])
            dialogue_count = 0
            
            # Record encoding time
            encode_start = time.time()
            
            for i, dialogue in enumerate(dialogues, 1):
                role = dialogue.get("role", "")
                content = dialogue.get("content", {})
                text = timeline_date + ":" + content.get("text", "")
                image_filename = content.get("image", "")
                
                # Get image path
                image_path = None
                if image_filename and session_dir_name in self.image_paths:
                    if image_filename in self.image_paths[session_dir_name]:
                        image_path = self.image_paths[session_dir_name][image_filename]
                # Create memory element
                memory = MemoryElement(
                    memory_id=0,  # Will be assigned when adding
                    session_id=session_id,
                    dialogue_index=i,
                    role=role,
                    text=text,
                    image_filename=image_filename,
                    image_path=image_path,
                    timestamp=session_data.get("timeline_date", "")
                )
                
                # Store in storage
                memory_id = self.storage.add(memory)
                
                # Encode and add to retriever
                observation = memory.to_observation()
                self.multimodal_retrieval.add(observation, memory_id=memory_id)
                
                # Record in all_dialogues (for statistics)
                dialogue_with_session = {
                    "session_id": session_id,
                    "session_title": session_data.get("session_title", ""),
                    "timeline_date": session_data.get("timeline_date", ""),
                    "session_dir_name": session_dir_name,
                    "dialogue_index": i,
                    "role": role,
                    "content": content,
                    "memory_id": memory_id,
                    "image_id": memory.image_id
                }
                self.all_dialogues.append(dialogue_with_session)
                
                dialogue_count += 1
            
            # Accumulate encoding time
            self.encoding_time += time.time() - encode_start
            
            # Update session information
            self.session_info[session_id]["dialogue_count"] = dialogue_count
            
            logger.debug(f"Successfully loaded and encoded {session_dir_name}, total {dialogue_count} memories")
            return dialogue_count
            
        except Exception as e:
            logger.error(f"Failed to load {conversation_file}: {e}")
            return 0
    
    def get_image_path(self, session_id: str, image_filename: str) -> Optional[str]:
        """Get complete path of an image"""
        if session_id in self.image_paths and image_filename in self.image_paths[session_id]:
            return self.image_paths[session_id][image_filename]
        
        for sid, info in self.session_info.items():
            if sid == session_id:
                dir_name = info.get("session_dir_name")
                if dir_name in self.image_paths and image_filename in self.image_paths[dir_name]:
                    return self.image_paths[dir_name][image_filename]
        
        return None
    
    def get_image_for_api(self, image_filename: str, session_id: str = None, 
                          dialogue_index: int = None, role: str = None,
                          dialogue_text: str = None,
                          is_question_image: bool = False, 
                          question_id: str = None) -> Optional[Dict]:
        """
        Get processed image data for API calls
        Mimics the get_image_for_api method from new code
        """
        # Get image path
        if session_id:
            image_path = self.get_image_path(session_id, image_filename)
        else:
            # Try to find in all sessions
            image_path = self.get_image_path(question_id, image_filename) if question_id else None
        print(image_path)
        if not image_path or not os.path.exists(image_path):
            logger.warning(f"Image file does not exist: {image_filename}")
            return None
        
        # Use ImageProcessor to process image
        return self.image_processor.process_image(
            image_path=image_path,
            session_id=session_id if not is_question_image else None,
            filename=image_filename,
            is_question_image=is_question_image,
            question_id=question_id,
            dialogue_index=dialogue_index,
            role=role,
            dialogue_text=dialogue_text
        )
    
    def retrieve_relevant_memories(self, query: Union[str, Dict], topk: int = None) -> Dict:
        """
        Retrieve memories relevant to the query
        
        Args:
            query: Query (text or dictionary containing text/image)
            topk: Number of results to return
        
        Returns:
            Dictionary containing text, image paths, and memory objects
        """
        return self.recall(query, topk)
    
    def get_memory_by_id(self, memory_id: int) -> Optional[Dict]:
        """Get memory by ID"""
        mem = self.storage.get_memory_element_by_mid(memory_id)
        return mem
    
    def get_session_context(self, target_session_id: str) -> Dict[str, Any]:
        """Get session context"""
        return {
            "target_session_id": target_session_id,
            "target_session_info": self.session_info.get(target_session_id, {}),
            "all_sessions": list(self.session_info.keys()),
            "total_dialogues": len(self.all_dialogues),
            "memory_system_type": "MultiModalRAG"
        }
    
    def get_full_memory_context(self) -> Dict[str, Any]:
        """Get complete memory information (metadata, not content)"""
        return {
            "all_sessions": list(self.session_info.keys()),
            "session_info": self.session_info,
            "total_dialogues": len(self.all_dialogues),
            "total_memories": len(self.storage),
            "retrieval_config": {
                "mode": self.retrieval_config.mode,
                "topk": self.retrieval_config.topk
            },
            "utilization_method": self.utilization_config.method
        }
    
    def get_statistics(self) -> Dict:
        """Get memory system statistics - with storage time"""
        memories = self.storage.get_all_memories()
        memories_with_images = sum(1 for m in memories if m.image_path)
        
        return {
            "total_sessions": len(self.session_info),
            "total_memories": len(memories),
            "memories_with_images": memories_with_images,
            "retrieval_mode": self.retrieval_config.mode,
            "topk": self.retrieval_config.topk,
            "utilization_method": self.utilization_config.method,
            "encoder_model": self.encoder_config.model_name,
            # Storage time statistics
            "storage_time": self.storage_time,
            "loading_time": self.loading_time,
            "encoding_time": self.encoding_time,
            "avg_time_per_memory": self.storage_time / len(memories) if memories else 0,
            "avg_encoding_time_per_memory": self.encoding_time / len(memories) if memories else 0
        }


# ==================== Prompt Template ====================

class MURAGPromptTemplate:
    """Standardized prompt template for MURAG retrieval method"""
    
    TEMPLATE = """You are a multimodal memory testing system using the MURAG method to retrieve relevant conversation chunks. {instruction}

Note: The conversation is between two people communicating in a specific scenario.

IMPORTANT: 
1. Provide only the answer without extensive reasoning. Keep answers concise.
2. Keep your answer within 100 words. Short answers are acceptable.
3. Answer in English. This is a strict requirement. Do not answer in any other language.

[Retrieved Relevant Memories] (may be incomplete)
{context}
{image_section}

[Question]
{question}
{image_note}

{format_requirement}

Please answer based on the above memory content (appropriate reasoning is allowed):"""

    def __init__(self, instruction: str, context: str, question: str, 
                 format_requirement: str, memory_images_count: int = 0, 
                 question_images_count: int = 0, has_question_images: bool = False):
        self.instruction = instruction
        self.context = context
        self.question = question
        self.format_requirement = format_requirement
        self.memory_images_count = memory_images_count
        self.question_images_count = question_images_count
        self.has_question_images = has_question_images
    
    def build(self) -> str:
        """Build the complete prompt"""
        
        # Build image section if there are images
        image_section = ""
        if self.memory_images_count > 0 or self.question_images_count > 0:
            image_lines = ["\n[Image Description]"]
            if self.memory_images_count > 0:
                image_lines.append(f"- First {self.memory_images_count} image(s) are [Memory Images] from the retrieved conversation context")
            if self.question_images_count > 0:
                image_lines.append(f"- Next {self.question_images_count} image(s) are [Question Images] related to the current question")
            image_lines.append("Please understand their content according to the image labels and order.")
            image_section = "\n".join(image_lines)
        
        # Add question image note if applicable
        image_note = "\n[Note: The question contains images. Please analyze them together with the question.]" if self.has_question_images else ""
        
        return self.TEMPLATE.format(
            instruction=self.instruction,
            context=self.context,
            image_section=image_section,
            question=self.question,
            image_note=image_note,
            format_requirement=self.format_requirement
        )


# ==================== Modified VLM Evaluator ====================

class VLMEvaluator:
    """VLM Evaluator - using multimodal RAG memory system (adopting new code style)"""
    
    def __init__(self, 
                 memory_system: MultiModalRAGMemorySystem,
                 api_key: str,
                 model: str = "qwen2.5-vl-7b-instruct",
                 base_url: str = "http://localhost:8000/v1",
                 verbose: bool = False,
                 max_retries: int = 3,
                 timeout: int = 60,
                 max_context_tokens: Optional[int] = None,
                 max_images: Optional[int] = None,
                 retrieval_topk: int = 5):
        
        self.memory_system = memory_system
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.verbose = verbose
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_context_tokens = max_context_tokens
        self.max_images = max_images
        self.retrieval_topk = retrieval_topk
        
        # Initialize token counter
        self.token_counter = TokenCounter()
        
        # Store statistics
        self.session_statistics = defaultdict(lambda: {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "processing_time": 0.0,
            "by_category": defaultdict(int),
            "by_difficulty": defaultdict(int),
            "avg_retrieval_count": 0,
            "total_retrieval_count": 0,
            "total_retrieval_time": 0,
            "total_images_prepare_time": 0,
            "total_prompt_build_time": 0,
            "total_api_call_time": 0,
            "images_limited_count": 0
        })
        
        self.global_statistics = {
            "total_sessions": 0,
            "total_questions": 0,
            "successful_questions": 0,
            "failed_questions": 0,
            "images_limited_questions": 0,
            "start_time": None,
            "end_time": None,
            "retrieval_topk": retrieval_topk,
            "max_images": max_images,
            "memory_system": "MultiModalRAG",
        }
        
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
                logger.info(f"Retrieval top-k: {self.retrieval_topk}")
                if self.max_images:
                    logger.info(f"Maximum images: {self.max_images}")
            else:
                logger.warning(f"API connection test returned non-200 status code: {response.status_code}")
        except Exception as e:
            logger.error(f"API connection test failed: {e}")
    
    def load_questions(self, conversations_dir: str) -> Dict[str, Dict]:
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
                conversation_file = session_dir / "session.json"
                session_id = session_dir_name
                
                with open(question_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                questions = data.get("questions", [])
                question_pairs = []
                
                # Image directory
                image_dir = session_dir / "image"
                
                for q in questions:
                    q_id = q.get("question_id", f"MURAG_{len(question_pairs)}")
                    
                    # Get question image filename
                    question_image_filename = q.get("question", {}).get("image", "")
                    
                    # If question has image, build full path and save to image_context
                    image_context_list = []

                    if question_image_filename:
                        if str(session_dir) == "session0":
                            fold, img_file = question_image_filename.split("/", 1)
                            full_path = scenes_dir / fold / "image" / img_file
                            image_context_list.append(str(full_path))
                            logger.debug(f"Question {q_id} image full path: {full_path}")
                            qa_pair = QuestionAnswerPair(
                                question_id=q_id,
                                session_id=fold,
                                dialogue_name=dialogue_name,
                                question_text=q.get("question", {}).get("text", ""),
                                question_image=question_image_filename,  # Keep original filename
                                original_answer=q.get("original_answer", ""),
                                answer_source=q.get("answer_source", "unknown"),
                                answer_session=q.get("answer_session", []),
                                question_type=q.get("question_type", {}),
                                difficulty=q.get("difficulty", "medium"),
                                supporting_evidence=q.get("supporting_evidence", []),
                                image_context=image_context_list  # Save full path here
                            )
                        else:
                            full_path = image_dir / question_image_filename
                            image_context_list.append(str(full_path))
                            logger.debug(f"Question {q_id} image full path: {full_path}")
                    
                            qa_pair = QuestionAnswerPair(
                                question_id=q_id,
                                session_id=session_id,
                                dialogue_name=dialogue_name,
                                question_text=q.get("question", {}).get("text", ""),
                                question_image=question_image_filename,  # Keep original filename
                                original_answer=q.get("original_answer", ""),
                                answer_source=q.get("answer_source", "unknown"),
                                answer_session=q.get("answer_session", []),
                                question_type=q.get("question_type", {}),
                                difficulty=q.get("difficulty", "medium"),
                                supporting_evidence=q.get("supporting_evidence", []),
                                image_context=image_context_list  # Save full path here
                            )
                    else:
                        qa_pair = QuestionAnswerPair(
                            question_id=q_id,
                            session_id=session_id,
                            dialogue_name=dialogue_name,
                            question_text=q.get("question", {}).get("text", ""),
                            question_image=None,
                            original_answer=q.get("original_answer", ""),
                            answer_source=q.get("answer_source", "unknown"),
                            answer_session=q.get("answer_session", []),
                            question_type=q.get("question_type", {}),
                            difficulty=q.get("difficulty", "medium"),
                            supporting_evidence=q.get("supporting_evidence", []),
                            image_context=[]  # No image
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
        
    def _prepare_query_from_question(self, question_pair: QuestionAnswerPair) -> Union[str, Dict]:
        """
        Prepare multimodal query from question
        """
        query_text = question_pair.question_text
        if question_pair.question_image and question_pair.image_context:
            image_path = question_pair.image_context[0]
            if os.path.exists(image_path):
                return {
                    'text': query_text,
                    'image': {'path': image_path}
                }
        
        return query_text
    
    def _get_instruction(self, question_pair: QuestionAnswerPair) -> str:
        """Get instruction based on question type"""
        question_type = question_pair.question_type.get("sub_type", "") if question_pair.question_type else ""
        
        # Instructions for 9 question types (only CD, AR, TTL include abbreviation)
        instructions = {
            "Unimodal Precise Recall": "Accurately recall specific information from the retrieved conversation chunks and answer directly.",
            "Cross-modal Related Retrieval": "Retrieve related information across different modalities (text and images) from the retrieved conversation chunks.",
            "Knowledge Resolution": "Resolve and maintain knowledge consistency based on the retrieved conversation chunks.",
            "Temporal Reasoning": "Reason about temporal relationships and time-based information in the retrieved conversation chunks.",
            "Multimodal Causal Reasoning": "Perform causal reasoning using both text and image information from the retrieved conversation chunks.",
            "Reference & Evolution Tracking": "Track references and their evolution throughout the retrieved conversation chunks.",
            "Test-Time Learning": "Learn and adapt from the retrieved conversation chunks at test time to answer the question.",
            "Conflict Detection": "Check whether this information conflicts with the retrieved conversation chunks.",
            "Answer Refusal": "Determine if the question can be answered based on the retrieved conversation chunks."
        }
        
        # Return instruction based on question type
        if question_type in instructions:
            return instructions[question_type]
        else:
            return "Answer the question based on the retrieved conversation chunks (appropriate reasoning is allowed)."


    def _get_format_requirement(self, question_type: str) -> str:
        """Get format requirement based on question type"""
        
        format_requirements = {
            "Conflict Detection": "Response format: Reply strictly with either 'Yes' or 'No' only.",
            "Answer Refusal": "Response format: If the information is present in the retrieved conversation chunks, provide answer based on that information; if not present, reply with: 'Not mentioned.'",
        }
        
        if question_type in format_requirements:
            return format_requirements[question_type]
        else:
            return "Response format: Provide clear and accurate answers based on the retrieved conversation chunks."
    
    def _prepare_images(self, question_pair: QuestionAnswerPair, 
                    retrieved_memory_objects: List[MemoryElement]) -> Tuple[List[Dict], bool, int, int]:
        """
        Prepare all images, organized in order: memory images first, question images after
        Return format: List[Dict] containing all images, order [memory images..., question images...]
        """
        all_images = []
        memory_images_list = []
        question_images_list = []
        
        question_images_count = 0
        context_images_count = 0
        images_limited = False
        
        # 1. Process memory images first (placed first)
        # Collect all memory images
        memory_images = []
        for mem in retrieved_memory_objects:
            if mem.image_path and mem.image_filename:
                memory_images.append(mem)
        
        # Calculate total image counts
        total_memory_images = len(memory_images)
        total_question_images = len([f.strip() for f in question_pair.question_image.split(',')]) if question_pair.question_image else 0
        
        # If max_images is set, calculate how many each category can keep
        if self.max_images:
            # Prioritize keeping question images, remaining for memory images
            if total_question_images >= self.max_images:
                # Question images exceed limit, keep only some question images, discard all memory images
                memory_images_to_keep = 0
                question_images_to_keep = self.max_images
                images_limited = True
                logger.info(f"Total images exceed limit, keeping only {self.max_images} question images, discarding all {total_memory_images} memory images")
            else:
                # Question images within limit, remaining for memory images
                question_images_to_keep = total_question_images
                remaining_slots = self.max_images - question_images_to_keep
                memory_images_to_keep = min(total_memory_images, remaining_slots)
                
                if memory_images_to_keep < total_memory_images:
                    images_limited = True
                    logger.info(f"Memory images limited from {total_memory_images} to {memory_images_to_keep}")
        else:
            # No limit, keep all
            memory_images_to_keep = total_memory_images
            question_images_to_keep = total_question_images
        
        # 2. Process memory images (keep only required number)
        if memory_images_to_keep > 0:
            memory_images_selected = memory_images[:memory_images_to_keep]
            
            for mem in memory_images_selected:
                img_info = self.memory_system.get_image_for_api(
                    mem.image_filename,
                    session_id=mem.session_id,
                    dialogue_index=mem.dialogue_index,
                    role=mem.role,
                    dialogue_text=mem.text,
                    is_question_image=False
                )
                if img_info:
                    # Add special marker for memory images
                    img_info["image_type"] = "memory"
                    img_info["marker"] = f"【memory_image-{mem.session_id}-{mem.dialogue_index}-{mem.role}】"
                    memory_images_list.append(img_info)
                    context_images_count += 1
                    logger.debug(f"Added memory image: {mem.image_filename} [marker: {img_info.get('marker', '')}]")
        
        # 3. Process question images (placed after memory images)
        if question_images_to_keep > 0 and question_pair.question_image:
            if str(question_pair.session_id) == "session0":
                print("session0 question_image:", question_pair.question_image)
                fold, img_file = question_pair.question_image.split('/', 1)
                question_files_selected = [img_file]
                img_info = self.memory_system.get_image_for_api(
                        img_file,
                        is_question_image=True,
                        question_id=fold
                    )
                if img_info:
                    # Add special marker for question images
                    img_info["image_type"] = "question"
                    # Mark sequence number to help LLM understand which image corresponds to which question
                    if len(question_files_selected) > 1:
                        img_info["marker"] = f"【question_image-{question_pair.question_id}-{i+1}/{len(question_files_selected)}】"
                    else:
                        img_info["marker"] = f"【question_image-{question_pair.question_id}】"
                    question_images_list.append(img_info)
                    question_images_count += 1
                    logger.debug(f"Added question image: {img_file} [marker: {img_info.get('marker', '')}]")
            else:
                image_files = [f.strip() for f in question_pair.question_image.split(',') if f.strip()]
                
                # Keep only required number of question images
                question_files_selected = image_files[:question_images_to_keep]
                
                for i, img_file in enumerate(question_files_selected):
                    img_info = self.memory_system.get_image_for_api(
                        img_file,
                        is_question_image=True,
                        question_id=question_pair.session_id
                    )
                    if img_info:
                        # Add special marker for question images
                        img_info["image_type"] = "question"
                        # Mark sequence number to help LLM understand which image corresponds to which question
                        if len(question_files_selected) > 1:
                            img_info["marker"] = f"【question_image-{question_pair.question_id}-{i+1}/{len(question_files_selected)}】"
                        else:
                            img_info["marker"] = f"【question_image-{question_pair.question_id}】"
                        question_images_list.append(img_info)
                        question_images_count += 1
                        logger.debug(f"Added question image: {img_file} [marker: {img_info.get('marker', '')}]")
        
        # 4. Combine image lists: memory images first, question images after
        all_images = memory_images_list + question_images_list
        
        original_count = total_memory_images + total_question_images
        limited_count = len(all_images)
        
        # Log image organization information
        logger.info(f"Image organization complete: {len(memory_images_list)} memory images, {len(question_images_list)} question images, total {len(all_images)} images")
        
        return all_images, images_limited, original_count, limited_count
    
    def _build_prompt(self, question_pair: QuestionAnswerPair, 
                 memory_text: str, has_question_images: bool,
                 memory_images_count: int, question_images_count: int) -> str:
        """
        Build prompt with clear explanation of image order and meaning
        """
        
        # Get question type
        question_type = question_pair.question_type.get("sub_type", "") if question_pair.question_type else ""
        
        # Get instruction based on question type
        instruction = self._get_instruction(question_pair)
        
        # Get format requirement for special types
        format_requirement = self._get_format_requirement(question_type)
        
        # Build prompt using template
        prompt = MURAGPromptTemplate(
            instruction=instruction,
            context=memory_text,
            question=question_pair.question_text,
            format_requirement=format_requirement,
            memory_images_count=memory_images_count,
            question_images_count=question_images_count,
            has_question_images=has_question_images
        )
        
        return prompt.build()
    
    def _call_vlm_api(self, prompt: str, images: List[Dict]) -> Dict[str, Any]:
        start_time = time.time()
        
        # Build message
        content_list = [{"type": "text", "text": prompt}]
        content_list.extend(images)
        
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": content_list
            }],
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
    
    def _calculate_confidence(self, prediction: str, reference: str, answer_source: str) -> float:
        return 0.7
    
    def evaluate_single_question(self, 
                           question_pair: QuestionAnswerPair,
                           session_id: str,
                           question_file_path: str = None) -> EvaluationResult:
        """Evaluate a single question - with detailed timing calculation and error recording"""
        start_time = time.time()
        
        # Timing variables
        retrieval_time = 0.0
        images_prepare_time = 0.0
        prompt_build_time = 0.0
        api_call_time = 0.0
        
        try:
            logger.debug(f"Processing question: {session_id} - {question_pair.question_id} ({question_pair.category})")
            
            # 1. Prepare multimodal query
            query = self._prepare_query_from_question(question_pair)
            
            # 2. Retrieve relevant memories (record time)
            retrieval_start = time.time()
            retrieved_result = self.memory_system.retrieve_relevant_memories(
                query, 
                topk=self.retrieval_topk
            )
            retrieval_time = time.time() - retrieval_start
            
            memory_text = retrieved_result.get('text', 'No relevant memories')
            memory_objects = retrieved_result.get('memory_objects', [])
            retrieved_chunks = retrieved_result.get('retrieved_chunks', [])
            retrieved_ids = [mem.memory_id for mem in memory_objects if hasattr(mem, 'memory_id')]
            
            # 3. Prepare images (record time)
            images_start = time.time()
            images, images_limited, orig_img_count, limited_img_count = self._prepare_images(
                question_pair, memory_objects
            )
            images_prepare_time = time.time() - images_start
            
            # 4. Count various image types
            memory_images_count = sum(1 for img in images if img.get("image_type") == "memory")
            question_images_count = sum(1 for img in images if img.get("image_type") == "question")
            has_question_images = question_images_count > 0
            
            # 5. Build prompt (record time)
            prompt_start = time.time()
            prompt = self._build_prompt(
                question_pair, 
                memory_text, 
                has_question_images,
                memory_images_count,
                question_images_count
            )
            prompt_build_time = time.time() - prompt_start
            
            # 6. Token truncation check
            if self.max_context_tokens:
                prompt_tokens = self.token_counter.count_tokens(prompt)
                if prompt_tokens > self.max_context_tokens:
                    prompt, original_tokens, truncated_tokens = self.token_counter.truncate_text(
                        prompt, self.max_context_tokens
                    )
                    was_truncated = True
                else:
                    was_truncated = False
                    original_tokens = prompt_tokens
                    truncated_tokens = prompt_tokens
            else:
                was_truncated = False
                original_tokens = self.token_counter.count_tokens(prompt)
                truncated_tokens = original_tokens
            
            # 7. Call VLM API (record time)
            api_start = time.time()
            vlm_response = self._call_vlm_api(prompt=prompt, images=images)
            api_call_time = vlm_response.get("processing_time", time.time() - api_start)
            
            system_answer = vlm_response.get("answer", "").strip()
            processing_time = vlm_response.get("processing_time", 0)
            success = vlm_response.get("success", False)
            
            # 8. Calculate confidence
            confidence = self._calculate_confidence(
                system_answer, 
                question_pair.original_answer,
                question_pair.answer_source
            )
            
            # Total processing time
            total_processing_time = time.time() - start_time
            
            # 9. Create evaluation result
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
                memory_type="MultiModalRAG",
                vlm_model=self.model,
                processing_time=total_processing_time,
                confidence=confidence,
                supporting_evidence=question_pair.supporting_evidence,
                memory_context_summary=f"Retrieved {len(memory_objects)} relevant memories",
                recall_method="multimodal_rag",
                success=success,
                error_message=None if success else vlm_response.get("error", ""),
                truncated=was_truncated,
                original_context_length=original_tokens if was_truncated else None,
                truncated_context_length=truncated_tokens if was_truncated else None,
                retrieved_memory_ids=retrieved_ids,
                retrieval_scores=[chunk.get('similarity', 0.0) for chunk in retrieved_chunks],
                images_limited=images_limited,
                original_image_count=orig_img_count,
                limited_image_count=limited_img_count,
                retrieved_chunks=retrieved_chunks,
                # New timing fields
                retrieval_time=retrieval_time,
                images_prepare_time=images_prepare_time,
                prompt_build_time=prompt_build_time,
                api_call_time=api_call_time
            )
            
            # Update session statistics
            self.session_statistics[session_id]["successful"] += 1
            self.session_statistics[session_id]["processing_time"] += total_processing_time
            self.session_statistics[session_id]["total_retrieval_count"] += len(memory_objects)
            # Accumulate timing statistics
            self.session_statistics[session_id]["total_retrieval_time"] += retrieval_time
            self.session_statistics[session_id]["total_images_prepare_time"] += images_prepare_time
            self.session_statistics[session_id]["total_prompt_build_time"] += prompt_build_time
            self.session_statistics[session_id]["total_api_call_time"] += api_call_time
            if images_limited:
                self.session_statistics[session_id]["images_limited_count"] += 1
            
            logger.info(f"✓ Successfully processed: {session_id} - {question_pair.question_id} "
                        f"(Total: {total_processing_time:.2f}s, Retrieval: {retrieval_time:.3f}s, "
                        f"Images: {images_prepare_time:.3f}s, Prompt: {prompt_build_time:.3f}s, "
                        f"API: {api_call_time:.2f}s, Retrieved {len(memory_objects)} memories)")
            
            return result
            
        except Exception as e:
            total_processing_time = time.time() - start_time
            error_msg = str(e)[:200]
            logger.error(f"✗ Error processing question {session_id} - {question_pair.question_id}: {error_msg}")
            
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
                memory_type="MultiModalRAG",
                vlm_model=self.model,
                processing_time=total_processing_time,
                confidence=0.0,
                supporting_evidence=question_pair.supporting_evidence,
                memory_context_summary="Error: Unable to retrieve memories",
                recall_method="multimodal_rag",
                success=False,
                error_message=error_msg,
                truncated=False,
                retrieval_time=retrieval_time,
                images_prepare_time=images_prepare_time,
                prompt_build_time=prompt_build_time,
                api_call_time=api_call_time
            )
            
            self.session_statistics[session_id]["failed"] += 1
            return result
    
    def evaluate_session_questions(self,
                                session_id: str,
                                session_data: Dict,
                                max_questions: Optional[int] = None) -> List[Dict[str, Any]]:
        questions = session_data["questions"]
        session_path = Path(session_data["session_path"])
        session_dir_name = session_data.get("session_dir_name", session_id)
        question_file_path = session_data.get("question_file", "")
        
        if max_questions and max_questions < len(questions):
            questions = questions[:max_questions]
        
        total_questions = len(questions)
        logger.info(f"Starting evaluation of {total_questions} questions for {session_id} (single-threaded mode)")
        logger.info(f"Using multimodal RAG retrieval, top-k: {self.retrieval_topk}")
        
        # Initialize session statistics
        self.session_statistics[session_id]["total"] = total_questions
        for qa in questions:
            self.session_statistics[session_id]["by_category"][qa.category] += 1
            self.session_statistics[session_id]["by_difficulty"][qa.difficulty] += 1
        
        results = []
        
        # Process each question sequentially
        for idx, qa in enumerate(questions, 1):
            logger.debug(f"Processing question {idx}/{total_questions}: {qa.question_id}")
            result = self.evaluate_single_question(qa, session_id, question_file_path)
            results.append(asdict(result))
            
            if idx % max(1, total_questions // 10) == 0:
                logger.info(f"[{session_id}] Progress: {idx}/{total_questions}")
        
        # Calculate average retrieval count
        if total_questions > 0:
            self.session_statistics[session_id]["avg_retrieval_count"] = (
                self.session_statistics[session_id]["total_retrieval_count"] / total_questions
            )
        
        # Save results
        self._save_session_results(session_id, session_dir_name, session_path, results)
        
        # Update global statistics
        self.global_statistics["total_questions"] += total_questions
        self.global_statistics["successful_questions"] += self.session_statistics[session_id]["successful"]
        self.global_statistics["failed_questions"] += self.session_statistics[session_id]["failed"]
        self.global_statistics["images_limited_questions"] += self.session_statistics[session_id]["images_limited_count"]
        
        # Output session timing statistics
        successful = self.session_statistics[session_id]["successful"]
        if successful > 0:
            logger.info(f"Session {session_id} timing statistics - "
                        f"Avg retrieval: {self.session_statistics[session_id]['total_retrieval_time']/successful:.3f}s, "
                        f"Avg image prepare: {self.session_statistics[session_id]['total_images_prepare_time']/successful:.3f}s, "
                        f"Avg prompt: {self.session_statistics[session_id]['total_prompt_build_time']/successful:.3f}s, "
                        f"Avg API: {self.session_statistics[session_id]['total_api_call_time']/successful:.2f}s")
        
        logger.info(f"Completed evaluation for {session_id}: Success {self.session_statistics[session_id]['successful']}, "
                    f"Failed {self.session_statistics[session_id]['failed']}, "
                    f"Avg retrieval {self.session_statistics[session_id]['avg_retrieval_count']:.1f} memories")
        
        return results
    
    # Modified evaluate_all_sessions: process each session sequentially
    def evaluate_all_sessions(self,
                        sessions_questions: Dict[str, Dict],
                        max_questions_per_session: Optional[int] = None,
                        conversations_dir: str = None
                        ):
        self.global_statistics["start_time"] = time.time()
        self.global_statistics["total_sessions"] = len(sessions_questions)
        
        logger.info(f"Starting evaluation of {len(sessions_questions)} sessions (single-threaded mode)")
        logger.info(f"Multimodal RAG configuration: top-k={self.retrieval_topk}, max token={self.max_context_tokens}")
        if self.max_images:
            logger.info(f"Maximum images: {self.max_images}")
        
        # Get memory system statistics
        if hasattr(self.memory_system, 'get_statistics'):
            mem_stats = self.memory_system.get_statistics()
            logger.info(f"Memory system: {mem_stats['total_memories']} memories, "
                        f"{mem_stats['memories_with_images']} containing images")
        
        # Process each session sequentially
        for session_id, session_data in sessions_questions.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing Session: {session_id}")
            logger.info(f"Number of questions: {len(session_data['questions'])}")
            logger.info(f"{'='*60}")
            
            results = self.evaluate_session_questions(
                session_id,
                session_data,
                max_questions_per_session
            )
            logger.info(f"Session {session_id} processing complete, generated {len(results)} results")
        
        self.global_statistics["end_time"] = time.time()
    
    # Modified _save_session_results: removed file lock
    def _save_session_results(self,
                            session_id: str,
                            session_dir_name: str,
                            session_path: Path,
                            results: List[Dict[str, Any]]):
        session_results_dir = session_path / "evaluation_results"
        session_results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_filename = "results_MURAG.json"
        json_file = session_results_dir / json_filename
        
        full_results = {
            "metadata": {
                "session_id": session_id,
                "session_dir_name": session_dir_name,
                "session_path": str(session_path),
                "vlm_model": self.model,
                "memory_type": type(self.memory_system).__name__,
                "base_url": self.base_url,
                "context_type": "multimodal_rag",
                "retrieval_topk": self.retrieval_topk,
                "max_context_tokens": self.max_context_tokens,
                "max_images": self.max_images,
                "evaluation_time": timestamp
            },
            "results": results
        }
        
        if hasattr(self.memory_system, 'get_statistics'):
            full_results["memory_statistics"] = self.memory_system.get_statistics()
        
        # Direct file write, no lock
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"Saved results for {session_id} to: {json_file}")
    
    

# ==================== Factory Function for Creating Memory System ====================

def create_memory_system(memory_type: str, conversations_dir: str, **kwargs):
    """Factory function for creating memory system"""
    if memory_type == "multimodal_rag":
        config = kwargs.get('config', {})
        return MultiModalRAGMemorySystem(conversations_dir, config)
    else:
        raise ValueError(f"Unsupported memory type: {memory_type}")


# ==================== Main Function ====================

def main():
    parser = argparse.ArgumentParser(description="VLM Memory Capability Evaluator (Multimodal RAG Edition)")
    parser.add_argument("--conversations_dir", type=str, required=True,
                       help="Conversation data directory (required)")
    parser.add_argument("--api_key", type=str, required=True,
                       help="VLM API key (required)")
    parser.add_argument("--model", type=str, required=True,
                       help="VLM model name (required)")
    parser.add_argument("--base_url", type=str, required=True,
                       help="API base URL (required)")
    parser.add_argument("--memory_type", type=str, default="multimodal_rag",
                       choices=["multimodal_rag"],
                       help="Memory system type")
    parser.add_argument("--max_questions_per_session", type=int, default=None,
                       help="Maximum questions to process per session")
    parser.add_argument("--max_sessions", type=int, default=None,
                       help="Maximum sessions to process")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose logging output")
    parser.add_argument("--test_mode", action="store_true",
                       help="Test mode")
    parser.add_argument("--max_retries", type=int, default=3,
                       help="Maximum API call retries")
    parser.add_argument("--timeout", type=int, default=60,
                       help="API call timeout (seconds)")
    parser.add_argument("--max_context_tokens", type=int, default=4096,
                       help="Maximum context tokens")
    
    # RAG specific parameters
    parser.add_argument("--retrieval_topk", type=int, default=10,
                       help="Number of memories to retrieve")
    parser.add_argument("--encoder_model", type=str, default="Alibaba-NLP/gme-Qwen2-VL-7B-Instruct",
                       help="Multimodal encoder model")
    parser.add_argument("--retrieval_mode", type=str, default="cosine",
                       choices=["cosine", "dot", "L2"],
                       help="Retrieval similarity calculation mode")
    parser.add_argument("--utilization_method", type=str, default="MultiModalUtilization",
                       choices=["ConcateUtilization", "MultiModalUtilization"],
                       help="Memory utilization strategy")
    
    # Image limit parameter
    parser.add_argument("--max_images", type=int, default=None,
                       help="Maximum number of images")
    
    args = parser.parse_args()
    
    # Configure log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    print("=" * 70)
    print("VLM Memory Capability Evaluator (Multimodal RAG Edition)")
    print(f"Model: {args.model}")
    print(f"Memory system: {args.memory_type}")
    if args.memory_type == "multimodal_rag":
        print(f"  Encoder: {args.encoder_model}")
        print(f"  Retrieval mode: {args.retrieval_mode}, top-k: {args.retrieval_topk}")
        print(f"  Utilization strategy: {args.utilization_method}")
    if args.max_context_tokens:
        print(f"Dialogue truncation: {args.max_context_tokens} tokens")
    if args.max_images:
        print(f"Image limit: {args.max_images} images")
    print("=" * 70)
    
    # Test mode setting
    if args.test_mode:
        args.max_questions_per_session = 2
        print("Test mode: Processing only first 2 questions per session")
    
    # 1. Initialize memory system
    print(f"\n[1] Initializing memory system ({args.memory_type})...")
    
    if args.memory_type == "multimodal_rag":
        rag_config = {
            'encoder': {
                'method': 'GMEEncoder',
                'model_name': args.encoder_model
            },
            'retrieval': {
                'mode': args.retrieval_mode,
                'topk': args.retrieval_topk
            },
            'utilization': {
                'method': args.utilization_method,
                'max_tokens': args.max_context_tokens
            },
            'truncation': {
                'method': 'SimpleTruncation',
                'max_tokens': args.max_context_tokens
            }
        }
        memory_system = create_memory_system(args.memory_type, args.conversations_dir, config=rag_config)
    else:
        memory_system = create_memory_system(args.memory_type, args.conversations_dir)
    
    print(f"   Loading and encoding all conversations...")
    memory_system.load_all_conversations()
    
    # Display statistics
    if args.memory_type == "multimodal_rag":
        stats = memory_system.get_statistics()
        print(f"   Loaded {stats['total_sessions']} sessions, total {stats['total_memories']} memories")
        print(f"   Memories containing images: {stats['memories_with_images']}")
    else:
        print(f"   Loaded {len(memory_system.session_info)} sessions, total {len(memory_system.all_dialogues)} dialogue turns")
    
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
        max_images=args.max_images,
        retrieval_topk=args.retrieval_topk,
    )
    
    # 3. Load questions
    print(f"\n[3] Loading question files...")
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
    print(f"\n[4] Starting session-by-session evaluation (using multimodal RAG memory)...")
    print(f"   Processing {len(sessions_to_process)} sessions")
    print(f"   Total questions: {total_questions}")
    print("-" * 70)
    evaluator.evaluate_all_sessions(
        sessions_questions=sessions_to_process,
        max_questions_per_session=args.max_questions_per_session,
        conversations_dir=args.conversations_dir
    )
    
    # 5. Output statistics
    print(f"\n[5] Evaluation complete!")
    print("-" * 70)
    

if __name__ == "__main__":
    main()