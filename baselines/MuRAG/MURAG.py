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

# 导入图像处理相关库
from PIL import Image
import base64
from io import BytesIO
import requests

from transformers import AutoModel
import logging

# 导入多模态模型相关库
from transformers import CLIPModel, CLIPProcessor

# 多线程相关导入
import concurrent.futures
from threading import Lock, Semaphore

# 尝试导入tiktoken用于token计数
try:
    import tiktoken
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False
    logging.warning("tiktoken未安装，将使用简单的字符数估算token数")

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==================== 数据类定义 ====================

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
class MemoryElement:
    """记忆单元 - 存储单轮对话的内容"""
    memory_id: int
    session_id: str
    dialogue_index: int
    role: str
    text: str
    image_filename: Optional[str] = None
    image_path: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: Optional[Dict] = None
    image_id: Optional[str] = None  # 图片唯一ID
    
    def __post_init__(self):
        # 生成图片ID
        if self.image_filename and self.session_id and self.dialogue_index:
            self.image_id = f"{self.image_filename}"
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
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
        转换为可用于编码的观察格式
        将角色信息加入到文本中，以便编码时能够区分不同角色的发言
        """
        # 将角色信息加入到文本中
        role_text = f"[{self.role}] {self.text}"
        
        if self.image_filename and self.image_path:
            # print("self.image_path")
            # print(self.image_path)
            return {
                'text': role_text,  # 使用带角色的文本
                'image': {'path': self.image_path}
            }
        else:
            return role_text  # 返回带角色的文本
    
    def get_image_info(self) -> Optional[Dict]:
        """获取图片信息（用于API调用）"""
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
    
    # 新增时间字段
    retrieval_time: float = 0.0      # 检索记忆时间
    images_prepare_time: float = 0.0 # 准备图片时间
    prompt_build_time: float = 0.0   # 构建提示词时间
    api_call_time: float = 0.0       # API调用时间


# ==================== Token计数器 ====================

class TokenCounter:
    """Token计数器"""
    
    def __init__(self, model_name: str = "cl100k_base"):
        self.model_name = model_name
        self.encoding = None
        
        if TOKENIZER_AVAILABLE:
            try:
                self.encoding = tiktoken.get_encoding(model_name)
                logger.info(f"成功加载tokenizer: {model_name}")
            except Exception as e:
                logger.warning(f"加载tokenizer失败: {e}，将使用估算方法")
    
    def count_tokens(self, text: str) -> int:
        """计算文本的token数量"""
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
        """截断文本到指定token数"""
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
            truncated_text = text[:keep_chars] + "... [内容已截断]"
            truncated_tokens = self.count_tokens(truncated_text)
            return truncated_text, original_tokens, truncated_tokens


# ==================== 图片处理器 ====================

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
                      is_question_image: bool = False, question_id: str = None,
                      dialogue_index: int = None, role: str = None,
                      dialogue_text: str = None) -> Dict:
        """
        处理图片，返回包含Base64数据和元信息的字典
        
        Args:
            image_path: 图片路径
            session_id: 图片所属session
            filename: 图片文件名
            is_question_image: 是否为问题图片
            question_id: 问题ID（如果是问题图片）
            dialogue_index: 对话轮次索引
            role: 说话角色
            dialogue_text: 对应的对话文本
        """
        # 检查缓存
        base64_data = None
        with self._cache_lock:
            if self.cache_enabled and image_path in self.image_cache:
                base64_data = self.image_cache[image_path]
                logger.debug(f"使用缓存的图片: {filename}")
        
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
            image_info["question_id"] = question_id
            image_info["marker"] = f"【question_image-{question_id}】"
        else:
            # 上下文图片保留完整的对话上下文标记
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
        logger.info("图片缓存已清空")


# ==================== 多模态编码器 ====================

class BaseMultiModalEncoder(ABC):
    """多模态编码器基类"""
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
        # print("image_path_or_url")
        # print(image_path_or_url)
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # print("inputs")
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

# 在CLIPEncoder类后面添加GME编码器类

class GMEEncoder(BaseMultiModalEncoder):
    """
    GME (General Multimodal Embedding) Qwen2-VL-based encoder for text and images.
    Supports unified multimodal representations for Any2Any Search.
    """
    def __init__(self, config):
        super().__init__(config)
        
        model_name = getattr(config, 'model_name', 'Alibaba-NLP/gme-Qwen2-VL-7B-Instruct')
        # 也支持通过 'path' 参数指定
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
        
        # 获取模型维度
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
            # 您可能需要设置默认的图片根目录
            final_path = os.path.join("/path/to/your/images", image_path_or_url)  # 修改为您的图片路径
        
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
# ==================== 配置类 ====================

class Config:
    """简单配置类"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


# 修改EncoderConfig类，添加GME支持
class EncoderConfig:
    """编码器配置"""
    def __init__(self, method='GMEEncoder', model_name='BAAI/bge-m3'):  # 默认改为GME
        self.method = method
        self.model_name = model_name


class RetrievalConfig:
    """检索配置"""
    def __init__(self, mode='cosine', topk=10):
        self.mode = mode
        self.topk = topk
        self.encoder = EncoderConfig()


class UtilizationConfig:
    """利用策略配置"""
    def __init__(self, method='ConcateUtilization', max_tokens=4000):
        self.method = method
        self.max_tokens = max_tokens


class TruncationConfig:
    """截断配置"""
    def __init__(self, method='SimpleTruncation', max_tokens=4000):
        self.method = method
        self.max_tokens = max_tokens


# ==================== 存储类 ====================

class BaseStore:
    """基础存储类"""
    def __init__(self, config):
        self.config = config
    
    def reset(self):
        pass


class SimpleStorage(BaseStore):
    """简单的内存存储"""
    def __init__(self, config):
        super().__init__(config)
        self.memories = []  # 存储MemoryElement对象
        self.memory_counter = 0
    
    def reset(self):
        self.memories = []
        self.memory_counter = 0
    
    def is_empty(self) -> bool:
        return len(self.memories) == 0
    
    def add(self, observation, metadata=None) -> int:
        """
        添加记忆
        
        Args:
            observation: MemoryElement对象或dict
            metadata: 额外元数据
        
        Returns:
            记忆ID
        """
        memory_id = self.memory_counter
        self.memory_counter += 1
        
        if isinstance(observation, MemoryElement):
            memory = observation
            memory.memory_id = memory_id
        else:
            # 创建MemoryElement
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
        """通过ID获取记忆"""
        if 0 <= memory_id < len(self.memories):
            return self.memories[memory_id]
        return None
    
    def get_memory_element_by_mid(self, mid: int) -> Optional[Dict]:
        """兼容接口：通过mid获取记忆（返回字典格式）"""
        memory = self.get_memory_by_id(mid)
        if memory:
            return memory.to_dict()
        return None
    
    def get_all_memories(self) -> List[MemoryElement]:
        return self.memories
    
    def __len__(self):
        return len(self.memories)


# ==================== 多模态检索器 ====================

class MultiModalRetrieval:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化编码器
        encoder_method = getattr(config.encoder, 'method', 'GMEEncoder')  # 默认使用GMEEncoder
        if encoder_method == 'CLIPEncoder':
            self.encoder = CLIPEncoder(config.encoder)
        elif encoder_method == 'GMEEncoder':  # 添加GMEEncoder支持
            self.encoder = GMEEncoder(config.encoder)
        else:
            raise ValueError(f"Unsupported encoder method: {encoder_method}")
        
        # 获取编码器输出维度
        self.encoder_dim = getattr(self.encoder, 'dimension', 1024)
        logger.info(f"Encoder output dimension: {self.encoder_dim}")
        
        # 存储所有记忆的嵌入向量
        self.tensorstore = None
        
        # 存储每个记忆的元数据
        self.memory_metadata = []
        
        # 存储记忆ID到索引的映射
        self.id_to_index = {}
        self.index_to_id = []
        
        # 最后一次检索的得分和索引
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
        添加记忆嵌入到检索器
        
        Args:
            obj: str或dict {'text': ..., 'image': ...}
            memory_id: 记忆ID（如果为None，则使用当前索引）
        
        Returns:
            嵌入向量
        """
        embedding = self.encoder(obj, return_type='tensor')
        
        if self.config.mode == 'cosine':
            embedding = self.__normalize__(embedding)
        
        # 确定索引
        if self.tensorstore is None:
            self.tensorstore = embedding
            index = 0
        else:
            self.tensorstore = torch.cat([self.tensorstore, embedding], dim=0)
            index = self.tensorstore.size(0) - 1
        
        # 存储元数据
        metadata = {
            'has_text': isinstance(obj, str) or (isinstance(obj, dict) and 'text' in obj and obj['text']),
            'has_image': isinstance(obj, dict) and 'image' in obj and obj['image']
        }
        self.memory_metadata.append(metadata)
        
        # 记录ID映射
        if memory_id is not None:
            self.id_to_index[memory_id] = index
            # 确保index_to_id足够长
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
        计算查询与所有存储记忆之间的相似度得分
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
        搜索与查询最相似的记忆
        
        Args:
            query: 查询（字符串或字典）
            topk: 返回数量
            with_score: 是否返回得分
            sort: 是否排序
            return_ids: 是否返回记忆ID（否则返回索引）
        
        Returns:
            记忆ID列表或(得分, 索引)元组
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
        
        # 确定返回数量
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
            # 将索引转换为记忆ID
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
        获取检索到的块详细信息，包括相似度得分和元数据
        """
        if self.tensorstore is None or self.tensorstore.size(0) == 0:
            return []
        
        # 获取带得分的检索结果
        if topk is None:
            topk = self.config.topk
        
        scores, indices = self.__call__(query, topk=topk, with_score=True, sort=True, return_ids=True)
        
        retrieved_chunks = []
        for i, (score, idx) in enumerate(zip(scores, indices)):
            memory_id = idx.item() if torch.is_tensor(idx) else idx
            chunk_info = {
                "chunk_index": i + 1,  # 检索结果中的序号
                "similarity": float(score),
                "embedding_similarity": True
            }
            
            # 获取记忆的详细信息
            if hasattr(self, 'storage') and self.storage:
                mem = self.storage.get_memory_by_id(memory_id)
                if mem:
                    chunk_info["session_id"] = mem.session_id
                    chunk_info["dialogue_indices"] = [mem.dialogue_index]
                    chunk_info["has_image"] = mem.image_path is not None
            
            retrieved_chunks.append(chunk_info)
        
        return retrieved_chunks
    
    def update(self, index, obj, memory_id=None):
        """更新指定位置的嵌入"""
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
        """删除指定位置的嵌入"""
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
        """通过ID列表获取嵌入张量"""
        indices = []
        for mid in id_list:
            if mid in self.id_to_index:
                indices.append(self.id_to_index[mid])
        if indices:
            return self.tensorstore[indices]
        return None


# ==================== 利用策略 ====================

class BaseUtilization:
    """利用策略基类"""
    def __init__(self, config):
        self.config = config
    
    def reset(self):
        pass
    
    def __call__(self, memories):
        raise NotImplementedError


class ConcateUtilization(BaseUtilization):
    """简单拼接利用策略 - 返回文本字符串"""
    def __init__(self, config):
        super().__init__(config)
        self.token_counter = TokenCounter()
    
    def __call__(self, memories):
        """
        将记忆列表拼接成文本
        
        Args:
            memories: MemoryElement对象列表或字典列表
        
        Returns:
            拼接后的文本字符串
        """
        if not memories:
            return "无相关记忆"
        
        context_parts = []
        
        for i, mem in enumerate(memories):
            if isinstance(mem, dict):
                session_id = mem.get('session_id', 'unknown')
                role = mem.get('role', 'unknown')
                text = mem.get('text', '')
                image = mem.get('image', '')
                dialogue_index = mem.get('dialogue_index', i+1)
                image_id = mem.get('image_id', '')
                
                prefix = f"[记忆{i+1}] Session {session_id} 第{dialogue_index}轮 - {role}: "
                if image:
                    prefix += f"[图片: {image_id}] "
                context_parts.append(prefix + text)
            elif hasattr(mem, 'to_dict'):
                mem_dict = mem.to_dict()
                session_id = mem_dict.get('session_id', 'unknown')
                role = mem_dict.get('role', 'unknown')
                text = mem_dict.get('text', '')
                image = mem_dict.get('image', '')
                image_id = mem_dict.get('image_id', '')
                dialogue_index = mem_dict.get('dialogue_index', i+1)
                
                prefix = f"[记忆{i+1}] Session {session_id} 第{dialogue_index}轮 - {role}: "
                if image:
                    prefix += f"[图片: {image_id}] "
                context_parts.append(prefix + text)
            else:
                context_parts.append(str(mem))
        
        full_text = "\n".join(context_parts)
        token_count = self.token_counter.count_tokens(full_text)
        
        if token_count > getattr(self.config, 'max_tokens', 4000):
            logger.debug(f"记忆文本超过token限制 ({token_count} > {self.config.max_tokens})，进行截断")
            truncated_text, _, _ = self.token_counter.truncate_text(
                full_text, 
                getattr(self.config, 'max_tokens', 4000)
            )
            return truncated_text
        
        return full_text


class MultiModalUtilization(BaseUtilization):
    """多模态利用策略 - 返回包含文本和图片的字典"""
    def __init__(self, config):
        super().__init__(config)
        self.token_counter = TokenCounter()
    
    def __call__(self, memories):
        """
        将记忆列表组织成多模态格式
        
        Args:
            memories: MemoryElement对象列表或字典列表
        
        Returns:
            字典: {'text': str, 'images': list of image paths, 'memory_objects': list}
        """
        if not memories:
            return {'text': '无相关记忆', 'images': [], 'memory_objects': []}
        
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
                
                prefix = f"[记忆{i+1}] Session {session_id} 第{dialogue_index}轮 - {role}: "
                if image:
                    prefix += f"[图片: {image_id}] "
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
                
                prefix = f"[记忆{i+1}] Session {session_id} 第{dialogue_index}轮 - {role}: "
                if image:
                    prefix += f"[图片: {image_id}] "
                text_parts.append(prefix + text)
                if mem.image_path and os.path.exists(mem.image_path):
                    image_paths.append(mem.image_path)
                memory_objects.append(mem)
        
        full_text = "\n".join(text_parts)
        
        token_count = self.token_counter.count_tokens(full_text)
        if token_count > getattr(self.config, 'max_tokens', 4000):
            logger.debug(f"记忆文本超过token限制 ({token_count} > {self.config.max_tokens})，进行截断")
            full_text, _, _ = self.token_counter.truncate_text(
                full_text, 
                getattr(self.config, 'max_tokens', 4000)
            )
        
        return {
            'text': full_text,
            'images': image_paths,
            'memory_objects': memory_objects
        }


# ==================== 截断策略 ====================

class BaseTruncation:
    """截断策略基类"""
    def __init__(self, config):
        self.config = config
    
    def reset(self):
        pass
    
    def __call__(self, text):
        raise NotImplementedError


class SimpleTruncation(BaseTruncation):
    """简单截断策略"""
    def __init__(self, config):
        super().__init__(config)
        self.token_counter = TokenCounter()
    
    def __call__(self, text):
        """截断文本到指定长度"""
        if not text:
            return text
        
        max_tokens = getattr(self.config, 'max_tokens', 4000)
        truncated_text, _, _ = self.token_counter.truncate_text(text, max_tokens)
        return truncated_text


# ==================== 多模态记忆召回 ====================

class MMMemoryRecall:
    """多模态记忆召回"""
    def __init__(self, config, **kwargs):
        self.config = config
        
        self.storage = kwargs['storage']
        
        # 初始化截断策略
        truncation_method = getattr(config.truncation, 'method', 'SimpleTruncation')
        if truncation_method == 'SimpleTruncation':
            self.truncation = SimpleTruncation(config.truncation)
        else:
            self.truncation = SimpleTruncation(config.truncation)
        
        # 初始化利用策略
        utilization_method = getattr(config.utilization, 'method', 'ConcateUtilization')
        if utilization_method == 'ConcateUtilization':
            self.utilization = ConcateUtilization(config.utilization)
        elif utilization_method == 'MultiModalUtilization':
            self.utilization = MultiModalUtilization(config.utilization)
        else:
            self.utilization = ConcateUtilization(config.utilization)
        
        # 初始化多模态检索器
        self.multimodal_retrieval = kwargs['multimodal_retrieval']
        
        # 将storage引用添加到检索器，以便获取详细信息
        self.multimodal_retrieval.storage = self.storage
        
        # 记录上次检索的ID和详细信息
        self.last_retrieved_ids = []
        self.last_retrieved_chunks = []
    
    def reset(self):
        self.last_retrieved_ids = []
        self.last_retrieved_chunks = []
    
    def __call__(self, query, topk=None) -> Dict:
        """
        基于多模态查询召回记忆
        
        Returns:
            字典包含：
            - 'text': 格式化的文本上下文
            - 'images': 图片路径列表
            - 'memory_objects': 完整的记忆对象列表
            - 'retrieved_chunks': 检索块的详细信息
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
        
        # 获取检索块的详细信息
        retrieved_chunks = self.multimodal_retrieval.get_retrieved_chunks_info(query, topk)
        self.last_retrieved_chunks = retrieved_chunks
        
        # 提取记忆ID
        ranking_ids = [chunk.get('memory_id') for chunk in retrieved_chunks if 'memory_id' in chunk]
        if not ranking_ids:
            # 如果没有memory_id，尝试用其他方式获取
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
        
        # 收集记忆对象
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
                
                # 如果retrieved_chunks中还没有memory_id，补充进去
                if i < len(retrieved_chunks) and 'memory_id' not in retrieved_chunks[i]:
                    retrieved_chunks[i]['memory_id'] = mem.memory_id
                    retrieved_chunks[i]['session_id'] = mem.session_id
                    retrieved_chunks[i]['dialogue_indices'] = [mem.dialogue_index]
                    retrieved_chunks[i]['has_image'] = mem.image_path is not None
        
        # 记录检索到的ID
        self.last_retrieved_ids = retrieved_ids
        
        # 使用利用策略格式化结果
        result = self.utilization(memories)
        
        # 确保返回格式包含完整信息
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

# ==================== 多模态RAG记忆系统 ====================

class MultiModalRAGMemorySystem:
    """
    多模态RAG记忆系统
    - 使用CLIP编码器将对话内容（文本+图片）编码为向量
    - 将向量存储在检索器中
    - 查询时检索最相关的记忆
    - 支持多模态查询和召回
    """
    
    def __init__(self, conversations_dir: str, config: Dict = None):
        self.conversations_dir = conversations_dir
        self.all_dialogues = []  # 所有对话的原始数据
        self.session_info = {}    # session额外信息
        self.image_paths = {}      # 图片路径映射 {session_id: {image_filename: full_path}}
        
        # 新增：图片处理器
        self.image_processor = ImageProcessor()
        
        # 新增：存储时间记录
        self.storage_time = 0.0      # 总存储时间
        self.loading_time = 0.0      # 数据加载时间
        self.encoding_time = 0.0     # 向量化编码时间
        
        # 默认配置
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
            # 合并配置
            for key, value in config.items():
                if key in default_config and isinstance(value, dict):
                    default_config[key].update(value)
                else:
                    default_config[key] = value
        
        # 创建配置对象
        self.encoder_config = EncoderConfig(**default_config['encoder'])
        self.retrieval_config = RetrievalConfig(**default_config['retrieval'])
        self.retrieval_config.encoder = self.encoder_config
        
        self.utilization_config = UtilizationConfig(**default_config['utilization'])
        self.truncation_config = TruncationConfig(**default_config['truncation'])
        
        # 初始化组件
        self._init_components()
        
        logger.info(f"多模态RAG记忆系统初始化完成")
        logger.info(f"  编码器: {self.encoder_config.method} ({self.encoder_config.model_name})")
        logger.info(f"  检索模式: {self.retrieval_config.mode}, top-k: {self.retrieval_config.topk}")
        logger.info(f"  利用策略: {self.utilization_config.method}")
    
    def _init_components(self):
        """初始化所有RAG组件"""
        # 创建存储
        self.storage = SimpleStorage(Config())
        
        # 创建多模态检索器
        self.multimodal_retrieval = MultiModalRetrieval(self.retrieval_config)
        
        # 创建召回器
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
        """加载所有对话并编码存储 - 添加时间记录"""
        overall_start = time.time()
        
        scenes_dir = os.path.join(self.conversations_dir, "scenes")
        if not os.path.exists(scenes_dir):
            raise ValueError(f"找不到scenes目录: {scenes_dir}")
        
        session_dirs = natsorted([
            d for d in os.listdir(scenes_dir) 
            if os.path.isdir(os.path.join(scenes_dir, d))
        ])
        
        total_memories = 0
        
        # 1. 数据加载时间
        loading_start = time.time()
        
        for session_dir_name in session_dirs:
            session_dir = os.path.join(scenes_dir, session_dir_name)
            
            # 扫描图片目录
            image_dir = os.path.join(session_dir, "image")
            if os.path.exists(image_dir):
                self.image_paths[session_dir_name] = {}
                for img_file in os.listdir(image_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                        self.image_paths[session_dir_name][img_file] = os.path.join(image_dir, img_file)
                logger.debug(f"session {session_dir_name} 扫描到 {len(self.image_paths[session_dir_name])} 张图片")
            
            # 加载会话数据
            session_memories = self._load_and_encode_session(session_dir_name, session_dir)
            if session_memories > 0:
                total_memories += session_memories
        
        self.loading_time = time.time() - loading_start
        logger.info(f"数据加载耗时: {self.loading_time:.2f}秒")
        
        # 2. 向量化时间（_load_and_encode_session中已经包含编码时间）
        # 这里记录总的存储时间
        self.storage_time = time.time() - overall_start
        
        logger.info(f"已加载 {len(self.session_info)} 个session，共 {len(self.all_dialogues)} 轮对话")
        logger.info(f"已编码存储 {total_memories} 条记忆向量")
        logger.info(f"记忆存储总耗时: {self.storage_time:.2f}秒 (加载: {self.loading_time:.2f}s)")
        
        # 统计包含图片的记忆
        memories_with_images = sum(1 for mem in self.storage.get_all_memories() if mem.image_path)
        logger.info(f"包含图片的记忆: {memories_with_images} 条")
    
    def _load_and_encode_session(self, session_dir_name: str, session_dir: str) -> int:
        """加载单个session并编码存储 - 添加编码时间记录"""
        conversation_file = os.path.join(session_dir, "enhanced_session_en.json")
        if not os.path.exists(conversation_file):   
            logger.warning(f"未找到enhanced_session_en.json文件: {conversation_file}")
            return 0
        
        try:
            with open(conversation_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            session_id = session_dir_name
            
            # 存储session信息
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
            # 处理对话
            dialogues = session_data.get("dialogue", [])
            dialogue_count = 0
            
            # 记录编码时间
            encode_start = time.time()
            
            for i, dialogue in enumerate(dialogues, 1):
                role = dialogue.get("role", "")
                content = dialogue.get("content", {})
                text = timeline_date + ":" + content.get("text", "")
                image_filename = content.get("image", "")
                
                # 获取图片路径
                image_path = None
                if image_filename and session_dir_name in self.image_paths:
                    if image_filename in self.image_paths[session_dir_name]:
                        image_path = self.image_paths[session_dir_name][image_filename]
                # 创建记忆元素
                memory = MemoryElement(
                    memory_id=0,  # 将在添加时分配
                    session_id=session_id,
                    dialogue_index=i,
                    role=role,
                    text=text,
                    image_filename=image_filename,
                    image_path=image_path,
                    timestamp=session_data.get("timeline_date", "")
                )
                
                # 存储到storage
                memory_id = self.storage.add(memory)
                
                # 编码并添加到检索器
                observation = memory.to_observation()
                self.multimodal_retrieval.add(observation, memory_id=memory_id)
                
                # 记录到all_dialogues（用于统计）
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
            
            # 累加编码时间
            self.encoding_time += time.time() - encode_start
            
            # 更新session信息
            self.session_info[session_id]["dialogue_count"] = dialogue_count
            
            logger.debug(f"成功加载并编码 {session_dir_name}，共 {dialogue_count} 条记忆")
            return dialogue_count
            
        except Exception as e:
            logger.error(f"加载 {conversation_file} 失败: {e}")
            return 0
    
    def get_image_path(self, session_id: str, image_filename: str) -> Optional[str]:
        """获取图片的完整路径"""
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
        获取处理好的图片数据（用于API调用）
        模仿新代码的get_image_for_api方法
        """
        # 获取图片路径
        if session_id:
            image_path = self.get_image_path(session_id, image_filename)
        else:
            # 尝试在所有session中查找
            image_path = self.get_image_path(question_id, image_filename) if question_id else None
        print(image_path)
        if not image_path or not os.path.exists(image_path):
            logger.warning(f"图片文件不存在: {image_filename}")
            return None
        
        # 使用ImageProcessor处理图片
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
        检索与查询相关的记忆
        
        Args:
            query: 查询（文本或包含text/image的字典）
            topk: 返回数量
        
        Returns:
            包含文本、图片路径和记忆对象的字典
        """
        return self.recall(query, topk)
    
    def get_memory_by_id(self, memory_id: int) -> Optional[Dict]:
        """通过ID获取记忆"""
        mem = self.storage.get_memory_element_by_mid(memory_id)
        return mem
    
    def get_session_context(self, target_session_id: str) -> Dict[str, Any]:
        """获取session上下文"""
        return {
            "target_session_id": target_session_id,
            "target_session_info": self.session_info.get(target_session_id, {}),
            "all_sessions": list(self.session_info.keys()),
            "total_dialogues": len(self.all_dialogues),
            "memory_system_type": "MultiModalRAG"
        }
    
    def get_full_memory_context(self) -> Dict[str, Any]:
        """获取完整记忆信息（元数据，不是内容）"""
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
        """获取记忆系统统计信息 - 添加存储时间"""
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
            # 新增存储时间统计
            "storage_time": self.storage_time,
            "loading_time": self.loading_time,
            "encoding_time": self.encoding_time,
            "avg_time_per_memory": self.storage_time / len(memories) if memories else 0,
            "avg_encoding_time_per_memory": self.encoding_time / len(memories) if memories else 0
        }

# ==================== 提示词模版 ====================

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

# ==================== 修改后的VLM评估器 ====================

class VLMEvaluator:
    """VLM评估器 - 使用多模态RAG记忆系统（采用新代码风格）"""
    
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
        
        
        # 初始化token计数器
        self.token_counter = TokenCounter()
        
        # 存储统计信息（单线程直接操作，无需锁）
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
                logger.info(f"检索top-k: {self.retrieval_topk}")
                if self.max_images:
                    logger.info(f"最大图片数量: {self.max_images}")
            else:
                logger.warning(f"API连接测试返回非200状态码: {response.status_code}")
        except Exception as e:
            logger.error(f"API连接测试失败: {e}")
    
    def load_questions(self, conversations_dir: str) -> Dict[str, Dict]:
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
                conversation_file = session_dir / "session.json"
                session_id = session_dir_name
                
                with open(question_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                questions = data.get("questions", [])
                question_pairs = []
                
                # 图片目录
                image_dir = session_dir / "image"
                
                for q in questions:
                    q_id = q.get("question_id", f"unknown_{len(question_pairs)}")
                    
                    # 获取问题图片文件名
                    question_image_filename = q.get("question", {}).get("image", "")
                    
                    # 如果问题有图片，构建完整路径并保存到image_context

                    image_context_list = []

                    if question_image_filename:
                        if str(session_dir) == "session0":
                            fold, img_file = question_image_filename.split("/", 1)
                            full_path = scenes_dir / fold / "image" / img_file
                            image_context_list.append(str(full_path))
                            logger.debug(f"问题 {q_id} 的图片完整路径: {full_path}")
                            qa_pair = QuestionAnswerPair(
                                question_id=q_id,
                                session_id=fold,
                                dialogue_name=dialogue_name,
                                question_text=q.get("question", {}).get("text", ""),
                                question_image=question_image_filename,  # 保持原始文件名
                                original_answer=q.get("original_answer", ""),
                                answer_source=q.get("answer_source", "unknown"),
                                answer_session=q.get("answer_session", []),
                                question_type=q.get("question_type", {}),
                                difficulty=q.get("difficulty", "medium"),
                                supporting_evidence=q.get("supporting_evidence", []),
                                image_context=image_context_list  # 这里保存完整路径
                            )
                        else:
                            full_path = image_dir / question_image_filename
                            image_context_list.append(str(full_path))
                            logger.debug(f"问题 {q_id} 的图片完整路径: {full_path}")
                    
                            qa_pair = QuestionAnswerPair(
                                question_id=q_id,
                                session_id=session_id,
                                dialogue_name=dialogue_name,
                                question_text=q.get("question", {}).get("text", ""),
                                question_image=question_image_filename,  # 保持原始文件名
                                original_answer=q.get("original_answer", ""),
                                answer_source=q.get("answer_source", "unknown"),
                                answer_session=q.get("answer_session", []),
                                question_type=q.get("question_type", {}),
                                difficulty=q.get("difficulty", "medium"),
                                supporting_evidence=q.get("supporting_evidence", []),
                                image_context=image_context_list  # 这里保存完整路径
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
                            image_context=[]  # 没有图片
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
        
    def _prepare_query_from_question(self, question_pair: QuestionAnswerPair) -> Union[str, Dict]:
        """
        从问题准备多模态查询
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
        question_type = question_pair.question_type.get("subsub_type", "") if question_pair.question_type else ""
        
        # Instructions for 9 question types (only CD, AR, TTL include abbreviation)
        instructions = {
            "Unimodal Precise Recall": "Accurately recall specific information from the retrieved conversation chunks and answer directly.",
            "Cross-modal Related Retrieval": "Retrieve related information across different modalities (text and images) from the retrieved conversation chunks.",
            "Knowledge Resolution": "Resolve and maintain knowledge consistency based on the retrieved conversation chunks.",
            "Temporal Reasoning": "Reason about temporal relationships and time-based information in the retrieved conversation chunks.",
            "Multimodal Causal Reasoning": "Perform causal reasoning using both text and image information from the retrieved conversation chunks.",
            "Reference & Evolution Tracking": "Track references and their evolution throughout the retrieved conversation chunks.",
            "Test-Time Learning (TTL)": "Learn and adapt from the retrieved conversation chunks at test time to answer the question.",
            "Conflict Detection (CD)": "Check whether this information conflicts with the retrieved conversation chunks.",
            "Answer Refusal (AR)": "Determine if the question can be answered based on the retrieved conversation chunks."
        }
        
        # Return instruction based on question type
        if question_type in instructions:
            return instructions[question_type]
        else:
            return "Answer the question based on the retrieved conversation chunks (appropriate reasoning is allowed)."


    def _get_format_requirement(self, question_type: str) -> str:
        """Get format requirement based on question type"""
        
        format_requirements = {
            "Conflict Detection (CD)": "Response format: Reply strictly with either 'Yes' or 'No' only.",
            "Answer Refusal (AR)": "Response format: If the information is present in the retrieved conversation chunks, provide answer based on that information; if not present, reply with: 'Not mentioned.'",
        }
        
        if question_type in format_requirements:
            return format_requirements[question_type]
        else:
            return "Response format: Provide clear and accurate answers based on the retrieved conversation chunks."
    
    def _prepare_images(self, question_pair: QuestionAnswerPair, 
                    retrieved_memory_objects: List[MemoryElement]) -> Tuple[List[Dict], bool, int, int]:
        """
        准备所有图片，按顺序组织：记忆图片在前，问题图片在后
        返回格式：List[Dict] 包含所有图片，顺序为 [记忆图片..., 问题图片...]
        """
        all_images = []
        memory_images_list = []
        question_images_list = []
        
        question_images_count = 0
        context_images_count = 0
        images_limited = False
        
        # 1. 先处理记忆图片（放在前面）
        # 收集所有记忆图片
        memory_images = []
        for mem in retrieved_memory_objects:
            if mem.image_path and mem.image_filename:
                memory_images.append(mem)
        
        # 计算总图片数量
        total_memory_images = len(memory_images)
        total_question_images = len([f.strip() for f in question_pair.question_image.split(',')]) if question_pair.question_image else 0
        
        # 如果设置了max_images，计算每个部分可以保留的数量
        if self.max_images:
            # 优先保留问题图片，剩余给记忆图片
            if total_question_images >= self.max_images:
                # 问题图片已经超过限制，只保留部分问题图片，记忆图片全部丢弃
                memory_images_to_keep = 0
                question_images_to_keep = self.max_images
                images_limited = True
                logger.info(f"图片总数超过限制，只保留 {self.max_images} 张问题图片，丢弃所有 {total_memory_images} 张记忆图片")
            else:
                # 问题图片未超限，剩余给记忆图片
                question_images_to_keep = total_question_images
                remaining_slots = self.max_images - question_images_to_keep
                memory_images_to_keep = min(total_memory_images, remaining_slots)
                
                if memory_images_to_keep < total_memory_images:
                    images_limited = True
                    logger.info(f"记忆图片数量从 {total_memory_images} 限制到 {memory_images_to_keep} 张")
        else:
            # 无限制，全部保留
            memory_images_to_keep = total_memory_images
            question_images_to_keep = total_question_images
        
        # 2. 处理记忆图片（只保留需要数量的记忆图片）
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
                    # 为记忆图片添加特殊标记
                    img_info["image_type"] = "memory"
                    img_info["marker"] = f"【memory_image-{mem.session_id}-{mem.dialogue_index}-{mem.role}】"
                    memory_images_list.append(img_info)
                    context_images_count += 1
                    logger.debug(f"添加记忆图片: {mem.image_filename} [标记: {img_info.get('marker', '')}]")
        
        # 3. 处理问题图片（放在后面）
        if question_images_to_keep > 0 and question_pair.question_image:
            if str(question_pair.session_id) == "session0":
                print("session0的question_image:", question_pair.question_image)
                fold, img_file = question_pair.question_image.split('/', 1)
                question_files_selected = [img_file]
                img_info = self.memory_system.get_image_for_api(
                        img_file,
                        is_question_image=True,
                        question_id=fold
                    )
                if img_info:
                    # 为问题图片添加特殊标记
                    img_info["image_type"] = "question"
                    # 标记顺序号，帮助LLM理解哪个图片对应哪个问题
                    if len(question_files_selected) > 1:
                        img_info["marker"] = f"【question_image-{question_pair.question_id}-{i+1}/{len(question_files_selected)}】"
                    else:
                        img_info["marker"] = f"【question_image-{question_pair.question_id}】"
                    question_images_list.append(img_info)
                    question_images_count += 1
                    logger.debug(f"添加问题图片: {img_file} [标记: {img_info.get('marker', '')}]")
            else:
                image_files = [f.strip() for f in question_pair.question_image.split(',') if f.strip()]
                
                # 只保留需要数量的问题图片
                question_files_selected = image_files[:question_images_to_keep]
                
                for i, img_file in enumerate(question_files_selected):
                    img_info = self.memory_system.get_image_for_api(
                        img_file,
                        is_question_image=True,
                        question_id=question_pair.session_id
                    )
                    if img_info:
                        # 为问题图片添加特殊标记
                        img_info["image_type"] = "question"
                        # 标记顺序号，帮助LLM理解哪个图片对应哪个问题
                        if len(question_files_selected) > 1:
                            img_info["marker"] = f"【question_image-{question_pair.question_id}-{i+1}/{len(question_files_selected)}】"
                        else:
                            img_info["marker"] = f"【question_image-{question_pair.question_id}】"
                        question_images_list.append(img_info)
                        question_images_count += 1
                        logger.debug(f"添加问题图片: {img_file} [标记: {img_info.get('marker', '')}]")
        
        # 4. 合并图片列表：记忆图片在前，问题图片在后
        all_images = memory_images_list + question_images_list
        
        original_count = total_memory_images + total_question_images
        limited_count = len(all_images)
        
        # 记录图片组织信息
        logger.info(f"图片组织完成: 记忆图片 {len(memory_images_list)} 张, 问题图片 {len(question_images_list)} 张, 总计 {len(all_images)} 张")
        
        return all_images, images_limited, original_count, limited_count
    def _build_prompt(self, question_pair: QuestionAnswerPair, 
                 memory_text: str, has_question_images: bool,
                 memory_images_count: int, question_images_count: int) -> str:
        """
        Build prompt with clear explanation of image order and meaning
        """
        
        # Get question type
        question_type = question_pair.question_type.get("subsub_type", "") if question_pair.question_type else ""
        
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
        
        # 构建消息
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
    
    def _calculate_confidence(self, prediction: str, reference: str, answer_source: str) -> float:
       return 0.7
    
    def evaluate_single_question(self, 
                           question_pair: QuestionAnswerPair,
                           session_id: str,
                           question_file_path: str = None) -> EvaluationResult:
        """评估单个问题 - 添加详细时间计算和错误记录"""
        start_time = time.time()
        
        # 时间记录变量
        retrieval_time = 0.0
        images_prepare_time = 0.0
        prompt_build_time = 0.0
        api_call_time = 0.0
        
        try:
            logger.debug(f"处理问题: {session_id} - {question_pair.question_id} ({question_pair.category})")
            
            # 1. 准备多模态查询
            query = self._prepare_query_from_question(question_pair)
            
            # 2. 检索相关记忆（记录时间）
            retrieval_start = time.time()
            retrieved_result = self.memory_system.retrieve_relevant_memories(
                query, 
                topk=self.retrieval_topk
            )
            retrieval_time = time.time() - retrieval_start
            
            memory_text = retrieved_result.get('text', '无相关记忆')
            memory_objects = retrieved_result.get('memory_objects', [])
            retrieved_chunks = retrieved_result.get('retrieved_chunks', [])
            retrieved_ids = [mem.memory_id for mem in memory_objects if hasattr(mem, 'memory_id')]
            
            # 3. 准备图片（记录时间）
            images_start = time.time()
            images, images_limited, orig_img_count, limited_img_count = self._prepare_images(
                question_pair, memory_objects
            )
            images_prepare_time = time.time() - images_start
            
            # 4. 统计各类图片数量
            memory_images_count = sum(1 for img in images if img.get("image_type") == "memory")
            question_images_count = sum(1 for img in images if img.get("image_type") == "question")
            has_question_images = question_images_count > 0
            
            # 5. 构建提示词（记录时间）
            prompt_start = time.time()
            prompt = self._build_prompt(
                question_pair, 
                memory_text, 
                has_question_images,
                memory_images_count,
                question_images_count
            )
            prompt_build_time = time.time() - prompt_start
            
            # 6. Token截断检查
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
            
            # 7. 调用VLM API（记录时间）
            api_start = time.time()
            vlm_response = self._call_vlm_api(prompt=prompt, images=images)
            api_call_time = vlm_response.get("processing_time", time.time() - api_start)
            
            system_answer = vlm_response.get("answer", "").strip()
            processing_time = vlm_response.get("processing_time", 0)
            success = vlm_response.get("success", False)
            
            # 8. 计算置信度
            confidence = self._calculate_confidence(
                system_answer, 
                question_pair.original_answer,
                question_pair.answer_source
            )
            
            # 总处理时间
            total_processing_time = time.time() - start_time
            
            # 9. 创建评估结果
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
                memory_context_summary=f"检索到 {len(memory_objects)} 条相关记忆",
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
                # 新增时间字段
                retrieval_time=retrieval_time,
                images_prepare_time=images_prepare_time,
                prompt_build_time=prompt_build_time,
                api_call_time=api_call_time
            )
            
            # 更新session统计
            with self.stats_lock:
                self.session_statistics[session_id]["successful"] += 1
                self.session_statistics[session_id]["processing_time"] += total_processing_time
                self.session_statistics[session_id]["total_retrieval_count"] += len(memory_objects)
                # 累计时间统计
                self.session_statistics[session_id]["total_retrieval_time"] += retrieval_time
                self.session_statistics[session_id]["total_images_prepare_time"] += images_prepare_time
                self.session_statistics[session_id]["total_prompt_build_time"] += prompt_build_time
                self.session_statistics[session_id]["total_api_call_time"] += api_call_time
                if images_limited:
                    self.session_statistics[session_id]["images_limited_count"] += 1
            
            logger.info(f"✓ 成功处理: {session_id} - {question_pair.question_id} "
                        f"(总: {total_processing_time:.2f}s, 检索: {retrieval_time:.3f}s, "
                        f"图片: {images_prepare_time:.3f}s, 提示词: {prompt_build_time:.3f}s, "
                        f"API: {api_call_time:.2f}s, 检索到 {len(memory_objects)} 条记忆)")
            
            return result
            
        except Exception as e:
            total_processing_time = time.time() - start_time
            error_msg = str(e)[:200]
            logger.error(f"✗ 处理问题 {session_id} - {question_pair.question_id} 时出错: {error_msg}")
            
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
                memory_type="MultiModalRAG",
                vlm_model=self.model,
                processing_time=total_processing_time,
                confidence=0.0,
                supporting_evidence=question_pair.supporting_evidence,
                memory_context_summary="错误: 无法检索记忆",
                recall_method="multimodal_rag",
                success=False,
                error_message=error_msg,
                truncated=False,
                retrieval_time=retrieval_time,
                images_prepare_time=images_prepare_time,
                prompt_build_time=prompt_build_time,
                api_call_time=api_call_time
            )
            
            with self.stats_lock:
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
        logger.info(f"开始评估 {session_id} 的 {total_questions} 个问题（单线程模式）")
        logger.info(f"使用多模态RAG检索，top-k: {self.retrieval_topk}")
        
        # 初始化session统计
        self.session_statistics[session_id]["total"] = total_questions
        for qa in questions:
            self.session_statistics[session_id]["by_category"][qa.category] += 1
            self.session_statistics[session_id]["by_difficulty"][qa.difficulty] += 1
        
        results = []
        
        # 顺序处理每个问题
        for idx, qa in enumerate(questions, 1):
            logger.debug(f"处理问题 {idx}/{total_questions}: {qa.question_id}")
            result = self.evaluate_single_question(qa, session_id, question_file_path)
            results.append(asdict(result))
            
            if idx % max(1, total_questions // 10) == 0:
                logger.info(f"[{session_id}] 进度: {idx}/{total_questions}")
        
        # 计算平均检索数量
        if total_questions > 0:
            self.session_statistics[session_id]["avg_retrieval_count"] = (
                self.session_statistics[session_id]["total_retrieval_count"] / total_questions
            )
        
        # 保存结果
        self._save_session_results(session_id, session_dir_name, session_path, results)
        
        # 更新全局统计
        self.global_statistics["total_questions"] += total_questions
        self.global_statistics["successful_questions"] += self.session_statistics[session_id]["successful"]
        self.global_statistics["failed_questions"] += self.session_statistics[session_id]["failed"]
        self.global_statistics["images_limited_questions"] += self.session_statistics[session_id]["images_limited_count"]
        
        # 输出session时间统计
        successful = self.session_statistics[session_id]["successful"]
        if successful > 0:
            logger.info(f"Session {session_id} 时间统计 - "
                        f"平均检索: {self.session_statistics[session_id]['total_retrieval_time']/successful:.3f}s, "
                        f"平均图片准备: {self.session_statistics[session_id]['total_images_prepare_time']/successful:.3f}s, "
                        f"平均提示词: {self.session_statistics[session_id]['total_prompt_build_time']/successful:.3f}s, "
                        f"平均API: {self.session_statistics[session_id]['total_api_call_time']/successful:.2f}s")
        
        logger.info(f"完成评估 {session_id}: 成功 {self.session_statistics[session_id]['successful']}, "
                    f"失败 {self.session_statistics[session_id]['failed']}, "
                    f"平均检索 {self.session_statistics[session_id]['avg_retrieval_count']:.1f} 条记忆")
        
        return results
    
    # 修改 evaluate_all_sessions：顺序处理每个session
    def evaluate_all_sessions(self,
                        sessions_questions: Dict[str, Dict],
                        max_questions_per_session: Optional[int] = None,
                        conversations_dir: str = None
                        ):
        self.global_statistics["start_time"] = time.time()
        self.global_statistics["total_sessions"] = len(sessions_questions)
        
        logger.info(f"开始评估 {len(sessions_questions)} 个session（单线程模式）")
        logger.info(f"多模态RAG配置: top-k={self.retrieval_topk}, 最大token={self.max_context_tokens}")
        if self.max_images:
            logger.info(f"最大图片数量: {self.max_images}")
        
        # 获取记忆系统统计信息
        if hasattr(self.memory_system, 'get_statistics'):
            mem_stats = self.memory_system.get_statistics()
            logger.info(f"记忆系统: {mem_stats['total_memories']} 条记忆, "
                        f"{mem_stats['memories_with_images']} 条包含图片")
        
        # 顺序处理每个session
        for session_id, session_data in sessions_questions.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"处理 Session: {session_id}")
            logger.info(f"问题数: {len(session_data['questions'])}")
            logger.info(f"{'='*60}")
            
            results = self.evaluate_session_questions(
                session_id,
                session_data,
                max_questions_per_session
            )
            logger.info(f"Session {session_id} 处理完成，生成 {len(results)} 条结果")
        
        self.global_statistics["end_time"] = time.time()
    
    # 修改 _save_session_results：移除文件锁
    def _save_session_results(self,
                            session_id: str,
                            session_dir_name: str,
                            session_path: Path,
                            results: List[Dict[str, Any]]):
        session_results_dir = session_path / "evaluation_results"
        session_results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存JSON结果
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
        
        # 直接写文件，无锁
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"已保存 {session_id} 的结果到: {json_file}")
    
    

# ==================== 创建记忆系统的工厂函数 ====================

def create_memory_system(memory_type: str, conversations_dir: str, **kwargs):
    """创建记忆系统的工厂函数"""
    if memory_type == "multimodal_rag":
        config = kwargs.get('config', {})
        return MultiModalRAGMemorySystem(conversations_dir, config)
    else:
        raise ValueError(f"不支持的记忆类型: {memory_type}")


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="VLM记忆能力评估器（多模态RAG版）")
    parser.add_argument("--conversations_dir", type=str, required=True,
                       help="对话数据目录（必需）")
    parser.add_argument("--api_key", type=str, required=True,
                       help="VLM API密钥（必需）")
    parser.add_argument("--model", type=str, required=True,
                       help="VLM模型名称（必需）")
    parser.add_argument("--base_url", type=str, required=True,
                       help="API基础URL（必需）")
    parser.add_argument("--memory_type", type=str, default="multimodal_rag",
                       choices=["multimodal_rag"],
                       help="记忆系统类型")
    parser.add_argument("--max_questions_per_session", type=int, default=None,
                       help="每个session最大处理问题数")
    parser.add_argument("--max_sessions", type=int, default=None,
                       help="最大处理session数")
    parser.add_argument("--verbose", action="store_true",
                       help="详细日志输出")
    parser.add_argument("--test_mode", action="store_true",
                       help="测试模式")
    parser.add_argument("--max_retries", type=int, default=3,
                       help="API调用最大重试次数")
    parser.add_argument("--timeout", type=int, default=60,
                       help="API调用超时时间（秒）")
    parser.add_argument("--max_context_tokens", type=int, default=4096,
                       help="最大上下文token数")
    
    # RAG特定参数
    parser.add_argument("--retrieval_topk", type=int, default=10,
                       help="检索返回的记忆数量")
    parser.add_argument("--encoder_model", type=str, default="Alibaba-NLP/gme-Qwen2-VL-7B-Instruct",
                       help="多模态编码器模型")
    parser.add_argument("--retrieval_mode", type=str, default="cosine",
                       choices=["cosine", "dot", "L2"],
                       help="检索相似度计算模式")
    parser.add_argument("--utilization_method", type=str, default="MultiModalUtilization",
                       choices=["ConcateUtilization", "MultiModalUtilization"],
                       help="记忆利用策略")
    
    # 图片限制参数
    parser.add_argument("--max_images", type=int, default=None,
                       help="最大图片数量")
    
    args = parser.parse_args()
    
    # 配置日志级别
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    print("=" * 70)
    print("VLM记忆能力评估器（多模态RAG版）")
    print(f"模型: {args.model}")
    print(f"记忆系统: {args.memory_type}")
    if args.memory_type == "multimodal_rag":
        print(f"  编码器: {args.encoder_model}")
        print(f"  检索模式: {args.retrieval_mode}, top-k: {args.retrieval_topk}")
        print(f"  利用策略: {args.utilization_method}")
    print(f"Session并行: {args.max_workers}, API并发: {args.max_api_concurrency}")
    if args.max_context_tokens:
        print(f"对话截断: {args.max_context_tokens} tokens")
    if args.max_images:
        print(f"图片限制: {args.max_images} 张")
    print("=" * 70)
    
    # 测试模式设置
    if args.test_mode:
        args.max_questions_per_session = 2
        print("测试模式：每个session只处理前2个问题")
    
    # 1. 初始化记忆系统
    print(f"\n[1] 初始化记忆系统 ({args.memory_type})...")
    
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
    
    print(f"   加载并编码所有对话...")
    memory_system.load_all_conversations()
    
    # 显示统计信息
    if args.memory_type == "multimodal_rag":
        stats = memory_system.get_statistics()
        print(f"   已加载 {stats['total_sessions']} 个session，共 {stats['total_memories']} 条记忆")
        print(f"   包含图片的记忆: {stats['memories_with_images']} 条")
    else:
        print(f"   已加载 {len(memory_system.session_info)} 个session，共 {len(memory_system.all_dialogues)} 轮对话")
    
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
        max_images=args.max_images,
        retrieval_topk=args.retrieval_topk,
        max_workers=args.max_workers,
        max_api_concurrency=args.max_api_concurrency
    )
    
    # 3. 加载问题
    print(f"\n[3] 加载intra-session问题文件...")
    try:
        sessions_questions = evaluator.load_questions(args.conversations_dir)
    except Exception as e:
        print(f"   加载问题失败: {e}")
        return
    
    if not sessions_questions:
        print("   未找到任何session的问题文件")
        return
    
    print(f"   成功从 {len(sessions_questions)} 个session加载了问题")
    total_questions = sum(len(data["questions"]) for data in sessions_questions.values())
    print(f"   总问题数: {total_questions}")
    
    # 限制处理的session数
    if args.max_sessions and args.max_sessions < len(sessions_questions):
        sessions_to_process = dict(list(sessions_questions.items())[:args.max_sessions])
        print(f"   限制处理前 {args.max_sessions} 个session")
    else:
        sessions_to_process = sessions_questions
    
    # 4. 执行评估
    print(f"\n[4] 开始按session评估（使用多模态RAG记忆）...")
    print(f"   处理session数: {len(sessions_to_process)}")
    print(f"   总问题数: {total_questions}")
    print("-" * 70)
    evaluator.evaluate_all_sessions(
        sessions_questions=sessions_to_process,
        max_questions_per_session=args.max_questions_per_session,
        conversations_dir=args.conversations_dir
    )
    
    # 5. 输出统计
    print(f"\n[5] 评估完成!")
    print("-" * 70)
    

if __name__ == "__main__":
    main()