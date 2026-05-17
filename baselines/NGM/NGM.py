import os
import json
import logging
import torch
import numpy as np
import argparse
import requests
import time
import base64
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from collections import deque, defaultdict
from datetime import datetime
from pathlib import Path
from io import BytesIO
from PIL import Image
from dataclasses import dataclass, asdict, field
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor
from natsort import natsorted
# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==================== 编码器基类 ====================

class BaseMultiModalEncoder:
    """多模态编码器基类"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.processor = None
    
    def _load_image(self, image_path_or_url: str) -> Image.Image:
        """加载图像（由子类实现）"""
        raise NotImplementedError
    
    def encode_text(self, text: str, return_type: str = 'numpy') -> Union[np.ndarray, torch.Tensor]:
        """编码文本（由子类实现）"""
        raise NotImplementedError
    
    def encode_image(self, image_path_or_url: str, return_type: str = 'numpy') -> Union[np.ndarray, torch.Tensor]:
        """编码图像（由子类实现）"""
        raise NotImplementedError
    
    def encode_multimodal(self, text: Optional[str] = None, image: Optional[Dict] = None, 
                         return_type: str = 'numpy') -> Union[np.ndarray, torch.Tensor]:
        """编码多模态数据（由子类实现）"""
        raise NotImplementedError
    
    def __call__(self, obj: Union[str, Dict], return_type: str = 'numpy') -> Union[np.ndarray, torch.Tensor]:
        """统一调用接口（由子类实现）"""
        raise NotImplementedError


# ==================== GME编码器实现 ====================

class GMEEncoder(BaseMultiModalEncoder):
    """
    GME (General Multimodal Embedding) Qwen2-VL-based encoder for text and images.
    Supports unified multimodal representations for Any2Any Search.
    """
    def __init__(self, config):
        super().__init__(config)
        
        model_name = getattr(config, 'path', 'Alibaba-NLP/gme-Qwen2-VL-7B-Instruct')
        self.image_base_path = getattr(config, 'image_base_path', '')
        self.embedding_dim = getattr(config, 'embedding_dim', 4096)  # GME-Qwen2-VL-7B维度
        
        print(f"正在加载GME模型: {model_name}")
        print(f"使用设备: {self.device}")
        
        # 加载GME模型 with trust_remote_code=True
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
            device_map='auto',
            trust_remote_code=True
        )
        self.model.eval()  # Set to evaluation mode
        
        # 加载处理器
        try:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        except:
            self.processor = None
            
        print(f"GME模型加载成功！")
        print(f"  嵌入维度: {self.embedding_dim}")
    
    def _load_image(self, image_path_or_url):
        """Load image from local path or URL."""
        # If already absolute path or URL, use directly
        if os.path.isabs(image_path_or_url) or image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://'):
            final_path = image_path_or_url
        else:
            # Only add prefix for relative paths
            final_path = os.path.join(self.image_base_path, image_path_or_url)
        
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
            print(f"Error loading image {image_path_or_url} (tried {final_path}): {e}")
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
            if isinstance(image, dict):
                image_path = image.get('path')
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
            if isinstance(image, dict):
                image_path = image.get('path')
            else:
                image_path = image
            return self.encode_image(image_path, return_type)
        
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


# ==================== CLIP编码器实现（备选） ====================

class CLIPEncoder(BaseMultiModalEncoder):
    """
    CLIP-based multimodal encoder for text and images.
    """
    def __init__(self, config):
        super().__init__(config)
        
        model_name = getattr(config, 'path', 'openai/clip-vit-base-patch32')
        self.image_base_path = getattr(config, 'image_base_path', '')
        self.embedding_dim = getattr(config, 'embedding_dim', 512)
        
        print(f"正在加载CLIP模型: {model_name}")
        print(f"使用设备: {self.device}")
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        
        print(f"CLIP模型加载成功！")
        print(f"  嵌入维度: {self.embedding_dim}")
    
    def _load_image(self, image_path_or_url):
        """Load image from local path or URL."""
        if os.path.isabs(image_path_or_url) or image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://'):
            final_path = image_path_or_url
        else:
            final_path = os.path.join(self.image_base_path, image_path_or_url)
        
        try:
            if final_path.startswith('http://') or final_path.startswith('https://'):
                response = requests.get(final_path, timeout=10)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                if not os.path.exists(final_path):
                    raise FileNotFoundError(f"Image file not found: {final_path}")
                image = Image.open(final_path).convert('RGB')
            return image
        except Exception as e:
            print(f"Error loading image {image_path_or_url}: {e}")
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
        Encode multimodal data (text and/or image).
        If both are provided, average the embeddings.
        """
        embeddings = []
        
        if text is not None and text.strip() != '':
            text_emb = self.encode_text(text, return_type='tensor')
            embeddings.append(text_emb)
        
        if image is not None:
            if isinstance(image, dict):
                image_path = image.get('path')
            else:
                image_path = image
            image_emb = self.encode_image(image_path, return_type='tensor')
            embeddings.append(image_emb)
        
        if not embeddings:
            return self.encode_text(" ", return_type=return_type)
        
        # Average the embeddings if both modalities are present
        if len(embeddings) > 1:
            combined = torch.mean(torch.stack(embeddings), dim=0)
        else:
            combined = embeddings[0]
        
        # Normalize
        combined = combined / combined.norm(dim=-1, keepdim=True)
        
        if return_type == 'numpy':
            return combined.cpu().numpy()
        elif return_type == 'tensor':
            return combined
        else:
            raise ValueError(f"Unrecognized return type: {return_type}")
    
    def __call__(self, obj, return_type='numpy'):
        if isinstance(obj, str):
            return self.encode_text(obj, return_type)
        elif isinstance(obj, dict):
            text = obj.get('text', '')
            image = obj.get('image', None)
            return self.encode_multimodal(text, image, return_type)
        else:
            raise ValueError(f"Unsupported input type: {type(obj)}")


# ==================== 图存储部分 ====================

class GraphStorage:
    """
    图结构记忆存储
    """
    def __init__(self, config):
        self.config = config
        self.node = {}  # node_id -> node_data
        self.edge = {}  # source_id -> {target_id -> edge_data}
        self.node_counter = 0
        self.edge_counter = 0
        self.memory_order_map = []  # 按插入顺序的node_id列表

    def reset(self):
        self.node = {}
        self.edge = {}
        self.node_counter = 0
        self.edge_counter = 0
        self.memory_order_map = []

    def get_element_number(self):
        return len(self.node)

    def is_empty(self):
        return self.get_element_number() == 0
    
    def get_node_id_by_mid(self, mid):
        """根据记忆索引获取节点ID"""
        if 0 <= mid < len(self.memory_order_map):
            return self.memory_order_map[mid]
        raise IndexError(f"Memory index {mid} out of range")

    def get_mid_by_node_id(self, node_id):
        """根据节点ID获取记忆索引"""
        if node_id in self.node:
            return self.node[node_id].get('mid', -1)
        raise KeyError(f"Node {node_id} not found")

    def get_memory_element_by_node_id(self, node_id):
        """根据节点ID获取记忆元素"""
        return self.node.get(node_id, {})

    def get_memory_element_by_mid(self, mid):
        """根据记忆索引获取记忆元素"""
        node_id = self.get_node_id_by_mid(mid)
        return self.node[node_id]

    def get_memory_text_by_node_id(self, node_id):
        """获取节点文本"""
        return self.node[node_id].get('text', '')

    def get_memory_image_by_node_id(self, node_id):
        """获取节点图片信息"""
        return self.node[node_id].get('image', None)
    
    def get_neighbors(self, node_id):
        """获取节点的邻居节点ID列表"""
        if node_id in self.edge:
            return list(self.edge[node_id].keys())
        return []
    
    def get_edges_from(self, node_id):
        """获取从节点出发的所有边"""
        return self.edge.get(node_id, {})
    
    def get_edges_to(self, node_id):
        """获取指向节点的所有边"""
        edges_to = {}
        for source, targets in self.edge.items():
            if node_id in targets:
                edges_to[source] = targets[node_id]
        return edges_to
    
    def get_node_degree(self, node_id):
        """获取节点的度数（出度+入度）"""
        out_degree = len(self.get_neighbors(node_id))
        in_degree = len(self.get_edges_to(node_id))
        return out_degree + in_degree

    def add_node(self, obj):
        """添加节点"""
        if 'text' not in obj and 'image' not in obj:
            raise ValueError("Node must contain at least 'text' or 'image'")
        
        obj['node_id'] = self.node_counter
        obj['mid'] = len(self.memory_order_map)
        obj['timestamp'] = obj.get('timestamp', time.time())
        
        self.node[self.node_counter] = obj
        self.memory_order_map.append(self.node_counter)
        self.node_counter += 1
        return self.node_counter - 1

    def add_edge(self, s, t, obj):
        """添加边"""
        obj['edge_id'] = self.edge_counter
        if s not in self.edge:
            self.edge[s] = {}
        self.edge[s][t] = obj
        self.edge_counter += 1
        return self.edge_counter - 1
    
    def display(self):
        """显示图信息"""
        return f"GraphStorage: {self.get_element_number()} nodes, {self.edge_counter} edges"


# ==================== 多模态检索部分 ====================

class MultiModalRetrieval:
    """
    多模态检索器
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化编码器
        encoder_method = getattr(config.encoder, 'method', 'GMEEncoder')
        if encoder_method == 'CLIPEncoder':
            self.encoder = CLIPEncoder(config.encoder)
        elif encoder_method == 'GMEEncoder':
            self.encoder = GMEEncoder(config.encoder)
        else:
            raise ValueError(f"Unsupported encoder: {encoder_method}")
        
        # 存储所有记忆的嵌入向量
        self.tensorstore = None
        
        # 存储元数据
        self.memory_metadata = []
    
    def reset(self):
        self.tensorstore = None
        self.memory_metadata = []
    
    def __normalize__(self, embedding):
        """L2归一化"""
        return torch.nn.functional.normalize(embedding, dim=-1)
    
    def add(self, obj):
        """
        添加记忆到检索器
        
        Args:
            obj: 记忆对象（文本、图片路径或字典）
        
        Returns:
            嵌入向量
        """
        embedding = self.encoder(obj, return_type='tensor')
        
        if self.config.mode == 'cosine':
            embedding = self.__normalize__(embedding)
        
        if self.tensorstore is None:
            self.tensorstore = embedding
        else:
            self.tensorstore = torch.cat([self.tensorstore, embedding], dim=0)
        
        # 记录元数据
        metadata = {
            'has_text': isinstance(obj, str) or (isinstance(obj, dict) and 'text' in obj),
            'has_image': isinstance(obj, dict) and 'image' in obj
        }
        self.memory_metadata.append(metadata)
        
        return embedding
    
    def __calculate_scores__(self, query):
        """计算查询与所有记忆的相似度"""
        query_embedding = self.encoder(query, return_type='tensor')
        
        if self.config.mode == 'cosine':
            query_embedding = self.__normalize__(query_embedding)
        
        if self.config.mode in ['cosine', 'dot']:
            scores = torch.matmul(self.tensorstore, query_embedding.squeeze())
        elif self.config.mode == 'L2':
            scores = -torch.norm(self.tensorstore - query_embedding.squeeze(), p=2, dim=1)
        else:
            raise ValueError(f"Unsupported mode: {self.config.mode}")
        
        return scores
    
    def __call__(self, query, topk='config', with_score=False, sort=True):
        """
        检索最相似的记忆
        
        Args:
            query: 查询（文本、图片路径或字典）
            topk: 返回数量
            with_score: 是否返回分数
            sort: 是否排序
        
        Returns:
            记忆索引列表或(分数, 索引)元组
        """
        if self.tensorstore is None or self.tensorstore.size(0) == 0:
            return torch.tensor([]) if not with_score else (torch.tensor([]), torch.tensor([]))
        
        scores = self.__calculate_scores__(query)
        
        if sort:
            scores, indices = torch.sort(scores, descending=True)
        else:
            indices = torch.arange(self.tensorstore.size(0))
        
        if topk == 'config':
            k = min(self.config.topk, self.tensorstore.size(0))
            scores = scores[:k]
            indices = indices[:k]
        elif isinstance(topk, int):
            k = min(topk, self.tensorstore.size(0))
            scores = scores[:k]
            indices = indices[:k]
        
        if with_score:
            return scores, indices
        else:
            return indices


# ==================== 存储操作 ====================

class NGMemorySystemStore:
    """
    Neural Graph Memory Store:
    将多模态事件存储为图节点，基于关系创建边
    """
    def __init__(self, config, **kwargs):
        self.config = config
        self.storage = kwargs['storage']
        self.multimodal_retrieval = kwargs['multimodal_retrieval']
        
        # 图构建参数
        self.similarity_threshold = getattr(config, 'similarity_threshold', 0.7)
        self.max_edges_per_node = getattr(config, 'max_edges_per_node', 5)
        self.temporal_decay_constant = getattr(config, 'temporal_decay_constant', 3600)
        
        # 添加存储时间统计
        self.store_times = []  # 记录每次存储的时间
        self.total_store_time = 0.0
        self.num_stores = 0

    def reset(self):
        self.store_times = []
        self.total_store_time = 0.0
        self.num_stores = 0

    def __calculate_similarity__(self, embedding1, embedding2):
        """计算余弦相似度"""
        emb1_norm = torch.nn.functional.normalize(embedding1, dim=-1)
        emb2_norm = torch.nn.functional.normalize(embedding2, dim=-1)
        return torch.matmul(emb1_norm, emb2_norm.T).item()
    
    def __calculate_temporal_weight__(self, source_node_id, target_node_id):
        """
        计算时间权重：使用指数衰减，时间越近权重越高
        """
        import numpy as np
        
        try:
            source_data = self.storage.get_memory_element_by_node_id(source_node_id)
            target_data = self.storage.get_memory_element_by_node_id(target_node_id)
            
            source_time = source_data.get('timestamp', 0)
            target_time = target_data.get('timestamp', 0)
            
            if isinstance(source_time, str):
                try:
                    source_time = datetime.fromisoformat(source_time.replace('Z', '+00:00')).timestamp()
                except:
                    source_time = float(source_data.get('mid', 0))
            
            if isinstance(target_time, str):
                try:
                    target_time = datetime.fromisoformat(target_time.replace('Z', '+00:00')).timestamp()
                except:
                    target_time = float(target_data.get('mid', 0))
            
            time_diff = abs(target_time - source_time)
            
            # 指数衰减：np.exp(-time_diff / decay_constant)
            return np.exp(-time_diff / self.temporal_decay_constant)
        except Exception as e:
            logger.debug(f"时间权重计算失败: {e}")
            return 1.0
    
    def __create_edges__(self, new_node_id, new_embedding):
        """为新节点创建边"""
        if self.storage.get_element_number() <= 1:
            return
        
        all_node_ids = [nid for nid in self.storage.memory_order_map if nid != new_node_id]
        
        if not all_node_ids:
            return

        # 1. 创建时间顺序边（与最近添加的节点）
        if len(all_node_ids) > 0:
            last_node_id = all_node_ids[-1]
            temporal_weight = self.__calculate_temporal_weight__(last_node_id, new_node_id)
            
            self.storage.add_edge(
                last_node_id,
                new_node_id,
                {
                    'type': 'temporal_succession',
                    'weight': float(temporal_weight),
                    'direction': 'forward'
                }
            )
        
        # 2. 创建语义相似边
        similarities = []
        for node_id in all_node_ids:
            node_mid = self.storage.get_mid_by_node_id(node_id)
            if (self.multimodal_retrieval.tensorstore is not None and
                node_mid < self.multimodal_retrieval.tensorstore.size(0)):
                
                node_embedding = self.multimodal_retrieval.tensorstore[node_mid:node_mid+1]
                similarity = self.__calculate_similarity__(new_embedding, node_embedding)
                similarities.append((node_id, similarity))

        # 选择最相似的节点创建边
        similarities.sort(key=lambda x: x[1], reverse=True)
        for node_id, similarity in similarities[:self.max_edges_per_node]:
            if similarity >= self.similarity_threshold:
                self.storage.add_edge(
                    node_id,
                    new_node_id,
                    {
                        'type': 'semantic_similarity',
                        'weight': float(similarity),
                        'direction': 'bidirectional'
                    }
                )

    def __call__(self, observation):
        """
        存储多模态观察为图节点
        
        Returns:
            tuple: (node_id, store_time) 节点ID和存储耗时
        """
        store_start_time = time.time()
        
        # 确保有时间戳
        if 'timestamp' not in observation:
            observation['timestamp'] = time.time()
        
        # 添加到图存储
        node_id = self.storage.add_node(observation)
        
        # 编码并添加到多模态检索器
        embedding = self.multimodal_retrieval.add(observation)
        
        # 创建边
        self.__create_edges__(node_id, embedding)
        
        store_time = time.time() - store_start_time
        
        # 记录存储时间
        self.store_times.append(store_time)
        self.total_store_time += store_time
        self.num_stores += 1
        
        logger.debug(f"已存储节点 {node_id}: {observation.get('text', '')[:50]}..., 耗时: {store_time:.4f}s")
        return node_id, store_time
    
    def get_store_statistics(self):
        """获取存储时间统计"""
        if self.num_stores == 0:
            return {
                'num_stores': 0,
                'total_time': 0.0,
                'avg_time': 0.0,
                'min_time': 0.0,
                'max_time': 0.0
            }
        
        return {
            'num_stores': self.num_stores,
            'total_time': self.total_store_time,
            'avg_time': self.total_store_time / self.num_stores,
            'min_time': min(self.store_times) if self.store_times else 0.0,
            'max_time': max(self.store_times) if self.store_times else 0.0
        }
    def get_all_store_times(self):
        """获取所有存储时间记录"""
        return {
            'store_times': self.store_times,
            'total_store_time': self.total_store_time,
            'num_stores': self.num_stores,
            'avg_store_time': self.total_store_time / self.num_stores if self.num_stores > 0 else 0,
            'min_store_time': min(self.store_times) if self.store_times else 0,
            'max_store_time': max(self.store_times) if self.store_times else 0
        }


# ==================== 召回操作 ====================

class NGMemorySystemRecall:
    """
    Neural Graph Memory Recall:
    基于查询的图遍历记忆检索
    """
    def __init__(self, config, **kwargs):
        self.config = config
        self.storage = kwargs['storage']
        self.multimodal_retrieval = kwargs['multimodal_retrieval']
        
        # 图遍历参数
        self.max_depth = getattr(config, 'max_depth', 3)
        self.max_nodes = getattr(config, 'max_nodes', 10)
        self.traversal_threshold = getattr(config, 'traversal_threshold', 0.5)
        self.traversal_strategy = getattr(config, 'traversal_strategy', 'breadth_first')
        self.initial_candidate_multiplier = getattr(config, 'initial_candidate_multiplier', 2)
        
        # 检索结果缓存
        self.last_retrieved_ids = []
        self.last_reasoning_paths = {}
        self.last_reasoning_details = {}

    def reset(self):
        self.last_retrieved_ids = []
        self.last_reasoning_paths = {}
        self.last_reasoning_details = {}

    def __graph_traversal_depth_first__(self, query_embedding, start_node_ids, visited=None, depth=0):
        """
        深度优先图遍历
        """
        if visited is None:
            visited = set()
        
        if depth >= self.max_depth:
            return []
        
        candidates = []
        
        for node_id in start_node_ids:
            if node_id in visited:
                continue
            
            visited.add(node_id)

            # 获取节点嵌入
            node_mid = self.storage.get_mid_by_node_id(node_id)
            if node_mid >= self.multimodal_retrieval.tensorstore.size(0):
                continue
            
            node_embedding = self.multimodal_retrieval.tensorstore[node_mid:node_mid+1]

            # 计算与查询的相似度
            query_norm = torch.nn.functional.normalize(query_embedding, dim=-1)
            node_norm = torch.nn.functional.normalize(node_embedding, dim=-1)
            similarity = torch.matmul(query_norm, node_norm.T).item()

            # 如果超过阈值，加入候选
            if similarity >= self.traversal_threshold:
                candidates.append((node_id, similarity))

            # 继续遍历邻居
            neighbors = self.storage.get_neighbors(node_id)
            if neighbors:
                neighbor_candidates = self.__graph_traversal_depth_first__(
                    query_embedding, neighbors, visited, depth + 1
                )
                candidates.extend(neighbor_candidates)
        
        return candidates
    
    def __graph_traversal_breadth_first__(self, query_embedding, initial_node_ids):
        """
        广度优先图遍历（匹配原始实现）
        """
        import torch

        # 1. 扩展搜索：添加所有初始节点及其邻居
        expanded_nodes = set()
        for node_id in initial_node_ids:
            expanded_nodes.add(node_id)
            neighbors = self.storage.get_neighbors(node_id)
            for neighbor in neighbors:
                expanded_nodes.add(neighbor)

        # 2. 为所有扩展节点重新计算相似度
        final_similarities = []
        for node_id in expanded_nodes:
            try:
                node_mid = self.storage.get_mid_by_node_id(node_id)
                if node_mid >= self.multimodal_retrieval.tensorstore.size(0):
                    continue
                
                node_embedding = self.multimodal_retrieval.tensorstore[node_mid:node_mid+1]

                # 计算余弦相似度
                query_norm = torch.nn.functional.normalize(query_embedding, dim=-1)
                node_norm = torch.nn.functional.normalize(node_embedding, dim=-1)
                similarity = torch.matmul(query_norm, node_norm.T).item()
                
                # 应用度数提升
                node_degree = self.storage.get_node_degree(node_id)
                degree_boost = 1 + (node_degree * 0.1)
                boosted_similarity = similarity * degree_boost
                
                final_similarities.append((node_id, boosted_similarity))
            except Exception:
                continue

        # 3. 按相似度排序
        final_similarities.sort(key=lambda x: x[1], reverse=True)
        return final_similarities
    
    def __graph_traversal__(self, query_embedding, start_node_ids, visited=None, depth=0):
        """
        图遍历入口方法
        """
        if self.traversal_strategy == 'breadth_first':
            return self.__graph_traversal_breadth_first__(query_embedding, start_node_ids)
        else:
            return self.__graph_traversal_depth_first__(query_embedding, start_node_ids, visited, depth)

    def __call__(self, query):
        """
        基于查询的图遍历记忆召回
        """
        if self.storage.is_empty():
            return []
        
        logger.debug(f"召回查询: {query if isinstance(query, str) else str(query)[:100]}...")

        # 1. 首先使用嵌入相似度找到最相关的起始节点
        initial_topk = 3
        if self.traversal_strategy == 'breadth_first':
            initial_topk = min(initial_topk * self.initial_candidate_multiplier, 
                              self.storage.get_element_number())
        ranking_ids = self.multimodal_retrieval(
            query, 
            topk=min(initial_topk, self.storage.get_element_number())
        )
        if len(ranking_ids) == 0:
            return []

        # 2. 编码查询
        query_embedding = self.multimodal_retrieval.encoder(query, return_type='tensor')
        if self.multimodal_retrieval.config.mode == 'cosine':
            query_embedding = self.multimodal_retrieval.__normalize__(query_embedding)

        # 3. 获取起始节点ID
        start_node_ids = []
        for mid in ranking_ids:
            try:
                node_id = self.storage.get_node_id_by_mid(int(mid))
                start_node_ids.append(node_id)
            except (KeyError, IndexError):
                continue
        # 4. 执行图遍历
        traversal_results = self.__graph_traversal__(query_embedding, start_node_ids)

        # 5. 合并初始检索结果和遍历结果
        all_candidates = {}

        if self.traversal_strategy == 'breadth_first':
            # 广度优先：直接使用遍历结果
            for node_id, boosted_similarity in traversal_results:
                all_candidates[node_id] = boosted_similarity
        else:
            # 深度优先：合并结果并应用度数提升
            import torch
            
            # 添加初始检索结果
            for mid in ranking_ids:
                try:
                    node_id = self.storage.get_node_id_by_mid(int(mid))
                    node_mid = self.storage.get_mid_by_node_id(node_id)
                    
                    if node_mid < self.multimodal_retrieval.tensorstore.size(0):
                        node_embedding = self.multimodal_retrieval.tensorstore[node_mid:node_mid+1]
                        query_norm = torch.nn.functional.normalize(query_embedding, dim=-1)
                        node_norm = torch.nn.functional.normalize(node_embedding, dim=-1)
                        similarity = torch.matmul(query_norm, node_norm.T).item()
                        
                        # 应用度数提升
                        node_degree = self.storage.get_node_degree(node_id)
                        degree_boost = 1 + (node_degree * 0.1)
                        all_candidates[node_id] = similarity * degree_boost
                except Exception:
                    all_candidates[node_id] = 1.0

            # 添加遍历结果
            for node_id, similarity in traversal_results:
                node_degree = self.storage.get_node_degree(node_id)
                degree_boost = 1 + (node_degree * 0.1)
                boosted_similarity = similarity * degree_boost
                
                if node_id not in all_candidates:
                    all_candidates[node_id] = boosted_similarity
                else:
                    all_candidates[node_id] = max(all_candidates[node_id], boosted_similarity)

        # 6. 排序并选择top-k
        sorted_candidates = sorted(all_candidates.items(), key=lambda x: x[1], reverse=True)
        selected_node_ids = [node_id for node_id, _ in sorted_candidates[:self.max_nodes]]
        
        if not selected_node_ids:
            return []

        # 7. 收集记忆
        memories = []
        retrieved_ids = []
        for node_id in selected_node_ids:
            mem = self.storage.get_memory_element_by_node_id(node_id)
            memories.append(mem)
            if 'dialogue_id' in mem:
                retrieved_ids.append(mem['dialogue_id'])

        # 记录检索结果
        self.last_retrieved_ids = retrieved_ids
        
        logger.debug(f"召回 {len(memories)} 个记忆节点")
        return memories


# ==================== 主NGM内存类 ====================

class NGMemorySystem:
    """
    Neural Graph Memory (NGM)
    神经图记忆系统
    
    Reference: Neural Graph Memory: A Structured Approach to Long-Term Memory in Multimodal Agents
    """
    def __init__(self, config):
        self.config = config
        
        # 图存储
        self.storage = GraphStorage(config.storage)
        
        # 多模态检索器
        self.multimodal_retrieval = MultiModalRetrieval(config.multimodal_retrieval)
        
        # 存储操作
        self.store_op = NGMemorySystemStore(
            config.store,
            storage=self.storage,
            multimodal_retrieval=self.multimodal_retrieval
        )
        
        # 召回操作
        self.recall_op = NGMemorySystemRecall(
            config.recall,
            storage=self.storage,
            multimodal_retrieval=self.multimodal_retrieval
        )
    
    def reset(self):
        """重置所有组件"""
        self.storage.reset()
        self.multimodal_retrieval.reset()
        self.store_op.reset()
        self.recall_op.reset()

    def store(self, observation):
        """
        存储观察（文本、图片或两者）为图节点
        
        Returns:
            tuple: (node_id, store_time) 节点ID和存储耗时
        """
        return self.store_op(observation)
    
    def recall(self, query):
        """
        基于查询召回相关记忆
        """
        return self.recall_op(query)
    
    def get_graph_statistics(self):
        """获取图统计信息"""
        if self.storage.get_element_number() == 0:
            return {
                'total_nodes': 0,
                'total_edges': 0,
                'avg_degree': 0,
                'config': {
                    'similarity_threshold': self.store_op.similarity_threshold,
                    'max_edges_per_node': self.store_op.max_edges_per_node,
                    'traversal_strategy': self.recall_op.traversal_strategy,
                    'max_depth': self.recall_op.max_depth
                }
            }
        
        return {
            'total_nodes': self.storage.get_element_number(),
            'total_edges': self.storage.edge_counter,
            'avg_degree': sum(self.storage.get_node_degree(nid) 
                             for nid in self.storage.node) / self.storage.get_element_number(),
            'config': {
                'similarity_threshold': self.store_op.similarity_threshold,
                'max_edges_per_node': self.store_op.max_edges_per_node,
                'traversal_strategy': self.recall_op.traversal_strategy,
                'max_depth': self.recall_op.max_depth
            }
        }
    
    def get_store_statistics(self):
        """获取存储时间统计"""
        return self.store_op.get_store_statistics()


# ==================== 配置类 ====================

class Config:
    """配置类"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                setattr(self, key, Config(**value))
            else:
                setattr(self, key, value)


# ==================== 对话加载器 ====================

class DialogueLoader:
    """对话加载器"""
    
    @staticmethod
    def load_conversations(conversations_dir: str) -> List[Dict]:
        """加载所有对话"""
        conversations = []
        base_dir = Path(conversations_dir)
        
        scenes_dir = os.path.join(base_dir, "scenes")
        if not os.path.exists(scenes_dir):
            raise ValueError(f"找不到scenes目录: {scenes_dir}")
            
        session_dirs = natsorted([
            d for d in os.listdir(scenes_dir) 
            if os.path.isdir(os.path.join(scenes_dir, d))
        ])
        # 遍历所有session
        for session_dir_name in session_dirs:
            session_dir = os.path.join(scenes_dir, session_dir_name)
            session_dir = Path(session_dir)
            conv_file = os.path.join(session_dir, "session.json")
            if not os.path.exists(conv_file):
                logger.warning(f"未找到session.json文件: {conv_file}")
                continue
            
            try:
                with open(conv_file, 'r', encoding='utf-8') as f:
                    conv_data = json.load(f)
                
                session_id = session_dir_name
                dialogues = conv_data.get("dialogue", [])
                timeline_date = conv_data.get("timeline_date", "")
                for i, dialogue in enumerate(dialogues):
                    # 构建观察对象
                    observation = {
                        'session_id': session_id,
                        'dialogue_id': f"{session_id}_{i}",
                        'dialogue_index': i,
                        'role': dialogue.get('role', ''),
                        'text': timeline_date + ":" + dialogue.get('content', {}).get('text', ''),
                        'timestamp': dialogue.get('timeline_date', time.time())
                    }
                    
                    # 处理图片
                    image_path = dialogue.get('content', {}).get('image', '')
                    prefix = os.path.join(session_dir, "image")
                    if image_path:
                        full_image_path = os.path.join(prefix, image_path)
                        if os.path.exists(full_image_path):
                            observation['image'] = {'path': str(full_image_path)}
                            observation['image_name'] = dialogue.get('content', {}).get('image', ''),
                    conversations.append(observation)
                    
            except Exception as e:
                logger.error(f"加载对话失败 {conv_file}: {e}")
        
        logger.info(f"加载了 {len(conversations)} 条对话")
        return conversations


# ==================== VLM评估器 ====================

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
            self.category = subsub_type or self.question_type.get("sub_type", "general")
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
    recall_method: str = "ngm_graph_traversal"
    success: bool = True
    error_message: Optional[str] = None
    retrieved_nodes: Optional[List[str]] = None
    graph_stats: Optional[Dict] = None
    
    # 新增时间字段
    recall_time: float = 0.0        # 召回记忆时间
    prompt_build_time: float = 0.0  # 构建提示词时间
    api_call_time: float = 0.0      # API调用时间

class SimpleRetrievalPromptTemplate:
    """Standardized prompt template for simple retrieval-based questions"""
    
    # Instructions for 9 question types (only CD, AR, TTL include abbreviation)
    INSTRUCTIONS = {
        "Unimodal Precise Recall": "Accurately recall specific information from the retrieved conversation memories and answer directly.",
        "Cross-modal Related Retrieval": "Retrieve related information across different modalities (text and images) from the retrieved conversation memories.",
        "Knowledge Resolution": "Resolve and maintain knowledge consistency based on the retrieved conversation memories.",
        "Temporal Reasoning": "Reason about temporal relationships and time-based information in the retrieved conversation memories.",
        "Multimodal Causal Reasoning": "Perform causal reasoning using both text and image information from the retrieved conversation memories.",
        "Reference & Evolution Tracking": "Track references and their evolution throughout the retrieved conversation memories.",
        "Test-Time Learning (TTL)": "Learn and adapt from the retrieved conversation memories at test time to answer the question.",
        "Conflict Detection (CD)": "Check whether this information conflicts with the retrieved conversation memories.",
        "Answer Refusal (AR)": "Determine if the question can be answered based on the retrieved conversation memories."
    }
    
    # Response format requirements
    FORMAT_REQUIREMENTS = {
        "Conflict Detection (CD)": "Response format: Reply strictly with either 'Yes' or 'No' only.",
        "Answer Refusal (AR)": "Response format: If the information is present in the retrieved conversation memories, provide answer based on that information; if not present, reply with: 'Not mentioned.'",
        "default": "Response format: Provide clear and accurate answers based on the retrieved conversation memories."
    }
    
    # Base template
    TEMPLATE = """You are a memory testing system. Answer the question based on the retrieved conversation memories.

{instruction}

IMPORTANT: 
1. Provide only the answer without any reasoning process. Give the answer directly in English.
2. Keep your answer within 100 words. Short and concise answers are acceptable.
3. Answer in English. This is a strict requirement. Do not answer in any other language.

Retrieved conversation memories:
{context}

Question: {question}

{format_requirement}

Examples:
Question: What is the cat's name?
Correct answer: Almond

Incorrect answer example (DO NOT answer like this):
We need answer: cat name is Almond because..."""

    def __init__(self, question_type: str, context: str, question: str):
        self.question_type = question_type
        self.context = context
        self.question = question
    
    def build(self) -> str:
        """Build the complete prompt"""
        
        # Get instruction for question type
        if self.question_type in self.INSTRUCTIONS:
            instruction = self.INSTRUCTIONS[self.question_type]
        else:
            instruction = self.INSTRUCTIONS["Unimodal Precise Recall"]
        
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


class VLMEvaluator:
    """
    VLM评估器 - 使用NGM记忆系统
    """
    
    def __init__(self, 
                 memory_system: NGMemorySystem,
                 api_key: str,
                 model: str = "",
                 base_url: str = "",
                 verbose: bool = False,
                 max_retries: int = 3,
                 timeout: int = 60,
                 test_mode: bool = False,
                 max_questions_per_session: Optional[int] = None):
        
        self.memory_system = memory_system
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.verbose = verbose
        self.max_retries = max_retries
        self.timeout = timeout
        self.test_mode = test_mode
        self.max_questions_per_session = max_questions_per_session


        # 新增：记录失败的问题文件路径（使用set自动去重）
        self.failed_json_files = set()
        
        # 统计信息
        self.session_statistics = defaultdict(lambda: {
            "total": 0, 
            "successful": 0, 
            "failed": 0, 
            "processing_time": 0.0,
            "total_recall_time": 0.0,      # 新增
            "total_prompt_build_time": 0.0, # 新增
            "total_api_call_time": 0.0      # 新增
        })
    

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
                    q_id = q.get("question_id", "")
                    
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
                            question_image="",  # 没有图片
                            original_answer=q.get("original_answer", ""),
                            answer_source=q.get("answer_source", "unknown"),
                            answer_session=q.get("answer_session", []),
                            question_type=q.get("question_type", {}),
                            difficulty=q.get("difficulty", "medium"),
                            supporting_evidence=q.get("supporting_evidence", []),
                            image_context=[]  # 没有图片上下文
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

    def _format_memory_context(self, memories: List[Dict]) -> str:
        if not memories:
            return "无可用记忆"
        
        context_parts = []
        context_parts.append("【神经图记忆 (NGM) 检索结果】")
        context_parts.append(f"检索到 {len(memories)} 个相关记忆节点")
        
        # 按session分组
        sessions = defaultdict(list)
        for mem in memories:
            sessions[mem.get('session_id', 'unknown')].append(mem)
        
        for session_id, session_mems in sessions.items():
            context_parts.append(f"\n【Session {session_id}】")
            
            # 按对话索引排序
            session_mems.sort(key=lambda x: x.get('dialogue_index', 0))
            for mem in session_mems:
                role = mem.get('role', 'unknown')
                text = mem.get('text', '')
                has_image = 'image' in mem
                
                line = f"  第{mem.get('dialogue_index', 0)}轮 - {role}: {text}"
                if has_image:
                    line += f"[发送图片:{mem.get('image_name', '')}]"
                
                context_parts.append(line)
        
        return "\n".join(context_parts)
    
    def _prepare_image_for_api(self, image_path: str) -> str:
        """准备图片用于API"""
        try:
            with Image.open(image_path) as img:
                # 缩放到合适大小
                img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                
                # 转换为RGB
                if img.mode in ('RGBA', 'LA', 'P'):
                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    rgb_img.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = rgb_img
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 转换为base64
                buffer = BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                return img_base64
                
        except Exception as e:
            logger.error(f"处理图片失败 {image_path}: {e}")
            raise
    
    def _call_vlm_api(self, prompt: str, images) -> Dict[str, Any]:
        """调用VLM API"""
        start_time = time.time()
        
        messages = [{
            "role": "user",
            "content": []
        }]
        
        # 添加文本
        messages[0]["content"].append({
            "type": "text",
            "text": prompt
        })
        # 添加图片
        if images:
            if images['memory_image']:
                for image in images['memory_image']:
                    try:
                        img_base64 = self._prepare_image_for_api(image["image"]["path"])
                        messages[0]["content"].append({
                            "type":"image_url",
                            "image_url": {
                                "url":f"data:image/jpeg;base64,{img_base64}"
                            },
                            "session_id":image["session_id"],
                            "dialogue_idx":image["dialogue_idx"],
                            "marker":"memory_image",
                        })
                    except Exception as e:
                        logger.error(f"添加图片失败 {image}: {e}")
            if images["question_image"]:
                for image in images["question_image"]:  
                    try:
                        img_base64 = self._prepare_image_for_api(image["image"])
                        messages[0]["content"].append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            },
                            "marker":"question_image"
                        })
                    except Exception as e:
                        logger.error(f"添加图片失败 {image}: {e}")
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.1
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    resp_data = response.json()
                    answer = resp_data["choices"][0]["message"]["content"].strip()
                    
                    return {
                        "answer": answer,
                        "processing_time": time.time() - start_time,
                        "success": True
                    }
                else:
                    logger.warning(f"API返回错误码 {response.status_code}: {response.text}")
                    
            except Exception as e:
                logger.warning(f"API调用失败 (尝试 {attempt+1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2)
        
        return {
            "answer": "[API调用失败]",
            "processing_time": time.time() - start_time,
            "success": False,
            "error": "所有重试都失败"
        }
    
    def _construct_prompt(self, question: QuestionAnswerPair, memories: List[Dict]) -> str:
        """Build prompt for question with retrieved memories"""
        
        # Format memory context
        context = self._format_memory_context(memories)
        
        # Get question type
        question_type = question.question_type.get("subsub_type", "") if question.question_type else ""
        
        # Build combined question text (with image context if available)
        combined_question = self._build_combined_question(question)
        
        # Build prompt using template
        prompt = SimpleRetrievalPromptTemplate(
            question_type=question_type,
            context=context,
            question=combined_question
        )
        
        return prompt.build()


    def _build_combined_question(self, question: QuestionAnswerPair) -> str:
        """Build combined question text with image context if available"""
        
        # Check if image exists and is valid
        has_valid_image = (question.image_context and 
                        question.image_context[0] and 
                        os.path.exists(question.image_context[0]))
        
        if has_valid_image:
            return f"Question image sent. {question.question_text}"
        else:
            return question.question_text
    
    def evaluate_single_question(self, question: QuestionAnswerPair) -> EvaluationResult:
        start_time = time.time()
        
        # 时间记录变量
        recall_time = 0.0
        prompt_build_time = 0.0
        api_call_time = 0.0
        
        try:
            # 1. 构建查询
            if question.image_context and os.path.exists(question.image_context[0]):
                query = {
                    'text': question.question_text,
                    'image': {'path': question.image_context[0]}
                }
            else:
                query = question.question_text
            
            # 2. 基于查询召回相关记忆（记录时间）
            recall_start = time.time()
            memories = self.memory_system.recall(query)
            recall_time = time.time() - recall_start
            
            # 3. 构建提示词（记录时间）
            prompt_start = time.time()
            prompt = self._construct_prompt(question, memories)
            prompt_build_time = time.time() - prompt_start
            
            # 4. 准备图片
            images = {"memory_image": [], "question_image": []}
            if question.image_context and os.path.exists(question.image_context[0]):
                images["question_image"].append({"image": question.image_context[0]})
            for memory in memories:
                if memory.get("image", None):
                    dic1 = {
                        "image": memory["image"],
                        "session_id": memory["session_id"],
                        "dialogue_idx": memory["dialogue_index"]
                    }
                    images['memory_image'].append(dic1)
            
            # 5. 调用VLM（记录时间）
            api_start = time.time()
            vlm_response = self._call_vlm_api(prompt, images)
            api_call_time = vlm_response.get("processing_time", time.time() - api_start)
            
            # 总处理时间
            total_processing_time = time.time() - start_time
            
            # 6. 创建结果
            result = EvaluationResult(
                sample_id=f"{question.session_id}_{question.question_id}_{int(time.time())}",
                session_id=question.session_id,
                dialogue_name=question.dialogue_name,
                question_id=question.question_id,
                question_text=question.question_text,
                question_image=question.question_image,
                system_answer=vlm_response.get("answer", ""),
                original_answer=question.original_answer,
                answer_source=question.answer_source,
                question_type=question.question_type,
                category=question.category,
                difficulty=question.difficulty,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                memory_type="NGM",
                vlm_model=self.model,
                processing_time=total_processing_time,
                confidence=0.7 if vlm_response.get("success") else 0.0,
                retrieved_nodes=[memories[i]["dialogue_id"] for i in range(len(memories))],
                graph_stats=self.memory_system.get_graph_statistics(),
                success=vlm_response.get("success", False),
                # 新增时间字段
                recall_time=recall_time,
                prompt_build_time=prompt_build_time,
                api_call_time=api_call_time
            )
            
            # 更新统计
            stats = self.session_statistics[question.session_id]
            stats["successful"] += 1
            stats["processing_time"] += total_processing_time
            stats["total_recall_time"] = stats.get("total_recall_time", 0) + recall_time
            stats["total_prompt_build_time"] = stats.get("total_prompt_build_time", 0) + prompt_build_time
            stats["total_api_call_time"] = stats.get("total_api_call_time", 0) + api_call_time
                        
            logger.info(f"✓ 成功处理: {question.question_id} "
                        f"(总: {total_processing_time:.2f}s, 召回: {recall_time:.3f}s, "
                        f"提示词: {prompt_build_time:.3f}s, API: {api_call_time:.2f}s)")
            
            return result
            
        except Exception as e:
            total_processing_time = time.time() - start_time
            error_msg = str(e)[:200]
            logger.error(f"处理问题 {question.question_id} 失败: {error_msg}")
            
            
            return EvaluationResult(
                sample_id=f"error_{question.question_id}_{int(time.time())}",
                session_id=question.session_id,
                dialogue_name=question.dialogue_name,
                question_id=question.question_id,
                question_text=question.question_text,
                question_image=question.question_image,
                system_answer=f"[处理错误: {error_msg}]",
                original_answer=question.original_answer,
                answer_source=question.answer_source,
                question_type=question.question_type,
                category=question.category,
                difficulty=question.difficulty,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                memory_type="NGM",
                vlm_model=self.model,
                processing_time=total_processing_time,
                confidence=0.0,
                success=False,
                error_message=error_msg,
                recall_time=recall_time,
                prompt_build_time=prompt_build_time,
                api_call_time=api_call_time
            )
    
    def evaluate_session(self, session_id: str, questions: List[QuestionAnswerPair], session_path: Path) -> List[Dict]:
        self.session_statistics[session_id]["total"] = len(questions)
        
        results = []
        for i, q in enumerate(questions):
            logger.info(f"  处理问题 {i+1}/{len(questions)}: {q.question_id}")
            result = self.evaluate_single_question(q)
            results.append(asdict(result))
            
            # 保存中间结果
            self._save_intermediate_results(session_id, results, session_path)
        
        # 输出session时间统计
        stats = self.session_statistics[session_id]
        successful = stats["successful"]
        if successful > 0:
            logger.info(f"Session {session_id} 时间统计 - "
                        f"平均召回: {stats['total_recall_time']/successful:.3f}s, "
                        f"平均提示词: {stats['total_prompt_build_time']/successful:.3f}s, "
                        f"平均API: {stats['total_api_call_time']/successful:.2f}s")
        
        return results
    
    def _save_intermediate_results(self, session_id: str, results: List[Dict], session_path: Path):
        """保存中间结果"""
        session_dir = session_path / "evaluation_results"
        session_dir.mkdir(exist_ok=True)
        
        output_file = os.path.join(session_dir, "results_NGM.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "session_id": session_id,
                "results": results,
                "statistics": self.session_statistics[session_id]
            }, f, ensure_ascii=False, indent=2)


# ==================== 主函数 ====================

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="NGM记忆系统 - 完整实现 (GME编码器)")
    
    # 路径参数 – 全部改为必需
    parser.add_argument("--conversations_dir", type=str, required=True,
                       help="对话数据目录（必需）")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="输出目录（必需）")
    parser.add_argument("--image_base_path", type=str, default="",
                       help="图片基础路径，用于相对路径图片（可选）")
    
    # 编码器参数
    parser.add_argument("--encoder_method", type=str, default="GMEEncoder",
                       choices=["GMEEncoder", "CLIPEncoder"],
                       help="编码器类型")
    parser.add_argument("--encoder_path", type=str, 
                       default="Alibaba-NLP/gme-Qwen2-VL-7B-Instruct",
                       help="编码器模型路径或名称")
    parser.add_argument("--embedding_dim", type=int, default=1024,
                       help="嵌入向量维度")
    parser.add_argument("--retrieval_mode", type=str, default="cosine",
                       choices=["cosine", "dot", "L2"],
                       help="检索模式")
    parser.add_argument("--retrieval_topk", type=int, default=10,
                       help="检索返回的最相似记忆数量")
    
    # 图构建参数
    parser.add_argument("--similarity_threshold", type=float, default=0.7,
                       help="语义相似度阈值，用于创建边")
    parser.add_argument("--max_edges_per_node", type=int, default=5,
                       help="每个节点的最大边数")
    parser.add_argument("--temporal_decay_constant", type=int, default=3600,
                       help="时间衰减常数（秒），默认1小时")
    
    # 图遍历参数
    parser.add_argument("--max_depth", type=int, default=3,
                       help="图遍历最大深度")
    parser.add_argument("--max_nodes", type=int, default=10,
                       help="召回的最大节点数")
    parser.add_argument("--traversal_threshold", type=float, default=0.5,
                       help="图遍历相似度阈值")
    parser.add_argument("--traversal_strategy", type=str, default="breadth_first",
                       choices=["depth_first", "breadth_first"],
                       help="图遍历策略")
    parser.add_argument("--initial_candidate_multiplier", type=int, default=2,
                       help="初始候选节点乘数")
    
    # VLM API参数
    parser.add_argument("--api_key", type=str, required=True,
                       help="VLM API密钥（必需）")
    parser.add_argument("--vlm_model", type=str, required=True,
                       help="VLM模型名称（必需）")
    parser.add_argument("--base_url", type=str, required=True,
                       help="API基础URL（必需）")
    parser.add_argument("--max_retries", type=int, default=3,
                       help="API调用最大重试次数")
    parser.add_argument("--timeout", type=int, default=60,
                       help="API调用超时时间（秒）")
    
    # 评估参数
    parser.add_argument("--test_mode", action="store_true",
                       help="测试模式，每个session只处理前2个问题")
    parser.add_argument("--max_questions_per_session", type=int, default=None,
                       help="每个session处理的最大问题数")
    parser.add_argument("--verbose", action="store_true",
                       help="详细日志输出")
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()
    
    # 设置日志级别
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    print("=" * 70)
    print("NGM记忆系统 - 完整实现 (GME编码器)")
    print("=" * 70)
    print(f"配置参数:")
    print(f"  编码器: {args.encoder_method}")
    print(f"  模型路径: {args.encoder_path}")
    print(f"  检索模式: {args.retrieval_mode}")
    print(f"  相似度阈值: {args.similarity_threshold}")
    print(f"  遍历策略: {args.traversal_strategy}")
    print(f"  测试模式: {args.test_mode}")
    print("=" * 70)
    
    # 构建配置对象
    from types import SimpleNamespace
    
    # 创建配置对象
    encoder_config = SimpleNamespace(
        method=args.encoder_method,
        path=args.encoder_path,
        embedding_dim=args.embedding_dim,
        image_base_path=args.image_base_path
    )
    
    multimodal_retrieval_config = SimpleNamespace(
        mode=args.retrieval_mode,
        topk=args.retrieval_topk,
        encoder=encoder_config
    )
    
    store_config = SimpleNamespace(
        similarity_threshold=args.similarity_threshold,
        max_edges_per_node=args.max_edges_per_node,
        temporal_decay_constant=args.temporal_decay_constant
    )
    
    recall_config = SimpleNamespace(
        max_depth=args.max_depth,
        max_nodes=args.max_nodes,
        traversal_threshold=args.traversal_threshold,
        traversal_strategy=args.traversal_strategy,
        initial_candidate_multiplier=args.initial_candidate_multiplier
    )
    
    storage_config = SimpleNamespace()
    
    config = SimpleNamespace(
        storage=storage_config,
        multimodal_retrieval=multimodal_retrieval_config,
        store=store_config,
        recall=recall_config
    )
    
    # 1. 初始化NGM记忆系统
    print("\n[1] 初始化NGM记忆系统...")
    memory_system = NGMemorySystem(config)
    
    # 2. 加载并存储对话（添加时间统计）
    print("\n[2] 加载对话并构建记忆图...")
    store_start_total = time.time()
    
    conversations = DialogueLoader.load_conversations(args.conversations_dir)
    
    store_times = []  # 记录每次存储时间
    encoding_times = []  # 编码时间（来自multimodal_retrieval.add）
    
    for i, conv in enumerate(conversations):
        if i % 10 == 0:
            print(f"   已处理 {i}/{len(conversations)} 条对话")
        
        # 存储并获取耗时
        node_id, store_time = memory_system.store(conv)
        store_times.append(store_time)
    
    store_total_time = time.time() - store_start_total
    
    stats = memory_system.get_graph_statistics()
    store_stats = memory_system.get_store_statistics()
    
    print(f"\n   记忆构建完成!")
    print(f"   总节点数: {stats['total_nodes']}")
    print(f"   总边数: {stats['total_edges']}")
    print(f"   平均节点度数: {stats['avg_degree']:.2f}")
    
    # 输出存储时间统计
    print(f"\n   【存储时间统计】:")
    print(f"   总存储对话数: {len(conversations)}")
    print(f"   总存储耗时: {store_total_time:.2f}秒")
    print(f"   平均每条存储: {store_total_time/len(conversations):.4f}秒")
    print(f"   最快存储: {min(store_times):.4f}秒")
    print(f"   最慢存储: {max(store_times):.4f}秒")
    
    # 分阶段统计
    if store_stats['num_stores'] > 0:
        print(f"   StoreOp总耗时: {store_stats['total_time']:.2f}秒")
        print(f"   StoreOp平均耗时: {store_stats['avg_time']:.4f}秒")
    
    # 3. 初始化评估器
    print("\n[3] 初始化VLM评估器...")
    print(args.vlm_model, args.base_url)
    evaluator = VLMEvaluator(
        memory_system=memory_system,
        api_key=args.api_key,
        model=args.vlm_model,
        base_url=args.base_url,
        verbose=args.verbose,
        max_retries=args.max_retries,
        timeout=args.timeout,
        test_mode=args.test_mode,
        max_questions_per_session=args.max_questions_per_session
    )
    
    # 4. 加载问题
    print("\n[4] 加载问题文件...")
    sessions_questions = evaluator.load_questions(args.conversations_dir)
    
    total_questions = sum(len(data["questions"]) for data in sessions_questions.values())
    print(f"   从 {len(sessions_questions)} 个session加载了 {total_questions} 个问题")
    
    # 5. 评估
    print("\n[5] 开始评估...")
    print("-" * 70)
    for session_id, session_data in sessions_questions.items():
        print(f"\n处理 Session: {session_id}")
        session_path = Path(args.conversations_dir) / "scenes" / str(session_id)
        questions = session_data["questions"]
        
        # 确定要处理的问题数量
        if args.test_mode:
            process_questions = questions[:2]
            print(f"  测试模式: 处理 {len(process_questions)}/{len(questions)} 个问题")
        elif args.max_questions_per_session:
            process_questions = questions[:args.max_questions_per_session]
            print(f"  限制模式: 处理 {len(process_questions)}/{len(questions)} 个问题")
        else:
            process_questions = questions
            print(f"  处理全部 {len(process_questions)} 个问题")
        
        results = evaluator.evaluate_session(session_id, process_questions, session_path)
        
        print(f"  完成 {len(results)} 个问题")
    
    # 6. 输出统计
    print("\n" + "=" * 70)
    print("评估完成!")
    print("=" * 70)
    
    total_processed = sum(s["total"] for s in evaluator.session_statistics.values())
    total_successful = sum(s["successful"] for s in evaluator.session_statistics.values())
    
    print(f"\n统计信息:")
    print(f"   处理问题数: {total_processed}")
    print(f"   成功数: {total_successful}")
    if total_processed > 0:
        print(f"   成功率: {total_successful/total_processed*100:.1f}%")
    
    print(f"\nNGM图统计:")
    print(f"   总节点数: {stats['total_nodes']}")
    print(f"   总边数: {stats['total_edges']}")
    print(f"   平均节点度数: {stats['avg_degree']:.2f}")
    print(f"   相似度阈值: {stats['config']['similarity_threshold']}")
    print(f"   遍历策略: {stats['config']['traversal_strategy']}")
    
    # 输出存储时间总览
    print(f"\n【存储时间总览】:")
    print(f"   总存储对话数: {len(conversations)}")
    print(f"   总存储耗时: {store_total_time:.2f}秒")
    print(f"   平均每条存储: {store_total_time/len(conversations):.4f}秒")
    
    if store_stats['num_stores'] > 0:
        print(f"   StoreOp总耗时: {store_stats['total_time']:.2f}秒")
        print(f"   存储占比: {store_stats['total_time']/store_total_time*100:.1f}%")
    
    print("=" * 70)


if __name__ == "__main__":
    main()