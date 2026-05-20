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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==================== Encoder Base Class ====================

class BaseMultiModalEncoder:
    """Base class for multimodal encoder"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.processor = None
    
    def _load_image(self, image_path_or_url: str) -> Image.Image:
        """Load image (to be implemented by subclass)"""
        raise NotImplementedError
    
    def encode_text(self, text: str, return_type: str = 'numpy') -> Union[np.ndarray, torch.Tensor]:
        """Encode text (to be implemented by subclass)"""
        raise NotImplementedError
    
    def encode_image(self, image_path_or_url: str, return_type: str = 'numpy') -> Union[np.ndarray, torch.Tensor]:
        """Encode image (to be implemented by subclass)"""
        raise NotImplementedError
    
    def encode_multimodal(self, text: Optional[str] = None, image: Optional[Dict] = None, 
                         return_type: str = 'numpy') -> Union[np.ndarray, torch.Tensor]:
        """Encode multimodal data (to be implemented by subclass)"""
        raise NotImplementedError
    
    def __call__(self, obj: Union[str, Dict], return_type: str = 'numpy') -> Union[np.ndarray, torch.Tensor]:
        """Unified call interface (to be implemented by subclass)"""
        raise NotImplementedError


# ==================== GME Encoder Implementation ====================

class GMEEncoder(BaseMultiModalEncoder):
    """
    GME (General Multimodal Embedding) Qwen2-VL-based encoder for text and images.
    Supports unified multimodal representations for Any2Any Search.
    """
    def __init__(self, config):
        super().__init__(config)
        
        model_name = getattr(config, 'encoder_model', 'Alibaba-NLP/gme-Qwen2-VL-7B-Instruct')
        self.image_base_path = getattr(config, 'image_base_path', '')
        self.embedding_dim = getattr(config, 'embedding_dim', 4096)  # GME-Qwen2-VL-7B dimension
        
        print(f"Loading GME model: {model_name}")
        print(f"Using device: {self.device}")
        
        # Load GME model with trust_remote_code=True
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
            device_map='auto',
            trust_remote_code=True
        )
        self.model.eval()  # Set to evaluation mode
        
        # Load processor
        try:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        except:
            self.processor = None
            
        print(f"GME model loaded successfully!")
        print(f"  Embedding dimension: {self.embedding_dim}")
    
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


# ==================== CLIP Encoder Implementation (Alternative) ====================

class CLIPEncoder(BaseMultiModalEncoder):
    """
    CLIP-based multimodal encoder for text and images.
    """
    def __init__(self, config):
        super().__init__(config)
        
        model_name = getattr(config, 'encoder_model', 'openai/clip-vit-base-patch32')
        self.image_base_path = getattr(config, 'image_base_path', '')
        self.embedding_dim = getattr(config, 'embedding_dim', 512)
        
        print(f"Loading CLIP model: {model_name}")
        print(f"Using device: {self.device}")
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        
        print(f"CLIP model loaded successfully!")
        print(f"  Embedding dimension: {self.embedding_dim}")
    
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


# ==================== Graph Storage Section ====================

class GraphStorage:
    """
    Graph structure memory storage
    """
    def __init__(self, config):
        self.config = config
        self.node = {}  # node_id -> node_data
        self.edge = {}  # source_id -> {target_id -> edge_data}
        self.node_counter = 0
        self.edge_counter = 0
        self.memory_order_map = []  # List of node_ids in insertion order

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
        """Get node ID by memory index"""
        if 0 <= mid < len(self.memory_order_map):
            return self.memory_order_map[mid]
        raise IndexError(f"Memory index {mid} out of range")

    def get_mid_by_node_id(self, node_id):
        """Get memory index by node ID"""
        if node_id in self.node:
            return self.node[node_id].get('mid', -1)
        raise KeyError(f"Node {node_id} not found")

    def get_memory_element_by_node_id(self, node_id):
        """Get memory element by node ID"""
        return self.node.get(node_id, {})

    def get_memory_element_by_mid(self, mid):
        """Get memory element by memory index"""
        node_id = self.get_node_id_by_mid(mid)
        return self.node[node_id]

    def get_memory_text_by_node_id(self, node_id):
        """Get node text"""
        return self.node[node_id].get('text', '')

    def get_memory_image_by_node_id(self, node_id):
        """Get node image information"""
        return self.node[node_id].get('image', None)
    
    def get_neighbors(self, node_id):
        """Get neighbor node IDs of the node"""
        if node_id in self.edge:
            return list(self.edge[node_id].keys())
        return []
    
    def get_edges_from(self, node_id):
        """Get all edges starting from the node"""
        return self.edge.get(node_id, {})
    
    def get_edges_to(self, node_id):
        """Get all edges pointing to the node"""
        edges_to = {}
        for source, targets in self.edge.items():
            if node_id in targets:
                edges_to[source] = targets[node_id]
        return edges_to
    
    def get_node_degree(self, node_id):
        """Get node degree (out-degree + in-degree)"""
        out_degree = len(self.get_neighbors(node_id))
        in_degree = len(self.get_edges_to(node_id))
        return out_degree + in_degree

    def add_node(self, obj):
        """Add node"""
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
        """Add edge"""
        obj['edge_id'] = self.edge_counter
        if s not in self.edge:
            self.edge[s] = {}
        self.edge[s][t] = obj
        self.edge_counter += 1
        return self.edge_counter - 1
    
    def display(self):
        """Display graph information"""
        return f"GraphStorage: {self.get_element_number()} nodes, {self.edge_counter} edges"


# ==================== Multimodal Retrieval Section ====================

class MultiModalRetrieval:
    """
    Multimodal retriever
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize encoder
        encoder_method = getattr(config.encoder, 'method', 'GMEEncoder')
        if encoder_method == 'CLIPEncoder':
            self.encoder = CLIPEncoder(config.encoder)
        elif encoder_method == 'GMEEncoder':
            self.encoder = GMEEncoder(config.encoder)
        else:
            raise ValueError(f"Unsupported encoder: {encoder_method}")
        
        # Store embedding vectors for all memories
        self.tensorstore = None
        
        # Store metadata
        self.memory_metadata = []
    
    def reset(self):
        self.tensorstore = None
        self.memory_metadata = []
    
    def __normalize__(self, embedding):
        """L2 normalization"""
        return torch.nn.functional.normalize(embedding, dim=-1)
    
    def add(self, obj):
        """
        Add memory to retriever
        
        Args:
            obj: Memory object (text, image path, or dictionary)
        
        Returns:
            Embedding vector
        """
        embedding = self.encoder(obj, return_type='tensor')
        
        if self.config.mode == 'cosine':
            embedding = self.__normalize__(embedding)
        
        if self.tensorstore is None:
            self.tensorstore = embedding
        else:
            self.tensorstore = torch.cat([self.tensorstore, embedding], dim=0)
        
        # Record metadata
        metadata = {
            'has_text': isinstance(obj, str) or (isinstance(obj, dict) and 'text' in obj),
            'has_image': isinstance(obj, dict) and 'image' in obj
        }
        self.memory_metadata.append(metadata)
        
        return embedding
    
    def __calculate_scores__(self, query):
        """Calculate similarity between query and all memories"""
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
        Retrieve most similar memories
        
        Args:
            query: Query (text, image path, or dictionary)
            topk: Number of results to return
            with_score: Whether to return scores
            sort: Whether to sort
        
        Returns:
            List of memory indices or (scores, indices) tuple
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


# ==================== Storage Operations ====================

class NGMMemorySystemStore:
    """
    Neural Graph Memory Store:
    Store multimodal events as graph nodes, create edges based on relationships
    """
    def __init__(self, config, **kwargs):
        self.config = config
        self.storage = kwargs['storage']
        self.multimodal_retrieval = kwargs['multimodal_retrieval']
        
        # Graph construction parameters
        self.similarity_threshold = getattr(config, 'similarity_threshold', 0.7)
        self.max_edges_per_node = getattr(config, 'max_edges_per_node', 5)
        self.temporal_decay_constant = getattr(config, 'temporal_decay_constant', 3600)
        
        # Store timing statistics
        self.store_times = []  # Record each store operation time
        self.total_store_time = 0.0
        self.num_stores = 0

    def reset(self):
        self.store_times = []
        self.total_store_time = 0.0
        self.num_stores = 0

    def __calculate_similarity__(self, embedding1, embedding2):
        """Calculate cosine similarity"""
        emb1_norm = torch.nn.functional.normalize(embedding1, dim=-1)
        emb2_norm = torch.nn.functional.normalize(embedding2, dim=-1)
        return torch.matmul(emb1_norm, emb2_norm.T).item()
    
    def __calculate_temporal_weight__(self, source_node_id, target_node_id):
        """
        Calculate temporal weight: exponential decay, closer time = higher weight
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
            
            # Exponential decay: np.exp(-time_diff / decay_constant)
            return np.exp(-time_diff / self.temporal_decay_constant)
        except Exception as e:
            logger.debug(f"Temporal weight calculation failed: {e}")
            return 1.0
    
    def __create_edges__(self, new_node_id, new_embedding):
        """Create edges for the new node"""
        if self.storage.get_element_number() <= 1:
            return
        
        all_node_ids = [nid for nid in self.storage.memory_order_map if nid != new_node_id]
        
        if not all_node_ids:
            return

        # 1. Create temporal edge (with the most recently added node)
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
        
        # 2. Create semantic similarity edges
        similarities = []
        for node_id in all_node_ids:
            node_mid = self.storage.get_mid_by_node_id(node_id)
            if (self.multimodal_retrieval.tensorstore is not None and
                node_mid < self.multimodal_retrieval.tensorstore.size(0)):
                
                node_embedding = self.multimodal_retrieval.tensorstore[node_mid:node_mid+1]
                similarity = self.__calculate_similarity__(new_embedding, node_embedding)
                similarities.append((node_id, similarity))

        # Select most similar nodes to create edges
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
        Store multimodal observation as graph node
        
        Returns:
            tuple: (node_id, store_time) node ID and store duration
        """
        store_start_time = time.time()
        
        # Ensure timestamp exists
        if 'timestamp' not in observation:
            observation['timestamp'] = time.time()
        
        # Add to graph storage
        node_id = self.storage.add_node(observation)
        
        # Encode and add to multimodal retriever
        embedding = self.multimodal_retrieval.add(observation)
        
        # Create edges
        self.__create_edges__(node_id, embedding)
        
        store_time = time.time() - store_start_time
        
        # Record store time
        self.store_times.append(store_time)
        self.total_store_time += store_time
        self.num_stores += 1
        return node_id, store_time
    
    def get_store_statistics(self):
        """Get storage time statistics"""
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
        """Get all store time records"""
        return {
            'store_times': self.store_times,
            'total_store_time': self.total_store_time,
            'num_stores': self.num_stores,
            'avg_store_time': self.total_store_time / self.num_stores if self.num_stores > 0 else 0,
            'min_store_time': min(self.store_times) if self.store_times else 0,
            'max_store_time': max(self.store_times) if self.store_times else 0
        }


# ==================== Recall Operations ====================

class NGMMemorySystemRecall:
    """
    Neural Graph Memory Recall:
    Query-based graph traversal memory retrieval
    """
    def __init__(self, config, **kwargs):
        self.config = config
        self.storage = kwargs['storage']
        self.multimodal_retrieval = kwargs['multimodal_retrieval']
        
        # Graph traversal parameters
        self.max_depth = getattr(config, 'max_depth', 3)
        self.max_nodes = getattr(config, 'max_nodes', 10)
        self.traversal_threshold = getattr(config, 'traversal_threshold', 0.5)
        self.traversal_strategy = getattr(config, 'traversal_strategy', 'breadth_first')
        self.initial_candidate_multiplier = getattr(config, 'initial_candidate_multiplier', 2)
        
        # Retrieval result cache
        self.last_retrieved_ids = []
        self.last_reasoning_paths = {}
        self.last_reasoning_details = {}

    def reset(self):
        self.last_retrieved_ids = []
        self.last_reasoning_paths = {}
        self.last_reasoning_details = {}

    def __graph_traversal_depth_first__(self, query_embedding, start_node_ids, visited=None, depth=0):
        """
        Depth-first graph traversal
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

            # Get node embedding
            node_mid = self.storage.get_mid_by_node_id(node_id)
            if node_mid >= self.multimodal_retrieval.tensorstore.size(0):
                continue
            
            node_embedding = self.multimodal_retrieval.tensorstore[node_mid:node_mid+1]

            # Calculate similarity with query
            query_norm = torch.nn.functional.normalize(query_embedding, dim=-1)
            node_norm = torch.nn.functional.normalize(node_embedding, dim=-1)
            similarity = torch.matmul(query_norm, node_norm.T).item()

            # Add to candidates if above threshold
            if similarity >= self.traversal_threshold:
                candidates.append((node_id, similarity))

            # Continue traversing neighbors
            neighbors = self.storage.get_neighbors(node_id)
            if neighbors:
                neighbor_candidates = self.__graph_traversal_depth_first__(
                    query_embedding, neighbors, visited, depth + 1
                )
                candidates.extend(neighbor_candidates)
        
        return candidates
    
    def __graph_traversal_breadth_first__(self, query_embedding, initial_node_ids):
        """
        Breadth-first graph traversal (matches original implementation)
        """
        import torch

        # 1. Expand search: add all initial nodes and their neighbors
        expanded_nodes = set()
        for node_id in initial_node_ids:
            expanded_nodes.add(node_id)
            neighbors = self.storage.get_neighbors(node_id)
            for neighbor in neighbors:
                expanded_nodes.add(neighbor)

        # 2. Recalculate similarity for all expanded nodes
        final_similarities = []
        for node_id in expanded_nodes:
            try:
                node_mid = self.storage.get_mid_by_node_id(node_id)
                if node_mid >= self.multimodal_retrieval.tensorstore.size(0):
                    continue
                
                node_embedding = self.multimodal_retrieval.tensorstore[node_mid:node_mid+1]

                # Calculate cosine similarity
                query_norm = torch.nn.functional.normalize(query_embedding, dim=-1)
                node_norm = torch.nn.functional.normalize(node_embedding, dim=-1)
                similarity = torch.matmul(query_norm, node_norm.T).item()
                
                # Apply degree boost
                node_degree = self.storage.get_node_degree(node_id)
                degree_boost = 1 + (node_degree * 0.1)
                boosted_similarity = similarity * degree_boost
                
                final_similarities.append((node_id, boosted_similarity))
            except Exception:
                continue

        # 3. Sort by similarity
        final_similarities.sort(key=lambda x: x[1], reverse=True)
        return final_similarities
    
    def __graph_traversal__(self, query_embedding, start_node_ids, visited=None, depth=0):
        """
        Graph traversal entry method
        """
        if self.traversal_strategy == 'breadth_first':
            return self.__graph_traversal_breadth_first__(query_embedding, start_node_ids)
        else:
            return self.__graph_traversal_depth_first__(query_embedding, start_node_ids, visited, depth)

    def __call__(self, query):
        """
        Query-based graph traversal memory recall
        """
        if self.storage.is_empty():
            return []
        

        # 1. First find most relevant starting nodes using embedding similarity
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

        # 2. Encode query
        query_embedding = self.multimodal_retrieval.encoder(query, return_type='tensor')
        if self.multimodal_retrieval.config.mode == 'cosine':
            query_embedding = self.multimodal_retrieval.__normalize__(query_embedding)

        # 3. Get starting node IDs
        start_node_ids = []
        for mid in ranking_ids:
            try:
                node_id = self.storage.get_node_id_by_mid(int(mid))
                start_node_ids.append(node_id)
            except (KeyError, IndexError):
                continue
        # 4. Execute graph traversal
        traversal_results = self.__graph_traversal__(query_embedding, start_node_ids)

        # 5. Merge initial retrieval results and traversal results
        all_candidates = {}

        if self.traversal_strategy == 'breadth_first':
            # Breadth-first: use traversal results directly
            for node_id, boosted_similarity in traversal_results:
                all_candidates[node_id] = boosted_similarity
        else:
            # Depth-first: merge results and apply degree boost
            import torch
            
            # Add initial retrieval results
            for mid in ranking_ids:
                try:
                    node_id = self.storage.get_node_id_by_mid(int(mid))
                    node_mid = self.storage.get_mid_by_node_id(node_id)
                    
                    if node_mid < self.multimodal_retrieval.tensorstore.size(0):
                        node_embedding = self.multimodal_retrieval.tensorstore[node_mid:node_mid+1]
                        query_norm = torch.nn.functional.normalize(query_embedding, dim=-1)
                        node_norm = torch.nn.functional.normalize(node_embedding, dim=-1)
                        similarity = torch.matmul(query_norm, node_norm.T).item()
                        
                        # Apply degree boost
                        node_degree = self.storage.get_node_degree(node_id)
                        degree_boost = 1 + (node_degree * 0.1)
                        all_candidates[node_id] = similarity * degree_boost
                except Exception:
                    all_candidates[node_id] = 1.0

            # Add traversal results
            for node_id, similarity in traversal_results:
                node_degree = self.storage.get_node_degree(node_id)
                degree_boost = 1 + (node_degree * 0.1)
                boosted_similarity = similarity * degree_boost
                
                if node_id not in all_candidates:
                    all_candidates[node_id] = boosted_similarity
                else:
                    all_candidates[node_id] = max(all_candidates[node_id], boosted_similarity)

        # 6. Sort and select top-k
        sorted_candidates = sorted(all_candidates.items(), key=lambda x: x[1], reverse=True)
        selected_node_ids = [node_id for node_id, _ in sorted_candidates[:self.max_nodes]]
        
        if not selected_node_ids:
            return []

        # 7. Collect memories
        memories = []
        retrieved_ids = []
        for node_id in selected_node_ids:
            mem = self.storage.get_memory_element_by_node_id(node_id)
            memories.append(mem)
            if 'dialogue_id' in mem:
                retrieved_ids.append(mem['dialogue_id'])

        # Record retrieval results
        self.last_retrieved_ids = retrieved_ids
        
        logger.debug(f"Recalled {len(memories)} memory nodes")
        return memories


# ==================== Main NGM Memory Class ====================

class NGMMemorySystem:
    """
    Neural Graph Memory (NGM)
    Neural Graph Memory System
    
    Reference: Neural Graph Memory: A Structured Approach to Long-Term Memory in Multimodal Agents
    """
    def __init__(self, config):
        self.config = config
        
        # Graph storage
        self.storage = GraphStorage(config.storage)
        
        # Multimodal retriever
        self.multimodal_retrieval = MultiModalRetrieval(config.multimodal_retrieval)
        
        # Storage operations
        self.store_op = NGMMemorySystemStore(
            config.store,
            storage=self.storage,
            multimodal_retrieval=self.multimodal_retrieval
        )
        
        # Recall operations
        self.recall_op = NGMMemorySystemRecall(
            config.recall,
            storage=self.storage,
            multimodal_retrieval=self.multimodal_retrieval
        )
    
    def reset(self):
        """Reset all components"""
        self.storage.reset()
        self.multimodal_retrieval.reset()
        self.store_op.reset()
        self.recall_op.reset()

    def store(self, observation):
        """
        Store observation (text, image, or both) as graph node
        
        Returns:
            tuple: (node_id, store_time) node ID and store duration
        """
        return self.store_op(observation)
    
    def recall(self, query):
        """
        Recall relevant memories based on query
        """
        return self.recall_op(query)
    
    def get_graph_statistics(self):
        """Get graph statistics"""
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
        """Get storage time statistics"""
        return self.store_op.get_store_statistics()


# ==================== Configuration Class ====================

class Config:
    """Configuration class"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                setattr(self, key, Config(**value))
            else:
                setattr(self, key, value)


# ==================== Dialogue Loader ====================

class DialogueLoader:
    """Dialogue loader"""
    
    @staticmethod
    def load_conversations(conversations_dir: str) -> List[Dict]:
        """Load all conversations"""
        conversations = []
        base_dir = Path(conversations_dir)
        
        scenes_dir = os.path.join(base_dir, "scenes")
        if not os.path.exists(scenes_dir):
            raise ValueError(f"Scenes directory not found: {scenes_dir}")
            
        session_dirs = natsorted([
            d for d in os.listdir(scenes_dir) 
            if os.path.isdir(os.path.join(scenes_dir, d))
        ])
        # Iterate through all sessions
        for session_dir_name in session_dirs:
            session_dir = os.path.join(scenes_dir, session_dir_name)
            session_dir = Path(session_dir)
            conv_file = os.path.join(session_dir, "session.json")
            if not os.path.exists(conv_file):
                logger.warning(f"session.json file not found: {conv_file}")
                continue
            
            try:
                with open(conv_file, 'r', encoding='utf-8') as f:
                    conv_data = json.load(f)
                
                session_id = session_dir_name
                dialogues = conv_data.get("dialogue", [])
                timeline_date = conv_data.get("timeline_date", "")
                for i, dialogue in enumerate(dialogues):
                    # Build observation object
                    observation = {
                        'session_id': session_id,
                        'dialogue_id': f"{session_id}_{i}",
                        'dialogue_index': i,
                        'role': dialogue.get('role', ''),
                        'text': timeline_date + ":" + dialogue.get('content', {}).get('text', ''),
                        'timestamp': dialogue.get('timeline_date', time.time())
                    }
                    
                    # Process image
                    image_path = dialogue.get('content', {}).get('image', '')
                    prefix = os.path.join(session_dir, "image")
                    if image_path:
                        full_image_path = os.path.join(prefix, image_path)
                        if os.path.exists(full_image_path):
                            observation['image'] = {'path': str(full_image_path)}
                            observation['image_name'] = dialogue.get('content', {}).get('image', ''),
                    conversations.append(observation)
                    
            except Exception as e:
                logger.error(f"Failed to load conversation {conv_file}: {e}")
        
        logger.info(f"Loaded {len(conversations)} conversations")
        return conversations


# ==================== VLM Evaluator ====================

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
    supporting_evidence: List[Dict]
    image_context: Optional[List[str]] = None
    metadata: Optional[Dict] = None
    category: str = field(init=False)
    
    def __post_init__(self):
        if self.question_type:
            sub_type = self.question_type.get("sub_type", "")
            self.category = sub_type or self.question_type.get("sub_type", "general")
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
    recall_method: str = "ngm_graph_traversal"
    success: bool = True
    error_message: Optional[str] = None
    retrieved_nodes: Optional[List[str]] = None
    graph_stats: Optional[Dict] = None
    
    # New timing fields
    recall_time: float = 0.0        # Memory recall time
    prompt_build_time: float = 0.0  # Prompt building time
    api_call_time: float = 0.0      # API call time


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
        "Test-Time Learning": "Learn and adapt from the retrieved conversation memories at test time to answer the question.",
        "Conflict Detection": "Check whether this information conflicts with the retrieved conversation memories.",
        "Answer Refusal": "Determine if the question can be answered based on the retrieved conversation memories."
    }
    
    # Response format requirements
    FORMAT_REQUIREMENTS = {
        "Conflict Detection": "Response format: Reply strictly with either 'Yes' or 'No' only.",
        "Answer Refusal": "Response format: If the information is present in the retrieved conversation memories, provide answer based on that information; if not present, reply with: 'Not mentioned.'",
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
    VLM Evaluator - using NGM memory system
    """
    def __init__(self, 
                memory_system: NGMMemorySystem,
                api_key: str,
                model: str = "",
                base_url: str = "",
                verbose: bool = False,
                max_retries: int = 3,
                timeout: int = 60,
                test_mode: bool = False,
                retrieval_topk: int = 10,  # New
                max_context_tokens: int = 4096,  # New
                max_images: int = 5):  # New
        
        self.memory_system = memory_system
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.verbose = verbose
        self.max_retries = max_retries
        self.timeout = timeout
        self.test_mode = test_mode
        
        # New configuration attributes
        self.retrieval_topk = retrieval_topk
        self.max_context_tokens = max_context_tokens
        self.max_images = max_images
        
        # Statistics
        self.session_statistics = defaultdict(lambda: {
            "total": 0, 
            "successful": 0, 
            "failed": 0, 
            "processing_time": 0.0,
            "total_recall_time": 0.0,
            "total_prompt_build_time": 0.0,
            "total_api_call_time": 0.0
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
                    q_id = q.get("question_id", f"NGM_{len(question_pairs)}")
                    
                    # Get question image filename
                    question_image_filename = q.get("question", {}).get("image", "")
                    
                    # If question has image, build full path and save to image_context
                    image_context_list = []
                    if question_image_filename:
                        if str(session_dir) == "session0":
                            fold, img_file = question_image_filename.split("/", 1)
                            full_path = scenes_dir / fold / "image" / img_file
                            image_context_list.append(str(full_path))

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
                            question_image="",  # No image
                            original_answer=q.get("original_answer", ""),
                            answer_source=q.get("answer_source", "unknown"),
                            answer_session=q.get("answer_session", []),
                            question_type=q.get("question_type", {}),
                            difficulty=q.get("difficulty", "medium"),
                            supporting_evidence=q.get("supporting_evidence", []),
                            image_context=[]  # No image context
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

    def _format_memory_context(self, memories: List[Dict]) -> str:
        if not memories:
            return "No available memory"
        
        context_parts = []
        context_parts.append("[Neural Graph Memory (NGM) Retrieval Results]")
        context_parts.append(f"Retrieved {len(memories)} relevant memory nodes")
        
        # Group by session
        sessions = defaultdict(list)
        for mem in memories:
            sessions[mem.get('session_id', 'unknown')].append(mem)
        
        for session_id, session_mems in sessions.items():
            context_parts.append(f"\n[Session {session_id}]")
            
            # Sort by dialogue index
            session_mems.sort(key=lambda x: x.get('dialogue_index', 0))
            for mem in session_mems:
                role = mem.get('role', 'unknown')
                text = mem.get('text', '')
                has_image = 'image' in mem
                
                line = f"  Turn {mem.get('dialogue_index', 0)} - {role}: {text}"
                if has_image:
                    line += f"[Sent image: {mem.get('image_name', '')}]"
                
                context_parts.append(line)
        
        return "\n".join(context_parts)
    
    def _prepare_image_for_api(self, image_path: str) -> str:
        """Prepare image for API"""
        try:
            with Image.open(image_path) as img:
                # Resize to appropriate size
                img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                
                # Convert to RGB
                if img.mode in ('RGBA', 'LA', 'P'):
                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    rgb_img.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = rgb_img
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Convert to base64
                buffer = BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                return img_base64
                
        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            raise
    
    def _call_vlm_api(self, prompt: str, images) -> Dict[str, Any]:
        """Call VLM API"""
        start_time = time.time()
        
        messages = [{
            "role": "user",
            "content": []
        }]
        
        # Add text
        messages[0]["content"].append({
            "type": "text",
            "text": prompt
        })
        # Add images
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
                        logger.error(f"Failed to add image {image}: {e}")
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
                        logger.error(f"Failed to add image {image}: {e}")
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
                    logger.warning(f"API returned error code {response.status_code}: {response.text}")
                    
            except Exception as e:
                logger.warning(f"API call failed (attempt {attempt+1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2)
        
        return {
            "answer": "[API call failed]",
            "processing_time": time.time() - start_time,
            "success": False,
            "error": "All retries failed"
        }
    
    def _construct_prompt(self, question: QuestionAnswerPair, memories: List[Dict]) -> str:
        """Build prompt for question with retrieved memories"""
        
        # Format memory context
        context = self._format_memory_context(memories)
        
        # Get question type
        question_type = question.question_type.get("sub_type", "") if question.question_type else ""
        
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
        
        # Timing variables
        recall_time = 0.0
        prompt_build_time = 0.0
        api_call_time = 0.0
        
        try:
            # 1. Build query
            if question.image_context and os.path.exists(question.image_context[0]):
                query = {
                    'text': question.question_text,
                    'image': {'path': question.image_context[0]}
                }
            else:
                query = question.question_text
            
            # 2. Recall relevant memories based on query (record time)
            recall_start = time.time()
            memories = self.memory_system.recall(query)
            recall_time = time.time() - recall_start
            
            # 3. Build prompt (record time)
            prompt_start = time.time()
            prompt = self._construct_prompt(question, memories)
            prompt_build_time = time.time() - prompt_start
            
            # 4. Prepare images
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
            
            # 5. Call VLM (record time)
            api_start = time.time()
            vlm_response = self._call_vlm_api(prompt, images)
            api_call_time = vlm_response.get("processing_time", time.time() - api_start)
            
            # Total processing time
            total_processing_time = time.time() - start_time
            
            # 6. Create result
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
                memory_type="NGMMemorySystem",
                vlm_model=self.model,
                processing_time=total_processing_time,
                confidence=0.7 if vlm_response.get("success") else 0.0,
                retrieved_nodes=[memories[i]["dialogue_id"] for i in range(len(memories))],
                graph_stats=self.memory_system.get_graph_statistics(),
                success=vlm_response.get("success", False),
                # New timing fields
                recall_time=recall_time,
                prompt_build_time=prompt_build_time,
                api_call_time=api_call_time
            )
            
            # Update statistics
            stats = self.session_statistics[question.session_id]
            stats["successful"] += 1
            stats["processing_time"] += total_processing_time
            stats["total_recall_time"] = stats.get("total_recall_time", 0) + recall_time
            stats["total_prompt_build_time"] = stats.get("total_prompt_build_time", 0) + prompt_build_time
            stats["total_api_call_time"] = stats.get("total_api_call_time", 0) + api_call_time
                        
            logger.info(f"✓ Successfully processed: {question.question_id} "
                        f"(Total: {total_processing_time:.2f}s, Recall: {recall_time:.3f}s, "
                        f"Prompt: {prompt_build_time:.3f}s, API: {api_call_time:.2f}s)")
            
            return result
            
        except Exception as e:
            total_processing_time = time.time() - start_time
            error_msg = str(e)[:200]
            logger.error(f"Failed to process question {question.question_id}: {error_msg}")
            
            
            return EvaluationResult(
                sample_id=f"error_{question.question_id}_{int(time.time())}",
                session_id=question.session_id,
                dialogue_name=question.dialogue_name,
                question_id=question.question_id,
                question_text=question.question_text,
                question_image=question.question_image,
                system_answer=f"[Processing error: {error_msg}]",
                original_answer=question.original_answer,
                answer_source=question.answer_source,
                question_type=question.question_type,
                category=question.category,
                difficulty=question.difficulty,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                memory_type="NGMMemorySystem",
                vlm_model=self.model,
                processing_time=total_processing_time,
                confidence=0.0,
                success=False,
                error_message=error_msg,
                recall_time=recall_time,
                prompt_build_time=prompt_build_time,
                api_call_time=api_call_time
            )
    
    def evaluate_session(self, session_id: str, questions: List[QuestionAnswerPair], session_path: Path) -> Dict:
        """Evaluate entire session, return dictionary containing results and metadata"""
        self.session_statistics[session_id]["total"] = len(questions)
        
        results = []
        for i, q in enumerate(questions):
            logger.info(f"  Processing question {i+1}/{len(questions)}: {q.question_id}")
            result = self.evaluate_single_question(q)
            results.append(asdict(result))
            
            # Save intermediate results
            self._save_intermediate_results(session_id, results, session_path)
        
        # Output session timing statistics
        stats = self.session_statistics[session_id]
        successful = stats["successful"]
        if successful > 0:
            logger.info(f"Session {session_id} timing statistics - "
                        f"Avg recall: {stats['total_recall_time']/successful:.3f}s, "
                        f"Avg prompt: {stats['total_prompt_build_time']/successful:.3f}s, "
                        f"Avg API: {stats['total_api_call_time']/successful:.2f}s")
        
        # Build metadata
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        graph_stats = self.memory_system.get_graph_statistics()
        
        metadata = {
            "session_id": session_id,
            "session_path": str(session_path),
            "vlm_model": self.model,
            "memory_type": type(self.memory_system).__name__,
            "base_url": self.base_url,
            "context_type": "multimodal_ngm",
            "retrieval_topk": getattr(self, 'retrieval_topk', 10),
            "max_context_tokens": getattr(self, 'max_context_tokens', 4096),
            "max_images": getattr(self, 'max_images', 5),
            "evaluation_time": timestamp,
            "total_questions": len(questions),
            "successful_questions": successful,
            "graph_total_nodes": graph_stats.get('total_nodes', 0) if graph_stats else 0,
            "graph_total_edges": graph_stats.get('total_edges', 0) if graph_stats else 0,
            "graph_avg_degree": graph_stats.get('avg_degree', 0) if graph_stats else 0,
            "similarity_threshold": getattr(self.memory_system.store_op, 'similarity_threshold', 0.7),
            "traversal_strategy": getattr(self.memory_system.recall_op, 'traversal_strategy', 'breadth_first'),
            "max_depth": getattr(self.memory_system.recall_op, 'max_depth', 3),
            "max_nodes": getattr(self.memory_system.recall_op, 'max_nodes', 10),
            "test_mode": self.test_mode,
        }
        
        return {
            "metadata": metadata,
            "results": results,
            "statistics": dict(stats)  # Convert to regular dictionary
        }
    
    def _save_intermediate_results(self, session_id: str, results: List[Dict], session_path: Path):
        """Save intermediate results, including metadata"""
        session_dir = session_path / "evaluation_results"
        session_dir.mkdir(exist_ok=True)
        
        output_file = os.path.join(session_dir, "results_NGM.json")
        
        # Build metadata
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        graph_stats = self.memory_system.get_graph_statistics()
        
        metadata = {
            "session_id": session_id,
            "session_path": str(session_path),
            "vlm_model": self.model,
            "memory_type": type(self.memory_system).__name__,
            "base_url": self.base_url,
            "context_type": "multimodal_ngm",
            "retrieval_topk": getattr(self, 'retrieval_topk', 10),
            "max_context_tokens": getattr(self, 'max_context_tokens', 4096),
            "max_images": getattr(self, 'max_images', 5),
            "evaluation_time": timestamp,
            "total_questions": len(results),
            "successful_questions": sum(1 for r in results if r.get('success', False)),
            "graph_total_nodes": graph_stats.get('total_nodes', 0) if graph_stats else 0,
            "graph_total_edges": graph_stats.get('total_edges', 0) if graph_stats else 0,
            "graph_avg_degree": graph_stats.get('avg_degree', 0) if graph_stats else 0,
            "similarity_threshold": getattr(self.memory_system.store_op, 'similarity_threshold', 0.7),
            "traversal_strategy": getattr(self.memory_system.recall_op, 'traversal_strategy', 'breadth_first'),
            "max_depth": getattr(self.memory_system.recall_op, 'max_depth', 3),
            "max_nodes": getattr(self.memory_system.recall_op, 'max_nodes', 10),
            "test_mode": self.test_mode,
        }
        
        # Get current statistics
        current_stats = self.session_statistics[session_id]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": metadata,
                "results": results,
                "statistics": dict(current_stats)
            }, f, ensure_ascii=False, indent=2)


# ==================== Main Function ====================

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="NGM Memory System - Complete Implementation (GME Encoder)")
    
    # Path parameters - all required
    parser.add_argument("--conversations_dir", type=str, required=True,
                       help="Conversation data directory (required)")
    parser.add_argument("--image_base_path", type=str, default="",
                       help="Base path for images, used for relative image paths (optional)")
    
    # Encoder parameters
    parser.add_argument("--encoder_method", type=str, default="GMEEncoder",
                       choices=["GMEEncoder", "CLIPEncoder"],
                       help="Encoder type")
    parser.add_argument("--encoder_model", type=str, 
                       default="Alibaba-NLP/gme-Qwen2-VL-7B-Instruct",
                       help="Encoder model path or name")
    parser.add_argument("--embedding_dim", type=int, default=1024,
                       help="Embedding vector dimension")
    parser.add_argument("--retrieval_mode", type=str, default="cosine",
                       choices=["cosine", "dot", "L2"],
                       help="Retrieval mode")
    parser.add_argument("--retrieval_topk", type=int, default=10,
                       help="Number of most similar memories to retrieve")
    
    # Graph construction parameters
    parser.add_argument("--similarity_threshold", type=float, default=0.7,
                       help="Semantic similarity threshold for creating edges")
    parser.add_argument("--max_edges_per_node", type=int, default=5,
                       help="Maximum number of edges per node")
    parser.add_argument("--temporal_decay_constant", type=int, default=3600,
                       help="Temporal decay constant (seconds), default 1 hour")
    
    # Graph traversal parameters
    parser.add_argument("--max_depth", type=int, default=3,
                       help="Maximum depth for graph traversal")
    parser.add_argument("--max_nodes", type=int, default=10,
                       help="Maximum number of nodes to recall")
    parser.add_argument("--traversal_threshold", type=float, default=0.5,
                       help="Similarity threshold for graph traversal")
    parser.add_argument("--traversal_strategy", type=str, default="breadth_first",
                       choices=["depth_first", "breadth_first"],
                       help="Graph traversal strategy")
    parser.add_argument("--initial_candidate_multiplier", type=int, default=2,
                       help="Initial candidate node multiplier")
    
    # VLM API parameters
    parser.add_argument("--api_key", type=str, required=True,
                       help="VLM API key (required)")
    parser.add_argument("--vlm_model", type=str, required=True,
                       help="VLM model name (required)")
    parser.add_argument("--base_url", type=str, required=True,
                       help="API base URL (required)")
    parser.add_argument("--max_retries", type=int, default=3,
                       help="Maximum number of API call retries")
    parser.add_argument("--timeout", type=int, default=60,
                       help="API call timeout (seconds)")
    
    # Evaluation parameters
    parser.add_argument("--test_mode", action="store_true",
                       help="Test mode, process only first 2 questions per session")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose logging output")
    
    parser.add_argument("--max_context_tokens", type=int, default=4096,
                       help="Maximum context tokens")
    parser.add_argument("--max_images", type=int, default=5,
                       help="Maximum number of images to include")
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    print("=" * 70)
    print("NGM Memory System - Complete Implementation (GME Encoder)")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Encoder: {args.encoder_method}")
    print(f"  Model Path: {args.encoder_model}")
    print(f"  Retrieval Mode: {args.retrieval_mode}")
    print(f"  Similarity Threshold: {args.similarity_threshold}")
    print(f"  Traversal Strategy: {args.traversal_strategy}")
    print(f"  Test Mode: {args.test_mode}")
    print("=" * 70)
    
    # Build configuration object
    from types import SimpleNamespace
    
    # Create configuration objects
    encoder_config = SimpleNamespace(
        method=args.encoder_method,
        encoder_model=args.encoder_model,
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
    
    # 1. Initialize NGM memory system
    print("\n[1] Initializing NGM memory system...")
    memory_system = NGMMemorySystem(config)
    
    # 2. Load and store conversations (with timing statistics)
    print("\n[2] Loading conversations and building memory graph...")
    store_start_total = time.time()
    
    conversations = DialogueLoader.load_conversations(args.conversations_dir)
    
    store_times = []  # Record each store operation time
    
    for i, conv in enumerate(conversations):
        if i % 10 == 0:
            print(f"   Processed {i}/{len(conversations)} conversations")
        
        # Store and get duration
        node_id, store_time = memory_system.store(conv)
        store_times.append(store_time)
    
    store_total_time = time.time() - store_start_total
    
    stats = memory_system.get_graph_statistics()
    store_stats = memory_system.get_store_statistics()
    
    print(f"\n   Memory construction complete!")
    print(f"   Total nodes: {stats['total_nodes']}")
    print(f"   Total edges: {stats['total_edges']}")
    print(f"   Average node degree: {stats['avg_degree']:.2f}")
    
    # Output storage time statistics
    print(f"\n   [Storage Time Statistics]:")
    print(f"   Total conversations stored: {len(conversations)}")
    print(f"   Total storage time: {store_total_time:.2f} seconds")
    print(f"   Average per storage: {store_total_time/len(conversations):.4f} seconds")
    print(f"   Fastest storage: {min(store_times):.4f} seconds")
    print(f"   Slowest storage: {max(store_times):.4f} seconds")
    
    # Stage statistics
    if store_stats['num_stores'] > 0:
        print(f"   StoreOp total time: {store_stats['total_time']:.2f} seconds")
        print(f"   StoreOp average time: {store_stats['avg_time']:.4f} seconds")
    
    # 3. Initialize evaluator
    print("\n[3] Initializing VLM evaluator...")
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
        retrieval_topk=args.retrieval_topk,  # New
        max_context_tokens=getattr(args, 'max_context_tokens', 4096),  # New
        max_images=getattr(args, 'max_images', 5)  # New
    )
    
    # 4. Load questions
    print("\n[4] Loading question files...")
    sessions_questions = evaluator.load_questions(args.conversations_dir)
    
    total_questions = sum(len(data["questions"]) for data in sessions_questions.values())
    print(f"   Loaded {total_questions} questions from {len(sessions_questions)} sessions")
    
    # 5. Evaluate
    print("\n[5] Starting evaluation...")
    print("-" * 70)
    for session_id, session_data in sessions_questions.items():
        print(f"\nProcessing Session: {session_id}")
        session_path = Path(args.conversations_dir) / "scenes" / str(session_id)
        questions = session_data["questions"]
        
        # Determine number of questions to process
        if args.test_mode:
            process_questions = questions[:2]
            print(f"  Test mode: Processing {len(process_questions)}/{len(questions)} questions")
        else:
            process_questions = questions
            print(f"  Processing all {len(process_questions)} questions")
        
        # Now evaluate_session returns dictionary containing metadata and results
        session_result = evaluator.evaluate_session(session_id, process_questions, session_path)
        
        print(f"  Completed {len(session_result['results'])} questions")

    
    # 6. Output statistics
    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)
    
    total_processed = sum(s["total"] for s in evaluator.session_statistics.values())
    total_successful = sum(s["successful"] for s in evaluator.session_statistics.values())
    
    print(f"\nStatistics:")
    print(f"   Questions processed: {total_processed}")
    print(f"   Successful: {total_successful}")
    if total_processed > 0:
        print(f"   Success rate: {total_successful/total_processed*100:.1f}%")
    
    print(f"\nNGM Graph Statistics:")
    print(f"   Total nodes: {stats['total_nodes']}")
    print(f"   Total edges: {stats['total_edges']}")
    print(f"   Average node degree: {stats['avg_degree']:.2f}")
    print(f"   Similarity threshold: {stats['config']['similarity_threshold']}")
    print(f"   Traversal strategy: {stats['config']['traversal_strategy']}")
    
    # Output storage time summary
    print(f"\n[Storage Time Summary]:")
    print(f"   Total conversations stored: {len(conversations)}")
    print(f"   Total storage time: {store_total_time:.2f} seconds")
    print(f"   Average per storage: {store_total_time/len(conversations):.4f} seconds")
    
    if store_stats['num_stores'] > 0:
        print(f"   StoreOp total time: {store_stats['total_time']:.2f} seconds")
        print(f"   Storage percentage: {store_stats['total_time']/store_total_time*100:.1f}%")
    
    print("=" * 70)


if __name__ == "__main__":
    main()