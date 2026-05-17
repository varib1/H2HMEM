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

# 在文件开头的导入部分添加
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

# 在文件开头的导入部分添加
import concurrent.futures
import threading
from threading import Lock, Semaphore
from tqdm import tqdm  # 可选，用于进度条，需要安装：pip install tqdm

# 导入API相关库
import requests
from PIL import Image
import base64
from io import BytesIO

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
    recall_method: str = "naive_rag"
    retrieved_chunks: Optional[List[Dict]] = None
    success: bool = True
    error_message: Optional[str] = None
    reasoning_process: Optional[str] = None
    
    # 新增：时间相关字段
    memory_retrieve_time: float = 0.0    # 记忆检索时间
    prompt_build_time: float = 0.0       # 提示词构建时间
    api_call_time: float = 0.0           # API调用时间
    retrieval_timing: Dict = field(default_factory=dict)  # 检索时间详情


class NaiveRAGMemorySystem:
    """
    基于Naive RAG的记忆系统
    - 将对话分割成块
    - 使用关键词检索
    - 只检索与问题相关的部分
    """
    
    def __init__(self, conversations_dir: str, chunk_size: int = 1, top_k: int = 3, 
             embedding_model: str = "all-MiniLM-L6-v2"):
        """
        初始化Naive RAG记忆系统（使用BERT embedding）
        
        Args:
            conversations_dir: 对话数据目录
            chunk_size: 每个块包含的对话轮数
            top_k: 检索的块数量
            embedding_model: 使用的embedding模型名称
        """
        self.conversations_dir = conversations_dir
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.embedding_model_name = embedding_model
        
        self.memory_storage = {}  # 存储所有session的内容
        self.all_chunks = []      # 所有对话块
        self.chunk_metadata = []  # 每个块的元数据
        self.chunk_embeddings = None  # 所有块的embeddings
        self.session_info = {}     # session额外信息
        
        # 新增：存储时间记录
        self.storage_time = 0.0      # 总存储时间
        self.loading_time = 0.0      # 数据加载时间
        self.chunking_time = 0.0     # 分块时间
        self.embedding_time = 0.0    # 向量化时间
        
        # 加载embedding模型
        self._load_embedding_model()
        
    def _load_embedding_model(self):
        """加载BERT embedding模型"""
        try:
            logger.info(f"正在加载embedding模型: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            # 检查是否有GPU可用
            if torch.cuda.is_available():
                self.embedding_model = self.embedding_model.to('cuda')
                logger.info("使用GPU加速embedding计算")
            else:
                logger.info("使用CPU计算embedding")
                
            logger.info("embedding模型加载完成")
        except Exception as e:
            logger.error(f"加载embedding模型失败: {e}")
            logger.warning("将使用备用的关键词匹配方法")
            self.embedding_model = None
        
    def load_all_conversations(self):
        """加载整个对话的所有session数据并分块，计算embeddings - 添加时间记录"""
        overall_start = time.time()
        
        scenes_dir = os.path.join(self.conversations_dir, "scenes")
        if not os.path.exists(scenes_dir):
            raise ValueError(f"找不到scenes目录: {scenes_dir}")
        
        session_dirs = natsorted([
            d for d in os.listdir(scenes_dir) 
            if os.path.isdir(os.path.join(scenes_dir, d))
        ])
        
        all_chunks = []
        chunk_metadata = []
        
        # 1. 数据加载时间
        loading_start = time.time()
        
        for session_dir_name in session_dirs:
            session_dir = os.path.join(scenes_dir, session_dir_name)
            session_data = self._load_single_session(session_dir_name, session_dir)
            
            if session_data:
                session_id = session_dir_name
                self.memory_storage[session_id] = session_data
                
                # 获取caption目录路径
                caption_dir = os.path.join(session_dir, "caption")
                caption_files_exist = os.path.exists(caption_dir)
                
                # 提取对话内容并添加session信息
                dialogues = session_data.get("dialogue", [])
                processed_dialogues = []
                timeline_date = session_data.get("timeline_date", "")
                for i, dialogue in enumerate(dialogues, 1):
                    role = dialogue.get("role", "")
                    content = dialogue.get("content", {})
                    text = timeline_date + ":" + content.get("text", "")
                    image_filename = content.get("image", "")
                    
                    # 处理图片描述信息
                    image_description = self._load_image_description(image_filename, caption_dir, caption_files_exist)
                    
                    # 创建包含图片描述的对话内容
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
                
                # 存储session信息
                self.session_info[session_id] = {
                    "session_dir_name": session_dir_name,
                    "session_title": session_data.get("session_title", ""),
                    "timeline_date": session_data.get("timeline_date", ""),
                    "generated_at": session_data.get("generated_at", ""),
                    "dialogue_count": len(dialogues),
                    "has_caption_dir": caption_files_exist,
                    "session_path": session_dir
                }
                
                # 将当前session的对话分块（分块操作计入chunking时间）
                chunk_start = time.time()
                session_chunks, session_metadata = self._chunk_dialogues(processed_dialogues, session_id)
                all_chunks.extend(session_chunks)
                chunk_metadata.extend(session_metadata)
                self.chunking_time += time.time() - chunk_start
        
        self.loading_time = time.time() - loading_start
        logger.info(f"数据加载耗时: {self.loading_time:.2f}秒")
        
        # 2. 存储块和元数据
        self.all_chunks = all_chunks
        self.chunk_metadata = chunk_metadata
        
        logger.info(f"分块处理耗时: {self.chunking_time:.2f}秒")
        
        # 3. 向量化时间
        embedding_start = time.time()
        self._compute_chunk_embeddings()
        self.embedding_time = time.time() - embedding_start
        logger.info(f"向量化耗时: {self.embedding_time:.2f}秒")
        
        # 总存储时间
        self.storage_time = time.time() - overall_start
        
        logger.info(f"已加载 {len(self.memory_storage)} 个session")
        logger.info(f"对话分块完成: {len(self.all_chunks)} 个块，每块最多{self.chunk_size}轮对话")
        logger.info(f"检索配置: top_k={self.top_k}, embedding_model={self.embedding_model_name}")
        logger.info(f"记忆存储总耗时: {self.storage_time:.2f}秒 (加载: {self.loading_time:.2f}s, 分块: {self.chunking_time:.2f}s, 向量化: {self.embedding_time:.2f}s)")
        
        # 统计信息
        chunks_with_images = sum(1 for meta in self.chunk_metadata if meta["has_image"])
        logger.info(f"包含图片的块: {chunks_with_images}")

    def _combine_dialogue_text(self, role: str, text: str, image_description: str) -> str:
        """组合对话文本用于embedding计算"""
        combined = []
        if role:
            combined.append(f"{role}:")
        if text:
            combined.append(text)
        if image_description:
            combined.append(f"[图片描述: {image_description}]")
        return " ".join(combined)

    def _compute_chunk_embeddings(self):
        """计算所有块的embeddings"""
        if self.embedding_model is None:
            logger.warning("没有可用的embedding模型，跳过embedding计算")
            self.chunk_embeddings = None
            return
        
        logger.info(f"开始计算 {len(self.all_chunks)} 个块的embeddings...")
        
        # 准备文本列表
        texts = []
        for i, chunk in enumerate(self.all_chunks):
            # 从chunk文本中提取纯文本部分（去掉标记）
            text = self._extract_text_from_chunk(chunk)
            texts.append(text)
        
        # 批量计算embeddings
        try:
            embeddings = self.embedding_model.encode(
                texts, 
                show_progress_bar=True,
                batch_size=32,
                convert_to_numpy=True
            )
            self.chunk_embeddings = embeddings
            logger.info(f"Embedding计算完成，维度: {embeddings.shape}")
        except Exception as e:
            logger.error(f"计算embeddings失败: {e}")
            self.chunk_embeddings = None

    def _extract_text_from_chunk(self, chunk: str) -> str:
        """从chunk文本中提取纯文本（去掉标记信息）"""
        lines = chunk.split('\n')
        text_lines = []
        
        for line in lines:
            # 跳过session标记行
            if line.startswith('[Session') or line.startswith('---') or not line.strip():
                continue
            
            # 提取对话内容
            # 格式："第X轮 role: text"
            match = re.match(r'第\d+轮\s+(\w+):\s*(.*)', line)
            if match:
                role, content = match.groups()
                text_lines.append(content)
            else:
                # 如果不是标准格式，保留原行
                text_lines.append(line)
        
        return " ".join(text_lines)
        
    def _load_image_description(self, image_filename: str, caption_dir: str, caption_files_exist: bool) -> str:
        """加载图片描述信息"""
        if not image_filename or not caption_files_exist:
            return ""
        
        image_description = ""
        # 提取文件名中的数字部分
        caption_json = Path(image_filename).stem + ".json"  # 获取文件名（不带扩展名）并添加.json扩展名
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
                        image_description = " | ".join(description_texts)
                        
                except Exception as e:
                    logger.error(f"加载图片描述文件 {caption_file_path} 失败: {e}")
        
        return image_description
    
    def _load_single_session(self, session_dir_name: str, session_dir: str) -> Optional[Dict]:
        """加载单个session的数据"""
        conversation_file = os.path.join(session_dir, "session.json")
        
        if not os.path.exists(conversation_file):
            logger.warning(f"未找到session.json文件: {conversation_file}")
            return None
        
        try:
            with open(conversation_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            logger.debug(f"成功加载 {session_dir_name} 的对话数据")
            return session_data
            
        except Exception as e:
            logger.error(f"加载 {conversation_file} 失败: {e}")
            return None
    
    def _chunk_dialogues(self, dialogues: List[Dict], session_id: str) -> Tuple[List[str], List[Dict]]:
        """
        将对话分块
        
        Args:
            dialogues: 对话列表
            session_id: session标识
        
        Returns:
            chunks: 文本块列表
            metadata: 每个块的元数据
        """
        chunks = []
        metadata = []
        
        for i in range(0, len(dialogues), self.chunk_size):
            chunk_dialogues = dialogues[i:i + self.chunk_size]
            
            # 构建块文本（用于显示）
            chunk_text = f"[Session {session_id} - 对话块 {i//self.chunk_size + 1}]\n"
            
            # 构建用于embedding的组合文本
            combined_texts = []
            
            # 记录块信息
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
                    chunk_text += f"第{dialogue_index}轮 {role}: [图片: {image_filename}] {image_description} {text}\n"
                else:
                    chunk_text += f"第{dialogue_index}轮 {role}: {text}\n"
                
                # 收集用于embedding的文本
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
                "combined_text": " ".join(combined_texts),  # 保存组合文本用于embedding
                "embedding": None  # 将在后面计算
            })
        
        return chunks, metadata
        
    def _extract_keywords(self, text: str) -> List[str]:
        """
        从文本中提取关键词
        
        Args:
            text: 输入文本
        
        Returns:
            关键词列表
        """
        # 简单分词和过滤
        words = re.findall(r'[\u4e00-\u9fff\w]+', text)
        
        # 过滤停用词（可以扩展）
        stopwords = {"的", "了", "在", "是", "我", "你", "他", "她", "它", "我们", "你们", "他们", 
                     "这", "那", "和", "与", "也", "都", "就", "还", "但", "而", "并且", "或者"}
        
        # 过滤过短的词
        keywords = [w for w in words if len(w) > 1 and w not in stopwords]
        
        return keywords[:10]  # 最多返回10个关键词
    
    def _calculate_similarity(self, question_embedding: np.ndarray, chunk_embedding: np.ndarray) -> float:
        """
        计算问题和块的cosine相似度
        
        Args:
            question_embedding: 问题的embedding向量
            chunk_embedding: 块的embedding向量
        
        Returns:
            cosine相似度分数（0-1之间）
        """
        if question_embedding is None or chunk_embedding is None:
            return 0.0
        
        # 计算cosine相似度
        similarity = cosine_similarity(
            question_embedding.reshape(1, -1),
            chunk_embedding.reshape(1, -1)
        )[0][0]
        
        return float(similarity)

    # 在 NaiveRAGMemorySystem 类中添加 retrieve_relevant_context 方法的修改版本
    def retrieve_relevant_context(self, question_text: str, target_session_id: str) -> Dict[str, Any]:
        """
        检索与问题相关的上下文（使用BERT embedding）- 添加时间记录
        """
        start_time = time.time()
        
        # 1. 计算问题的embedding
        embedding_start = time.time()
        question_embedding = self._get_embedding(question_text)
        embedding_time = time.time() - embedding_start
        
        retrieval_time = 0
        similarity_time = 0
        
        if question_embedding is None or self.chunk_embeddings is None:
            logger.warning("无法计算embedding，使用备用的关键词匹配方法")
            return self._retrieve_by_keywords_fallback(question_text, target_session_id)
        
        # 2. 计算与所有块的相似度
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
        
        # 3. 按相似度排序并选择top_k
        retrieval_start = time.time()
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        top_chunks = similarities[:self.top_k]
        
        # 4. 构建检索到的上下文
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
        
        # 5. 如果没有检索到相关块，使用目标session的最近块作为fallback
        if not retrieved_context:
            logger.warning(f"未检索到相关内容，使用目标session {target_session_id} 的最近块作为fallback")
            return self._get_fallback_context(target_session_id)
        
        retrieval_time = time.time() - retrieval_start
        total_time = time.time() - start_time
        
        # 6. 构建完整的上下文信息
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
        获取文本的embedding向量
        
        Args:
            text: 输入文本
        
        Returns:
            embedding向量
        """
        if self.embedding_model is None:
            return None
        
        try:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"计算embedding失败: {e}")
            return None
    
    def get_session_context(self, target_session_id: str, question_text: str) -> Dict[str, Any]:
        """
        获取针对特定问题的检索增强上下文
        
        Args:
            target_session_id: 目标session ID
            question_text: 问题文本
        
        Returns:
            检索增强的上下文
        """
        # 执行检索
        retrieval_result = self.retrieve_relevant_context(question_text, target_session_id)
        
        # 构建返回的上下文格式
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
        """获取记忆系统的统计信息"""
        return {
            "total_sessions": len(self.memory_storage),
            "total_chunks": len(self.all_chunks),
            "chunk_size": self.chunk_size,
            "top_k": self.top_k,
            "embedding_model": self.embedding_model_name,
            "embedding_available": self.embedding_model is not None,
            "embedding_dimension": self.chunk_embeddings.shape[1] if self.chunk_embeddings is not None else 0,
            "session_info": self.session_info,
            "chunks_with_images": sum(1 for meta in self.chunk_metadata if meta["has_image"])
        }
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆系统的统计信息 - 添加存储时间"""
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
            # 新增存储时间统计
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
        "Test-Time Learning (TTL)": "Learn and adapt from the retrieved conversation context at test time to answer the question.",
        "Conflict Detection (CD)": "Check whether this information conflicts with the retrieved conversation chunks.",
        "Answer Refusal (AR)": "Determine if the question can be answered based on the retrieved conversation chunks."
    }
    
    # Response format requirements
    FORMAT_REQUIREMENTS = {
        "Conflict Detection (CD)": "Response format: Reply strictly with either 'Yes' or 'No' only.",
        "Answer Refusal (AR)": "Response format: If the information is present in the retrieved conversation chunks, provide answer based on that information; if not present, reply with: 'Not mentioned.'",
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
             max_workers: int = 3,  # 新增：最大工作线程数
             max_api_concurrency: int = 2):  # 新增：最大API并发数
        """
        初始化VLM评估器（多线程版本）
        
        Args:
            memory_system: 记忆系统实例（Naive RAG）
            api_key: VLM API密钥
            model: VLM模型名称
            base_url: API基础URL
            verbose: 详细日志输出
            max_retries: 最大重试次数
            timeout: 请求超时时间（秒）
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
        
        # 新增：线程控制相关属性
        self.max_workers = max_workers
        self.max_api_concurrency = max_api_concurrency
        self.api_semaphore = Semaphore(max_api_concurrency)  # 控制API并发
        self.file_lock = Lock()  # 文件写入锁
        self.stats_lock = Lock()  # 统计信息更新锁
        
        
        # 存储统计信息
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
            "max_workers": max_workers,  # 新增
            "max_api_concurrency": max_api_concurrency  # 新增
        }
        
        # 测试API连接
        self._test_api_connection()

        # 新增：记录失败的问题文件路径（使用set自动去重）
        self.failed_json_files = set()  # 使用set自动去重
        self.failed_lock = Lock()  # 线程安全的锁

        
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
            按session_id分组的字典:{session_id: {"questions": [], "session_path": str}}
        """
        sessions_questions = {}
        
        # 解析目录结构
        base_dir = Path(conversations_dir)
        
        # 检查是否是"dialogueX"这样的顶层目录
        if base_dir.name.startswith("dialogue"):
            dialogue_name = base_dir.name
            scenes_dir = base_dir / "scenes"
        else:
            # 尝试在目录下查找包含"dialogue"的子目录
            dialogue_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("dialogue")]
            if not dialogue_dirs:
                raise ValueError(f"找不到对话目录: {base_dir}")
            
            dialogue_name = dialogue_dirs[0].name
            scenes_dir = dialogue_dirs[0] / "scenes"
        
        if not scenes_dir.exists():
            raise ValueError(f"找不到scenes目录: {scenes_dir}")
        
        logger.info(f"正在从 {scenes_dir} 加载问题文件...")
        
        # 遍历所有session目录
        session_dirs = [d for d in scenes_dir.iterdir() if d.is_dir()]
        # session_dirs = [
        #     scenes_dir / "session0"
        # ]
        
        for session_dir in session_dirs:
            session_dir_name = session_dir.name
            question_file = session_dir / "questions.json"
            
            if question_file.exists():
                try:
                    # 首先读取session的session.json获取session_id
                    conversation_file = session_dir / "session.json"
                    session_id = session_dir_name  # 默认为目录名
                    
                    if conversation_file.exists():
                        with open(conversation_file, 'r', encoding='utf-8') as f:
                            conv_data = json.load(f)
                            session_id = conv_data.get("session_id", session_dir_name)
                    
                    # 加载问题文件
                    with open(question_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    questions = data.get("questions", [])
                    
                    # 转换格式为QuestionAnswerPair列表
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
                        
                        # 添加图片上下文
                        if qa_pair.question_image:
                            if str(session_id) == "session0":
                                print("处理session0的图片路径", qa_pair.question_image)
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
    
    def _format_retrieved_context(self, context: Dict[str, Any]) -> str:
        """
        格式化检索到的上下文为文本
        
        Args:
            context: 包含检索结果的上下文
        
        Returns:
            格式化后的文本
        """
        if not context:
            return "无可用记忆"
        
        context_parts = []
        
        # 添加检索信息
        retrieval_info = context.get("retrieval_info", {})
        retrieved_metadata = retrieval_info.get("retrieved_metadata", [])
        similarity_method = retrieval_info.get("similarity_method", "unknown")
        
        context_parts.append("【Naive RAG检索结果 (BERT Embedding)】")
        context_parts.append(f"相似度计算方法: {similarity_method}")
        context_parts.append(f"检索到 {len(retrieved_metadata)} 个相关块 (top_k={self.memory_system.top_k})")
        
        # 显示检索到的块信息
        context_parts.append("\n【检索到的相关块】")
        for i, meta in enumerate(retrieved_metadata, 1):
            session_id = meta.get("session_id", "未知")
            chunk_idx = meta.get("chunk_index", 0)
            dial_indices = meta.get("dialogue_indices", [])
            similarity = meta.get("similarity", 0)
            has_image = meta.get("has_image", False)
            is_fallback = meta.get("fallback", False)
            method = meta.get("method", "embedding")
            
            if is_fallback:
                context_parts.append(f"\n块 {i} [Session {session_id} 块 {chunk_idx} - Fallback] (对话轮次: {dial_indices[0]}-{dial_indices[-1]})")
            else:
                context_parts.append(f"\n块 {i} [Session {session_id} 块 {chunk_idx}] (相似度: {similarity:.4f}, 方法: {method}, 对话轮次: {dial_indices[0]}-{dial_indices[-1]})")
            
            if has_image:
                context_parts.append(f"  (包含图片)")
        
        # 添加检索到的具体内容
        retrieved_chunks = retrieval_info.get("retrieved_chunks", [])
        if retrieved_chunks:
            context_parts.append("\n【检索到的对话内容】")
            for i, chunk in enumerate(retrieved_chunks, 1):
                context_parts.append(f"\n--- 块 {i} ---")
                context_parts.append(chunk)
        
        return "\n".join(context_parts)
    
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
                print("成功处理")
                return img_base64
                
        except Exception as e:
            logger.error(f"处理图片 {image_path} 失败: {e}")
            raise
    
    def _call_vlm_api(self, prompt: str, images: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        调用真实的VLM API（带并发控制）
        
        Args:
            prompt: 提示词文本
            images: 图片路径列表
        
        Returns:
            API响应结果
        """
        start_time = time.time()
        # 新增：使用信号量控制API并发
        acquired = False
        if self.api_semaphore:
            self.api_semaphore.acquire()
            acquired = True
            if self.verbose:
                logger.debug(f"API信号量已获取，当前可用: {self.api_semaphore._value}")
        
        try:
            # 准备消息
            messages = []
            
            # 如果有图片，将图片作为消息的一部分
            if images and len(images) > 0:
                # 处理第一张图片
                try:
                    image_base64 = self._prepare_image_for_api(images[0])
                    # 构建消息（支持图片格式）
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
                # 仅文本
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
                    
                    logger.debug(f"调用API (尝试 {attempt + 1}/{self.max_retries}): {api_url}")
                    
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
                            
                            logger.debug(f"API调用成功，响应时间: {processing_time:.2f}秒")
                            
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
            
            processing_time = time.time() - start_time
            return {
                "answer": f"[API调用失败: 所有{self.max_retries}次重试都失败]",
                "processing_time": processing_time,
                "model": self.model,
                "success": False,
                "error": "所有重试都失败"
            }
        finally:
            # 新增：释放信号量
            if acquired:
                self.api_semaphore.release()
                if self.verbose:
                    logger.debug(f"API信号量已释放，当前可用: {self.api_semaphore._value}")
    
    def _construct_prompt_for_question(self, 
                                  question_pair: QuestionAnswerPair,
                                  memory_context: Dict[str, Any]) -> str:
        """Build prompt for specific question using Naive RAG retrieved context"""
        
        # Format retrieved context
        context_str = self._format_retrieved_context(memory_context)
        
        # Extract question components
        question_text = question_pair.question_text
        question_type = question_pair.question_type.get("subsub_type", "")
        
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
        """评估单个问题 - 添加详细时间记录和错误记录"""
        start_time = time.time()
        
        # 时间记录变量
        memory_retrieve_time = 0
        prompt_build_time = 0
        api_call_time = 0
        
        try:
            logger.debug(f"处理问题: {session_id} - {question_pair.question_id} ({question_pair.category})")
            
            # 1. 准备查询文本（包括图片caption信息）
            query_text = question_pair.question_text
            
            # 如果有图片，添加图片caption信息
            if question_pair.question_image and question_pair.image_context:
                captions = []
                for img_path in question_pair.image_context:
                    # 从图片路径生成对应的caption文件路径
                    img_dir = os.path.dirname(img_path)
                    img_filename = os.path.basename(img_path)
                    img_name_without_ext = os.path.splitext(img_filename)[0]
                    
                    # 构建caption路径
                    caption_dir = img_dir.replace('\\image\\', '\\caption\\')
                    caption_path = os.path.join(caption_dir, f"{img_name_without_ext}.json")
                    
                    # 如果上述路径不存在，尝试其他可能的路径结构
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
                                        logger.debug(f"成功读取caption: {caption_path}")
                        except Exception as e:
                            logger.warning(f"读取caption文件失败 {caption_path}: {e}")
                
                # 将图片caption信息添加到查询文本中
                if captions:
                    captions_text = ' '.join([f"[image{i+1}description]: {cap}" for i, cap in enumerate(captions)])
                    query_text = f"[question image description]: {captions_text}\n[user question]: {question_pair.question_text}"
                    logger.debug(f"增强后的查询文本长度: {len(query_text)}")
            
            # 2. 获取检索增强的上下文（记录检索时间）
            memory_retrieve_start = time.time()
            rag_context = self.memory_system.get_session_context(session_id, query_text)
            memory_retrieve_time = time.time() - memory_retrieve_start
            
            # 获取检索时间详情
            retrieval_timing = rag_context.get("retrieval_info", {}).get("timing", {})
            
            # 创建记忆上下文摘要
            retrieval_info = rag_context.get("retrieval_info", {})
            retrieved_metadata = retrieval_info.get("retrieved_metadata", [])
            
            memory_context_summary = f"检索到 {len(retrieved_metadata)} 个相关块"
            if retrieval_timing:
                memory_context_summary += f" (检索耗时: {retrieval_timing.get('total_time', 0):.3f}秒)"
            
            # 3. 构建提示词（记录构建时间）
            prompt_build_start = time.time()
            prompt = self._construct_prompt_for_question(question_pair, rag_context)
            prompt_build_time = time.time() - prompt_build_start
            
            # 4. 准备图片（用于VLM的视觉输入）
            images = []
            if question_pair.question_image and question_pair.image_context:
                for img_path in question_pair.image_context:
                    if os.path.exists(img_path):
                        images.append(img_path)
                    else:
                        logger.warning(f"图片文件不存在: {img_path}")
            
            # 5. 调用VLM API（记录API调用时间）
            api_call_start = time.time()
            vlm_response = self._call_vlm_api(prompt=prompt, images=images)
            api_call_time = vlm_response.get("processing_time", time.time() - api_call_start)
            
            response_text = vlm_response.get("answer", "").strip()
            success = vlm_response.get("success", False)
            
            # 6. 解析JSON响应，提取推理过程和答案
            reasoning_process = None
            system_answer = response_text
            
            if success:
                try:
                    # 查找JSON内容（可能被markdown代码块包裹）
                    json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        # 尝试直接查找JSON对象
                        json_match = re.search(r'(\{.*?\})', response_text, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(1)
                        else:
                            json_str = response_text
                    
                    # 解析JSON
                    parsed_response = json.loads(json_str)
                    
                    # 提取推理过程和答案
                    reasoning_process = parsed_response.get("reasoning_process", "")
                    system_answer = parsed_response.get("system_answer", response_text)
                    
                    logger.debug(f"成功解析JSON响应 - 推理过程长度: {len(reasoning_process) if reasoning_process else 0}")
                    
                except json.JSONDecodeError as e:
                    # 如果无法解析为JSON，使用原始响应作为答案
                    logger.warning(f"无法解析API响应为JSON: {e}")
                    logger.debug(f"原始响应: {response_text[:200]}...")
                    system_answer = response_text
                    reasoning_process = None
            
            # 7. 计算置信度
            confidence = 0.7
            
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
                recall_method="naive_rag",
                retrieved_chunks=retrieved_metadata,
                success=success,
                error_message=None if success else vlm_response.get("error", ""),
                reasoning_process=reasoning_process,
                # 新增时间字段
                memory_retrieve_time=memory_retrieve_time,
                prompt_build_time=prompt_build_time,
                api_call_time=api_call_time,
                retrieval_timing=retrieval_timing
            )
            
            # 更新session统计
            with self.stats_lock:
                self.session_statistics[session_id]["successful"] += 1
                self.session_statistics[session_id]["processing_time"] += total_processing_time
                
                # 累计时间统计
                if "total_memory_retrieve_time" not in self.session_statistics[session_id]:
                    self.session_statistics[session_id]["total_memory_retrieve_time"] = 0
                    self.session_statistics[session_id]["total_prompt_build_time"] = 0
                    self.session_statistics[session_id]["total_api_call_time"] = 0
                
                self.session_statistics[session_id]["total_memory_retrieve_time"] += memory_retrieve_time
                self.session_statistics[session_id]["total_prompt_build_time"] += prompt_build_time
                self.session_statistics[session_id]["total_api_call_time"] += api_call_time
            
            logger.info(f"✓ 成功处理: {session_id} - {question_pair.question_id} (总: {total_processing_time:.2f}秒, 检索: {memory_retrieve_time:.3f}秒, API: {api_call_time:.2f}秒)")
            
            return result
            
        except Exception as e:
            total_processing_time = time.time() - start_time
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
                processing_time=total_processing_time,
                confidence=0.0,
                supporting_evidence=question_pair.supporting_evidence,
                memory_context_summary="错误: 无法检索上下文",
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
            
            # 更新session统计
            with self.stats_lock:
                self.session_statistics[session_id]["failed"] += 1
            
            return result
    
    def evaluate_session_questions(self,
                                session_id: str,
                                session_data: Dict,
                                max_questions: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        并行评估一个session内的所有问题
        """
        questions = session_data["questions"]
        session_path = Path(session_data["session_path"])
        session_dir_name = session_data.get("session_dir_name", session_id)
        question_file_path = session_data.get("question_file", "")  # 获取问题文件路径
        
        if max_questions and max_questions < len(questions):
            questions = questions[:max_questions]
        
        total_questions = len(questions)
        logger.info(f"开始并行评估 {session_id} 的 {total_questions} 个问题")
        logger.info(f"使用Naive RAG方法：块大小={self.memory_system.chunk_size}, top_k={self.memory_system.top_k}")
        logger.info(f"API并发数: {self.max_api_concurrency}")
        
        # 线程安全地初始化session统计
        with self.stats_lock:
            self.session_statistics[session_id]["total"] = total_questions
            for qa in questions:
                self.session_statistics[session_id]["by_category"][qa.category] += 1
                self.session_statistics[session_id]["by_difficulty"][qa.difficulty] += 1
            
            # 初始化时间统计字段
            if "total_memory_retrieve_time" not in self.session_statistics[session_id]:
                self.session_statistics[session_id]["total_memory_retrieve_time"] = 0
                self.session_statistics[session_id]["total_prompt_build_time"] = 0
                self.session_statistics[session_id]["total_api_call_time"] = 0
        
        # 用于存储结果的线程安全列表
        results = []
        results_lock = Lock()
        processed_count = 0
        processed_lock = Lock()
        
        # 使用线程池并行处理问题
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_api_concurrency) as executor:
            # 提交所有问题任务
            future_to_question = {}
            for question_pair in questions:
                future = executor.submit(
                    self._evaluate_question_with_stats,
                    question_pair,
                    session_id,
                    question_file_path  # 传递文件路径
                )
                future_to_question[future] = question_pair
            
            # 使用tqdm显示进度（可选）
            try:
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
                                pbar.set_postfix({"成功": "✓", "ID": question_pair.question_id[:20]})
                            else:
                                pbar.set_postfix({"成功": "✗", "ID": question_pair.question_id[:20]})
                            
                        except Exception as e:
                            logger.error(f"问题 {question_pair.question_id} 处理失败: {e}")
                            
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
                # 如果没有tqdm，使用简单进度显示
                for future in concurrent.futures.as_completed(future_to_question):
                    question_pair = future_to_question[future]
                    try:
                        result_dict = future.result()
                        
                        with results_lock:
                            results.append(result_dict)
                        
                        with processed_lock:
                            processed_count += 1
                            if processed_count % 5 == 0 or processed_count == total_questions:
                                logger.info(f"[{session_id}] 进度: {processed_count}/{total_questions}")
                        
                    except Exception as e:
                        logger.error(f"问题 {question_pair.question_id} 处理失败: {e}")
                        
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
        
        # 最终保存结果（需要线程安全）
        with self.file_lock:
            self._save_session_results(
                session_id,
                session_dir_name,
                session_path,
                results,
                final=True
            )
        
        successful = len([r for r in results if r.get('success', False)])
        logger.info(f"Session {session_id} 并行处理完成: 成功 {successful}/{total_questions}")
        
        # 输出session时间统计
        with self.stats_lock:
            session_stats = self.session_statistics[session_id]
            if successful > 0:
                logger.info(f"  Session时间统计 - 平均检索: {session_stats.get('total_memory_retrieve_time', 0)/successful:.3f}秒, "
                        f"平均构建: {session_stats.get('total_prompt_build_time', 0)/successful:.3f}秒, "
                        f"平均API: {session_stats.get('total_api_call_time', 0)/successful:.2f}秒")
        
        # 更新全局统计
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
        """保存单个session的结果到对应session目录"""
        # 在session目录下创建结果目录
        session_results_dir = session_path / "evaluation_results"
        session_results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存JSON结果
        json_filename = f"results_naive_rag.json"
        json_file = session_results_dir / json_filename
        
        # 构建完整的结果数据结构
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
        
        logger.debug(f"已保存 {session_id} 的Naive RAG结果到: {json_file}")
        
    
    def evaluate_all_sessions(self,
                        sessions_questions: Dict[str, Dict],
                        max_questions_per_session: Optional[int] = None):
        """
        并行评估所有session的问题（多线程版本）
        """
        self.global_statistics["start_time"] = time.time()
        self.global_statistics["total_sessions"] = len(sessions_questions)
        
        memory_stats = self.memory_system.get_memory_stats()
        logger.info(f"开始并行评估 {len(sessions_questions)} 个session")
        logger.info(f"记忆系统: {type(self.memory_system).__name__}")
        logger.info(f"  总块数: {memory_stats['total_chunks']}")
        logger.info(f"  块大小: {memory_stats['chunk_size']}")
        logger.info(f"  检索top_k: {memory_stats['top_k']}")
        logger.info(f"线程配置: max_workers={self.max_workers}, max_api_concurrency={self.max_api_concurrency}")
        
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
            successful_sessions = 0
            failed_sessions = 0
            
            # 使用tqdm显示session进度
            try:
                from tqdm import tqdm
                with tqdm(total=len(sessions_questions), desc="总体进度", unit="session") as pbar:
                    for future in concurrent.futures.as_completed(future_to_session):
                        session_id = future_to_session[future]
                        completed += 1
                        try:
                            results = future.result()
                            successful_sessions += 1
                            pbar.set_postfix({"成功": "✓", "session": session_id[:10]})
                            logger.info(f"[{completed}/{len(sessions_questions)}] Session {session_id} 处理完成，成功处理 {len(results)} 个问题")
                        except Exception as e:
                            failed_sessions += 1
                            logger.error(f"Session {session_id} 处理失败: {e}")
                            pbar.set_postfix({"成功": "✗", "session": session_id[:10]})
                        pbar.update(1)
            except ImportError:
                # 如果没有tqdm，使用简单进度显示
                for future in concurrent.futures.as_completed(future_to_session):
                    session_id = future_to_session[future]
                    completed += 1
                    try:
                        results = future.result()
                        successful_sessions += 1
                        logger.info(f"[{completed}/{len(sessions_questions)}] Session {session_id} 处理完成，成功处理 {len(results)} 个问题")
                    except Exception as e:
                        failed_sessions += 1
                        logger.error(f"Session {session_id} 处理失败: {e}")
        
        self.global_statistics["end_time"] = time.time()
        self.global_statistics["successful_sessions"] = successful_sessions
        self.global_statistics["failed_sessions"] = failed_sessions
        
        # 输出总体统计
        total_time = self.global_statistics["end_time"] - self.global_statistics["start_time"]
        logger.info(f"\n{'='*60}")
        logger.info(f"评估完成统计:")
        logger.info(f"  - 总Session数: {len(sessions_questions)}")
        logger.info(f"  - 成功Session数: {successful_sessions}")
        logger.info(f"  - 失败Session数: {failed_sessions}")
        logger.info(f"  - 总问题数: {self.global_statistics['total_questions']}")
        logger.info(f"  - 成功问题数: {self.global_statistics['successful_questions']}")
        logger.info(f"  - 失败问题数: {self.global_statistics['failed_questions']}")
        logger.info(f"  - 总耗时: {total_time:.2f}秒")
        if self.global_statistics['total_questions'] > 0:
            logger.info(f"  - 平均每问题: {total_time/self.global_statistics['total_questions']:.2f}秒")
        logger.info(f"{'='*60}")
        
    def _evaluate_session_parallel(self, session_id: str, session_data: Dict, 
                             max_questions_per_session: Optional[int]) -> List[Dict]:
        """
        session评估的包装方法（用于线程池调用）
        """
        import threading
        thread_name = threading.current_thread().name
        logger.info(f"线程 [{thread_name}] 开始处理 session: {session_id}")
        
        try:
            # 调用并行处理session内问题的方法
            results = self.evaluate_session_questions(
                session_id, session_data, max_questions_per_session
            )
            return results
        except Exception as e:
            logger.error(f"线程 [{thread_name}] 处理 session {session_id} 时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def _evaluate_question_with_stats(self, question_pair: QuestionAnswerPair, 
                                 session_id: str,
                                 question_file_path: str = None) -> Dict[str, Any]:
        """
        评估单个问题并更新统计信息（线程安全）
        
        Args:
            question_pair: 问题-答案对
            session_id: session ID
            question_file_path: 问题文件的路径
        """
        try:
            # 调用原有的评估方法，传递文件路径
            result = self.evaluate_single_question(question_pair, session_id, question_file_path)
            result_dict = asdict(result)
            
            # 线程安全地更新统计信息
            with self.stats_lock:
                if result.success:
                    self.session_statistics[session_id]["successful"] += 1
                else:
                    self.session_statistics[session_id]["failed"] += 1
                
                # 累计时间统计
                self.session_statistics[session_id]["total_memory_retrieve_time"] += getattr(result, 'memory_retrieve_time', 0)
                self.session_statistics[session_id]["total_prompt_build_time"] += getattr(result, 'prompt_build_time', 0)
                self.session_statistics[session_id]["total_api_call_time"] += getattr(result, 'api_call_time', 0)
                self.session_statistics[session_id]["processing_time"] += result.processing_time
            
            return result_dict
            
        except Exception as e:
            logger.error(f"评估问题 {question_pair.question_id} 时发生未捕获异常: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
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
                supporting_evidence=question_pair.supporting_evidence,
                memory_context_summary="错误: 处理异常",
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
            
            # 更新失败统计
            with self.stats_lock:
                self.session_statistics[session_id]["failed"] += 1
            
            return asdict(error_result)


def create_memory_system(memory_type: str, conversations_dir: str, **kwargs):
    """创建记忆系统"""

    if memory_type == "naive_rag":
        chunk_size = kwargs.get("chunk_size", 5)
        top_k = kwargs.get("top_k", 3)
        return NaiveRAGMemorySystem(conversations_dir, chunk_size=chunk_size, top_k=top_k)
    else:
        raise ValueError(f"不支持的记忆类型: {memory_type}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="VLM intra-session记忆能力评估器（使用Naive RAG）")
    parser.add_argument("--conversations_dir", required=True, help="Path to dialogue folder ")
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
    

    # 并行处理
    parser.add_argument("--max_workers", type=int, default=3, help="Parallel sessions")
    parser.add_argument("--max_api_concurrency", type=int, default=2, help="Parallel questions per session")
    
    parser.add_argument("--embedding_model", default="all-MiniLM-L6-v2", help="Sentence-Transformer model name")
    
    args = parser.parse_args()
    
    # 配置日志级别
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    print("=" * 70)
    print("VLM Intra-Session记忆能力评估器（使用Naive RAG）")
    print(f"模型: {args.model}")
    print(f"API端点: {args.base_url}")
    print(f"记忆系统: {args.memory_type}")
    if args.memory_type == "naive_rag":
        print(f"  块大小: {args.chunk_size}")
        print(f"  检索top_k: {args.top_k}")
    print("=" * 70)
    
    # 测试模式设置
    if args.test_mode:
        args.max_questions_per_session = 2
        print("测试模式：每个session只处理前2个问题")
    
    # 1. 初始化记忆系统
    print(f"\n[1] 初始化记忆系统 ({args.memory_type})...")
    print(f"   加载整个对话的所有session内容...")
    
    # 在创建记忆系统时
    if args.memory_type == "naive_rag":
        memory_system = NaiveRAGMemorySystem(
            args.conversations_dir, 
            chunk_size=args.chunk_size,
            top_k=args.top_k,
            embedding_model=args.embedding_model  # 新增
        )
    
    memory_system.load_all_conversations()
    
    # 显示记忆系统统计
    if args.memory_type == "naive_rag":
        stats = memory_system.get_memory_stats()
        print(f"   已加载 {stats['total_sessions']} 个session")
        print(f"   分块完成: {stats['total_chunks']} 个块")
        print(f"   包含图片的块: {stats['chunks_with_images']}")
    
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
        max_workers=args.max_workers,  # 新增
        max_api_concurrency=args.max_api_concurrency  # 新增
    )
    
    # 3. 加载intra-session问题
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
    
    total_questions = sum(len(data["questions"]) for data in sessions_questions.values())
    print(f"   总问题数: {total_questions}")
    
    # 限制处理的session数
    if args.max_sessions and args.max_sessions < len(sessions_questions):
        sessions_to_process = dict(list(sessions_questions.items())[:args.max_sessions])
        print(f"   限制处理前 {args.max_sessions} 个session")
    else:
        sessions_to_process = sessions_questions
    
    # 4. 执行评估
    print(f"\n[4] 开始按session评估（使用Naive RAG检索）...")
    print(f"   处理session数: {len(sessions_to_process)}")
    print(f"   总问题数: {total_questions}")
    print(f"   检索配置: chunk_size={args.chunk_size}, top_k={args.top_k}")
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