#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
retrievers.py
记忆检索器实现 - 包含SimpleEmbeddingRetriever和HybridRetriever
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import numpy as np
import pickle
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize

# 下载nltk数据（如果需要）
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def simple_tokenize(text):
    """简单的分词函数"""
    return word_tokenize(text.lower())


class SimpleEmbeddingRetriever:
    """基于嵌入向量的简单检索器"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """初始化检索器
        
        Args:
            model_name: SentenceTransformer模型名称
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.corpus: List[str] = []  # 文档列表
        self.embeddings: Optional[np.ndarray] = None  # 文档嵌入
    
    def add_documents(self, documents: List[str]):
        """批量添加文档
        
        Args:
            documents: 文档内容列表
        """
        if not documents:
            return
        
        # 如果没有现有文档，直接替换
        if not self.corpus:
            self.corpus = documents
            self.embeddings = self.model.encode(documents)
        else:
            # 追加新文档
            self.corpus.extend(documents)
            
            # 计算新文档的嵌入
            new_embeddings = self.model.encode(documents)
            
            if self.embeddings is None:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
    
    def search(self, query: str, k: int = 5) -> List[int]:
        """搜索相似文档
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            文档索引列表
        """
        if not self.corpus or self.embeddings is None:
            return []
        # 编码查询
        query_embedding = self.model.encode([query])[0]
        
        # 计算余弦相似度
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # 获取top-k索引
        k = min(k, len(self.corpus))
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        return top_k_indices.tolist()
    
    def search_with_scores(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """搜索并返回带分数的结果"""
        if not self.corpus or self.embeddings is None:
            return []
        
        query_embedding = self.model.encode([query])[0]
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        k = min(k, len(self.corpus))
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            results.append({
                "index": idx,
                "document": self.corpus[idx],
                "score": float(similarities[idx])
            })
        
        return results
    
    def save(self, cache_file: str, embeddings_file: str):
        """保存检索器状态"""
        # 保存嵌入向量
        if self.embeddings is not None:
            np.save(embeddings_file, self.embeddings)
        
        # 保存其他状态
        state = {
            'model_name': self.model_name,
            'corpus': self.corpus
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, cache_file: str, embeddings_file: str) -> 'SimpleEmbeddingRetriever':
        """加载检索器状态"""
        # 加载嵌入向量
        if os.path.exists(embeddings_file):
            self.embeddings = np.load(embeddings_file)
        
        # 加载其他状态
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                state = pickle.load(f)
                self.model_name = state.get('model_name', self.model_name)
                self.corpus = state.get('corpus', [])
        
        return self
    
    def clear(self):
        """清空检索器"""
        self.corpus = []
        self.embeddings = None


class HybridRetriever:
    """混合检索系统 - 结合BM25关键词匹配和语义搜索"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', alpha: float = 0.5):
        """初始化混合检索器
        
        Args:
            model_name: SentenceTransformer模型名称
            alpha: 混合权重 (0 = 仅BM25, 1 = 仅语义)
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.alpha = alpha
        self.bm25 = None
        self.corpus: List[str] = []  # 文档列表
        self.embeddings: Optional[np.ndarray] = None  # 文档嵌入
        self.document_ids: Dict[str, int] = {}  # 文档内容到索引的映射
    
    def add_documents(self, documents: List[str]):
        """批量添加文档到BM25和语义索引
        
        Args:
            documents: 文档内容列表
        """
        if not documents:
            return
        
        start_idx = len(self.corpus)
        self.corpus.extend(documents)
        
        # 更新文档ID映射
        for i, doc in enumerate(documents):
            self.document_ids[doc] = start_idx + i
        
        # 为BM25分词
        tokenized_docs = [simple_tokenize(doc) for doc in self.corpus]
        
        # 重新构建BM25（需要完整重建）
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # 更新语义嵌入
        new_embeddings = self.model.encode(documents)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
    
    def add_document(self, document: str) -> bool:
        """添加单个文档
        
        Args:
            document: 文档内容
            
        Returns:
            bool: 是否成功添加
        """
        if document in self.document_ids:
            return False
        
        # 添加到语料库
        doc_idx = len(self.corpus)
        self.corpus.append(document)
        self.document_ids[document] = doc_idx
        
        # 重新构建BM25（需要完整重建）
        tokenized_docs = [simple_tokenize(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # 更新嵌入
        doc_embedding = self.model.encode([document])[0]
        if self.embeddings is None:
            self.embeddings = doc_embedding.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, doc_embedding])
        
        return True
    
    def search(self, query: str, k: int = 5) -> List[int]:
        """使用混合评分检索文档
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            文档索引列表
        """
        if not self.corpus or self.bm25 is None or self.embeddings is None:
            print("search", flush=True)
            return []
        print("search_ok", flush=True)
        # 获取BM25分数
        tokenized_query = simple_tokenize(query)
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))
        
        # 归一化BM25分数
        if len(bm25_scores) > 0 and bm25_scores.max() > bm25_scores.min():
            bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-6)
        
        # 获取语义分数
        query_embedding = self.model.encode([query])[0]
        semantic_scores = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # 组合分数
        hybrid_scores = self.alpha * bm25_scores + (1 - self.alpha) * semantic_scores
        
        # 获取top-k索引
        k = min(k, len(self.corpus))
        top_k_indices = np.argsort(hybrid_scores)[-k:][::-1]
        
        return top_k_indices.tolist()
    
    def search_with_scores(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        print("search_with_scores", flush=True)
        """搜索并返回详细的分数信息"""
        if not self.corpus or self.bm25 is None or self.embeddings is None:
            return []
        
        tokenized_query = simple_tokenize(query)
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))
        
        # 保存原始BM25分数用于显示
        raw_bm25_scores = bm25_scores.copy()
        
        # 归一化BM25分数用于混合
        if len(bm25_scores) > 0 and bm25_scores.max() > bm25_scores.min():
            bm25_scores_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-6)
        else:
            bm25_scores_norm = bm25_scores
        
        query_embedding = self.model.encode([query])[0]
        semantic_scores = cosine_similarity([query_embedding], self.embeddings)[0]
        
        hybrid_scores = self.alpha * bm25_scores_norm + (1 - self.alpha) * semantic_scores
        
        k = min(k, len(self.corpus))
        top_k_indices = np.argsort(hybrid_scores)[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            results.append({
                "index": idx,
                "document": self.corpus[idx],
                "bm25_score": float(raw_bm25_scores[idx]),
                "semantic_score": float(semantic_scores[idx]),
                "hybrid_score": float(hybrid_scores[idx]),
                "alpha": self.alpha
            })
        
        return results
    
    def save(self, cache_file: str, embeddings_file: str):
        """保存检索器状态到磁盘"""
        
        # 保存嵌入向量
        if self.embeddings is not None:
            np.save(embeddings_file, self.embeddings)
        
        # 保存其他状态（BM25不能直接pickle，需要特殊处理）
        # 保存BM25的参数
        bm25_state = None
        if self.bm25 is not None:
            # 保存BM25的文档频率等信息
            bm25_state = {
                'doc_freqs': self.bm25.doc_freqs,
                'idf': self.bm25.idf,
                'doc_len': self.bm25.doc_len,
                'avgdl': self.bm25.avgdl
            }
        
        state = {
            'alpha': self.alpha,
            'bm25_state': bm25_state,
            'corpus': self.corpus,
            'document_ids': self.document_ids,
            'model_name': self.model_name
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(state, f)
    
    @classmethod
    def load(cls, cache_file: str, embeddings_file: str) -> 'HybridRetriever':
        """从磁盘加载检索器状态"""
        
        # 加载状态
        with open(cache_file, 'rb') as f:
            state = pickle.load(f)
        
        # 创建新实例
        retriever = cls(
            model_name=state['model_name'], 
            alpha=state['alpha']
        )
        
        # 恢复基本属性
        retriever.corpus = state['corpus']
        retriever.document_ids = state.get('document_ids', {})
        
        # 恢复BM25
        bm25_state = state.get('bm25_state')
        if bm25_state and retriever.corpus:
            # 重新构建BM25
            tokenized_corpus = [simple_tokenize(doc) for doc in retriever.corpus]
            retriever.bm25 = BM25Okapi(tokenized_corpus)
        
        # 加载嵌入向量
        if os.path.exists(embeddings_file):
            retriever.embeddings = np.load(embeddings_file)
        
        return retriever
    
    def clear(self):
        """清空检索器"""
        self.corpus = []
        self.bm25 = None
        self.embeddings = None
        self.document_ids = {}
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取检索器统计信息"""
        return {
            "total_documents": len(self.corpus),
            "alpha": self.alpha,
            "model_name": self.model_name,
            "has_bm25": self.bm25 is not None,
            "has_embeddings": self.embeddings is not None,
            "embeddings_shape": self.embeddings.shape if self.embeddings is not None else None
        }