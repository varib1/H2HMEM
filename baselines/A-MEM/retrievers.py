#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
retrievers.py
Memory Retriever Implementations - Includes SimpleEmbeddingRetriever and HybridRetriever
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

# Download nltk data (if needed)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def simple_tokenize(text):
    """Simple tokenization function"""
    return word_tokenize(text.lower())


class SimpleEmbeddingRetriever:
    """Simple retriever based on embedding vectors"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize retriever
        
        Args:
            model_name: SentenceTransformer model name
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.corpus: List[str] = []  # Document list
        self.embeddings: Optional[np.ndarray] = None  # Document embeddings
    
    def add_documents(self, documents: List[str]):
        """Batch add documents
        
        Args:
            documents: List of document contents
        """
        if not documents:
            return
        
        # If no existing documents, replace directly
        if not self.corpus:
            self.corpus = documents
            self.embeddings = self.model.encode(documents)
        else:
            # Append new documents
            self.corpus.extend(documents)
            
            # Calculate embeddings for new documents
            new_embeddings = self.model.encode(documents)
            
            if self.embeddings is None:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
    
    def search(self, query: str, k: int = 5) -> List[int]:
        """Search for similar documents
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of document indices
        """
        if not self.corpus or self.embeddings is None:
            return []
        # Encode query
        query_embedding = self.model.encode([query])[0]
        
        # Calculate cosine similarity
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get top-k indices
        k = min(k, len(self.corpus))
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        return top_k_indices.tolist()
    
    def search_with_scores(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search and return results with scores"""
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
        """Save retriever state"""
        # Save embeddings
        if self.embeddings is not None:
            np.save(embeddings_file, self.embeddings)
        
        # Save other state
        state = {
            'model_name': self.model_name,
            'corpus': self.corpus
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, cache_file: str, embeddings_file: str) -> 'SimpleEmbeddingRetriever':
        """Load retriever state"""
        # Load embeddings
        if os.path.exists(embeddings_file):
            self.embeddings = np.load(embeddings_file)
        
        # Load other state
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                state = pickle.load(f)
                self.model_name = state.get('model_name', self.model_name)
                self.corpus = state.get('corpus', [])
        
        return self
    
    def clear(self):
        """Clear retriever"""
        self.corpus = []
        self.embeddings = None


class HybridRetriever:
    """Hybrid retrieval system - combines BM25 keyword matching and semantic search"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', alpha: float = 0.5):
        """Initialize hybrid retriever
        
        Args:
            model_name: SentenceTransformer model name
            alpha: Hybrid weight (0 = BM25 only, 1 = semantic only)
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.alpha = alpha
        self.bm25 = None
        self.corpus: List[str] = []  # Document list
        self.embeddings: Optional[np.ndarray] = None  # Document embeddings
        self.document_ids: Dict[str, int] = {}  # Document content to index mapping
    
    def add_documents(self, documents: List[str]):
        """Batch add documents to BM25 and semantic index
        
        Args:
            documents: List of document contents
        """
        if not documents:
            return
        
        start_idx = len(self.corpus)
        self.corpus.extend(documents)
        
        # Update document ID mapping
        for i, doc in enumerate(documents):
            self.document_ids[doc] = start_idx + i
        
        # Tokenize for BM25
        tokenized_docs = [simple_tokenize(doc) for doc in self.corpus]
        
        # Rebuild BM25 (requires full rebuild)
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # Update semantic embeddings
        new_embeddings = self.model.encode(documents)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
    
    def add_document(self, document: str) -> bool:
        """Add a single document
        
        Args:
            document: Document content
            
        Returns:
            bool: Whether addition was successful
        """
        if document in self.document_ids:
            return False
        
        # Add to corpus
        doc_idx = len(self.corpus)
        self.corpus.append(document)
        self.document_ids[document] = doc_idx
        
        # Rebuild BM25 (requires full rebuild)
        tokenized_docs = [simple_tokenize(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # Update embeddings
        doc_embedding = self.model.encode([document])[0]
        if self.embeddings is None:
            self.embeddings = doc_embedding.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, doc_embedding])
        
        return True
    
    def search(self, query: str, k: int = 5) -> List[int]:
        """Search documents using hybrid scoring
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of document indices
        """
        if not self.corpus or self.bm25 is None or self.embeddings is None:
            return []
        # Get BM25 scores
        tokenized_query = simple_tokenize(query)
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))
        
        # Normalize BM25 scores
        if len(bm25_scores) > 0 and bm25_scores.max() > bm25_scores.min():
            bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-6)
        
        # Get semantic scores
        query_embedding = self.model.encode([query])[0]
        semantic_scores = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Combine scores
        hybrid_scores = self.alpha * bm25_scores + (1 - self.alpha) * semantic_scores
        
        # Get top-k indices
        k = min(k, len(self.corpus))
        top_k_indices = np.argsort(hybrid_scores)[-k:][::-1]
        
        return top_k_indices.tolist()
    
    def search_with_scores(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search and return detailed score information"""
        if not self.corpus or self.bm25 is None or self.embeddings is None:
            return []
        
        tokenized_query = simple_tokenize(query)
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))
        
        # Save raw BM25 scores for display
        raw_bm25_scores = bm25_scores.copy()
        
        # Normalize BM25 scores for hybrid combination
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
        """Save retriever state to disk"""
        
        # Save embeddings
        if self.embeddings is not None:
            np.save(embeddings_file, self.embeddings)
        
        # Save other state (BM25 cannot be pickled directly, needs special handling)
        # Save BM25 parameters
        bm25_state = None
        if self.bm25 is not None:
            # Save BM25 document frequencies, etc.
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
        """Load retriever state from disk"""
        
        # Load state
        with open(cache_file, 'rb') as f:
            state = pickle.load(f)
        
        # Create new instance
        retriever = cls(
            model_name=state['model_name'], 
            alpha=state['alpha']
        )
        
        # Restore basic attributes
        retriever.corpus = state['corpus']
        retriever.document_ids = state.get('document_ids', {})
        
        # Restore BM25
        bm25_state = state.get('bm25_state')
        if bm25_state and retriever.corpus:
            # Rebuild BM25
            tokenized_corpus = [simple_tokenize(doc) for doc in retriever.corpus]
            retriever.bm25 = BM25Okapi(tokenized_corpus)
        
        # Load embeddings
        if os.path.exists(embeddings_file):
            retriever.embeddings = np.load(embeddings_file)
        
        return retriever
    
    def clear(self):
        """Clear retriever"""
        self.corpus = []
        self.bm25 = None
        self.embeddings = None
        self.document_ids = {}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get retriever statistics"""
        return {
            "total_documents": len(self.corpus),
            "alpha": self.alpha,
            "model_name": self.model_name,
            "has_bm25": self.bm25 is not None,
            "has_embeddings": self.embeddings is not None,
            "embeddings_shape": self.embeddings.shape if self.embeddings is not None else None
        }