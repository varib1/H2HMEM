#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
memory_system.py
智能记忆系统 - 支持混合检索器
"""

import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
import logging

from core import MemoryNote
from retrievers import SimpleEmbeddingRetriever, HybridRetriever
from llm_controller import LLMController

logger = logging.getLogger(__name__)


class AgenticMemorySystem:
    """智能记忆系统 - 管理记忆的获取、加工和进化"""
    
    def __init__(self, 
                 dialogue_name: str = "",
                 embedding_model_name: str = 'all-MiniLM-L6-v2',
                 llm_model: str = "gpt-4o-mini",
                 evo_threshold: int = 100,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 retriever_type: str = "hybrid",  # "simple" 或 "hybrid"
                 hybrid_alpha: float = 0.5):  # 混合检索的权重
        
        self.dialogue_name = dialogue_name
        self.memories: Dict[str, MemoryNote] = {}  # id -> MemoryNote
        self.retriever_type = retriever_type
        self.hybrid_alpha = hybrid_alpha
        
        # 初始化检索器
        if retriever_type == "hybrid":
            self.retriever = HybridRetriever(model_name=embedding_model_name, alpha=hybrid_alpha)
            logger.info(f"使用混合检索器: alpha={hybrid_alpha}")
        else:
            self.retriever = SimpleEmbeddingRetriever(model_name=embedding_model_name)
            logger.info("使用简单嵌入检索器")
        
        # 初始化LLM控制器
        self.llm_controller = LLMController(
            model=llm_model,
            api_key=api_key,
            base_url=base_url,
        )
        
        # 进化系统提示词
        self.evolution_system_prompt = '''
You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
Analyze the the new memory note according to keywords and context, also with their several nearest neighbors memory.
Make decisions about its evolution.  

The new memory context:
{context}
content: {content}
keywords: {keywords}

The nearest neighbors memories:
{nearest_neighbors_memories}

Based on this information, determine:
1. Should this memory be evolved? Consider its relationships with other memories.
2. What specific actions should be taken (strengthen, update_neighbor)?
   2.1 If choose to strengthen the connection, which memory should it be connected to? Can you give the updated tags of this memory?
   2.2 If choose to update_neighbor, you can update the context and tags of these memories based on the understanding of these memories. If the context and the tags are not updated, the new context and tags should be the same as the original ones. Generate the new context and tags in the sequential order of the input neighbors.
Tags should be determined by the content of these characteristic of these memories, which can be used to retrieve them later and categorize them.
Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.
The number of neighbors is {neighbor_number}.

Return your decision in JSON format with the following structure:
{{
    "should_evolve": true or false,
    "actions": ["strengthen", "update_neighbor"],
    "suggested_connections": ["neighbor_memory_ids"],
    "tags_to_update": ["tag_1",..."tag_n"], 
    "new_context_neighborhood": ["new context",...,"new context"],
    "new_tags_neighborhood": [["tag_1",...,"tag_n"],...["tag_1",...,"tag_n"]]
}}
'''
        
        # 进化响应格式
        self.evolution_response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "should_evolve": {
                            "type": "boolean"
                        },
                        "actions": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        },
                        "suggested_connections": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        },
                        "new_context_neighborhood": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        },
                        "tags_to_update": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        },
                        "new_tags_neighborhood": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            }
                        }
                    },
                    "required": [
                        "should_evolve",
                        "actions",
                        "suggested_connections",
                        "tags_to_update",
                        "new_context_neighborhood",
                        "new_tags_neighborhood"
                    ],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
        
        self.evo_cnt = 0
        self.evo_threshold = evo_threshold
        
        # 用于检索器的映射
        self.memory_id_to_index: Dict[str, int] = {}
        self.index_to_memory_id: Dict[int, str] = {}
    
    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        """添加新的记忆笔记"""
        
        # 创建记忆节点（LLM分析在MemoryNote内部完成）
        note = MemoryNote(
            content=content, 
            llm_controller=self.llm_controller, 
            timestamp=time, 
            **kwargs
        )
        
        # 添加对话相关元数据
        if 'dialogue_name' in kwargs:
            note.dialogue_name = kwargs['dialogue_name']
        if 'session_id' in kwargs:
            note.session_id = kwargs['session_id']
        if 'dialogue_index' in kwargs:
            note.dialogue_index = kwargs['dialogue_index']
        if 'role' in kwargs:
            note.role = kwargs['role']
        
        # 处理记忆进化
        evo_label, note = self.process_memory(note)
        
        # 存储记忆
        self.memories[note.id] = note
        
        # 更新检索器
        search_text = note.get_searchable_text()
        
        # 记录索引映射
        current_idx = len(self.retriever.corpus)
        self.memory_id_to_index[note.id] = current_idx
        self.index_to_memory_id[current_idx] = note.id
        
        # 添加到检索器
        self.retriever.add_documents([search_text])
        
        # 进化计数和合并
        if evo_label:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()
        
        return note.id
    
    def consolidate_memories(self):
        """合并记忆 - 重新构建检索器"""
        logger.info("合并记忆...")
        
        # 保存检索器类型和参数
        retriever_type = self.retriever_type
        hybrid_alpha = self.hybrid_alpha if hasattr(self, 'hybrid_alpha') else 0.5
        
        try:
            model_name = self.retriever.model.get_config_dict()['model_name']
        except (AttributeError, KeyError):
            model_name = 'all-MiniLM-L6-v2'
        
        # 创建新检索器
        if retriever_type == "hybrid":
            self.retriever = HybridRetriever(model_name=model_name, alpha=hybrid_alpha)
        else:
            self.retriever = SimpleEmbeddingRetriever(model_name=model_name)
        
        self.memory_id_to_index.clear()
        self.index_to_memory_id.clear()
        
        # 重新添加所有记忆
        documents = []
        for memory in self.memories.values():
            documents.append(memory.get_searchable_text())
        
        if documents:
            self.retriever.add_documents(documents)
            
            # 重建索引映射
            for idx, memory in enumerate(self.memories.values()):
                if idx < len(documents):
                    self.memory_id_to_index[memory.id] = idx
                    self.index_to_memory_id[idx] = memory.id
        
        logger.info(f"合并完成，共 {len(self.memories)} 条记忆")
    
    def process_memory(self, note: MemoryNote) -> Tuple[bool, MemoryNote]:
        """处理记忆进化"""
        
        # 查找相关记忆
        neighbor_memory, indices = self.find_related_memories_for_evolution(note.content, k=5)
        if not neighbor_memory:
            return False, note
        
        # 准备prompt
        keywords_str = ", ".join(note.keywords) if isinstance(note.keywords, list) else str(note.keywords)
        
        prompt_memory = self.evolution_system_prompt.format(
            context=note.context,
            content=note.content,
            keywords=keywords_str,
            nearest_neighbors_memories=neighbor_memory,
            neighbor_number=len(indices)
        )
        
        logger.debug(f"进化prompt: {prompt_memory[:200]}...")
        
        # 调用LLM
        try:
            response = self.llm_controller.llm.get_completion(
                prompt_memory,
                response_format=self.evolution_response_format,
                temperature=0.3
            )
            # 清理响应
            response_cleaned = self._clean_json_response(response)
            response_json = json.loads(response_cleaned)
            logger.debug(f"进化响应: {response_json}")
            
        except Exception as e:

            logger.error(f"进化处理失败: {e}")
            logger.error(f"原始响应: {response if 'response' in locals() else 'No response'}")
            return False, note
        
        # 执行进化决策
        should_evolve = response_json.get("should_evolve", False)
        
        if should_evolve:
            actions = response_json.get("actions", [])
            
            for action in actions:
                if action == "strengthen":
                    self._apply_strengthen_action(note, response_json, indices)
                elif action == "update_neighbor":
                    self._apply_update_neighbor_action(note, response_json, indices)
            
            # 记录进化历史
            note.evolution_history.append({
                "timestamp": datetime.now().isoformat(),
                "action": actions,
                "response": response_json
            })
        
        return should_evolve, note
    
    def find_related_memories_for_evolution(self, query: str, k: int = 5) -> Tuple[str, List[int]]:
        """查找相关记忆并返回格式化的文本（用于进化prompt）"""
        if not self.memories:
            return "", []
        
        # 获取检索结果索引
        indices = self.retriever.search(query, k)
        
        # 构建格式化的记忆文本
        all_memories = list(self.memories.values())
        memory_str = ""
        
        for i in indices:
            if i < len(all_memories):
                memory = all_memories[i]
                memory_str += (
                    f"memory index:{i}\t"
                    f"talk start time:{memory.timestamp}\t"
                    f"memory content: {memory.content}\t"
                    f"memory context: {memory.context}\t"
                    f"memory keywords: {memory.keywords}\t"
                    f"memory tags: {memory.tags}\n"
                )
        
        return memory_str, indices
    
    def find_related_memories(self, query: str, k: int = 5) -> List[MemoryNote]:
        """查找相关记忆并返回记忆列表"""
        if not self.memories:
            return []
        
        indices = self.retriever.search(query, k)
        print(indices)
        all_memories = list(self.memories.values())
        
        result = []
        for idx in indices:
            if idx < len(all_memories):
                result.append(all_memories[idx])
        
        return result
    
    def find_related_memories_with_scores(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """查找相关记忆并返回带分数的详细信息"""
        if not self.memories:
            return []
        
        # 获取带分数的检索结果
        if hasattr(self.retriever, 'search_with_scores'):
            results = self.retriever.search_with_scores(query, k)
        else:
            # 简单检索器不支持带分数，手动计算
            indices = self.retriever.search(query, k)
            all_memories = list(self.memories.values())
            results = []
            for idx in indices:
                if idx < len(all_memories):
                    results.append({
                        "index": idx,
                        "document": self.retriever.corpus[idx] if idx < len(self.retriever.corpus) else "",
                        "score": 0.0  # 简单检索器不返回分数
                    })
        
        # 添加记忆信息
        for r in results:
            idx = r["index"]
            if idx in self.index_to_memory_id:
                mem_id = self.index_to_memory_id[idx]
                if mem_id in self.memories:
                    r["memory"] = self.memories[mem_id]
        
        return results
    
    def find_related_memories_raw(self, query: str, k: int = 5) -> str:
        """查找相关记忆并返回包含邻居链接的详细文本"""
        if not self.memories:
            return ""
        
        indices = self.retriever.search(query, k)
        all_memories = list(self.memories.values())
        memory_str = ""
        
        for i in indices:
            if i >= len(all_memories):
                continue
                
            memory = all_memories[i]
            memory_str += (
                f"talk start time:{memory.timestamp} "
                f"memory content: {memory.content} "
                f"memory context: {memory.context} "
                f"memory keywords: {memory.keywords} "
                f"memory tags: {memory.tags}\n"
            )
            
            # 添加链接的记忆
            j = 0
            for neighbor_id in memory.links:
                if neighbor_id in self.memories:
                    neighbor = self.memories[neighbor_id]
                    memory_str += (
                        f"talk start time:{neighbor.timestamp} "
                        f"memory content: {neighbor.content} "
                        f"memory context: {neighbor.context} "
                        f"memory keywords: {neighbor.keywords} "
                        f"memory tags: {neighbor.tags}\n"
                    )
                    j += 1
                    if j >= k:
                        break
        
        return memory_str
    
    def _apply_strengthen_action(self, note: MemoryNote, response_json: Dict, indices: List[int]):
        """应用strengthen动作"""
        suggest_connections = response_json.get("suggested_connections", [])
        new_tags = response_json.get("tags_to_update", [])
        
        # 处理连接建议
        for conn in suggest_connections:
            conn_str = str(conn)
            # 如果传入的是索引，尝试转换为ID
            if conn_str.isdigit() and int(conn_str) in self.index_to_memory_id:
                conn_id = self.index_to_memory_id[int(conn_str)]
            else:
                conn_id = conn_str
            
            if conn_id not in note.links and conn_id in self.memories:
                note.links.append(conn_id)
        
        # 更新标签
        if new_tags:
            note.tags = new_tags
    
    def _apply_update_neighbor_action(self, note: MemoryNote, response_json: Dict, indices: List[int]):
        """应用update_neighbor动作"""
        new_context_neighborhood = response_json.get("new_context_neighborhood", [])
        new_tags_neighborhood = response_json.get("new_tags_neighborhood", [])
        
        all_memories = list(self.memories.values())
        all_ids = list(self.memories.keys())
        
        for i in range(min(len(indices), len(new_tags_neighborhood))):
            if i >= len(indices):
                break
            
            idx = indices[i]
            if idx >= len(all_memories):
                continue
            
            # 获取标签和上下文
            tag = new_tags_neighborhood[i]
            context = (new_context_neighborhood[i] 
                      if i < len(new_context_neighborhood) 
                      else all_memories[idx].context)
            
            # 更新记忆
            memory_to_update = all_memories[idx]
            memory_to_update.tags = tag if isinstance(tag, list) else [tag]
            memory_to_update.context = context
            
            # 写回字典
            if idx < len(all_ids):
                self.memories[all_ids[idx]] = memory_to_update
    
    def _clean_json_response(self, response: str) -> str:
        """清理JSON响应"""
        response = re.sub(r'^```json\s*|\s*```$', '', response.strip(), flags=re.MULTILINE)
        
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            return response[start_idx:end_idx]
        
        return response
    
    def get_retriever_statistics(self) -> Dict[str, Any]:
        """获取检索器统计信息"""
        if hasattr(self.retriever, 'get_statistics'):
            return self.retriever.get_statistics()
        else:
            return {
                "total_documents": len(self.retriever.corpus),
                "retriever_type": self.retriever_type
            }
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        from collections import defaultdict
        
        tag_count = defaultdict(int)
        kw_count = defaultdict(int)
        
        for memory in self.memories.values():
            for tag in memory.tags:
                tag_count[tag] += 1
            for kw in memory.keywords:
                kw_count[kw] += 1
        
        total_links = sum(len(m.links) for m in self.memories.values())
        
        return {
            "total_memories": len(self.memories),
            "total_links": total_links,
            "avg_links_per_memory": total_links / max(1, len(self.memories)),
            "tags_distribution": dict(sorted(tag_count.items(), key=lambda x: x[1], reverse=True)[:20]),
            "keywords_distribution": dict(sorted(kw_count.items(), key=lambda x: x[1], reverse=True)[:20]),
            "evolution_count": self.evo_cnt,
            "retriever": self.get_retriever_statistics()
        }
    
    def save(self, save_dir: Path):
        """保存记忆系统状态"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存记忆数据
        memories_file = save_dir / f"{self.dialogue_name}_memories.json"
        memories_data = {
            "memories": {mid: m.to_dict() for mid, m in self.memories.items()},
            "evo_cnt": self.evo_cnt,
            "dialogue_name": self.dialogue_name,
            "retriever_type": self.retriever_type,
            "hybrid_alpha": self.hybrid_alpha if hasattr(self, 'hybrid_alpha') else 0.5
        }
        with open(memories_file, 'w', encoding='utf-8') as f:
            json.dump(memories_data, f, ensure_ascii=False, indent=2)
        
        # 保存检索器
        retriever_cache = save_dir / f"{self.dialogue_name}_retriever.pkl"
        retriever_embeddings = save_dir / f"{self.dialogue_name}_embeddings.npy"
        self.retriever.save(str(retriever_cache), str(retriever_embeddings))
        
        # 保存统计信息
        stats_file = save_dir / f"{self.dialogue_name}_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.get_statistics(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"记忆系统已保存到: {save_dir}")
    
    @classmethod
    def load(cls, load_dir: Path, dialogue_name: str) -> 'AgenticMemorySystem':
        """加载记忆系统状态"""
        load_dir = Path(load_dir)
        
        # 加载记忆数据
        memories_file = load_dir / f"{dialogue_name}_memories.json"
        if not memories_file.exists():
            raise FileNotFoundError(f"Memory file not found: {memories_file}")
        
        with open(memories_file, 'r', encoding='utf-8') as f:
            memories_data = json.load(f)
        
        # 获取检索器类型和参数
        retriever_type = memories_data.get("retriever_type", "hybrid")
        hybrid_alpha = memories_data.get("hybrid_alpha", 0.5)
        
        # 创建实例
        system = cls(
            dialogue_name=dialogue_name,
            retriever_type=retriever_type,
            hybrid_alpha=hybrid_alpha
        )
        system.evo_cnt = memories_data.get("evo_cnt", 0)
        print(system.evo_cnt)
        # 重建记忆
        for mid, mdata in memories_data.get("memories", {}).items():
            system.memories[mid] = MemoryNote.from_dict(mdata)
        
        # 加载检索器
        retriever_cache = load_dir / f"{dialogue_name}_retriever.pkl"
        retriever_embeddings = load_dir / f"{dialogue_name}_embeddings.npy"
        if retriever_cache.exists() and retriever_embeddings.exists():
            if retriever_type == "hybrid":
                system.retriever = HybridRetriever.load(str(retriever_cache), str(retriever_embeddings))
            else:
                simple_retriever = SimpleEmbeddingRetriever()
                system.retriever = simple_retriever.load(str(retriever_cache), str(retriever_embeddings))
            
            # 重建索引映射
            all_memories = list(system.memories.values())
            for idx, memory in enumerate(all_memories):
                if idx < len(system.retriever.corpus):
                    system.memory_id_to_index[memory.id] = idx
                    system.index_to_memory_id[idx] = memory.id
        
        return system