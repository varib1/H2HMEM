#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
core.py
Memory System Core Data Models
"""

import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field


@dataclass
class MemoryNote:
    """Memory node - core data unit"""
    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    keywords: List[str] = field(default_factory=list)
    links: List[str] = field(default_factory=list)  # Associated memory IDs
    importance_score: float = 1.0
    retrieval_count: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M"))
    last_accessed: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M"))
    context: str = "General"
    evolution_history: List[Dict] = field(default_factory=list)
    category: str = "Uncategorized"
    tags: List[str] = field(default_factory=list)
    
    # Dialogue-related metadata
    dialogue_name: str = ""
    session_id: str = ""
    dialogue_index: int = 0
    role: str = ""
    has_image: bool = False
    image_filename: str = ""
    
    # LLM controller (for content analysis)
    llm_controller: Optional[Any] = None
    
    def __post_init__(self):
        """Post-initialization processing, analyze content if LLM controller is provided"""
        if self.llm_controller and (not self.keywords or not self.context or not self.tags):
            analysis = self.analyze_content(self.content, self.llm_controller)
            self.keywords = analysis.get("keywords", self.keywords)
            self.context = analysis.get("context", self.context)
            self.tags = analysis.get("tags", self.tags)
    
    @staticmethod
    def analyze_content(content: str, llm_controller) -> Dict[str, Any]:
        """Analyze content using LLM"""
        try:
            # Call llm_controller's analyze_content method here
            return llm_controller.analyze_content(content)
        except Exception as e:
            print(f"Failed to analyze content: {e}")
            return {
                "keywords": ["conversation"],
                "context": "General",
                "tags": ["conversation"]
            }
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "content": self.content,
            "keywords": self.keywords,
            "links": self.links,
            "importance_score": self.importance_score,
            "retrieval_count": self.retrieval_count,
            "timestamp": self.timestamp,
            "last_accessed": self.last_accessed,
            "context": self.context,
            "evolution_history": self.evolution_history,
            "category": self.category,
            "tags": self.tags,
            "dialogue_name": self.dialogue_name,
            "session_id": self.session_id,
            "dialogue_index": self.dialogue_index,
            "role": self.role,
            "has_image": self.has_image,
            "image_filename": self.image_filename
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryNote':
        """Create from dictionary"""
        note = cls(content=data["content"])
        note.id = data["id"]
        note.keywords = data.get("keywords", [])
        note.links = data.get("links", [])
        note.importance_score = data.get("importance_score", 1.0)
        note.retrieval_count = data.get("retrieval_count", 0)
        note.timestamp = data.get("timestamp", note.timestamp)
        note.last_accessed = data.get("last_accessed", note.last_accessed)
        note.context = data.get("context", "General")
        note.evolution_history = data.get("evolution_history", [])
        note.category = data.get("category", "Uncategorized")
        note.tags = data.get("tags", [])
        note.dialogue_name = data.get("dialogue_name", "")
        note.session_id = data.get("session_id", "")
        note.dialogue_index = data.get("dialogue_index", 0)
        note.role = data.get("role", "")
        note.has_image = data.get("has_image", False)
        note.image_filename = data.get("image_filename", "")
        return note
    
    def get_searchable_text(self) -> str:
        """Get searchable text (for retrieval)"""
        return (f"content: {self.content} "
                f"context: {self.context} "
                f"keywords: {', '.join(self.keywords)} "
                f"tags: {', '.join(self.tags)}")
    
    def to_evolution_text(self, index: int) -> str:
        """Convert to text format for evolution"""
        return (f"memory index:{index}\t"
                f"talk start time:{self.timestamp}\t"
                f"memory content: {self.content}\t"
                f"memory context: {self.context}\t"
                f"memory keywords: {self.keywords}\t"
                f"memory tags: {self.tags}")


class ConversationMemoryGroup:
    """Conversation memory group - corresponds to all memories of a conversation folder"""
    
    def __init__(self, dialogue_name: str):
        self.dialogue_name = dialogue_name
        self.memories: Dict[str, MemoryNote] = {}
        self.session_order: List[str] = []
        self.session_info: Dict[str, Any] = {}
        self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
    
    @property
    def total_memories(self) -> int:
        return len(self.memories)
    
    def add_memory(self, memory: MemoryNote):
        """Add memory"""
        self.memories[memory.id] = memory
        self.updated_at = datetime.now().isoformat()
    
    def get_memory_list(self) -> List[MemoryNote]:
        """Get memory list"""
        return list(self.memories.values())
    
    def get_memory_by_index(self, index: int) -> Optional[MemoryNote]:
        """Get memory by index"""
        memories = self.get_memory_list()
        if 0 <= index < len(memories):
            return memories[index]
        return None
    
    def get_memories_by_session(self, session_id: str) -> List[MemoryNote]:
        """Get all memories of a specified session, sorted by dialogue index"""
        session_memories = [
            m for m in self.memories.values() 
            if m.session_id == session_id
        ]
        return sorted(session_memories, key=lambda x: x.dialogue_index)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary (for serialization)"""
        return {
            "dialogue_name": self.dialogue_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "total_memories": self.total_memories,
            "session_order": self.session_order,
            "session_info": self.session_info,
            "memories": {mid: m.to_dict() for mid, m in self.memories.items()}
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ConversationMemoryGroup':
        """Create from dictionary"""
        group = cls(dialogue_name=data["dialogue_name"])
        group.created_at = data.get("created_at", group.created_at)
        group.updated_at = data.get("updated_at", group.updated_at)
        group.session_order = data.get("session_order", [])
        group.session_info = data.get("session_info", {})
        
        for mid, mdata in data.get("memories", {}).items():
            group.memories[mid] = MemoryNote.from_dict(mdata)
        
        return group