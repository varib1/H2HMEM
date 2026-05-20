#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_with_memory_notes.py
Test using saved memory notes - pure loading mode
Only loads saved memory notes, does not rebuild
"""

import os
import json
import argparse
import logging
import random
import time
import base64
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm
import traceback

# Import memory system
from memory_system import AgenticMemorySystem
from llm_controller import LLMController

# ================================================================
# Configure logger - do not set level to avoid affecting print output
# ================================================================
import sys

# Create a console handler that outputs to console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Get root logger and configure
root_logger = logging.getLogger()
root_logger.addHandler(console_handler)
# Do not set level, use default NOTSET (all levels output)

# Get current module logger
logger = logging.getLogger(__name__)
# Do not set level, inherit root logger settings


@dataclass
class TestSample:
    """Test sample - loaded from question file"""
    question_id: str
    session_id: str
    session_dir: str
    dialogue_name: str
    question_text: str
    question_image: str  # Image filename
    original_answer: str
    answer_source: str
    question_type: Dict[str, str]
    category: str
    difficulty: str
    supporting_evidence: List[Dict]
    metadata: Optional[Dict] = None
    
    # Add image related fields
    image_path: Optional[str] = None
    image_caption: Optional[str] = None
    image_base64: Optional[str] = None


class MemoryOnlyLoader:
    """
    Pure memory loader - only loads saved memories, does not rebuild
    
    Loads previously processed memories from memory_data/ directory
    Provides retrieval interface for LLM to use
    """
    
    def __init__(self,
                 dialogue_path: Path,
                 backbone_model: str = "",
                 memoryconstruct_model: str = "gpt-4o-mini",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 embedding_model: str = "",  
                 retrieve_k: int = 10,
                 temperature_c5: float = 0.1,
                 verbose: bool = False):
        
        self.dialogue_path = Path(dialogue_path)
        self.dialogue_name = self.dialogue_path.name

        self.backbone_model = backbone_model
        self.memoryconstruct_model = memoryconstruct_model

        self.api_key = api_key
        self.base_url = base_url
        self.retrieve_k = retrieve_k
        self.temperature_c5 = temperature_c5
        self.verbose = verbose
        self.embedding_model = embedding_model
        
        # ================================================================
        # 1. Load saved memory notes - directly load using class method
        # ================================================================
        memory_dir = self.dialogue_path / "memory_data"
        if not memory_dir.exists():
            raise FileNotFoundError(f"Memory notes not found: {memory_dir}")
        
        print(f"📂 Loading saved memory notes from {memory_dir}")
        logger.info(f"Loading saved memory notes from {memory_dir}")
        
        # ✅ Directly load using class method, get already loaded instance
        self.memory_system = AgenticMemorySystem.load(
            load_dir=memory_dir,
            dialogue_name=self.dialogue_name,
            api_key=self.api_key,
            memoryconstruct_model=self.memoryconstruct_model,
            base_url=self.base_url,
            embedding_model=self.embedding_model,
        )
        
        # ================================================================
        # 2. Initialize LLM controller for evaluation
        # ================================================================
        self.llm_controller = LLMController(
            model=self.backbone_model,
            api_key=self.api_key,
            base_url=self.base_url,
        )
        
        # ================================================================
        # 3. Initialize statistics
        # ================================================================
        self.results = []               # Only store successful question results
        self.all_categories = []
        self.category_counts = defaultdict(int)
        self.total_questions = 0        # Only successful questions count
        self.failed_count = 0           # Failed questions count (only count, no details saved)
        self.session_results = defaultdict(list)
        self.session_stats = defaultdict(lambda: {
            "total": 0,
            "by_category": defaultdict(int),
            "by_difficulty": defaultdict(int),
            "total_retrieval_time": 0.0,
            "total_llm_time": 0.0,
            "avg_retrieval_time": 0.0,
            "avg_llm_time": 0.0
        })
        
        # Display memory system statistics
        memory_stats = self.memory_system.get_statistics()
        print(f"✅ Memory system loaded successfully:")
        print(f"  Total memories: {memory_stats['total_memories']}")
        print(f"  Evolution count: {memory_stats.get('evolution_count', 0)}")
        logger.info(f"Memory system loaded: total_memories={memory_stats['total_memories']}")
        
        # Display retriever information
        retriever_info = memory_stats.get('retriever', {})
        print(f"  Retriever type: {retriever_info.get('retriever_type', 'unknown')}")
        

    def _get_prompt_config(self, category: str) -> Tuple[str, str, float]:
        """
        Get corresponding instruction, format requirement, and temperature based on question type
        References the prompt building logic for 9 question types
        """
        # 1. Define instructions for 9 question types
        instructions = {
            "Unimodal Precise Recall": "Accurately recall specific information from the retrieved conversation chunks and answer directly.",
            "Cross-modal Related Retrieval": "Retrieve relevant information across modalities (text and image).",
            "Knowledge Resolution": "Resolve and maintain knowledge consistency based on the retrieved conversation chunks.",
            "Temporal Reasoning": "Reason about temporal relationships and time information based on the retrieved conversation chunks.",
            "Multimodal Causal Reasoning": "Perform causal reasoning using both text and image information.",
            "Reference & Evolution Tracking": "Track references and their evolution process.",
            "Test-Time Learning": "Learn and adapt from the retrieved context at test time to answer the question.",
            "Conflict Detection": "Check whether the information in the question conflicts with the retrieved conversation chunks.",
            "Answer Refusal": "Determine if the question can be answered based on the retrieved conversation chunks."
        }

        # 2. Define format requirements
        format_reqs = {
            "Conflict Detection": "Response format: Reply strictly with either 'Yes' or 'No' only.",
            "Answer Refusal": "Response format: If the information is present, answer the question; if not present, reply with: 'Not mentioned.'",
            "default": "Response format: Provide clear and accurate answers based on the retrieved conversation chunks."
        }

        # 3. Get configuration, default to general type
        instruction = instructions.get(category, instructions["Unimodal Precise Recall"])
        format_req = format_reqs.get(category, format_reqs["default"])
        
        # 4. Temperature setting
        temperature = self.temperature_c5 
        
        return instruction, format_req, temperature



    def _load_image_caption(self, session_dir: str, image_filename: str) -> Optional[str]:
        """
        Load image caption
        
        Args:
            session_dir: Session directory name
            image_filename: Image filename, e.g., "1.png"
            
        Returns:
            Caption text, returns None if not exists
        """
        if not image_filename:
            return None
        
        # Build caption file path
        # Change image filename extension to .json
        caption_filename = Path(image_filename).stem + ".json"
        caption_path = self.dialogue_path / "scenes" / session_dir / "caption" / caption_filename
        
        if not caption_path.exists():
            logger.warning(f"Caption file not found: {caption_path}")
            if self.verbose:
                print(f"⚠️ Caption file not found: {caption_path}")
            return None
        
        try:
            with open(caption_path, 'r', encoding='utf-8') as f:
                caption_data = json.load(f)
            
            # Extract final_text as caption
            caption = caption_data.get("description", {}).get("final_text", "")
            if caption:
                if self.verbose:
                    print(f"  📷 Loaded image caption: {caption[:50]}...")
                return caption
            else:
                logger.warning(f"No final_text field in caption file: {caption_path}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load caption {caption_path}: {e}")
            return None
    
    def _encode_image_to_base64(self, image_path: Path) -> Optional[str]:
        """
        Encode image to base64 format
        
        Args:
            image_path: Image file path
            
        Returns:
            Base64 encoded image string, returns None if failed
        """
        if not image_path.exists():
            logger.error(f"Image file not found: {image_path}")
            return None
        
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                
                # Determine MIME type based on file extension
                suffix = image_path.suffix.lower()
                if suffix in ['.png']:
                    mime_type = "image/png"
                elif suffix in ['.jpg', '.jpeg']:
                    mime_type = "image/jpeg"
                elif suffix in ['.gif']:
                    mime_type = "image/gif"
                elif suffix in ['.webp']:
                    mime_type = "image/webp"
                else:
                    mime_type = "image/png"  # Default
                
                return f"data:{mime_type};base64,{encoded_string}"
                
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            return None
    
    def load_all_questions(self) -> List[TestSample]:
        """
        Load all question files from scenes directory
        Also load image captions and encode images
        """
        scenes_dir = self.dialogue_path / "scenes"
        if not scenes_dir.exists():
            raise FileNotFoundError(f"Scenes directory not found: {scenes_dir}")
        
        samples = []
        session_dirs = sorted([d for d in scenes_dir.iterdir() if d.is_dir()])
        print(f"📁 Found {len(session_dirs)} session directories")
        logger.info(f"Found {len(session_dirs)} session directories")
        
        for session_dir in session_dirs:
            question_file = session_dir / "questions.json"
            
            if not question_file.exists():
                print(f"⚠️ Skipping {session_dir.name}, question file not found")
                logger.warning(f"Skipping {session_dir.name}, question file not found")
                continue
            
            try:
                session_samples = self._load_session_questions(
                    session_dir.name, 
                    session_dir,
                    question_file
                )
                samples.extend(session_samples)
                print(f"  ✓ Loaded {len(session_samples)} questions from {session_dir.name}")
                logger.info(f"Loaded {len(session_samples)} questions from {session_dir.name}")
                
            except Exception as e:
                print(f"  ✗ Failed to load question file for {session_dir.name}: {e}")
                logger.error(f"Failed to load question file for {session_dir.name}: {e}")
        
        print(f"\n📊 Loaded {len(samples)} test samples in total")
        
        # Count image loading statistics
        images_with_caption = sum(1 for s in samples if s.image_caption is not None)
        images_with_base64 = sum(1 for s in samples if s.image_base64 is not None)
        print(f"  📷 Images with caption: {images_with_caption}/{len(samples)}")
        print(f"  🖼️ Images with base64: {images_with_base64}/{len(samples)}")
        
        logger.info(f"Loaded {len(samples)} test samples, of which {images_with_caption} have captions")
        return samples
    
    def _load_session_questions(self, 
                               session_dir_name: str,
                               session_dir: Path, 
                               question_file: Path) -> List[TestSample]:
        """
        Load question file for a single session, also load image captions and encode images
        """
        with open(question_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        session_id = session_dir_name
        questions = data.get("questions", [])
        
        samples = []
        for q in questions:
            # Extract category information
            question_type = q.get("question_type", {})
            subsub_type = question_type.get("subsub_type", "")
            if subsub_type:
                category = subsub_type
            else:
                sub_type = question_type.get("sub_type", "")
                category = sub_type or question_type.get("main_type", "general")
            
            question_data = q.get("question", {})
            question_text = question_data.get("text", "")
            question_image = question_data.get("image", "")
            
            # Load image related information
            image_path = None
            image_caption = None
            image_base64 = None
            
            if question_image:
                if session_dir_name != "session0":
                    # Build complete image path
                    image_path = session_dir / "image" / question_image
                    image_full_path = self.dialogue_path / "scenes" / image_path
                    
                    # Load caption
                    image_caption = self._load_image_caption(session_dir_name, question_image)
                    
                    # Encode image to base64
                    if image_full_path.exists():
                        image_base64 = self._encode_image_to_base64(image_full_path)
                        if self.verbose and image_base64:
                            print(f"  🖼️ Encoded image: {question_image} -> {len(image_base64)} characters")
                    else:
                        logger.warning(f"Image file not found: {image_full_path}")
                        if self.verbose:
                            print(f"  ⚠️ Image file not found: {image_full_path}")
                else:
                    folder, img_filename = question_image.split('/')
                    image_path = Path(folder) / "image" / img_filename
                    image_full_path = self.dialogue_path / "scenes" / image_path
                    # Load caption
                    image_caption = self._load_image_caption(folder, img_filename)

                    # Encode image to base64
                    if image_full_path.exists():
                        image_base64 = self._encode_image_to_base64(image_full_path)
                        if self.verbose and image_base64:
                            print(f"  🖼️ Encoded image: {question_image} -> {len(image_base64)} characters")
                    else:
                        logger.warning(f"Image file not found: {image_full_path}")
                        if self.verbose:
                            print(f"  ⚠️ Image file not found: {image_full_path}")
            
            sample = TestSample(
                question_id=q.get("question_id", f"A_MEM_{len(samples)}"),
                session_id=session_id,
                session_dir=session_dir_name,
                dialogue_name=self.dialogue_name,
                question_text=question_text,
                question_image=question_image,
                original_answer=q.get("original_answer", ""),
                answer_source=q.get("answer_source", "unknown"),
                question_type=question_type,
                category=category,
                difficulty=q.get("difficulty", "medium"),
                supporting_evidence=q.get("supporting_evidence", []),
                metadata={
                    "validation_notes": q.get("validation_notes", ""),
                    "validated": q.get("validated", False),
                    "generated_at": q.get("generated_at", "")
                },
                image_path=str(image_path) if image_path else None,
                image_caption=image_caption,
                image_base64=image_base64
            )
            samples.append(sample)
        
        return samples
    
    def retrieve_memories(self, query: str, k: Optional[int] = None) -> Tuple[List[Dict], float]:
        """
        Retrieve relevant memories from loaded memories
        
        Args:
            query: Query text
            k: Number of memories to retrieve
        
        Returns:
            (List of retrieved memories, retrieval time)
        """
        if k is None:
            k = self.retrieve_k
        
        # Record retrieval start time
        retrieval_start = time.time()
        
        # Use memory system's retrieval function, returns list of MemoryNote objects
        memory_notes = self.memory_system.find_related_memories(query, k=k)
        
        # Calculate retrieval time
        retrieval_time = time.time() - retrieval_start
        
        print(f"🔍 Retrieved memories: '{query[:50]}...' -> {len(memory_notes)} items (time: {retrieval_time:.3f}s)")
        logger.debug(f"Retrieved memories: '{query}', results={len(memory_notes)}, time={retrieval_time:.3f}s")
        
        # Convert MemoryNote objects to dictionary format
        memories = []
        for note in memory_notes:
            # Get attributes from MemoryNote object
            memory_dict = {
                "content": note.content,
                "time": getattr(note, 'timestamp', ''),
                "session_id": getattr(note, 'session_id', ''),
                "score": getattr(note, 'score', 0.0),  # If score attribute exists
                "keywords": note.keywords,
                "tags": note.tags,
                "context": note.context,
                "id": note.id
            }
            memories.append(memory_dict)
            
            if self.verbose:
                print(f"   [{len(memories)}] {note.content[:80]}...")
                logger.debug(f"Memory {len(memories)}: {note.content[:50]}...")
        
        return memories, retrieval_time
    
    def format_memories_for_prompt(self, memories: List[Dict]) -> str:
        """
        Format retrieved memories into text usable for prompts
        """
        if not memories:
            return "No relevant memories"
        
        formatted = []
        for i, mem in enumerate(memories, 1):
            content = mem.get("content", "")
            time_str = mem.get("time", "")
            session = mem.get("session_id", "")
            score = mem.get("score", 0)
            
            memory_text = f"[{i}] "
            if time_str:
                memory_text += f"[Time:{time_str}] "
            if session:
                memory_text += f"[Session:{session}] "
            memory_text += content
            
            formatted.append(memory_text)
        
        return "\n".join(formatted)
    
    def build_prompt(self, question: str, image_caption: Optional[str], memories: List[Dict], category: str, answer_source: str, original_answer: str) -> Tuple[str, float]:
        """
        Build English prompt (Text-only)
        """
        # Get type-specific configuration
        instruction, format_req, temperature = self._get_prompt_config(category)

        # Build complete question text
        full_question = question
        if image_caption:
            full_question = f"{question}\n\nImage Description (for retrieval reference): {image_caption}"

        # Format memory context
        context_text = self.format_memories_for_prompt(memories)

        # Build system instruction
        system_prompt = (
            "You are a memory testing system using Naive RAG to retrieve relevant conversation chunks.\n"
            f"{instruction}\n"
            "**IMPORTANT:**\n"
            "1. Provide your answer based on the retrieved conversation chunks only. You may not have the complete conversation history.\n"
            "2. Keep your answer within 100 words.\n"
            "3. Answer in English. This is a strict requirement. Do not answer in any other language\n"
            f"{format_req}\n"
        )

        # Assemble final prompt
        prompt = f"""
{system_prompt}

---
Retrieved relevant conversation chunks:
{context_text}

Question: {full_question}
Please answer based on the above memory content(text and image):
"""
        return prompt, temperature
    
    def build_multimodal_messages(self, question: str, image_base64: Optional[str], memories: List[Dict], category: str, answer_source: str, original_answer: str) -> Tuple[List[Dict], float]:
        """
        Build multimodal messages (Multimodal)
        """
        # Get type-specific configuration
        instruction, format_req, temperature = self._get_prompt_config(category)

        # Format memory context (plain text)
        context_text = self.format_memories_for_prompt(memories)

        # Build system instruction (consistent with text mode)
        system_prompt = (
            "You are a memory testing system using Naive RAG to retrieve relevant conversation chunks.\n"
            f"{instruction}\n"
            "**IMPORTANT:**\n"
            "1. Provide your answer based on the retrieved conversation chunks only.\n"
            "2. Keep your answer within 100 words.\n"
            "3. Answer in English. This is a strict requirement. Do not answer in any other language\n"
            f"{format_req}\n"
        )

        # Build user message content
        user_content = []
        
        # Add image (if exists)
        if image_base64:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_base64,
                    "detail": "auto"
                }
            })

        # Build text part
        text_content = f"""
{system_prompt}

---
Retrieved relevant conversation chunks:
{context_text}

Question: {question}
Please answer based on the above memory content(text and image).:
"""
        user_content.append({
            "type": "text",
            "text": text_content
        })

        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."}, # Base system role
            {"role": "user", "content": user_content}
        ]

        
        return messages, temperature
    
    def evaluate_question(self, sample: TestSample) -> Dict:
        """
        Evaluate a single question
        
        Args:
            sample: Test sample
        
        Returns:
            Result dictionary containing success field
        """
        overall_start = time.time()
        
        try:
            # ================================================================
            # Step 1: Build retrieval query (question text + image caption)
            # ================================================================
            query_text = sample.question_text
            if sample.image_caption:
                query_text = f"{sample.question_text}\nimage description: {sample.image_caption}"
            
            # Execute retrieval and get retrieval time
            try:
                memories, retrieval_time = self.retrieve_memories(query_text, k=self.retrieve_k)
            except Exception as e:
                # Retrieval failed, return failure result (do not save details)
                return {
                    "success": False,
                    "question_id": sample.question_id,
                    "session_id": sample.session_id,
                    "session_dir": sample.session_dir,
                    "question_text": sample.question_text,
                    "question_image": sample.question_image,
                    "original_answer": sample.original_answer,
                    "prediction": f"[RETRIEVAL ERROR: {str(e)[:100]}]",
                    "question_type": sample.question_type,
                    "category": sample.category,
                    "difficulty": sample.difficulty,
                    "answer_source": sample.answer_source,
                    "processing_time": time.time() - overall_start,
                    "retrieval_time": 0.0,
                    "llm_time": 0.0,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            
            # ================================================================
            # Step 2: Choose calling method based on whether image exists
            # ================================================================
            has_image = sample.image_base64 is not None
            
            # Record LLM call start time
            llm_start = time.time()
            
            try:
                if has_image:
                    # Use multimodal approach (only pass image, not caption)
                    messages, temperature = self.build_multimodal_messages(
                        question=sample.question_text,
                        image_base64=sample.image_base64,
                        memories=memories,
                        category=sample.category,
                        answer_source=sample.answer_source,
                        original_answer=sample.original_answer
                    )
                    
                    try:
                        response = self.llm_controller.get_completion(
                            messages,
                            temperature=temperature
                        )
                        
                        if isinstance(response, str):
                            try:
                                response_data = json.loads(response)
                                prediction = response_data.get("answer", response)
                            except:
                                prediction = response.strip()
                        else:
                            prediction = str(response).strip()
                            
                    except AttributeError:
                        # If multimodal not supported, fall back to text mode (using caption)
                        logger.warning("LLM does not support multimodal, falling back to text mode (using caption)")
                        prompt, temperature = self.build_prompt(
                            question=sample.question_text,
                            image_caption=sample.image_caption,
                            memories=memories,
                            category=sample.category,
                            answer_source=sample.answer_source,
                            original_answer=sample.original_answer
                        )
                        
                        response = self.llm_controller.llm.get_completion(
                            prompt,
                            temperature=temperature
                        )
                        
                        if isinstance(response, str):
                            try:
                                response_data = json.loads(response)
                                prediction = response_data.get("answer", response)
                            except:
                                prediction = response.strip()
                        else:
                            prediction = str(response).strip()
                else:
                    # Use text-only mode
                    prompt, temperature = self.build_prompt(
                        question=sample.question_text,
                        image_caption=None,
                        memories=memories,
                        category=sample.category,
                        answer_source=sample.answer_source,
                        original_answer=sample.original_answer
                    )
                    
                    response = self.llm_controller.llm.get_completion(
                        prompt,
                        temperature=temperature
                    )
                    
                    if isinstance(response, str):
                        try:
                            response_data = json.loads(response)
                            prediction = response_data.get("answer", response)
                        except:
                            prediction = response.strip()
                    else:
                        prediction = str(response).strip()
                        
            except Exception as e:
                # LLM call failed, return failure result (do not save details)
                return {
                    "success": False,
                    "question_id": sample.question_id,
                    "session_id": sample.session_id,
                    "session_dir": sample.session_dir,
                    "question_text": sample.question_text,
                    "question_image": sample.question_image,
                    "original_answer": sample.original_answer,
                    "prediction": f"[LLM ERROR: {str(e)[:100]}]",
                    "category": sample.category,
                    "question_type": sample.question_type,
                    "difficulty": sample.difficulty,
                    "answer_source": sample.answer_source,
                    "processing_time": time.time() - overall_start,
                    "retrieval_time": retrieval_time,
                    "llm_time": 0.0,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            
            # Calculate LLM call time
            llm_time = time.time() - llm_start
            
            total_processing_time = time.time() - overall_start
            
            # ================================================================
            # Step 3: Record successful result
            # ================================================================
            result = {
                "success": True,
                "question_id": sample.question_id,
                "session_id": sample.session_id,
                "session_dir": sample.session_dir,
                "question_text": sample.question_text,
                "question_image": sample.question_image,
                "question_image_path": sample.image_path,
                "has_image_caption": sample.image_caption is not None,
                "has_image_base64": sample.image_base64 is not None,
                "original_answer": sample.original_answer,
                "system_answer": prediction,
                "category": sample.category,
                "difficulty": sample.difficulty,
                "question_type": sample.question_type,
                "answer_source": sample.answer_source,
                "processing_time": total_processing_time,
                "retrieval_time": retrieval_time,
                "llm_time": llm_time,
                "retrieved_count": len(memories),
                "retrieved_memories": [
                    {
                        "content": m.get("content", ""),
                        "score": m.get("score", 0),
                        "time": m.get("time", ""),
                        "session": m.get("session_id", "")
                    }
                    for m in memories
                ],
                "timestamp": datetime.now().isoformat()
            }
            
            if self.verbose:
                if has_image:
                    result["multimodal_messages"] = "used"
                else:
                    result["prompt"] = prompt
                result["full_retrieved"] = memories
            
            # Output result using both print and logger
            print(f"\n📝 Question [{sample.session_dir}:{sample.question_id}]: {sample.question_text[:50]}...")
            if sample.question_image:
                print(f"  📷 Image: {sample.question_image}")
                if sample.image_caption:
                    print(f"  💬 Caption (retrieval only): {sample.image_caption[:50]}...")
            print(f"  🤖 Prediction: {prediction[:50]}...")
            print(f"  ✅ Correct: {sample.original_answer[:50]}...")
            print(f"  🏷️ Category: {sample.category}")
            print(f"  📊 Retrieved: {len(memories)} memories (time: {retrieval_time:.3f}s)")
            print(f"  🤖 LLM: {llm_time:.3f}s")
            print(f"  ⏱️ Total time: {total_processing_time:.2f}s")
            
            logger.info(f"Question {sample.question_id} evaluation complete: retrieval={retrieval_time:.3f}s, LLM={llm_time:.3f}s")
            
            return result
            
        except Exception as e:
            # Unexpected exception, return failure result
            return {
                "success": False,
                "question_id": sample.question_id,
                "session_id": sample.session_id,
                "session_dir": sample.session_dir,
                "question_text": sample.question_text,
                "question_image": sample.question_image,
                "original_answer": sample.original_answer,
                "prediction": f"[UNEXPECTED ERROR: {str(e)[:100]}]",
                "category": sample.category,
                "difficulty": sample.difficulty,
                "question_type": sample.question_type,
                "answer_source": sample.answer_source,
                "processing_time": time.time() - overall_start,
                "retrieval_time": 0.0,
                "llm_time": 0.0,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def evaluate_all(self, 
                    samples: List[TestSample],
                    batch_size: int = 10) -> Dict:
        
        print(f"\n{'='*60}")
        print(f"Starting evaluation: {self.dialogue_name}")
        print(f"  Total samples: {len(samples)}")
        print(f"  Retrieval k: {self.retrieve_k}")
        print(f"  Model: {self.backbone_model}")
        print(f"{'='*60}")
        logger.info(f"Starting evaluation: {self.dialogue_name}, samples={len(samples)}")
        
        # Group by session
        session_samples = defaultdict(list)
        for sample in samples:
            session_samples[sample.session_dir].append(sample)
        
        print(f"Grouped by {len(session_samples)} sessions for evaluation")
        logger.info(f"Grouped by {len(session_samples)} sessions for evaluation")
        
        # Reset counters
        self.total_questions = 0
        self.failed_count = 0
        self.results = []
        self.session_results.clear()
        self.session_stats.clear()
        
        # Evaluate session by session
        for session_dir, session_samps in session_samples.items():
            print(f"\n📁 Evaluating session: {session_dir} ({len(session_samps)} questions)")
            logger.info(f"Evaluating session: {session_dir}, questions={len(session_samps)}")
            
            session_path = self.dialogue_path / "scenes" / session_dir
            
            # Process in batches
            for i in range(0, len(session_samps), batch_size):
                batch = session_samps[i:i+batch_size]
                
                for sample in batch:
                    result = self.evaluate_question(sample)
                    
                    # Only count successful questions
                    if result.get("success", False):
                        self.results.append(result)
                        self.session_results[sample.session_dir].append(result)
                        self.all_categories.append(sample.category)
                        self.category_counts[sample.category] += 1
                        self.total_questions += 1
                        
                        # Update statistics (including timing statistics)
                        stats = self.session_stats[sample.session_dir]
                        stats["total"] += 1
                        stats["by_category"][sample.category] += 1
                        stats["by_difficulty"][sample.difficulty] += 1
                        stats["total_retrieval_time"] += result.get("retrieval_time", 0)
                        stats["total_llm_time"] += result.get("llm_time", 0)
                        stats["avg_retrieval_time"] = stats["total_retrieval_time"] / stats["total"]
                        stats["avg_llm_time"] = stats["total_llm_time"] / stats["total"]
                    else:
                        # Failed questions only increase failure count
                        self.failed_count += 1
                        print(f"⚠️ Question {sample.question_id} failed, skipped from statistics")
                        logger.warning(f"Question {sample.question_id} failed, skipped from statistics")
            
            # Save current session results (only successful results)
            self._save_session_results(session_dir, session_path)
        
        # Build final results
        final_results = self._build_final_results()
        
        # Output summary
        self._print_summary()
        
        return final_results
    
    def _build_final_results(self) -> Dict:
        # Calculate overall timing statistics (based on successful questions)
        total_retrieval_time = sum(r.get("retrieval_time", 0) for r in self.results)
        total_llm_time = sum(r.get("llm_time", 0) for r in self.results)
        avg_retrieval_time = total_retrieval_time / self.total_questions if self.total_questions > 0 else 0
        avg_llm_time = total_llm_time / self.total_questions if self.total_questions > 0 else 0
        
        total_attempted = self.total_questions + self.failed_count
        
        return {
            "metadata": {
                "dialogue_name": self.dialogue_name,
                "vlm_model": self.backbone_model,
                "retrieve_k": self.retrieve_k,
                "temperature_c5": self.temperature_c5,
                "total_questions_attempted": total_attempted,
                "memory_type": "A_MEM",
                "successful_questions": self.total_questions,
                "failed_questions": self.failed_count,
                "success_rate": (self.total_questions / total_attempted * 100) if total_attempted > 0 else 0,
                "evaluation_time": datetime.now().isoformat(),
                "memory_stats": self.memory_system.get_statistics(),
                "timing_stats": {
                    "total_retrieval_time": total_retrieval_time,
                    "total_llm_time": total_llm_time,
                    "avg_retrieval_time_per_question": avg_retrieval_time,
                    "avg_llm_time_per_question": avg_llm_time,
                    "total_processing_time": total_retrieval_time + total_llm_time
                }
            },
            "category_distribution": {
                cat: count for cat, count in self.category_counts.items()
            },
            "session_statistics": {
                session_dir: {
                    "total": stats["total"],
                    "by_category": dict(stats["by_category"]),
                    "by_difficulty": dict(stats["by_difficulty"]),
                    "timing_stats": {
                        "total_retrieval_time": stats["total_retrieval_time"],
                        "total_llm_time": stats["total_llm_time"],
                        "avg_retrieval_time": stats["avg_retrieval_time"],
                        "avg_llm_time": stats["avg_llm_time"]
                    }
                }
                for session_dir, stats in self.session_stats.items()
            },
            "results": self.results   # Only successful results
        }
    
    def _save_session_results(self, session_dir: str, session_path: Path):
        """Save evaluation results for a single session (only save successful questions)"""
        results_dir = session_path / "evaluation_results"
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_results = self.session_results[session_dir]   # Already filtered, only successful results
        session_stats = self.session_stats[session_dir]
        
        # Complete results
        results_file = results_dir / "results_A_MEM.json"
        full_results = {
            "metadata": {
                "session_id": session_dir,
                "dialogue_name": self.dialogue_name,
                "vlm_model": self.backbone_model,
                "memory_type": "A_MEM",
                "retrieve_k": self.retrieve_k,
                "successful_questions": session_stats["total"],
                "evaluation_time": timestamp,
                "timing_stats": {
                    "total_retrieval_time": session_stats["total_retrieval_time"],
                    "total_llm_time": session_stats["total_llm_time"],
                    "avg_retrieval_time": session_stats["avg_retrieval_time"],
                    "avg_llm_time": session_stats["avg_llm_time"]
                }
            },
            "statistics": {
                "total": session_stats["total"],
                "by_category": dict(session_stats["by_category"]),
                "by_difficulty": dict(session_stats["by_difficulty"])
            },
            "results": session_results
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, ensure_ascii=False, indent=2)
        
        
        print(f"\n💾 Session {session_dir} results saved:")
        print(f"  - Complete results: {results_file}")
        print(f"  ⏱️ Timing stats: total_retrieval={session_stats['total_retrieval_time']:.2f}s, "
              f"total_LLM={session_stats['total_llm_time']:.2f}s, "
              f"avg_retrieval={session_stats['avg_retrieval_time']:.3f}s, "
              f"avg_LLM={session_stats['avg_llm_time']:.3f}s")
        logger.info(f"Session {session_dir} results saved")
    
    def _print_summary(self):
        """Print result summary (based on successful questions)"""
        total_attempted = self.total_questions + self.failed_count
        print(f"\n{'='*60}")
        print(f"✅ Evaluation Complete - {self.dialogue_name}")
        print(f"{'='*60}")
        
        print(f"\n📊 Overall Statistics:")
        print(f"  Total questions attempted: {total_attempted}")
        print(f"  Successful questions: {self.total_questions}")
        print(f"  Failed questions: {self.failed_count}")
        print(f"  Success rate: {(self.total_questions / total_attempted * 100):.1f}%" if total_attempted > 0 else "N/A")
        print(f"  Category distribution (based on successful questions):")
        for cat, count in sorted(self.category_counts.items()):
            print(f"    {cat}: {count} ({count/self.total_questions*100:.1f}%)")
        
        print(f"\n📁 Session Statistics (successful questions):")
        for session_dir, stats in self.session_stats.items():
            print(f"  {session_dir}: {stats['total']} successful")
        
        print(f"\n🧠 Memory System:")
        memory_stats = self.memory_system.get_statistics()
        print(f"  Total memories: {memory_stats['total_memories']}")
        print(f"  Evolution count: {memory_stats.get('evolution_count', 0)}")
        
        # Count image usage statistics (based on successful questions)
        images_count = sum(1 for r in self.results if r.get("question_image"))
        images_with_caption = sum(1 for r in self.results if r.get("has_image_caption", False))
        print(f"\n📷 Image Statistics (successful questions):")
        print(f"  Questions with images: {images_count}/{self.total_questions}")
        print(f"  Images with caption: {images_with_caption}/{images_count}")
        
        # Count timing information (based on successful questions)
        total_retrieval_time = sum(r.get("retrieval_time", 0) for r in self.results)
        total_llm_time = sum(r.get("llm_time", 0) for r in self.results)
        total_time = total_retrieval_time + total_llm_time
        avg_retrieval = total_retrieval_time / self.total_questions if self.total_questions > 0 else 0
        avg_llm = total_llm_time / self.total_questions if self.total_questions > 0 else 0
        
        print(f"\n⏱️ Timing Statistics (successful questions):")
        print(f"  Total retrieval time: {total_retrieval_time:.2f}s")
        print(f"  Total LLM time: {total_llm_time:.2f}s")
        print(f"  Total processing time: {total_time:.2f}s")
        print(f"  Average retrieval time/question: {avg_retrieval:.3f}s")
        print(f"  Average LLM time/question: {avg_llm:.3f}s")
        
        print(f"{'='*60}")
        
        logger.info(f"Evaluation complete: successful={self.total_questions}, failed={self.failed_count}, "
                   f"total_retrieval_time={total_retrieval_time:.2f}s, total_llm_time={total_llm_time:.2f}s")



def main():
    parser = argparse.ArgumentParser(description="Test using saved memory notes - pure loading mode")
    parser.add_argument("--dialogue_path", type=str, required=True,
                       help="Dialogue directory path (contains memory_data_en and scenes directories)")
    parser.add_argument("--backbone_model", type=str, required=True,
                       help="Backbone model name")
    parser.add_argument("--memoryconstruct_model", type=str, required=True,
                       help="Model name used for memory construction")
    parser.add_argument("--api_key", type=str, required=True,
                       help="API key (can also be set via environment variable OPENAI_API_KEY)")
    parser.add_argument("--base_url", type=str, required=True,
                       help="API base URL")
    parser.add_argument("--retrieve_k", type=int, required=True,
                       help="Number of memories to retrieve")
    parser.add_argument("--temperature_c5", type=float, default=0.1,
                       help="Temperature setting for language model responses")
    parser.add_argument("--batch_size", type=int, default=10,
                       help="Batch size")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose logging")
    args = parser.parse_args()

    # Set verbose logging
    if args.verbose:
        file_handler = logging.FileHandler('evaluation_debug.log', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

    print(f"\n{'='*70}")
    print(f"Memory Notes Test - Pure Loading Mode")
    print(f"{'='*70}")
    print(f"Dialogue directory: {args.dialogue_path}")
    print(f"Backbone model: {args.backbone_model}")
    print(f"Retrieval k: {args.retrieve_k}")
    print(f"{'='*70}\n")

    try:
        api_key = args.api_key
        if not api_key:
            raise ValueError("Please provide --api_key argument or set environment variable OPENAI_API_KEY")

        loader = MemoryOnlyLoader(
            dialogue_path=Path(args.dialogue_path),
            backbone_model=args.backbone_model,
            memoryconstruct_model=args.memoryconstruct_model,
            api_key=api_key,
            base_url=args.base_url,
            retrieve_k=args.retrieve_k,
            temperature_c5=args.temperature_c5,
            verbose=args.verbose
        )

        samples = loader.load_all_questions()
        if not samples:
            print(f"\n❌ No test questions found")
            return 1

        print(f"\nFound {len(samples)} test questions")
        session_counts = defaultdict(int)
        for s in samples:
            session_counts[s.session_dir] += 1
        print("Session distribution:")
        for session_dir, count in session_counts.items():
            print(f"  {session_dir}: {count} questions")

        loader.evaluate_all(
            samples=samples,
            batch_size=args.batch_size
        )

        print(f"\n✅ Evaluation complete!")
        print(f"\nResults saved to each session's evaluation_results_qwen_3B directory")
        return 0

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())