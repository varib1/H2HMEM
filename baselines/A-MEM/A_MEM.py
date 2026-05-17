#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_with_memory_notes.py
使用已保存的记忆笔记进行测试 - 纯加载模式
只加载已保存的记忆笔记，不重新构建
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

# 导入记忆系统
from memory_system import AgenticMemorySystem
from llm_controller import LLMController

# ================================================================
# 配置logger - 不设置级别，不影响print输出
# ================================================================
import sys

# 创建一个同时输出到控制台的处理器
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# 获取root logger并配置
root_logger = logging.getLogger()
root_logger.addHandler(console_handler)
# 不设置级别，使用默认的NOTSET（所有级别都输出）

# 获取当前模块的logger
logger = logging.getLogger(__name__)
# 不设置级别，继承root logger的设置


@dataclass
class TestSample:
    """测试样本 - 从问题文件加载"""
    question_id: str
    session_id: str
    session_dir: str
    dialogue_name: str
    question_text: str
    question_image: str  # 图片文件名
    original_answer: str
    answer_source: str
    question_type: Dict[str, str]
    category: str
    difficulty: str
    supporting_evidence: List[Dict]
    metadata: Optional[Dict] = None
    
    # 添加图片相关字段
    image_path: Optional[str] = None
    image_caption: Optional[str] = None
    image_base64: Optional[str] = None


class MemoryOnlyLoader:
    """
    纯记忆加载器 - 只加载已保存的记忆，不重新构建
    
    从 memory_data/ 目录加载之前处理好的记忆
    提供检索接口供LLM使用
    """
    
    def __init__(self,
                 dialogue_path: Path,
                 model: str = "gpt-4o-mini",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,  
                 retrieve_k: int = 10,
                 temperature_c5: float = 0.5,
                 verbose: bool = False):
        
        self.dialogue_path = Path(dialogue_path)
        self.dialogue_name = self.dialogue_path.name
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.retrieve_k = retrieve_k
        self.temperature_c5 = temperature_c5
        self.verbose = verbose
        
        # ================================================================
        # 1. 加载已保存的记忆笔记 - 使用类方法直接加载
        # ================================================================
        memory_dir = self.dialogue_path / "memory_data"
        if not memory_dir.exists():
            raise FileNotFoundError(f"记忆笔记不存在: {memory_dir}")
        
        print(f"📂 加载已保存的记忆笔记 from {memory_dir}")
        logger.info(f"加载已保存的记忆笔记 from {memory_dir}")
        
        # ✅ 使用类方法直接加载，得到已加载好的实例
        self.memory_system = AgenticMemorySystem.load(
            load_dir=memory_dir,
            dialogue_name=self.dialogue_name
        )
        
        # ================================================================
        # 2. 初始化LLM控制器用于评估
        # ================================================================
        self.llm_controller = LLMController(
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
        
        # ================================================================
        # 3. 初始化统计信息
        # ================================================================
        self.results = []               # 只存储成功的问题结果
        self.all_categories = []
        self.category_counts = defaultdict(int)
        self.total_questions = 0        # 仅成功的问题数
        self.failed_count = 0           # 失败的问题数（仅计数，不保存详情）
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
        
        # 显示记忆系统统计
        memory_stats = self.memory_system.get_statistics()
        print(f"✅ 记忆系统加载完成:")
        print(f"  总记忆数: {memory_stats['total_memories']}")
        print(f"  进化次数: {memory_stats.get('evolution_count', 0)}")
        logger.info(f"记忆系统加载完成: 总记忆数={memory_stats['total_memories']}")
        
        # 显示检索器信息
        retriever_info = memory_stats.get('retriever', {})
        print(f"  检索器类型: {retriever_info.get('retriever_type', 'unknown')}")
        

    def _get_prompt_config(self, category: str) -> Tuple[str, str, float]:
        """
        根据问题类型获取对应的指令、格式要求和温度设置
        参考了9种问题类型的提示词构建逻辑
        """
        # 1. 定义9种问题类型的指令
        instructions = {
            "Unimodal Precise Recall": "Accurately recall specific information from the retrieved conversation chunks and answer directly.",
            "Cross-modal Related Retrieval": "Retrieve relevant information across modalities (text and image).",
            "Knowledge Resolution": "Resolve and maintain knowledge consistency based on the retrieved conversation chunks.",
            "Temporal Reasoning": "Reason about temporal relationships and time information based on the retrieved conversation chunks.",
            "Multimodal Causal Reasoning": "Perform causal reasoning using both text and image information.",
            "Reference & Evolution Tracking": "Track references and their evolution process.",
            "Test-Time Learning (TTL)": "Learn and adapt from the retrieved context at test time to answer the question.",
            "Conflict Detection (CD)": "Check whether the information in the question conflicts with the retrieved conversation chunks.",
            "Answer Refusal (AR)": "Determine if the question can be answered based on the retrieved conversation chunks."
        }

        # 2. 定义格式要求
        format_reqs = {
            "Conflict Detection (CD)": "Response format: Reply strictly with either 'Yes' or 'No' only.",
            "Answer Refusal (AR)": "Response format: If the information is present, answer the question; if not present, reply with: 'Not mentioned.'",
            "default": "Response format: Provide clear and accurate answers based on the retrieved conversation chunks."
        }

        # 3. 获取配置，默认为通用类型
        instruction = instructions.get(category, instructions["Unimodal Precise Recall"])
        format_req = format_reqs.get(category, format_reqs["default"])
        
        # 4. 温度设置 (对抗性问题使用较低温度)
        temperature = self.temperature_c5 
        
        return instruction, format_req, temperature



    def _load_image_caption(self, session_dir: str, image_filename: str) -> Optional[str]:
        """
        加载图片的caption
        
        Args:
            session_dir: session目录名
            image_filename: 图片文件名，如 "1.png"
            
        Returns:
            caption文本，如果不存在则返回None
        """
        if not image_filename:
            return None
        
        # 构建caption文件路径
        # 将图片文件名后缀改为.json
        caption_filename = Path(image_filename).stem + ".json"
        caption_path =  self.dialogue_path / "scenes" / session_dir / "caption" / caption_filename
        
        if not caption_path.exists():
            logger.warning(f"Caption文件不存在: {caption_path}")
            if self.verbose:
                print(f"⚠️ Caption文件不存在: {caption_path}")
            return None
        
        try:
            with open(caption_path, 'r', encoding='utf-8') as f:
                caption_data = json.load(f)
            
            # 提取final_text作为caption
            caption = caption_data.get("description", {}).get("final_text", "")
            if caption:
                if self.verbose:
                    print(f"  📷 加载图片caption: {caption[:50]}...")
                return caption
            else:
                logger.warning(f"Caption文件中没有final_text字段: {caption_path}")
                return None
                
        except Exception as e:
            logger.error(f"加载caption失败 {caption_path}: {e}")
            return None
    
    def _encode_image_to_base64(self, image_path: Path) -> Optional[str]:
        """
        将图片编码为base64格式
        
        Args:
            image_path: 图片文件路径
            
        Returns:
            base64编码的图片字符串，如果失败则返回None
        """
        if not image_path.exists():
            logger.error(f"图片文件不存在: {image_path}")
            return None
        
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                
                # 根据文件扩展名确定MIME类型
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
                    mime_type = "image/png"  # 默认
                
                return f"data:{mime_type};base64,{encoded_string}"
                
        except Exception as e:
            logger.error(f"编码图片失败 {image_path}: {e}")
            return None
    
    def load_all_questions(self) -> List[TestSample]:
        """
        从scenes目录加载所有session的问题文件
        同时加载图片caption和编码图片
        """
        scenes_dir = self.dialogue_path / "scenes"
        if not scenes_dir.exists():
            raise FileNotFoundError(f"scenes目录不存在: {scenes_dir}")
        
        samples = []
        session_dirs = sorted([d for d in scenes_dir.iterdir() if d.is_dir()])
        print(f"📁 找到 {len(session_dirs)} 个session目录")
        logger.info(f"找到 {len(session_dirs)} 个session目录")
        
        for session_dir in session_dirs:
            question_file = session_dir / "questions.json"
            
            if not question_file.exists():
                print(f"⚠️ 跳过 {session_dir.name}，未找到问题文件")
                logger.warning(f"跳过 {session_dir.name}，未找到问题文件")
                continue
            
            try:
                session_samples = self._load_session_questions(
                    session_dir.name, 
                    session_dir,
                    question_file
                )
                samples.extend(session_samples)
                print(f"  ✓ 从 {session_dir.name} 加载了 {len(session_samples)} 个问题")
                logger.info(f"从 {session_dir.name} 加载了 {len(session_samples)} 个问题")
                
            except Exception as e:
                print(f"  ✗ 加载 {session_dir.name} 的问题文件失败: {e}")
                logger.error(f"加载 {session_dir.name} 的问题文件失败: {e}")
        
        print(f"\n📊 总共加载了 {len(samples)} 个测试样本")
        
        # 统计图片加载情况
        images_with_caption = sum(1 for s in samples if s.image_caption is not None)
        images_with_base64 = sum(1 for s in samples if s.image_base64 is not None)
        print(f"  📷 有caption的图片: {images_with_caption}/{len(samples)}")
        print(f"  🖼️ 有base64的图片: {images_with_base64}/{len(samples)}")
        
        logger.info(f"总共加载了 {len(samples)} 个测试样本，其中 {images_with_caption} 个有caption")
        return samples
    
    def _load_session_questions(self, 
                               session_dir_name: str,
                               session_dir: Path, 
                               question_file: Path) -> List[TestSample]:
        """
        加载单个session的问题文件，同时加载图片caption和编码图片
        """
        with open(question_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        session_id = session_dir_name
        questions = data.get("questions", [])
        
        samples = []
        for q in questions:
            # 提取类别信息
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
            
            # 加载图片相关信息
            image_path = None
            image_caption = None
            image_base64 = None
            
            if question_image:
                if session_dir_name != "session0":
                    # 构建完整的图片路径
                    image_path = session_dir / "image" / question_image
                    image_full_path = self.dialogue_path/ "scenes" / image_path
                    
                    # 加载caption
                    image_caption = self._load_image_caption(session_dir_name, question_image)
                    
                    # 编码图片为base64
                    if image_full_path.exists():
                        image_base64 = self._encode_image_to_base64(image_full_path)
                        if self.verbose and image_base64:
                            print(f"  🖼️ 编码图片: {question_image} -> {len(image_base64)} 字符")
                    else:
                        logger.warning(f"图片文件不存在: {image_full_path}")
                        if self.verbose:
                            print(f"  ⚠️ 图片文件不存在: {image_full_path}")
                else:
                    folder, img_filename = question_image.split('/')
                    image_path = Path(folder) / "image" / img_filename
                    image_full_path = self.dialogue_path / "scenes" / image_path
                    # 加载caption
                    image_caption = self._load_image_caption(folder, img_filename)

                    # 编码图片为base64
                    if image_full_path.exists():
                        image_base64 = self._encode_image_to_base64(image_full_path)
                        if self.verbose and image_base64:
                            print(f"  🖼️ 编码图片: {question_image} -> {len(image_base64)} 字符")
                    else:
                        logger.warning(f"图片文件不存在: {image_full_path}")
                        if self.verbose:
                            print(f"  ⚠️ 图片文件不存在: {image_full_path}")
            
            sample = TestSample(
                question_id=q.get("question_id", f"unknown_{len(samples)}"),
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
        从已加载的记忆中检索相关记忆
        
        Args:
            query: 查询文本
            k: 检索数量
        
        Returns:
            (检索到的记忆列表, 检索时间)
        """
        if k is None:
            k = self.retrieve_k
        
        # 记录检索开始时间
        retrieval_start = time.time()
        
        # 使用记忆系统的检索功能，返回的是 MemoryNote 对象列表
        memory_notes = self.memory_system.find_related_memories(query, k=k)
        
        # 计算检索时间
        retrieval_time = time.time() - retrieval_start
        
        print(f"🔍 检索记忆: '{query[:50]}...' -> {len(memory_notes)} 条 (耗时: {retrieval_time:.3f}秒)")
        logger.debug(f"检索记忆: '{query}', 结果数={len(memory_notes)}, 耗时={retrieval_time:.3f}s")
        
        # 将 MemoryNote 对象转换为字典格式
        memories = []
        for note in memory_notes:
            # 从 MemoryNote 对象获取属性
            memory_dict = {
                "content": note.content,
                "time": getattr(note, 'timestamp', ''),
                "session_id": getattr(note, 'session_id', ''),
                "score": getattr(note, 'score', 0.0),  # 如果有score属性的话
                "keywords": note.keywords,
                "tags": note.tags,
                "context": note.context,
                "id": note.id
            }
            memories.append(memory_dict)
            
            if self.verbose:
                print(f"   [{len(memories)}] {note.content[:80]}...")
                logger.debug(f"记忆 {len(memories)}: {note.content[:50]}...")
        
        return memories, retrieval_time
    
    def format_memories_for_prompt(self, memories: List[Dict]) -> str:
        """
        将检索到的记忆格式化为提示词可用的文本
        """
        if not memories:
            return "无相关记忆"
        
        formatted = []
        for i, mem in enumerate(memories, 1):
            content = mem.get("content", "")
            time_str = mem.get("time", "")
            session = mem.get("session_id", "")
            score = mem.get("score", 0)
            
            memory_text = f"[{i}] "
            if time_str:
                memory_text += f"[时间:{time_str}] "
            if session:
                memory_text += f"[会话:{session}] "
            memory_text += content
            
            formatted.append(memory_text)
        
        return "\n".join(formatted)
    
    def build_prompt(self, question: str, image_caption: Optional[str], memories: List[Dict], category: str, answer_source: str, original_answer: str) -> Tuple[str, float]:
        """
        构建英文提示词 (Text-only)
        """
        # 获取特定类型的配置
        instruction, format_req, temperature = self._get_prompt_config(category)

        # 构建完整问题文本
        full_question = question
        if image_caption:
            full_question = f"{question}\n\nImage Description (for retrieval reference): {image_caption}"

        # 格式化记忆上下文
        context_text = self.format_memories_for_prompt(memories)

        # 构建系统指令
        system_prompt = (
            "You are a memory testing system using Naive RAG to retrieve relevant conversation chunks.\n"
            f"{instruction}\n"
            "**IMPORTANT:**\n"
            "1. Provide your answer based on the retrieved conversation chunks only. You may not have the complete conversation history.\n"
            "2. Keep your answer within 100 words.\n"
            "3. Answer in English. This is a strict requirement. Do not answer in any other language\n"
            f"{format_req}\n"
        )

        # 组装最终Prompt
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
        构建多模态消息 (Multimodal)
        """
        # 获取特定类型的配置
        instruction, format_req, temperature = self._get_prompt_config(category)

        # 格式化记忆上下文 (纯文本)
        context_text = self.format_memories_for_prompt(memories)

        # 构建系统指令 (与文本模式保持一致)
        system_prompt = (
            "You are a memory testing system using Naive RAG to retrieve relevant conversation chunks.\n"
            f"{instruction}\n"
            "**IMPORTANT:**\n"
            "1. Provide your answer based on the retrieved conversation chunks only.\n"
            "2. Keep your answer within 100 words.\n"
            "3. Answer in English. This is a strict requirement. Do not answer in any other language\n"
            f"{format_req}\n"
        )

        # 构建用户消息内容
        user_content = []
        
        # 添加图片 (如果存在)
        if image_base64:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_base64,
                    "detail": "auto"
                }
            })

        # 构建文本部分
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
            {"role": "system", "content": "You are a helpful AI assistant."}, # 基础系统角色
            {"role": "user", "content": user_content}
        ]

        
        return messages, temperature
    
    def evaluate_question(self, sample: TestSample) -> Dict:
        """
        评估单个问题
        
        Args:
            sample: 测试样本
        
        Returns:
            包含 success 字段的结果字典
        """
        overall_start = time.time()
        
        try:
            # ================================================================
            # 步骤1: 构建检索查询（问题文本 + 图片caption）
            # ================================================================
            query_text = sample.question_text
            if sample.image_caption:
                query_text = f"{sample.question_text}\nimage description: {sample.image_caption}"
            
            # 执行检索并获取检索时间
            try:
                memories, retrieval_time = self.retrieve_memories(query_text, k=self.retrieve_k)
            except Exception as e:
                # 检索失败，返回失败结果（不保存详情）
                return {
                    "success": False,
                    "question_id": sample.question_id,
                    "session_id": sample.session_id,
                    "session_dir": sample.session_dir,
                    "question_text": sample.question_text,
                    "question_image": sample.question_image,
                    "original_answer": sample.original_answer,
                    "prediction": f"[RETRIEVAL ERROR: {str(e)[:100]}]",
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
            # 步骤2: 根据是否有图片选择调用方式
            # ================================================================
            has_image = sample.image_base64 is not None
            
            # 记录LLM调用开始时间
            llm_start = time.time()
            
            try:
                if has_image:
                    print(sample.question_text)
                    # 使用多模态方式调用（只传图片，不传caption）
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
                        # 如果不支持多模态，回退到文本方式
                        logger.warning("LLM不支持多模态，回退到文本方式（使用caption）")
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
                    # 使用纯文本方式
                    print(sample.question_text)
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
                # LLM调用失败，返回失败结果（不保存详情）
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
                    "difficulty": sample.difficulty,
                    "answer_source": sample.answer_source,
                    "processing_time": time.time() - overall_start,
                    "retrieval_time": retrieval_time,
                    "llm_time": 0.0,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            
            # 计算LLM调用时间
            llm_time = time.time() - llm_start
            
            total_processing_time = time.time() - overall_start
            
            # ================================================================
            # 步骤3: 记录成功结果
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
            
            # 同时使用print和logger输出结果
            print(f"\n📝 问题 [{sample.session_dir}:{sample.question_id}]: {sample.question_text[:50]}...")
            if sample.question_image:
                print(f"  📷 图片: {sample.question_image}")
                if sample.image_caption:
                    print(f"  💬 Caption (仅用于检索): {sample.image_caption[:50]}...")
            print(f"  🤖 预测: {prediction[:50]}...")
            print(f"  ✅ 正确: {sample.original_answer[:50]}...")
            print(f"  🏷️ 类别: {sample.category}")
            print(f"  📊 检索: {len(memories)}条记忆 (耗时: {retrieval_time:.3f}秒)")
            print(f"  🤖 LLM: {llm_time:.3f}秒")
            print(f"  ⏱️ 总用时: {total_processing_time:.2f}秒")
            
            logger.info(f"问题 {sample.question_id} 评估完成: 检索={retrieval_time:.3f}s, LLM={llm_time:.3f}s")
            
            return result
            
        except Exception as e:
            # 未预期的异常，返回失败结果
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
                "answer_source": sample.answer_source,
                "processing_time": time.time() - overall_start,
                "retrieval_time": 0.0,
                "llm_time": 0.0,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def evaluate_all(self, 
                    samples: List[TestSample],
                    max_samples: Optional[int] = None,
                    batch_size: int = 10) -> Dict:
        """
        评估所有问题，只统计成功的问题
        """
        if max_samples and max_samples < len(samples):
            samples = samples[:max_samples]
            print(f"📊 限制评估 {max_samples} 个样本")
            logger.info(f"限制评估 {max_samples} 个样本")
        
        print(f"\n{'='*60}")
        print(f"开始评估: {self.dialogue_name}")
        print(f"  样本总数: {len(samples)}")
        print(f"  检索数量: {self.retrieve_k}")
        print(f"  模型: {self.model}")
        print(f"{'='*60}")
        logger.info(f"开始评估: {self.dialogue_name}, 样本数={len(samples)}")
        
        # 按session分组
        session_samples = defaultdict(list)
        for sample in samples:
            session_samples[sample.session_dir].append(sample)
        
        print(f"按 {len(session_samples)} 个session分组评估")
        logger.info(f"按 {len(session_samples)} 个session分组评估")
        
        # 重置计数器
        self.total_questions = 0
        self.failed_count = 0
        self.results = []
        self.session_results.clear()
        self.session_stats.clear()
        
        # 逐session评估
        for session_dir, session_samps in session_samples.items():
            print(f"\n📁 评估 session: {session_dir} ({len(session_samps)} 个问题)")
            logger.info(f"评估 session: {session_dir}, 问题数={len(session_samps)}")
            
            session_path = self.dialogue_path / "scenes" / session_dir
            
            # 分批处理
            for i in range(0, len(session_samps), batch_size):
                batch = session_samps[i:i+batch_size]
                
                for sample in batch:
                    result = self.evaluate_question(sample)
                    
                    # 只统计成功的问题
                    if result.get("success", False):
                        self.results.append(result)
                        self.session_results[sample.session_dir].append(result)
                        self.all_categories.append(sample.category)
                        self.category_counts[sample.category] += 1
                        self.total_questions += 1
                        
                        # 更新统计（包括时间统计）
                        stats = self.session_stats[sample.session_dir]
                        stats["total"] += 1
                        stats["by_category"][sample.category] += 1
                        stats["by_difficulty"][sample.difficulty] += 1
                        stats["total_retrieval_time"] += result.get("retrieval_time", 0)
                        stats["total_llm_time"] += result.get("llm_time", 0)
                        stats["avg_retrieval_time"] = stats["total_retrieval_time"] / stats["total"]
                        stats["avg_llm_time"] = stats["total_llm_time"] / stats["total"]
                    else:
                        # 失败的问题仅增加失败计数
                        self.failed_count += 1
                        print(f"⚠️ 问题 {sample.question_id} 失败，已跳过统计")
                        logger.warning(f"问题 {sample.question_id} 失败，已跳过统计")
            
            # 保存当前session的结果（仅成功的结果）
            self._save_session_results(session_dir, session_path)
        
        # 构建最终结果
        final_results = self._build_final_results()
        
        # 输出总结
        self._print_summary()
        
        return final_results
    
    def _build_final_results(self) -> Dict:
        """构建最终结果（仅包含成功的问题）"""
        # 计算总体时间统计（基于成功的问题）
        total_retrieval_time = sum(r.get("retrieval_time", 0) for r in self.results)
        total_llm_time = sum(r.get("llm_time", 0) for r in self.results)
        avg_retrieval_time = total_retrieval_time / self.total_questions if self.total_questions > 0 else 0
        avg_llm_time = total_llm_time / self.total_questions if self.total_questions > 0 else 0
        
        total_attempted = self.total_questions + self.failed_count
        
        return {
            "metadata": {
                "dialogue_name": self.dialogue_name,
                "model": self.model,
                "retrieve_k": self.retrieve_k,
                "temperature_c5": self.temperature_c5,
                "total_questions_attempted": total_attempted,
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
            "results": self.results   # 仅成功的结果
        }
    
    def _save_session_results(self, session_dir: str, session_path: Path):
        """保存单个session的评估结果（仅保存成功的问题）"""
        results_dir = session_path / "evaluation_results"
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_results = self.session_results[session_dir]   # 已过滤，只包含成功结果
        session_stats = self.session_stats[session_dir]
        
        # 完整结果
        results_file = results_dir / "results_A_MEM.json"
        full_results = {
            "metadata": {
                "session_id": session_dir,
                "dialogue_name": self.dialogue_name,
                "model": self.model,
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
        
        
        print(f"\n💾 Session {session_dir} 结果已保存:")
        print(f"  - 完整结果: {results_file}")
        print(f"  ⏱️ 时间统计: 检索总时间={session_stats['total_retrieval_time']:.2f}s, "
              f"LLM总时间={session_stats['total_llm_time']:.2f}s, "
              f"平均检索={session_stats['avg_retrieval_time']:.3f}s, "
              f"平均LLM={session_stats['avg_llm_time']:.3f}s")
        logger.info(f"Session {session_dir} 结果已保存")
    
    def _print_summary(self):
        """打印结果摘要（基于成功的问题）"""
        total_attempted = self.total_questions + self.failed_count
        print(f"\n{'='*60}")
        print(f"✅ 评估完成 - {self.dialogue_name}")
        print(f"{'='*60}")
        
        print(f"\n📊 总体统计:")
        print(f"  尝试问题总数: {total_attempted}")
        print(f"  成功问题数: {self.total_questions}")
        print(f"  失败问题数: {self.failed_count}")
        print(f"  成功率: {(self.total_questions / total_attempted * 100):.1f}%" if total_attempted > 0 else "N/A")
        print(f"  类别分布（基于成功问题）:")
        for cat, count in sorted(self.category_counts.items()):
            print(f"    {cat}: {count} ({count/self.total_questions*100:.1f}%)")
        
        print(f"\n📁 Session统计（成功问题）:")
        for session_dir, stats in self.session_stats.items():
            print(f"  {session_dir}: {stats['total']} 个成功")
        
        print(f"\n🧠 记忆系统:")
        memory_stats = self.memory_system.get_statistics()
        print(f"  总记忆数: {memory_stats['total_memories']}")
        print(f"  进化次数: {memory_stats.get('evolution_count', 0)}")
        
        # 统计图片使用情况（基于成功问题）
        images_count = sum(1 for r in self.results if r.get("question_image"))
        images_with_caption = sum(1 for r in self.results if r.get("has_image_caption", False))
        print(f"\n📷 图片统计（成功问题）:")
        print(f"  包含图片的问题: {images_count}/{self.total_questions}")
        print(f"  有caption的图片: {images_with_caption}/{images_count}")
        
        # 统计时间信息（基于成功问题）
        total_retrieval_time = sum(r.get("retrieval_time", 0) for r in self.results)
        total_llm_time = sum(r.get("llm_time", 0) for r in self.results)
        total_time = total_retrieval_time + total_llm_time
        avg_retrieval = total_retrieval_time / self.total_questions if self.total_questions > 0 else 0
        avg_llm = total_llm_time / self.total_questions if self.total_questions > 0 else 0
        
        print(f"\n⏱️ 时间统计（成功问题）:")
        print(f"  总检索时间: {total_retrieval_time:.2f}秒")
        print(f"  总LLM时间: {total_llm_time:.2f}秒")
        print(f"  总处理时间: {total_time:.2f}秒")
        print(f"  平均检索时间/问题: {avg_retrieval:.3f}秒")
        print(f"  平均LLM时间/问题: {avg_llm:.3f}秒")
        
        print(f"{'='*60}")
        
        logger.info(f"评估完成: 成功问题数={self.total_questions}, 失败数={self.failed_count}, "
                   f"总检索时间={total_retrieval_time:.2f}s, 总LLM时间={total_llm_time:.2f}s")



def main():
    parser = argparse.ArgumentParser(description="使用已保存的记忆笔记进行测试 - 纯加载模式")
    parser.add_argument("--dialogue_path", type=str, required=True,
                       help="对话目录路径（包含memory_data_en和scenes目录）")
    parser.add_argument("--model", type=str, required=True,
                       help="LLM模型名称")
    parser.add_argument("--api_key", type=str, required=True,
                       help="API密钥（也可通过环境变量 OPENAI_API_KEY 设置）")
    parser.add_argument("--base_url", type=str, required=True,
                       help="API基础URL")
    parser.add_argument("--retrieve_k", type=int, required=True,
                       help="检索的记忆数量")
    parser.add_argument("--temperature_c5", type=float, default=0.5,
                       help="对抗性问题的温度")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="最大测试样本数")
    parser.add_argument("--batch_size", type=int, default=10,
                       help="批处理大小")
    parser.add_argument("--verbose", action="store_true",
                       help="详细日志")
    args = parser.parse_args()

    # 设置详细日志
    if args.verbose:
        file_handler = logging.FileHandler('evaluation_debug.log', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

    print(f"\n{'='*70}")
    print(f"记忆笔记测试 - 纯加载模式")
    print(f"{'='*70}")
    print(f"对话目录: {args.dialogue_path}")
    print(f"模型: {args.model}")
    print(f"检索数量: {args.retrieve_k}")
    print(f"{'='*70}\n")

    try:
        # 如果未提供 API key，尝试从环境变量读取
        api_key = args.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("请提供 --api_key 参数或设置环境变量 OPENAI_API_KEY")

        loader = MemoryOnlyLoader(
            dialogue_path=Path(args.dialogue_path),
            model=args.model,
            api_key=api_key,
            base_url=args.base_url,
            retrieve_k=args.retrieve_k,
            temperature_c5=args.temperature_c5,
            verbose=args.verbose
        )

        samples = loader.load_all_questions()
        if not samples:
            print(f"\n❌ 未找到任何测试问题")
            return 1

        print(f"\n找到 {len(samples)} 个测试问题")
        session_counts = defaultdict(int)
        for s in samples:
            session_counts[s.session_dir] += 1
        print("Session分布:")
        for session_dir, count in session_counts.items():
            print(f"  {session_dir}: {count} 个问题")

        loader.evaluate_all(
            samples=samples,
            max_samples=args.max_samples,
            batch_size=args.batch_size
        )

        print(f"\n✅ 评估完成!")
        print(f"\n结果已保存到各个session的 evaluation_results_qwen_3B 目录中")
        if loader.failed_questions:
            print(f"失败问题记录已保存到 evaluation_failures 目录中")
        return 0

    except Exception as e:
        print(f"\n❌ 评估失败: {e}")
        traceback.print_exc()
        return 1



if __name__ == "__main__":
    exit(main())