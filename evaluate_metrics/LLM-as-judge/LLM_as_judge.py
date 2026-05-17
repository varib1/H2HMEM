#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM-judge 评估器 - 多线程版本
使用大语言模型评估问答系统的答案质量
支持按记忆系统类型和问题类别汇总统计
支持批量评估，每次API调用评估一个session的全部结果
多线程并发处理多个结果文件
"""

import json
import time
import re
import logging
import argparse
import threading
from typing import Dict, List, Any, Optional, Union, Set
from pathlib import Path
from datetime import datetime
import csv
from collections import defaultdict
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
import tenacity
import numpy as np

# 尝试导入tqdm用于进度条
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # 简单的回退进度显示
    class tqdm:
        def __init__(self, *args, **kwargs): pass
        def update(self, *args, **kwargs): pass
        def close(self): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ThreadSafeCounter:
    """线程安全的计数器"""
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()
    
    def increment(self, amount=1):
        with self.lock:
            self.value += amount
            return self.value
    
    def get(self):
        with self.lock:
            return self.value


class ThreadSafeDict:
    """线程安全的字典"""
    def __init__(self):
        self.dict = {}
        self.lock = threading.Lock()
    
    def set(self, key, value):
        with self.lock:
            self.dict[key] = value
    
    def get(self, key, default=None):
        with self.lock:
            return self.dict.get(key, default)
    
    def update(self, key, update_func, *args, **kwargs):
        """原子地更新字典中的值"""
        with self.lock:
            if key not in self.dict:
                self.dict[key] = {}
            self.dict[key] = update_func(self.dict[key], *args, **kwargs)
    
    def get_all(self):
        with self.lock:
            return dict(self.dict)


class TextProcessor:
    """文本处理器 - 用于文本预处理（支持英文）"""
    
    def __init__(self, use_stopwords: bool = True):
        self.use_stopwords = use_stopwords
        self.stopwords = self._get_default_stopwords() if use_stopwords else set()
    
    def _get_default_stopwords(self) -> Set[str]:
        """获取默认停用词（仅标点符号）"""
        stopwords = {
            # 英文标点符号
            '.', ',', '!', '?', ';', ':', '"', "'", '(', ')', '[', ']', 
            '{', '}', '<', '>', '/', '\\', '|', '`', '~', '@', '#', '$', 
            '%', '^', '&', '*', '-', '_', '=', '+',
            # 空格和换行
            ' ', '\n', '\r', '\t'
        }
        return stopwords
    
    def clean_text(self, text: str) -> str:
        """清理文本：移除多余空白"""
        if not text:
            return ""
        # 将各种空白字符替换为单个空格
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def normalize_text(self, text: str) -> str:
        """标准化文本（用于精确匹配）"""
        if not text:
            return ""
        
        # 转为小写
        text = text.lower()
        
        # 移除所有标点符号
        import string
        # 英文标点
        english_punct = string.punctuation  # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
        
        # 创建翻译表移除标点
        translator = str.maketrans('', '', english_punct)
        text = text.translate(translator)
        
        # 移除所有空白字符
        text = re.sub(r'\s+', '', text)
        
        return text.strip()


@dataclass
class QuestionLLMJudgeResult:
    """单个问题的LLM评估结果"""
    # 问题基本信息
    sample_id: str
    session_id: str
    dialogue_name: str
    question_id: str
    question_text: str
    category: str
    difficulty: str
    
    # 答案信息
    original_answer: str
    system_answer: str
    memory_type: str
    vlm_model: str
    
    # LLM评估结果
    llm_score: float = 0.0
    llm_reasoning: str = ""
    llm_success: bool = True
    llm_error: str = ""
    
    # 原始系统成功标志
    system_success: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, (np.float32, np.float64)):
                result[key] = float(value)
            elif isinstance(value, (np.int32, np.int64)):
                result[key] = int(value)
            else:
                result[key] = value
        return result


class LLMJudgeBatchEvaluator:
    """LLM评判器 - 批量评估，每次API调用评估一个session的全部结果"""
    
    # 问题类别映射
    QUESTION_TYPES = {
        "Unimodal Precise Recall": "单模态精确回忆",
        "Cross-modal Related Retrieval": "跨模态相关检索",
        "Knowledge Resolution": "知识维持",
        "Temporal Reasoning": "时间推理",
        "Multimodal Causal Inference": "多模态因果推理",
        "Reference & Evolution Tracking": "指代与演变追踪",
        "Test-Time Learning (TTL)": "测试时学习",
        "Conflict Detection (CD)": "冲突检测",
        "Answer Refusal (AR)": "答案拒绝"
    }
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化LLM评判器
        
        Args:
            config: 配置字典，包含：
                - api_key: OpenAI API密钥
                - base_url: API基础URL
                - model_name: 模型名称
                - prompt_file: 提示词文件路径
                - temperature: 温度参数
                - timeout: 超时时间
                - batch_prompt_template: 批量评估的提示词模板（可选）
                - max_workers: 最大工作线程数（默认4）
        """
        self.api_key = config.get('api_key')
        self.base_url = config.get('base_url', 'https://api.openai.com/v1')
        self.model_name = config.get('model_name', 'gpt-4o-mini')
        self.prompt_file = config.get('prompt_file', 'llm_judge_prompt.txt')
        self.batch_prompt_file = config.get('batch_prompt_file', 'llm_judge_batch_prompt.txt')
        self.temperature = config.get('temperature', 0)
        self.timeout = config.get('timeout', 30)
        self.max_workers = config.get('max_workers', 4)
        
        # 验证配置
        if not self.api_key:
            raise ValueError("API密钥不能为空")
        
        # 初始化客户端（每个线程会创建自己的客户端实例）
        self.client_config = {
            'api_key': self.api_key,
            'base_url': self.base_url
        }
        
        # 加载提示词
        self.prompt_template = self._load_prompt(self.prompt_file)
        self.batch_prompt_template = self._load_prompt(self.batch_prompt_file, optional=True)
        
        # 如果没有批量提示词文件，创建一个默认的
        if not self.batch_prompt_template:
            self.batch_prompt_template = self._create_default_batch_prompt()
        
        # 初始化文本处理器（禁用停用词，只保留基本的文本清理功能）
        self.text_processor = TextProcessor(use_stopwords=False)
        
        # 线程安全的统计信息
        self.stats = {
            'total_api_calls': ThreadSafeCounter(),
            'total_questions': ThreadSafeCounter(),
            'successful_api_calls': ThreadSafeCounter(),
            'failed_api_calls': ThreadSafeCounter(),
            'total_time': ThreadSafeCounter()
        }
        
        # 线程安全的方法结果存储
        self.method_results_lock = threading.Lock()
        self.method_results: Dict[str, Dict] = defaultdict(lambda: {
            'metadata': {},
            'results': [],
            'dialogue_stats': defaultdict(lambda: {'count': 0, 'sessions': set()}),
            'session_stats': defaultdict(lambda: {'count': 0})
        })
        
        # 所有结果的列表（线程安全添加）
        self.all_results_lock = threading.Lock()
        self.all_results = []
        
        # 处理文件计数器
        self.processed_files_counter = ThreadSafeCounter()
        self.total_files_counter = ThreadSafeCounter()
        
        logger.info(f"LLM评判器初始化完成")
        logger.info(f"  模型: {self.model_name}")
        logger.info(f"  单条提示词文件: {self.prompt_file}")
        logger.info(f"  批量提示词文件: {self.batch_prompt_file}")
        logger.info(f"  最大工作线程: {self.max_workers}")
        logger.info(f"  Base URL: {self.base_url}")
        logger.info(f"  评估模式: 批量评估 (每个session调用一次API)")
    
    def _load_prompt(self, prompt_file: str, optional: bool = False) -> Optional[str]:
        """加载提示词模板"""
        prompt_path = Path(prompt_file)
        if not prompt_path.exists():
            if optional:
                return None
            raise FileNotFoundError(f"提示词文件不存在: {prompt_file}")
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _create_default_batch_prompt(self) -> str:
        """创建默认的批量评估提示词"""
        return """You are an impartial judge evaluating the memory capabilities of an AI assistant with the question-answering task.
Your task is to compare the Assistant's Answer against the Ground Truth and assign a score of 0, 0.25, 0.5, 0.75, or 1 for multiple questions.
Provide reasoning in English.

### Scoring Rubric

**Score 0 (Incorrect / Miss):**
- The answer contradicts the Ground Truth.
- For Yes/No questions: The answer has the wrong polarity (e.g., says "Yes" when Ground Truth is "No").
- For Open-ended questions: The answer provides factually wrong information or hallucinations.
- The assistant fails to provide the required information.

**Score 0.25 (Poor / Tangential):**
- The answer touches on the topic but misses the **core entity** or key value required.
- The answer contains a mix of minor correct details and **significant hallucinations** or wrong associations.
- The answer is excessively vague to the point of being useless (e.g., answering "a dog" instead of "a golden retriever").

**Score 0.5 (Partial / Vague):**
- The answer is technically correct, but lacks confidence or is incomplete.
- The answer captures the **main entity or concept** correctly but misses a part of the required supporting details.
- For Yes/No questions: The polarity is correct, but the reasoning is flawed (if have), or the assistant is uncertain (e.g., "I think it might be Yes").
- For Open-ended questions: The answer is too general or misses key adjectives/details present in the Ground Truth.

**Score 0.75 (Good / Minor Imperfection):**
- The answer is largely accurate and captures the core information confidently.
- It misses only **minor details** (e.g., specific adjectives or secondary details) that do not alter the main truth.
- The answer contains all the correct information but includes unnecessary "fluff" or slight conversational filler that reduces precision.

**Score 1 (Correct / Exact):**
- The answer is accurate, precise, and confident.
- For Yes/No questions: The polarity matches the Ground Truth perfectly.
- For Open-ended questions: The answer contains **all** the core information and necessary details required by the Ground Truth without hallucinations.

Please evaluate the following multiple questions. For each question, you will receive:
- Question
- Ground Truth
- Assistant Answer

### Output Format

Output strictly in the following JSON array format:
[
    {"question_id": "Q001", "score": 0.75, "reasoning": "<short explanation in English>"},
    {"question_id": "Q002", "score": 1.0, "reasoning": "<short explanation in English>"}
]

Ensure that:
1. Score must be one of: 0, 0.25, 0.5, 0.75, or 1
2. Provide a concise reasoning in English for each score
3. Include ALL questions in your response

Here are the questions to evaluate:
"""
    
    def _create_client(self):
        """为每个线程创建独立的客户端实例"""
        return OpenAI(**self.client_config)
    
    def parse_batch_response(self, response_text: str, expected_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        解析批量评估的模型响应
        
        Args:
            response_text: 模型响应文本
            expected_ids: 期望的问题ID列表
        
        Returns:
            字典，key为question_id，value为{'score': float, 'reasoning': str}
        """
        try:
            # 提取JSON
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group()
            
            # 解析JSON
            results = json.loads(response_text)
            
            if not isinstance(results, list):
                logger.warning(f"响应不是数组格式: {type(results)}")
                return {}
            
            # 转换为字典
            result_dict = {}
            valid_scores = [0, 0.25, 0.5, 0.75, 1]
            
            for item in results:
                q_id = item.get('question_id')
                score = item.get('score')
                reasoning = item.get('reasoning', '').strip()
                
                if not q_id:
                    continue
                
                # 验证分数
                if score not in valid_scores:
                    try:
                        score = float(score)
                        score = min(valid_scores, key=lambda x: abs(x - score))
                    except:
                        logger.warning(f"无效的分数值: {score} for {q_id}")
                        continue
                
                result_dict[q_id] = {
                    'score': float(score),
                    'reasoning': reasoning
                }
            
            # 检查是否有缺失的问题
            missing_ids = [qid for qid in expected_ids if qid not in result_dict]
            if missing_ids:
                logger.warning(f"以下问题ID在响应中缺失: {missing_ids}")
            
            return result_dict
            
        except Exception as e:
            logger.error(f"解析批量响应失败: {e}")
            logger.debug(f"响应文本: {response_text[:500]}")
            return {}
    
    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
        retry=tenacity.retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.info(
            f"重试第 {retry_state.attempt_number}/3 次..."
        )
    )
    def evaluate_batch(self, 
                       questions: List[Dict[str, Any]],
                       session_info: Dict[str, Any],
                       verbose: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        批量评估多个问答对
        
        Args:
            questions: 问题列表，每个元素包含：
                - question_id: 问题ID
                - question_text: 问题文本
                - ground_truth: 标准答案
                - model_output: 模型输出
                - category: 问题类别
                - difficulty: 难度
            session_info: session信息，包含：
                - dialogue_name: 对话名称
                - session_name: session名称
                - memory_type: 记忆类型
                - vlm_model: VLM模型
            verbose: 是否打印详细信息
        
        Returns:
            字典，key为question_id，value为评估结果
        """
        self.stats['total_api_calls'].increment()
        self.stats['total_questions'].increment(len(questions))
        start_time = time.time()
        
        # 为这个调用创建新的客户端
        client = self._create_client()
        
        try:
            # 构建批量提示词
            prompt = self._build_batch_prompt(questions, session_info)
            
            if verbose:
                logger.info(f"\n{'='*50}")
                logger.info(f"批量评估 Session: {session_info.get('session_name')}")
                logger.info(f"问题数量: {len(questions)}")
                logger.info(f"提示词长度: {len(prompt)} 字符")
            
            # 调用API
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                timeout=self.timeout
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # 解析结果
            expected_ids = [q['question_id'] for q in questions]
            results = self.parse_batch_response(response_text, expected_ids)
            
            if not results:
                raise ValueError(f"无法解析批量响应")
            
            # 更新统计
            self.stats['successful_api_calls'].increment()
            elapsed = time.time() - start_time
            self.stats['total_time'].increment(elapsed)
            
            if verbose:
                logger.info(f"成功解析 {len(results)}/{len(questions)} 个结果")
                logger.info(f"耗时: {elapsed:.2f}秒")
            
            return results
            
        except Exception as e:
            self.stats['failed_api_calls'].increment()
            elapsed = time.time() - start_time
            self.stats['total_time'].increment(elapsed)
            
            logger.error(f"批量评估失败: {e}")
            
            # 返回空结果（所有问题评估失败）
            return {}
    
    def _build_batch_prompt(self, questions: List[Dict[str, Any]], session_info: Dict[str, Any]) -> str:
        """构建批量评估的提示词"""
        prompt_parts = [self.batch_prompt_template]
        
        # 添加session信息（可选）
        prompt_parts.append(f"\nDialogue: {session_info.get('dialogue_name', 'unknown')}")
        prompt_parts.append(f"Session: {session_info.get('session_name', 'unknown')}")
        prompt_parts.append(f"Memory System: {session_info.get('memory_type', 'unknown')}")
        prompt_parts.append(f"VLM Model: {session_info.get('vlm_model', 'unknown')}")
        prompt_parts.append("")
        
        # 添加每个问题
        for i, q in enumerate(questions, 1):
            prompt_parts.append(f"--- Question {i} ---")
            prompt_parts.append(f"Question ID: {q['question_id']}")
            prompt_parts.append(f"Question: {q['question_text']}")
            prompt_parts.append(f"Ground Truth: {q['ground_truth']}")
            prompt_parts.append(f"Assistant Answer: {q['model_output']}")
            prompt_parts.append("")
        
        prompt_parts.append("Please output the results in JSON array format as specified above.")
        
        return "\n".join(prompt_parts)
    
    def _extract_dialogue_name(self, file_path: Path) -> str:
        """从文件路径中提取对话名称"""
        parts = file_path.parts
        for part in parts:
            if part.startswith('对话'):
                return part
        return "unknown"
    
    def process_session_file(self,
                            result_file_path: str,
                            delay: float = 0.5,
                            verbose: bool = False) -> Optional[List[QuestionLLMJudgeResult]]:
        """
        处理单个session的结果文件（批量评估该文件中的所有问题）
        
        Args:
            result_file_path: 结果文件路径
            delay: 请求间隔(秒)
            verbose: 是否打印详细信息
        
        Returns:
            评估结果列表
        """
        file_path = Path(result_file_path)
        
        try:
            # 读取JSON文件
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取元数据
            metadata = data.get('metadata', {})
            memory_type = metadata.get('memory_type', 'unknown')
            vlm_model = metadata.get('vlm_model', 'unknown')
            
            # 简化模型名称
            if '/' in vlm_model:
                vlm_model = vlm_model.split('/')[-1]
            
            session_id = metadata.get('session_id', 'unknown')
            session_dir_name = metadata.get('session_dir_name', 'unknown')
            dialogue_name = self._extract_dialogue_name(file_path)
            
            # 提取结果列表
            results_list = data.get('results', [])
            
            if not results_list:
                logger.warning(f"  文件中没有问题数据: {file_path}")
                return None
            
            # 准备批量评估的问题列表
            questions_for_eval = []
            for item in results_list:
                question_id = item.get('question_id', '')
                question_text = item.get('question_text', '')
                system_answer = item.get('system_answer', '').strip()
                original_answer = item.get('original_answer', '').strip()
                
                questions_for_eval.append({
                    'question_id': question_id,
                    'question_text': question_text,
                    'ground_truth': original_answer,
                    'model_output': system_answer,
                    'category': item.get('category', 'unknown'),
                    'difficulty': item.get('difficulty', 'unknown')
                })
            
            # session信息
            session_info = {
                'dialogue_name': dialogue_name,
                'session_name': session_dir_name,
                'memory_type': memory_type,
                'vlm_model': vlm_model
            }
            
            # 批量评估
            eval_results = self.evaluate_batch(
                questions=questions_for_eval,
                session_info=session_info,
                verbose=verbose
            )
            
            # 构建结果对象
            llm_results = []
            for item in results_list:
                question_id = item.get('question_id', '')
                
                # 获取LLM评估结果
                eval_result = eval_results.get(question_id, {})
                llm_score = eval_result.get('score', 0.0)
                llm_reasoning = eval_result.get('reasoning', 'Evaluation failed')
                llm_success = question_id in eval_results
                
                # 创建结果对象
                result = QuestionLLMJudgeResult(
                    sample_id=item.get('sample_id', ''),
                    session_id=session_id,
                    dialogue_name=dialogue_name,
                    question_id=question_id,
                    question_text=item.get('question_text', ''),
                    category=item.get('category', 'unknown'),
                    difficulty=item.get('difficulty', 'unknown'),
                    original_answer=item.get('original_answer', ''),
                    system_answer=item.get('system_answer', ''),
                    memory_type=memory_type,
                    vlm_model=vlm_model,
                    llm_score=llm_score,
                    llm_reasoning=llm_reasoning,
                    llm_success=llm_success,
                    system_success=item.get('success', True)
                )
                llm_results.append(result)
            
            # 请求间隔
            if delay > 0:
                time.sleep(delay)
            
            return llm_results
            
        except Exception as e:
            logger.error(f"处理文件 {file_path} 时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _process_file_wrapper(self, file_info: Dict[str, Any], delay: float, verbose: bool):
        """
        处理单个文件的包装函数（用于多线程）
        
        Args:
            file_info: 文件信息字典
            delay: 请求间隔
            verbose: 是否打印详细信息
        
        Returns:
            (file_info, session_results) 元组
        """
        file_path = file_info['file']
        logger.debug(f"线程处理文件: {file_path.name}")
        
        # 处理文件
        session_results = self.process_session_file(
            result_file_path=str(file_path),
            delay=delay,
            verbose=verbose
        )
        
        # 更新处理计数
        processed = self.processed_files_counter.increment()
        total = self.total_files_counter.get()
        
        if processed % 10 == 0 or processed == total:
            logger.info(f"进度: {processed}/{total} 文件处理完成")
        
        return file_info, session_results
    
    def _add_results_to_collection(self, file_info: Dict[str, Any], session_results: List[QuestionLLMJudgeResult]):
        """线程安全地添加结果到集合"""
        if not session_results:
            return
        
        # 添加到所有结果列表
        with self.all_results_lock:
            self.all_results.extend(session_results)
        
        # 添加到方法结果
        if session_results:
            method_key = f"{session_results[0].memory_type}_{session_results[0].vlm_model}"
            
            with self.method_results_lock:
                for r in session_results:
                    self.method_results[method_key]['results'].append(r)
                    self.method_results[method_key]['dialogue_stats'][r.dialogue_name]['count'] += 1
                    self.method_results[method_key]['dialogue_stats'][r.dialogue_name]['sessions'].add(r.session_id)
                    self.method_results[method_key]['session_stats'][f"{r.dialogue_name}_{r.session_id}"]['count'] += 1
    
    def scan_and_evaluate(self,
                         base_path: Union[str, Path],
                         output_folder: Optional[str] = None,
                         memory_types: List[str] = None,
                         dialogues: List[str] = None,
                         sessions: List[str] = None,
                         pattern: str = "results_*.json",
                         delay: float = 0.5,
                         verbose: bool = False,
                         max_workers: int = None) -> Dict[str, Any]:
        """
        扫描并评估所有结果文件（多线程版本）
        
        Args:
            base_path: 包含对话文件夹的根路径
            output_folder: 输出文件夹路径
            memory_types: 要评估的记忆系统类型列表
            dialogues: 要评估的对话列表
            sessions: 要评估的session列表
            pattern: 结果文件的命名模式
            delay: 请求间隔(秒)
            verbose: 是否打印详细信息
            max_workers: 最大工作线程数（覆盖初始化时的设置）
        
        Returns:
            汇总评估结果
        """
        base_path = Path(base_path)
        if not base_path.exists():
            raise FileNotFoundError(f"路径不存在: {base_path}")
        
        # 设置线程数
        if max_workers is None:
            max_workers = self.max_workers
        
        # 创建输出文件夹
        if output_folder is None:
            output_folder = base_path / "LLM_judge_results"
        else:
            output_folder = Path(output_folder)
        
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # 查找所有结果文件
        result_files = []
        
        # 确定要处理的对话
        if dialogues:
            dialogue_folders = [base_path / d for d in dialogues]
        else:
            # 查找所有对话文件夹
            dialogue_folders = []
            for i in range(1, 21):
                dialogue_folder = base_path / f"对话{i}"
                if dialogue_folder.exists():
                    dialogue_folders.append(dialogue_folder)
        
        logger.info(f"找到 {len(dialogue_folders)} 个对话文件夹")
        
        # 遍历每个对话文件夹
        for dialogue_folder in dialogue_folders:
            dialogue_name = dialogue_folder.name
            logger.info(f"扫描对话: {dialogue_name}")
            
            # 查找scenes文件夹
            scenes_folder = dialogue_folder / "scenes"
            if not scenes_folder.exists():
                logger.warning(f"  scenes文件夹不存在: {scenes_folder}")
                continue
            
            # 查找所有session文件夹
            session_folders = [f for f in scenes_folder.iterdir() if f.is_dir() and f.name.startswith('session')]
            
            logger.info(f"  找到 {len(session_folders)} 个session文件夹")
            
            for session_folder in session_folders:
                # 查找evaluation_results文件夹
                eval_results_folder = session_folder / "evaluation_results"
                if not eval_results_folder.exists():
                    logger.debug(f"    evaluation_results文件夹不存在: {eval_results_folder}")
                    continue
                
                # 查找符合模式的结果文件
                for result_file in eval_results_folder.glob(pattern):
                    # 从文件名提取记忆类型
                    memory_type = result_file.stem.replace("results_", "")
                    
                    # 如果指定了记忆类型，进行过滤
                    if memory_types and memory_type not in memory_types:
                        continue
                    
                    result_files.append({
                        'file': result_file,
                        'dialogue': dialogue_name,
                        'session': session_folder.name,
                        'memory_type': memory_type
                    })
        
        logger.info(f"总共找到 {len(result_files)} 个结果文件")
        
        if not result_files:
            logger.error("未找到任何结果文件")
            return {}
        
        # 设置总文件数
        self.total_files_counter = ThreadSafeCounter()
        self.total_files_counter.increment(len(result_files))
        self.processed_files_counter = ThreadSafeCounter()
        
        logger.info(f"开始多线程处理，使用 {max_workers} 个工作线程")
        
        # 使用线程池处理文件
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_file = {
                executor.submit(self._process_file_wrapper, file_info, delay, verbose): file_info
                for file_info in result_files
            }
            
            # 使用tqdm显示进度
            if TQDM_AVAILABLE:
                with tqdm(total=len(result_files), desc="处理文件", unit="file") as pbar:
                    for future in as_completed(future_to_file):
                        file_info = future_to_file[future]
                        try:
                            file_info, session_results = future.result(timeout=300)  # 5分钟超时
                            if session_results:
                                self._add_results_to_collection(file_info, session_results)
                        except Exception as e:
                            logger.error(f"处理文件 {file_info['file']} 时发生异常: {e}")
                        finally:
                            pbar.update(1)
            else:
                # 无tqdm时的简单进度显示
                completed = 0
                for future in as_completed(future_to_file):
                    file_info = future_to_file[future]
                    try:
                        file_info, session_results = future.result(timeout=300)
                        if session_results:
                            self._add_results_to_collection(file_info, session_results)
                    except Exception as e:
                        logger.error(f"处理文件 {file_info['file']} 时发生异常: {e}")
                    
                    completed += 1
                    if completed % 10 == 0 or completed == len(result_files):
                        logger.info(f"进度: {completed}/{len(result_files)} 文件处理完成")
        
        logger.info(f"\n扫描完成! 共处理 {len(result_files)} 个结果文件")
        logger.info(f"评估了 {len(self.all_results)} 个问题")
        logger.info(f"发现 {len(self.method_results)} 种不同的方法")
        
        # 为每种方法生成聚合报告
        self._generate_method_reports(output_folder)
        
        # 生成整体对比报告
        self._generate_comparison_report(output_folder)
        
        # 保存所有结果到CSV
        self._save_all_results_csv(self.all_results, output_folder / "LLM_judge_all_results.csv")
        
        # 打印统计信息
        self.print_stats()
        
        return {
            'total_files': len(result_files),
            'total_questions': len(self.all_results),
            'total_methods': len(self.method_results),
            'output_folder': str(output_folder)
        }
    
    def _generate_method_reports(self, output_folder: Path):
        """为每种方法生成报告"""
        for method_key, method_data in self.method_results.items():
            results = method_data['results']
            
            if not results:
                continue
            
            logger.info(f"\n生成方法报告: {method_key}")
            logger.info(f"  问题数: {len(results)}")
            
            # 创建方法输出目录
            method_dir = output_folder / f"LLM_judge_{method_key}"
            method_dir.mkdir(exist_ok=True)
            
            # 计算统计信息
            stats = self._calculate_statistics(results)
            
            # 保存详细结果（JSON）
            self._save_method_json(method_dir, method_key, results, stats)
            
            # 保存CSV
            self._save_method_csv(method_dir, method_key, results)
            
            # 生成文本报告
            self._save_method_report(method_dir, method_key, results, stats)
    
    def _calculate_statistics(self, results: List[QuestionLLMJudgeResult]) -> Dict:
        """计算统计信息"""
        # 整体指标
        llm_scores = [r.llm_score for r in results if r.llm_success]
        system_success = [r for r in results if r.system_success]
        
        stats = {
            'overall': {
                'total': len(results),
                'llm_success': len([r for r in results if r.llm_success]),
                'system_success': len(system_success),
                'avg_llm_score': sum(llm_scores) / len(llm_scores) if llm_scores else 0,
                'llm_score_std': float(np.std(llm_scores)) if llm_scores and len(llm_scores) > 1 else 0
            }
        }
        
        # 按类别统计
        category_stats = {}
        for category in self.QUESTION_TYPES.keys():
            cat_results = [r for r in results if r.category == category]
            cat_scores = [r.llm_score for r in cat_results if r.llm_success]
            
            category_stats[category] = {
                'count': len(cat_results),
                'llm_success': len([r for r in cat_results if r.llm_success]),
                'avg_score': sum(cat_scores) / len(cat_scores) if cat_scores else 0,
                'chinese_name': self.QUESTION_TYPES.get(category, category)
            }
        
        stats['by_category'] = category_stats
        
        # 按难度统计
        difficulty_stats = {}
        for r in results:
            diff = r.difficulty
            if diff not in difficulty_stats:
                difficulty_stats[diff] = {'count': 0, 'scores': []}
            difficulty_stats[diff]['count'] += 1
            if r.llm_success:
                difficulty_stats[diff]['scores'].append(r.llm_score)
        
        for diff in difficulty_stats:
            scores = difficulty_stats[diff]['scores']
            difficulty_stats[diff]['avg_score'] = sum(scores) / len(scores) if scores else 0
        
        stats['by_difficulty'] = difficulty_stats
        
        # 按对话统计
        dialogue_stats = {}
        for r in results:
            dialogue = r.dialogue_name
            if dialogue not in dialogue_stats:
                dialogue_stats[dialogue] = {'count': 0, 'scores': []}
            dialogue_stats[dialogue]['count'] += 1
            if r.llm_success:
                dialogue_stats[dialogue]['scores'].append(r.llm_score)
        
        for dialogue in dialogue_stats:
            scores = dialogue_stats[dialogue]['scores']
            dialogue_stats[dialogue]['avg_score'] = sum(scores) / len(scores) if scores else 0
        
        stats['by_dialogue'] = dialogue_stats
        
        return stats
    
    def _save_method_json(self, method_dir: Path, method_key: str, 
                          results: List[QuestionLLMJudgeResult], stats: Dict):
        """保存方法的JSON结果"""
        output_data = {
            'method': method_key,
            'evaluation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model': self.model_name,
            'statistics': stats,
            'results': [r.to_dict() for r in results]
        }
        
        json_file = method_dir / f"LLM_judge_{method_key}_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"  JSON结果已保存: {json_file}")
    
    def _save_method_csv(self, method_dir: Path, method_key: str, 
                         results: List[QuestionLLMJudgeResult]):
        """保存方法的CSV结果"""
        csv_file = method_dir / f"LLM_judge_{method_key}_results.csv"
        
        fieldnames = [
            'dialogue_name', 'session_id', 'question_id', 'category', 'difficulty',
            'question_text', 'original_answer', 'system_answer',
            'llm_score', 'llm_reasoning', 'llm_success', 'system_success'
        ]
        
        with open(csv_file, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for r in results:
                row = {
                    'dialogue_name': r.dialogue_name,
                    'session_id': r.session_id,
                    'question_id': r.question_id,
                    'category': r.category,
                    'difficulty': r.difficulty,
                    'question_text': r.question_text[:100] + '...' if len(r.question_text) > 100 else r.question_text,
                    'original_answer': r.original_answer,
                    'system_answer': r.system_answer,
                    'llm_score': r.llm_score,
                    'llm_reasoning': r.llm_reasoning[:200] + '...' if len(r.llm_reasoning) > 200 else r.llm_reasoning,
                    'llm_success': r.llm_success,
                    'system_success': r.system_success
                }
                writer.writerow(row)
        
        logger.info(f"  CSV结果已保存: {csv_file}")
    
    def _save_method_report(self, method_dir: Path, method_key: str,
                           results: List[QuestionLLMJudgeResult], stats: Dict):
        """生成方法的文本报告"""
        report_lines = []
        
        report_lines.append("=" * 80)
        report_lines.append(f"LLM-judge Evaluation Report - {method_key}")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"LLM Model: {self.model_name}")
        report_lines.append("")
        
        # 整体统计
        overall = stats['overall']
        report_lines.append("【Overall Statistics】")
        report_lines.append(f"  Total Questions: {overall['total']}")
        report_lines.append(f"  LLM Evaluation Success: {overall['llm_success']}")
        report_lines.append(f"  System Success: {overall['system_success']}")
        report_lines.append(f"  Average LLM Score: {overall['avg_llm_score']:.4f}")
        report_lines.append("")
        
        # 按类别统计
        report_lines.append("【Statistics by Question Category】")
        report_lines.append("-" * 100)
        report_lines.append(f"{'Category':<35} {'Count':<8} {'Avg Score':<10}")
        report_lines.append("-" * 100)
        
        for category, cat_stats in stats['by_category'].items():
            chinese_name = cat_stats['chinese_name']
            display = f"{chinese_name}/{category}"
            if len(display) > 34:
                display = display[:31] + "..."
            
            if cat_stats['count'] == 0:
                report_lines.append(f"{display:<35} {'(none)':<8} {'--':<10}")
            else:
                report_lines.append(f"{display:<35} {cat_stats['count']:<8} {cat_stats['avg_score']:<10.4f}")
        
        report_lines.append("")
        
        # 按难度统计
        report_lines.append("【Statistics by Difficulty】")
        report_lines.append("-" * 50)
        report_lines.append(f"{'Difficulty':<10} {'Count':<8} {'Avg Score':<10}")
        report_lines.append("-" * 50)
        
        for difficulty, diff_stats in stats['by_difficulty'].items():
            report_lines.append(f"{difficulty:<10} {diff_stats['count']:<8} {diff_stats['avg_score']:<10.4f}")
        
        report_lines.append("")
        
        # 按对话统计
        report_lines.append("【Statistics by Dialogue】")
        report_lines.append("-" * 50)
        report_lines.append(f"{'Dialogue':<10} {'Count':<8} {'Avg Score':<10}")
        report_lines.append("-" * 50)
        
        for dialogue, dia_stats in stats['by_dialogue'].items():
            report_lines.append(f"{dialogue:<10} {dia_stats['count']:<8} {dia_stats['avg_score']:<10.4f}")
        
        report_lines.append("")
        
        # 高分样本
        report_lines.append("【High Score Samples (LLM Score >= 0.8)】")
        high_score = [r for r in results if r.llm_score >= 0.8 and r.llm_success]
        for r in high_score[:5]:
            report_lines.append(f"  {r.dialogue_name}/{r.session_id}/{r.question_id}: Score={r.llm_score:.4f}")
            report_lines.append(f"    Question: {r.question_text[:100]}...")
            report_lines.append(f"    Reasoning: {r.llm_reasoning[:200]}...")
            report_lines.append("")
        
        if not high_score:
            report_lines.append("  No high score samples")
        
        report_lines.append("")
        
        # 低分样本
        report_lines.append("【Low Score Samples (LLM Score < 0.2)】")
        low_score = [r for r in results if r.llm_score < 0.2 and r.llm_score > 0 and r.llm_success]
        for r in low_score[:5]:
            report_lines.append(f"  {r.dialogue_name}/{r.session_id}/{r.question_id}: Score={r.llm_score:.4f}")
            report_lines.append(f"    Question: {r.question_text[:100]}...")
            report_lines.append(f"    Reasoning: {r.llm_reasoning[:200]}...")
            report_lines.append("")
        
        if not low_score:
            report_lines.append("  No low score samples")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # 保存报告
        report_file = method_dir / f"LLM_judge_{method_key}_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        
        logger.info(f"  报告已保存: {report_file}")
    
    def _generate_comparison_report(self, output_folder: Path):
        """生成所有方法的对比报告"""
        if len(self.method_results) < 2:
            logger.info("只有一种方法，跳过对比报告")
            return
        
        report_lines = []
        
        report_lines.append("=" * 100)
        report_lines.append("LLM-judge Method Comparison Report")
        report_lines.append("=" * 100)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Methods Compared: {len(self.method_results)}")
        report_lines.append("")
        
        # 整体指标对比
        report_lines.append("【Overall Metrics Comparison】")
        report_lines.append("-" * 120)
        header = f"{'Method Name':<35} {'Questions':<8} {'LLM Success Rate':<15} {'Avg Score':<10}"
        report_lines.append(header)
        report_lines.append("-" * 120)
        
        method_summaries = []
        for method_key, method_data in self.method_results.items():
            results = method_data['results']
            stats = self._calculate_statistics(results)
            overall = stats['overall']
            
            short_name = method_key[:33] + "..." if len(method_key) > 35 else method_key
            success_rate = overall['llm_success'] / overall['total'] if overall['total'] else 0
            
            report_lines.append(f"{short_name:<35} {overall['total']:<8} {success_rate*100:<14.1f}% {overall['avg_llm_score']:<10.4f}")
            
            method_summaries.append({
                'name': method_key,
                'avg_score': overall['avg_llm_score']
            })
        
        report_lines.append("")
        
        # 最佳方法
        if method_summaries:
            best_method = max(method_summaries, key=lambda x: x['avg_score'])
            report_lines.append(f"Best Method: {best_method['name']} (Avg Score: {best_method['avg_score']:.4f})")
        
        report_lines.append("")
        report_lines.append("=" * 100)
        
        # 保存对比报告
        comparison_file = output_folder / "LLM_judge_method_comparison.txt"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        
        logger.info(f"对比报告已保存: {comparison_file}")
        
        # 打印到控制台
        print("\n" + "\n".join(report_lines))
    
    def _save_all_results_csv(self, results: List[QuestionLLMJudgeResult], csv_file: Path):
        """保存所有结果到CSV"""
        with open(csv_file, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Memory System', 'VLM Model', 'Dialogue', 'Session', 'Question ID', 
                'Category', 'Difficulty', 'LLM Score', 'LLM Success', 'System Success',
                'Question', 'Ground Truth', 'Assistant Answer', 'Reasoning'
            ])
            
            for r in results:
                writer.writerow([
                    r.memory_type,
                    r.vlm_model,
                    r.dialogue_name,
                    r.session_id,
                    r.question_id,
                    r.category,
                    r.difficulty,
                    r.llm_score,
                    r.llm_success,
                    r.system_success,
                    r.question_text[:100] + '...' if len(r.question_text) > 100 else r.question_text,
                    r.original_answer[:100] + '...' if len(r.original_answer) > 100 else r.original_answer,
                    r.system_answer[:100] + '...' if len(r.system_answer) > 100 else r.system_answer,
                    r.llm_reasoning[:200] + '...' if len(r.llm_reasoning) > 200 else r.llm_reasoning
                ])
        
        logger.info(f"所有结果CSV已保存: {csv_file}")
    
    def print_stats(self):
        """打印统计信息"""
        logger.info("\n" + "="*50)
        logger.info("LLM-judge Statistics")
        logger.info("="*50)
        logger.info(f"Total API Calls: {self.stats['total_api_calls'].get()}")
        logger.info(f"Total Questions: {self.stats['total_questions'].get()}")
        logger.info(f"Successful API Calls: {self.stats['successful_api_calls'].get()}")
        logger.info(f"Failed API Calls: {self.stats['failed_api_calls'].get()}")
        
        if self.stats['successful_api_calls'].get() > 0:
            total_time = self.stats['total_time'].get()
            success_calls = self.stats['successful_api_calls'].get()
            total_questions = self.stats['total_questions'].get()
            
            avg_time = total_time / success_calls
            logger.info(f"Average Time per API Call: {avg_time:.2f} seconds")
            logger.info(f"Average Time per Question: {avg_time / total_questions * success_calls:.2f} seconds")
        
        logger.info("="*50)


def create_batch_prompt_file(prompt_file: str = "llm_judge_batch_prompt.txt"):
    """创建批量评估的提示词文件"""
    content = """You are an impartial judge evaluating the memory capabilities of an AI assistant with the question-answering task.
Your task is to compare the Assistant's Answer against the Ground Truth and assign a score of 0, 0.25, 0.5, 0.75, or 1 for multiple questions.
Provide reasoning in English.

### Scoring Rubric

**Score 0 (Incorrect / Miss):**
- The answer contradicts the Ground Truth.
- For Yes/No questions: The answer has the wrong polarity (e.g., says "Yes" when Ground Truth is "No").
- For Open-ended questions: The answer provides factually wrong information or hallucinations.
- The assistant fails to provide the required information.

**Score 0.25 (Poor / Tangential):**
- The answer touches on the topic but misses the **core entity** or key value required.
- The answer contains a mix of minor correct details and **significant hallucinations** or wrong associations.
- The answer is excessively vague to the point of being useless (e.g., answering "a dog" instead of "a golden retriever").

**Score 0.5 (Partial / Vague):**
- The answer is technically correct, but lacks confidence or is incomplete.
- The answer captures the **main entity or concept** correctly but misses a part of the required supporting details.
- For Yes/No questions: The polarity is correct, but the reasoning is flawed (if have), or the assistant is uncertain (e.g., "I think it might be Yes").
- For Open-ended questions: The answer is too general or misses key adjectives/details present in the Ground Truth.

**Score 0.75 (Good / Minor Imperfection):**
- The answer is largely accurate and captures the core information confidently.
- It misses only **minor details** (e.g., specific adjectives or secondary details) that do not alter the main truth.
- The answer contains all the correct information but includes unnecessary "fluff" or slight conversational filler that reduces precision.

**Score 1 (Correct / Exact):**
- The answer is accurate, precise, and confident.
- For Yes/No questions: The polarity matches the Ground Truth perfectly.
- For Open-ended questions: The answer contains **all** the core information and necessary details required by the Ground Truth without hallucinations.

Please evaluate the following multiple questions. For each question, you will receive:
- Question
- Ground Truth
- Assistant Answer

### Output Format

Output strictly in the following JSON array format:
[
  {"question_id": "Q001", "score": 0.75, "reasoning": "<short explanation in English>"},
  {"question_id": "Q002", "score": 1.0, "reasoning": "<short explanation in English>"}
]

Ensure that:
1. Score must be one of: 0, 0.25, 0.5, 0.75, or 1
2. Provide a concise reasoning in English for each score
3. Include ALL questions in your response

Here are the questions to evaluate:
"""
    
    with open(prompt_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"已创建批量提示词文件: {prompt_file}")


def load_config(config_file: str) -> Dict[str, Any]:
    """加载配置文件"""
    config_path = Path(config_file)
    if not config_path.exists():
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LLM-judge 评估器 - 多线程批量评估模式")
    
    # 输入输出参数
    parser.add_argument('--root_folder', '-r', type=str, required=True,
                       help='根文件夹路径，包含对话1-20等子文件夹')
    parser.add_argument('--output_folder', '-o', type=str,
                       help='输出文件夹路径 (默认: 根文件夹/LLM_judge_results)')
    
    # 过滤选项
    parser.add_argument('--memory_types', '-m', type=str, nargs='+',
                       help='要评估的记忆系统类型，如 NaiveRAGMemorySystem GraphRAGMemorySystem')
    parser.add_argument('--dialogues', '-d', type=str, nargs='+',
                       help='要评估的对话列表，如 对话1 对话2 (默认: 评估所有)')
    parser.add_argument('--sessions', '-s', type=str, nargs='+',
                       help='要评估的session列表，如 session1 session2 (默认: 评估所有)')
    parser.add_argument('--pattern', type=str, default='results_multimodal.json',
                       help='结果文件的命名模式 (默认: results_*.json)')
    
    # API配置
    parser.add_argument('--api_key', type=str,
                       help='OpenAI API密钥')
    parser.add_argument('--base_url', type=str, default='https://api.openai.com/v1',
                       help='API基础URL')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                       help='模型名称')
    parser.add_argument('--prompt_file', type=str, default='llm_judge_prompt.txt',
                       help='单条评估提示词文件路径')
    parser.add_argument('--batch_prompt_file', type=str, default='llm_judge_batch_prompt.txt',
                       help='批量评估提示词文件路径')
    
    # 多线程配置
    parser.add_argument('--max_workers', type=int, default=4,
                       help='最大工作线程数 (默认: 4)')
    
    # 其他参数
    parser.add_argument('--config', type=str,
                       help='配置文件路径 (JSON格式)')
    parser.add_argument('--delay', type=float, default=0.5,
                       help='请求间隔(秒)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='打印详细信息')
    parser.add_argument('--list_memory_types', action='store_true',
                       help='只列出找到的记忆系统类型，不进行评估')
    parser.add_argument('--create_batch_prompt', action='store_true',
                       help='创建批量评估提示词文件')
    
    args = parser.parse_args()
    
    # 创建批量提示词文件
    if args.create_batch_prompt:
        create_batch_prompt_file(args.batch_prompt_file)
        return
    
    # 加载配置文件
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # 命令行参数覆盖配置文件
    config['api_key'] = args.api_key or config.get('api_key')
    config['base_url'] = args.base_url or config.get('base_url', 'https://api.openai.com/v1')
    config['model_name'] = args.model or config.get('model_name', 'gpt-4o-mini')
    config['prompt_file'] = args.prompt_file or config.get('prompt_file', 'llm_judge_prompt.txt')
    config['batch_prompt_file'] = args.batch_prompt_file or config.get('batch_prompt_file', 'llm_judge_batch_prompt.txt')
    config['temperature'] = config.get('temperature', 0)
    config['timeout'] = config.get('timeout', 30)
    config['max_workers'] = args.max_workers or config.get('max_workers', 4)
    
    # 验证API密钥
    if not config['api_key']:
        parser.error("请提供API密钥 (通过 --api_key 或配置文件)")
    
    # 初始化评估器
    evaluator = LLMJudgeBatchEvaluator(config)
    
    # 如果只列出记忆类型
    if args.list_memory_types:
        memory_types = set()
        root_path = Path(args.root_folder)
        for dialogue in root_path.glob("对话*"):
            for session in (dialogue / "scenes").glob("session*"):
                eval_folder = session / "evaluation_results"
                if eval_folder.exists():
                    for result_file in eval_folder.glob("results_*.json"):
                        memory_type = result_file.stem.replace("results_", "")
                        memory_types.add(memory_type)
        
        print("\n找到的记忆系统类型:")
        for mt in sorted(memory_types):
            print(f"  - {mt}")
        return
    
    # 执行批量评估
    summary = evaluator.scan_and_evaluate(
        base_path=args.root_folder,
        output_folder=args.output_folder,
        memory_types=args.memory_types,
        dialogues=args.dialogues,
        sessions=args.sessions,
        pattern=args.pattern,
        delay=args.delay,
        verbose=args.verbose,
        max_workers=args.max_workers
    )
    
    if summary:
        print(f"\n{'='*60}")
        print(f"评估完成！")
        print(f"{'='*60}")
        print(f"处理文件数: {summary['total_files']}")
        print(f"总问题数: {summary['total_questions']}")
        print(f"发现方法数: {summary['total_methods']}")
        print(f"\n结果已保存到: {summary['output_folder']}")
    else:
        print("\n评估失败: 未找到任何结果文件")


if __name__ == "__main__":
    main()