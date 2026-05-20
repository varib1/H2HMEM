#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM-judge Evaluator - Multi-threaded Version
Uses large language models to evaluate answer quality of QA systems
Supports aggregated statistics by memory system type and question category
Supports batch evaluation, one API call evaluates all results of one session
Multi-threaded concurrent processing of multiple result files
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
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
import tenacity
import numpy as np

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    class tqdm:
        def __init__(self, *args, **kwargs): pass
        def update(self, *args, **kwargs): pass
        def close(self): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== Built-in Batch Evaluation Prompt ====================
DEFAULT_BATCH_PROMPT = """You are an impartial judge evaluating the memory capabilities of an AI assistant with the question-answering task.
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
# ============================================================


class ThreadSafeCounter:
    """Thread-safe counter"""
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
    """Thread-safe dictionary"""
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
        with self.lock:
            if key not in self.dict:
                self.dict[key] = {}
            self.dict[key] = update_func(self.dict[key], *args, **kwargs)
    
    def get_all(self):
        with self.lock:
            return dict(self.dict)


class TextProcessor:
    """Text processor - for text preprocessing (supports English)"""
    
    def __init__(self, use_stopwords: bool = True):
        self.use_stopwords = use_stopwords
        self.stopwords = self._get_default_stopwords() if use_stopwords else set()
    
    def _get_default_stopwords(self) -> Set[str]:
        stopwords = {
            '.', ',', '!', '?', ';', ':', '"', "'", '(', ')', '[', ']', 
            '{', '}', '<', '>', '/', '\\', '|', '`', '~', '@', '#', '$', 
            '%', '^', '&', '*', '-', '_', '=', '+',
            ' ', '\n', '\r', '\t'
        }
        return stopwords
    
    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def normalize_text(self, text: str) -> str:
        if not text:
            return ""
        text = text.lower()
        import string
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
        text = re.sub(r'\s+', '', text)
        return text.strip()


@dataclass
class QuestionLLMJudgeResult:
    """LLM evaluation result for a single question"""
    sample_id: str
    session_id: str
    dialogue_name: str
    question_id: str
    question_text: str
    category: str
    difficulty: str
    
    original_answer: str
    system_answer: str
    memory_type: str
    vlm_model: str
    
    llm_score: float = 0.0
    llm_reasoning: str = ""
    llm_success: bool = True
    llm_error: str = ""
    system_success: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
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
    """LLM Judge - Batch evaluation, one API call evaluates all results of one session"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM Judge
        
        Args:
            config: Configuration dictionary containing:
                - api_key: OpenAI API key
                - base_url: API base URL
                - model_name: Model name
                - temperature: Temperature parameter
                - timeout: Timeout
                - max_workers: Maximum number of worker threads (default 4)
        """
        self.api_key = config.get('api_key')
        self.base_url = config.get('base_url', '')
        self.model_name = config.get('model_name', '')
        self.temperature = config.get('temperature', 0)
        self.timeout = config.get('timeout', 30)
        self.max_workers = config.get('max_workers', 4)
        
        if not self.api_key:
            raise ValueError("API key cannot be empty")
        
        self.client_config = {
            'api_key': self.api_key,
            'base_url': self.base_url
        }
        
        self.batch_prompt_template = DEFAULT_BATCH_PROMPT
        self.text_processor = TextProcessor(use_stopwords=False)
        
        self.stats = {
            'total_api_calls': ThreadSafeCounter(),
            'total_questions': ThreadSafeCounter(),
            'successful_api_calls': ThreadSafeCounter(),
            'failed_api_calls': ThreadSafeCounter(),
            'total_time': ThreadSafeCounter()
        }
        
        self.method_results_lock = threading.Lock()
        self.method_results: Dict[str, Dict] = defaultdict(lambda: {
            'metadata': {},
            'results': [],
            'dialogue_stats': defaultdict(lambda: {'count': 0, 'sessions': set()}),
            'session_stats': defaultdict(lambda: {'count': 0})
        })
        
        self.all_results_lock = threading.Lock()
        self.all_results = []
        
        self.processed_files_counter = ThreadSafeCounter()
        self.total_files_counter = ThreadSafeCounter()
        
        logger.info(f"LLM Judge initialization complete")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Max workers: {self.max_workers}")
        logger.info(f"  Base URL: {self.base_url}")
        logger.info(f"  Evaluation mode: Batch evaluation (one API call per session)")
    
    def _create_client(self):
        return OpenAI(**self.client_config)
    
    def parse_batch_response(self, response_text: str, expected_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        try:
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group()
            
            results = json.loads(response_text)
            
            if not isinstance(results, list):
                logger.warning(f"Response is not an array format: {type(results)}")
                return {}
            
            result_dict = {}
            valid_scores = [0, 0.25, 0.5, 0.75, 1]
            
            for item in results:
                q_id = item.get('question_id')
                score = item.get('score')
                reasoning = item.get('reasoning', '').strip()
                
                if not q_id:
                    continue
                
                if score not in valid_scores:
                    try:
                        score = float(score)
                        score = min(valid_scores, key=lambda x: abs(x - score))
                    except:
                        logger.warning(f"Invalid score value: {score} for {q_id}")
                        continue
                
                result_dict[q_id] = {
                    'score': float(score),
                    'reasoning': reasoning
                }
            
            missing_ids = [qid for qid in expected_ids if qid not in result_dict]
            if missing_ids:
                logger.warning(f"Missing question IDs in response: {missing_ids}")
            
            return result_dict
            
        except Exception as e:
            logger.error(f"Failed to parse batch response: {e}")
            return {}
    
    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
        retry=tenacity.retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.info(
            f"Retrying attempt {retry_state.attempt_number}/3..."
        )
    )
    def evaluate_batch(self, 
                       questions: List[Dict[str, Any]],
                       session_info: Dict[str, Any],
                       verbose: bool = False) -> Dict[str, Dict[str, Any]]:
        
        self.stats['total_api_calls'].increment()
        self.stats['total_questions'].increment(len(questions))
        start_time = time.time()
        
        client = self._create_client()
        
        try:
            prompt = self._build_batch_prompt(questions, session_info)
            
            if verbose:
                logger.info(f"\n{'='*50}")
                logger.info(f"Batch evaluation Session: {session_info.get('session_name')}")
                logger.info(f"Number of questions: {len(questions)}")
            
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                timeout=self.timeout
            )
            
            response_text = response.choices[0].message.content.strip()
            
            expected_ids = [q['question_id'] for q in questions]
            results = self.parse_batch_response(response_text, expected_ids)
            
            if not results:
                raise ValueError(f"Unable to parse batch response")
            
            self.stats['successful_api_calls'].increment()
            elapsed = time.time() - start_time
            self.stats['total_time'].increment(elapsed)
            
            if verbose:
                logger.info(f"Successfully parsed {len(results)}/{len(questions)} results")
                logger.info(f"Time: {elapsed:.2f} seconds")
            
            return results
            
        except Exception as e:
            self.stats['failed_api_calls'].increment()
            elapsed = time.time() - start_time
            self.stats['total_time'].increment(elapsed)
            logger.error(f"Batch evaluation failed: {e}")
            return {}
    
    def _build_batch_prompt(self, questions: List[Dict[str, Any]], session_info: Dict[str, Any]) -> str:
        prompt_parts = [self.batch_prompt_template]
        
        prompt_parts.append(f"\nDialogue: {session_info.get('dialogue_name', 'unknown')}")
        prompt_parts.append(f"Session: {session_info.get('session_name', 'unknown')}")
        prompt_parts.append(f"Memory System: {session_info.get('memory_type', 'unknown')}")
        prompt_parts.append(f"VLM Model: {session_info.get('vlm_model', 'unknown')}")
        prompt_parts.append("")
        
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
        parts = file_path.parts
        for part in parts:
            if part.startswith('dialogue'):
                return part
        return "unknown"
    
    def process_session_file(self,
                            result_file_path: str,
                            delay: float = 0.5,
                            verbose: bool = False) -> Optional[List[QuestionLLMJudgeResult]]:
        
        file_path = Path(result_file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            metadata = data.get('metadata', {})
            memory_type = metadata.get('memory_type', 'unknown')
            vlm_model = metadata.get('vlm_model', 'unknown')
            
            if '/' in vlm_model:
                vlm_model = vlm_model.split('/')[-1]
            
            session_id = metadata.get('session_id', 'unknown')
            session_dir_name = metadata.get('session_dir_name', 'unknown')
            dialogue_name = self._extract_dialogue_name(file_path)
            
            results_list = data.get('results', [])
            
            if not results_list:
                logger.warning(f"  No question data in file: {file_path}")
                return None
            
            questions_for_eval = []
            for item in results_list:
                question_id = item.get('question_id', '')
                question_text = item.get('question_text', '')
                system_answer = item.get('system_answer', '').strip()
                original_answer = item.get('original_answer', '').strip()
                
                category = item.get('category', '')
                if not category:
                    qtype = item.get('question_type', {})
                    if isinstance(qtype, dict):
                        category = qtype.get('sub_type', 'unknown')
                
                questions_for_eval.append({
                    'question_id': question_id,
                    'question_text': question_text,
                    'ground_truth': original_answer,
                    'model_output': system_answer,
                    'category': category,
                    'difficulty': item.get('difficulty', 'unknown')
                })
            
            session_info = {
                'dialogue_name': dialogue_name,
                'session_name': session_dir_name,
                'memory_type': memory_type,
                'vlm_model': vlm_model
            }
            
            eval_results = self.evaluate_batch(
                questions=questions_for_eval,
                session_info=session_info,
                verbose=verbose
            )
            
            llm_results = []
            for item in results_list:
                question_id = item.get('question_id', '')
                
                category = item.get('category', '')
                if not category:
                    qtype = item.get('question_type', {})
                    if isinstance(qtype, dict):
                        category = qtype.get('sub_type', 'unknown')
                
                eval_result = eval_results.get(question_id, {})
                llm_score = eval_result.get('score', 0.0)
                llm_reasoning = eval_result.get('reasoning', 'Evaluation failed')
                llm_success = question_id in eval_results
                
                result = QuestionLLMJudgeResult(
                    sample_id=item.get('sample_id', ''),
                    session_id=session_id,
                    dialogue_name=dialogue_name,
                    question_id=question_id,
                    question_text=item.get('question_text', ''),
                    category=category,
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
            
            if delay > 0:
                time.sleep(delay)
            
            return llm_results
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _process_file_wrapper(self, file_info: Dict[str, Any], delay: float, verbose: bool):
        file_path = file_info['file']
        logger.debug(f"Thread processing file: {file_path.name}")
        
        session_results = self.process_session_file(
            result_file_path=str(file_path),
            delay=delay,
            verbose=verbose
        )
        
        processed = self.processed_files_counter.increment()
        total = self.total_files_counter.get()
        
        if processed % 10 == 0 or processed == total:
            logger.info(f"Progress: {processed}/{total} files processed")
        
        return file_info, session_results
    
    def _add_results_to_collection(self, file_info: Dict[str, Any], session_results: List[QuestionLLMJudgeResult]):
        if not session_results:
            return
        
        with self.all_results_lock:
            self.all_results.extend(session_results)
        
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
                         pattern: str = "results_*.json",
                         delay: float = 0.5,
                         verbose: bool = False,
                         max_workers: int = None) -> Dict[str, Any]:
        
        base_path = Path(base_path)
        if not base_path.exists():
            raise FileNotFoundError(f"Path does not exist: {base_path}")
        
        if max_workers is None:
            max_workers = self.max_workers
        
        if output_folder is None:
            output_folder = base_path / "LLM_judge_results"
        else:
            output_folder = Path(output_folder)
        
        output_folder.mkdir(parents=True, exist_ok=True)
        
        result_files = []
        
        # Scan all dialogue* folders
        dialogue_folders = sorted([f for f in base_path.glob("dialogue*") if f.is_dir()])
        
        logger.info(f"Found {len(dialogue_folders)} dialogue folders")
        
        for dialogue_folder in dialogue_folders:
            dialogue_name = dialogue_folder.name
            logger.info(f"Scanning dialogue: {dialogue_name}")
            
            scenes_folder = dialogue_folder / "scenes"
            if not scenes_folder.exists():
                logger.warning(f"  scenes folder does not exist: {scenes_folder}")
                continue
            
            # Scan all session folders
            session_folders = [f for f in scenes_folder.iterdir() if f.is_dir() and f.name.startswith('session')]
            session_folders = sorted(session_folders)
            
            logger.info(f"  Found {len(session_folders)} session folders")
            
            for session_folder in session_folders:
                eval_results_folder = session_folder / "evaluation_results"
                if not eval_results_folder.exists():
                    continue
                
                for result_file in eval_results_folder.glob(pattern):
                    memory_type = result_file.stem.replace("results_", "")
                    result_files.append({
                        'file': result_file,
                        'dialogue': dialogue_name,
                        'session': session_folder.name,
                        'memory_type': memory_type
                    })
        
        logger.info(f"Total found {len(result_files)} result files")
        
        if not result_files:
            logger.error("No result files found")
            return {}
        
        self.total_files_counter = ThreadSafeCounter()
        self.total_files_counter.increment(len(result_files))
        self.processed_files_counter = ThreadSafeCounter()
        
        logger.info(f"Starting multi-threaded processing with {max_workers} worker threads")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self._process_file_wrapper, file_info, delay, verbose): file_info
                for file_info in result_files
            }
            
            if TQDM_AVAILABLE:
                with tqdm(total=len(result_files), desc="Processing files", unit="file") as pbar:
                    for future in as_completed(future_to_file):
                        file_info = future_to_file[future]
                        try:
                            file_info, session_results = future.result(timeout=300)
                            if session_results:
                                self._add_results_to_collection(file_info, session_results)
                        except Exception as e:
                            logger.error(f"Exception occurred while processing file {file_info['file']}: {e}")
                        finally:
                            pbar.update(1)
            else:
                completed = 0
                for future in as_completed(future_to_file):
                    file_info = future_to_file[future]
                    try:
                        file_info, session_results = future.result(timeout=300)
                        if session_results:
                            self._add_results_to_collection(file_info, session_results)
                    except Exception as e:
                        logger.error(f"Exception occurred while processing file {file_info['file']}: {e}")
                    
                    completed += 1
                    if completed % 10 == 0 or completed == len(result_files):
                        logger.info(f"Progress: {completed}/{len(result_files)} files processed")
        
        logger.info(f"\nScan complete! Processed {len(result_files)} result files")
        logger.info(f"Evaluated {len(self.all_results)} questions")
        logger.info(f"Found {len(self.method_results)} different methods")
        
        self._generate_method_reports(output_folder)
        self._generate_comparison_report(output_folder)
        self.print_stats()
        
        return {
            'total_files': len(result_files),
            'total_questions': len(self.all_results),
            'total_methods': len(self.method_results),
            'output_folder': str(output_folder)
        }
    
    def _generate_method_reports(self, output_folder: Path):
        for method_key, method_data in self.method_results.items():
            results = method_data['results']
            
            if not results:
                continue
            
            logger.info(f"\nGenerating method report: {method_key}")
            logger.info(f"  Number of questions: {len(results)}")
            
            method_dir = output_folder / f"LLM_judge_{method_key}"
            method_dir.mkdir(exist_ok=True)
            
            stats = self._calculate_statistics(results)
            self._save_method_json(method_dir, method_key, results, stats)
            self._save_method_csv(method_dir, method_key, results)
            self._save_method_report(method_dir, method_key, results, stats)
    
    def _calculate_statistics(self, results: List[QuestionLLMJudgeResult]) -> Dict:
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
        
        # Statistics by category
        category_stats = {}
        category_counts = defaultdict(int)
        category_scores = defaultdict(list)
        
        for r in results:
            if r.llm_success:
                category_scores[r.category].append(r.llm_score)
            category_counts[r.category] += 1
        
        for cat, count in category_counts.items():
            scores = category_scores[cat]
            category_stats[cat] = {
                'count': count,
                'llm_success': len(scores),
                'avg_score': sum(scores) / len(scores) if scores else 0
            }
        
        stats['by_category'] = category_stats
        
        # Statistics by difficulty
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
        
        # Statistics by dialogue
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
        
        logger.info(f"  JSON results saved: {json_file}")
    
    def _save_method_csv(self, method_dir: Path, method_key: str, 
                         results: List[QuestionLLMJudgeResult]):
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
        
        logger.info(f"  CSV results saved: {csv_file}")
    
    def _save_method_report(self, method_dir: Path, method_key: str,
                           results: List[QuestionLLMJudgeResult], stats: Dict):
        report_lines = []
        
        report_lines.append("=" * 80)
        report_lines.append(f"LLM-judge Evaluation Report - {method_key}")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"LLM Model: {self.model_name}")
        report_lines.append("")
        
        overall = stats['overall']
        report_lines.append("[Overall Statistics]")
        report_lines.append(f"  Total Questions: {overall['total']}")
        report_lines.append(f"  LLM Evaluation Success: {overall['llm_success']}")
        report_lines.append(f"  System Success: {overall['system_success']}")
        report_lines.append(f"  Average LLM Score: {overall['avg_llm_score']:.4f}")
        report_lines.append("")
        
        report_lines.append("[Statistics by Question Category]")
        report_lines.append("-" * 80)
        report_lines.append(f"{'Category':<50} {'Count':<8} {'Avg Score':<10}")
        report_lines.append("-" * 80)
        
        for category, cat_stats in sorted(stats['by_category'].items()):
            if cat_stats['count'] == 0:
                report_lines.append(f"{category:<50} {'(none)':<8} {'--':<10}")
            else:
                report_lines.append(f"{category:<50} {cat_stats['count']:<8} {cat_stats['avg_score']:<10.4f}")
        
        report_lines.append("")
        
        report_lines.append("[Statistics by Difficulty]")
        report_lines.append("-" * 50)
        report_lines.append(f"{'Difficulty':<15} {'Count':<8} {'Avg Score':<10}")
        report_lines.append("-" * 50)
        
        for difficulty, diff_stats in stats['by_difficulty'].items():
            report_lines.append(f"{difficulty:<15} {diff_stats['count']:<8} {diff_stats['avg_score']:<10.4f}")
        
        report_lines.append("")
        
        report_lines.append("[Statistics by Dialogue]")
        report_lines.append("-" * 50)
        report_lines.append(f"{'Dialogue':<20} {'Count':<8} {'Avg Score':<10}")
        report_lines.append("-" * 50)
        
        for dialogue, dia_stats in stats['by_dialogue'].items():
            report_lines.append(f"{dialogue:<20} {dia_stats['count']:<8} {dia_stats['avg_score']:<10.4f}")
        
        report_lines.append("")
        
        report_lines.append("[High Score Samples (LLM Score >= 0.8)]")
        high_score = [r for r in results if r.llm_score >= 0.8 and r.llm_success]
        for r in high_score[:5]:
            report_lines.append(f"  {r.dialogue_name}/{r.session_id}/{r.question_id}: Score={r.llm_score:.4f}")
            report_lines.append(f"    Question: {r.question_text[:100]}...")
            report_lines.append(f"    Reasoning: {r.llm_reasoning[:200]}...")
            report_lines.append("")
        
        if not high_score:
            report_lines.append("  No high score samples")
        
        report_lines.append("")
        
        report_lines.append("[Low Score Samples (LLM Score < 0.2)]")
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
        
        report_file = method_dir / f"LLM_judge_{method_key}_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        
        logger.info(f"  Report saved: {report_file}")
    
    def _generate_comparison_report(self, output_folder: Path):
        if len(self.method_results) < 2:
            logger.info("Only one method, skipping comparison report")
            return
        
        report_lines = []
        
        report_lines.append("=" * 100)
        report_lines.append("LLM-judge Method Comparison Report")
        report_lines.append("=" * 100)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Methods Compared: {len(self.method_results)}")
        report_lines.append("")
        
        report_lines.append("[Overall Metrics Comparison]")
        report_lines.append("-" * 120)
        report_lines.append(f"{'Method Name':<50} {'Questions':<10} {'LLM Success Rate':<18} {'Avg Score':<10}")
        report_lines.append("-" * 120)
        
        method_summaries = []
        for method_key, method_data in self.method_results.items():
            results = method_data['results']
            stats = self._calculate_statistics(results)
            overall = stats['overall']
            
            short_name = method_key[:47] + "..." if len(method_key) > 50 else method_key
            success_rate = overall['llm_success'] / overall['total'] if overall['total'] else 0
            
            report_lines.append(f"{short_name:<50} {overall['total']:<10} {success_rate*100:<17.1f}% {overall['avg_llm_score']:<10.4f}")
            
            method_summaries.append({
                'name': method_key,
                'avg_score': overall['avg_llm_score']
            })
        
        report_lines.append("")
        
        if method_summaries:
            best_method = max(method_summaries, key=lambda x: x['avg_score'])
            report_lines.append(f"Best Method: {best_method['name']} (Avg Score: {best_method['avg_score']:.4f})")
        
        report_lines.append("")
        report_lines.append("=" * 100)
        
        comparison_file = output_folder / "LLM_judge_method_comparison.txt"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        
        logger.info(f"Comparison report saved: {comparison_file}")
        print("\n" + "\n".join(report_lines))
    
    def print_stats(self):
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


def load_config(config_file: str) -> Dict[str, Any]:
    config_path = Path(config_file)
    if not config_path.exists():
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="LLM-judge Evaluator - Multi-threaded Batch Evaluation Mode")
    
    parser.add_argument('--root_folder', type=str, required=True,
                       help='Root folder path, containing dialogue* subfolders')
    parser.add_argument('--output_folder', type=str, required=True,
                       help='Output folder path (default: root_folder/LLM_judge_results)')
    parser.add_argument('--pattern', type=str, default='results_*.json',
                       help='Pattern for result file names (default: results_*.json)')
    
    parser.add_argument('--api_key', type=str, required=True, help='OpenAI API key')
    parser.add_argument('--base_url', type=str, required=True, help='API base URL')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    
    parser.add_argument('--max_workers', type=int, default=4,
                       help='Maximum number of worker threads (default: 4)')
    parser.add_argument('--delay', type=float, default=0.5, help='Request interval (seconds)')
    parser.add_argument('--verbose', action='store_true', help='Print detailed information')
    
    args = parser.parse_args()
    
    config = {}
    
    config['api_key'] = args.api_key 
    config['base_url'] = args.base_url
    config['model_name'] = args.model
    config['max_workers'] = args.max_workers
    
    if not config['api_key']:
        parser.error("Please provide API key")
    
    evaluator = LLMJudgeBatchEvaluator(config)
    
    summary = evaluator.scan_and_evaluate(
        base_path=args.root_folder,
        output_folder=args.output_folder,
        pattern=args.pattern,
        delay=args.delay,
        verbose=args.verbose,
        max_workers=args.max_workers
    )
    
    if summary:
        print(f"\n{'='*60}")
        print(f"Evaluation complete!")
        print(f"{'='*60}")
        print(f"Files processed: {summary['total_files']}")
        print(f"Total questions: {summary['total_questions']}")
        print(f"Methods found: {summary['total_methods']}")
        print(f"\nResults saved to: {summary['output_folder']}")
    else:
        print("\nEvaluation failed: No result files found")


if __name__ == "__main__":
    main()