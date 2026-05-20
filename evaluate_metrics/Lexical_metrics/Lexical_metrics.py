import json
import re
import argparse
import logging
from pathlib import Path
from collections import defaultdict
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Set, Optional, Union

# English NLP tools
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    raise ImportError("Please install nltk: pip install nltk")

# Download nltk data (first time run)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class QuestionEvaluationResult:
    """Single question evaluation result (only four core metrics)"""
    sample_id: str
    session_id: str
    dialogue_name: str
    question_id: str
    question_text: str
    question_image: str
    category: str
    difficulty: str
    question_type: Dict[str, str]
    original_answer: str
    system_answer: str
    model_name: str
    memory_type: str
    vlm_model: str
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    bleu1_score: float = 0.0
    answer_source: str = ""
    supporting_evidence: List[Dict] = None
    confidence: float = 0.0
    processing_time: float = 0.0
    success: bool = True
    error_message: str = ""

    def __post_init__(self):
        if self.supporting_evidence is None:
            self.supporting_evidence = []

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, (np.float32, np.float64)):
                result[key] = float(value)
            elif isinstance(value, (np.int32, np.int64)):
                result[key] = int(value)
            else:
                result[key] = value
        return result


class EnglishTextProcessor:
    """English text processor"""
    
    # 9 question categories
    CATEGORIES = [
        "Unimodal Precise Recall",
        "Cross-modal Related Retrieval",
        "Knowledge Resolution",
        "Temporal Reasoning",
        "Multimodal Causal Reasoning",
        "Reference & Evolution Tracking",
        "Test-Time Learning",
        "Conflict Detection",
        "Answer Refusal"
    ]
    
    # Category display names
    CATEGORY_DISPLAY_NAMES = {cat: cat for cat in CATEGORIES}

    def __init__(self, use_stopwords: bool = True, stopwords_file: str = None):
        self.use_stopwords = use_stopwords
        self.stopwords = self._load_stopwords(stopwords_file) if use_stopwords else set()

    def _load_stopwords(self, stopwords_file: str = None) -> Set[str]:
        """Load English stopwords (NLTK built-in + optional external file)"""
        stop_words = set(stopwords.words('english'))
        # Add extra punctuation and common words
        extra = {'.', ',', '!', '?', ';', ':', '"', "'", '(', ')', '[', ']', '{', '}', '-', '–', '—', '...', '..'}
        stop_words.update(extra)
        # Optional load extra stopwords from external file
        if stopwords_file and Path(stopwords_file).exists():
            with open(stopwords_file, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip().lower()
                    if word and not word.startswith('#'):
                        stop_words.add(word)
            logger.info(f"Loaded extra stopwords from {stopwords_file}")
        return stop_words

    def tokenize(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """English tokenization"""
        if not text or not text.strip():
            return []
        # Lowercase
        text = text.lower()
        # Basic cleaning: remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        # Use nltk's word_tokenize
        try:
            tokens = word_tokenize(text)
        except Exception as e:
            logger.warning(f"nltk tokenization failed: {e}, using simple space split")
            tokens = text.split()
        # Filter
        filtered = []
        for t in tokens:
            t = t.strip()
            if not t:
                continue
            if remove_stopwords and self.use_stopwords and t in self.stopwords:
                continue
            # Filter pure punctuation (optional)
            if all(ch in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~" for ch in t):
                continue
            filtered.append(t)
        return filtered

    def normalize_text(self, text: str) -> str:
        """Normalize text for exact matching: lowercase, remove punctuation, remove spaces"""
        if not text:
            return ""
        text = text.lower()
        # Remove non-alphanumeric characters (keep alphanumeric and spaces)
        text = re.sub(r'[^a-z0-9\s]', '', text)
        # Remove extra spaces
        return re.sub(r'\s+', ' ', text).strip()


class EvaluationMetricsCalculator:
    def __init__(self, text_processor: EnglishTextProcessor = None):
        self.text_processor = text_processor or EnglishTextProcessor()

    def calculate_single_pair(self, prediction: str, reference: str) -> Dict[str, float]:
        return {
            'precision': self._precision(prediction, reference),
            'recall': self._recall(prediction, reference),
            'f1': self._f1(prediction, reference),
            'bleu1': self._bleu1(prediction, reference),
        }

    def _precision(self, pred: str, ref: str) -> float:
        p_tokens = self.text_processor.tokenize(pred)
        r_tokens = self.text_processor.tokenize(ref)
        
        if not p_tokens:
            return 0.0
        
        p_set = set(p_tokens)
        r_set = set(r_tokens)
        
        if not p_set:
            return 0.0
            
        intersection = p_set & r_set
        return len(intersection) / len(p_set)

    def _recall(self, pred: str, ref: str) -> float:
        p_tokens = self.text_processor.tokenize(pred)
        r_tokens = self.text_processor.tokenize(ref)
        
        if not r_tokens:
            return 0.0
        
        p_set = set(p_tokens)
        r_set = set(r_tokens)
        
        if not r_set:
            return 0.0
            
        intersection = p_set & r_set
        return len(intersection) / len(r_set)

    def _f1(self, pred: str, ref: str) -> float:
        precision = self._precision(pred, ref)
        recall = self._recall(pred, ref)
        
        if precision + recall == 0:
            return 0.0
            
        return 2 * precision * recall / (precision + recall)

    def _bleu1(self, prediction: str, reference: str, **kwargs) -> float:
        """
        Calculate BLEU-1 score using sentence_bleu
        
        Args:
            prediction: Model generated answer
            reference: Ground truth answer
            
        Returns:
            float: BLEU-1 score, range [0,1]
        """
        # Tokenize
        pred_tokens = self.text_processor.tokenize(prediction, remove_stopwords=False)
        ref_tokens = self.text_processor.tokenize(reference, remove_stopwords=False)

        if not pred_tokens or not ref_tokens:
            return 0.0
        if pred_tokens == ref_tokens:
            return 1.0
        
        # Use sentence_bleu to calculate BLEU-1 (weights=(1, 0, 0, 0) means use only 1-gram)
        try:
            # Add smoothing function to avoid zero score issues
            smoothie = SmoothingFunction().method4
            bleu_score = sentence_bleu(
                [ref_tokens],  # Reference needs to be a list of lists
                pred_tokens,
                weights=(1.0, 0, 0, 0),  # BLEU-1
                smoothing_function=smoothie
            )
            return float(bleu_score)
        except Exception as e:
            print(prediction, reference)
            logger.warning(f"Error calculating BLEU score: {e}")
            return 0.0


class MethodAggregatedEvaluator:
    def __init__(self, metrics_calc: EvaluationMetricsCalculator = None, output_dir: Union[str, Path] = "./results"):
        self.metrics_calc = metrics_calc or EvaluationMetricsCalculator()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.categories = EnglishTextProcessor.CATEGORIES
        self.category_display = EnglishTextProcessor.CATEGORY_DISPLAY_NAMES
        self.text_processor = EnglishTextProcessor()
        self.method_results = defaultdict(lambda: {
            'metadata': {}, 'results': [],
            'dialogue_stats': defaultdict(lambda: {'count': 0, 'sessions': set()}),
            'session_stats': defaultdict(int)
        })

    def scan_and_evaluate(self, base_path: Union[str, Path], pattern: str = "results_*.json"):
        base = Path(base_path)
        if not base.exists():
            raise FileNotFoundError(f"Path does not exist: {base_path}")
        
        # Support both dialogue and dialogue1 naming patterns
        dialogue_patterns = ["dialogue*", "对话*"]
        dialogues = []
        for pattern_name in dialogue_patterns:
            dialogues.extend(sorted(base.glob(pattern_name)))
        dialogues = sorted(set(dialogues))  # Remove duplicates
        
        logger.info(f"Found {len(dialogues)} dialogue folders")
        for dialogue in dialogues:
            logger.info(f"\nProcessing dialogue: {dialogue.name}")
            scenes = dialogue / "scenes"
            if not scenes.exists():
                logger.warning(f"  scenes directory does not exist: {scenes}")
                continue
            sessions = [d for d in scenes.iterdir() if d.is_dir() and d.name.startswith('session')]
            logger.info(f"  Found {len(sessions)} sessions")
            for sess in sessions:
                eval_dir = sess / "evaluation_results"
                if not eval_dir.exists():
                    continue
                for f in eval_dir.glob(pattern):
                    self._process_file(f, dialogue.name, sess.name)
        logger.info(f"Scan complete, found {len(self.method_results)} methods")
        for method in self.method_results:
            self._generate_report(method)
        self.generate_overall_comparison()

    def _process_file(self, file_path: Path, dialogue: str, session: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        meta = data.get('metadata', {})
        memory_type = meta.get('memory_type', 'unknown')
        vlm = meta.get('vlm_model', 'unknown')
        if '/' in vlm:
            vlm = vlm.split('/')[-1]
        method = f"{memory_type}_{vlm}"
        items = data.get('results', [])
        logger.info(f"    Processing {method}: {len(items)} questions")
        for it in items:
            eval_res = self._evaluate_item(it, dialogue, session, method, memory_type, vlm)
            if eval_res:
                self.method_results[method]['results'].append(eval_res)
                self.method_results[method]['dialogue_stats'][dialogue]['count'] += 1
                self.method_results[method]['dialogue_stats'][dialogue]['sessions'].add(session)
                self.method_results[method]['session_stats'][f"{dialogue}_{session}"] += 1
        if not self.method_results[method]['metadata']:
            self.method_results[method]['metadata'] = meta

    def _evaluate_item(self, item: Dict, dialogue: str, session: str, method: str, mem_type: str, vlm: str):
        try:
            # Get answers and perform safety check
            system_answer = item.get('system_answer', '').strip()
            original_answer = item.get('original_answer', '').strip()
            
            # If both answers are empty, return default values directly
            if not system_answer and not original_answer:
                logger.debug(f"Question {item.get('question_id', 'unknown')} has empty answers")
                metrics = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'bleu1': 0.0}
            else:
                metrics = self.metrics_calc.calculate_single_pair(system_answer, original_answer)
            
            # get question category
            category = item.get('category', 'unknown')

            # Get question_type
            qtype = item.get('question_type', {})
            if isinstance(qtype, str):
                qtype = {'main_type': qtype, 'sub_type': ''}
            elif isinstance(qtype, dict):
                # Ensure main_type and sub_type exist
                if 'main_type' not in qtype:
                    qtype['main_type'] = ''
                if 'sub_type' not in qtype:
                    qtype['sub_type'] = ''
            
            
            return QuestionEvaluationResult(
                sample_id=item.get('sample_id', ''),
                session_id=session,
                dialogue_name=dialogue,
                question_id=item.get('question_id', ''),
                question_text=item.get('question_text', ''),
                question_image=item.get('question_image', ''),
                category=category,
                difficulty=item.get('difficulty', 'medium'),
                question_type=qtype,
                original_answer=original_answer,
                system_answer=system_answer,
                model_name=method,
                memory_type=mem_type,
                vlm_model=vlm,
                precision=metrics['precision'],
                recall=metrics['recall'],
                f1_score=metrics['f1'],
                bleu1_score=metrics['bleu1'],
                answer_source=item.get('answer_source', ''),
                supporting_evidence=item.get('supporting_evidence', []),
                confidence=item.get('confidence', 0.0),
                processing_time=item.get('processing_time', 0.0),
                success=item.get('success', True),
                error_message=item.get('error_message', '')
            )
        except Exception as e:
            logger.error(f"Error evaluating question: {e}, item_id={item.get('question_id', 'unknown')}")
            return None

    def _generate_report(self, method: str):
        data = self.method_results[method]
        res = data['results']
        if not res:
            return
        stats = self._calc_stats(res)
        self._save_results(method, res, data['metadata'], stats, data['dialogue_stats'], data['session_stats'])
        self._save_report(method, res, data['metadata'], stats, data['dialogue_stats'], data['session_stats'])

    def _calc_stats(self, results: List[QuestionEvaluationResult]) -> Dict:
        metrics = {'precision': [], 'recall': [], 'f1': [], 'bleu1': []}
        for r in results:
            metrics['precision'].append(r.precision)
            metrics['recall'].append(r.recall)
            metrics['f1'].append(r.f1_score)
            metrics['bleu1'].append(r.bleu1_score)
        stats = {}
        for k, v in metrics.items():
            if v:
                stats[k] = {'mean': float(np.mean(v)), 'std': float(np.std(v)),
                            'min': float(np.min(v)), 'max': float(np.max(v)),
                            'median': float(np.median(v))}
        # Statistics by category
        cat_stats = {cat: {'count': 0, 'p': 0.0, 'r': 0.0, 'f1': 0.0, 'b': 0.0} for cat in self.categories}
        for r in results:
            if r.category in cat_stats:
                cat_stats[r.category]['count'] += 1
                cat_stats[r.category]['p'] += r.precision
                cat_stats[r.category]['r'] += r.recall
                cat_stats[r.category]['f1'] += r.f1_score
                cat_stats[r.category]['b'] += r.bleu1_score
        for cat in cat_stats:
            cnt = cat_stats[cat]['count']
            if cnt > 0:
                cat_stats[cat]['p_mean'] = cat_stats[cat]['p'] / cnt
                cat_stats[cat]['r_mean'] = cat_stats[cat]['r'] / cnt
                cat_stats[cat]['f1_mean'] = cat_stats[cat]['f1'] / cnt
                cat_stats[cat]['b_mean'] = cat_stats[cat]['b'] / cnt
            for k in ['p', 'r', 'f1', 'b']:
                cat_stats[cat].pop(k, None)
        stats['by_category'] = cat_stats
        # Statistics by difficulty
        diff_stats = {}
        for r in results:
            d = r.difficulty
            if d not in diff_stats:
                diff_stats[d] = {'count': 0, 'p': 0.0, 'r': 0.0, 'f1': 0.0, 'b': 0.0}
            diff_stats[d]['count'] += 1
            diff_stats[d]['p'] += r.precision
            diff_stats[d]['r'] += r.recall
            diff_stats[d]['f1'] += r.f1_score
            diff_stats[d]['b'] += r.bleu1_score
        for d in diff_stats:
            cnt = diff_stats[d]['count']
            diff_stats[d]['p_mean'] = diff_stats[d]['p'] / cnt
            diff_stats[d]['r_mean'] = diff_stats[d]['r'] / cnt
            diff_stats[d]['f1_mean'] = diff_stats[d]['f1'] / cnt
            diff_stats[d]['b_mean'] = diff_stats[d]['b'] / cnt
            for k in ['p', 'r', 'f1', 'b']:
                diff_stats[d].pop(k, None)
        stats['by_difficulty'] = diff_stats
        return stats

    def _save_results(self, method: str, results: List[QuestionEvaluationResult], metadata: Dict,
                      stats: Dict, dialogue_stats: Dict, session_stats: Dict):
        out_dir = self.output_dir / method
        out_dir.mkdir(parents=True, exist_ok=True)
        # JSON
        json_file = out_dir / f"{method}_aggregated_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                'method': method,
                'metadata': metadata,
                'evaluation_time': datetime.now().isoformat(),
                'total_questions': len(results),
                'total_dialogues': len(dialogue_stats),
                'total_sessions': len(session_stats),
                'statistics': stats,
                'dialogue_statistics': {d: {'count': v['count'], 'sessions': list(v['sessions'])} for d, v in dialogue_stats.items()},
                'results': [r.to_dict() for r in results]
            }, f, ensure_ascii=False, indent=2)
        # CSV
        import csv
        csv_file = out_dir / f"{method}_results.csv"
        with open(csv_file, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'dialogue_name', 'session_id', 'question_id', 'category', 'difficulty',
                'question_text', 'original_answer', 'system_answer',
                'precision', 'recall', 'f1_score', 'bleu1_score', 'confidence', 'success'
            ])
            writer.writeheader()
            for r in results:
                writer.writerow({
                    'dialogue_name': r.dialogue_name,
                    'session_id': r.session_id,
                    'question_id': r.question_id,
                    'category': r.category,
                    'difficulty': r.difficulty,
                    'question_text': r.question_text[:100] + '...' if len(r.question_text) > 100 else r.question_text,
                    'original_answer': r.original_answer,
                    'system_answer': r.system_answer,
                    'precision': r.precision,
                    'recall': r.recall,
                    'f1_score': r.f1_score,
                    'bleu1_score': r.bleu1_score,
                    'confidence': r.confidence,
                    'success': r.success
                })
        logger.info(f"Results saved to {out_dir}")

    def _save_report(self, method: str, results: List[QuestionEvaluationResult], metadata: Dict,
                     stats: Dict, dialogue_stats: Dict, session_stats: Dict):
        out_dir = self.output_dir / method
        out_dir.mkdir(parents=True, exist_ok=True)
        lines = []
        lines.append("=" * 80)
        lines.append(f"Multimodal Memory Evaluation Report - {method}")
        lines.append("=" * 80)
        lines.append(f"VLM Model: {metadata.get('vlm_model', 'unknown')}")
        lines.append(f"Memory Type: {metadata.get('memory_type', 'unknown')}")
        lines.append(f"Total Questions: {len(results)}")
        lines.append("")
        lines.append("[Overall Metrics]")
        lines.append(f"{'Metric':<12} {'Mean':>8} {'Std':>8} {'Median':>8} {'Min':>8} {'Max':>8}")
        lines.append("-" * 52)
        for m in ['precision', 'recall', 'f1', 'bleu1']:
            s = stats.get(m, {})
            name = {'precision':'Precision','recall':'Recall','f1':'F1','bleu1':'BLEU-1'}[m]
            lines.append(f"{name:<12} {s.get('mean',0):>8.4f} {s.get('std',0):>8.4f} {s.get('median',0):>8.4f} {s.get('min',0):>8.4f} {s.get('max',0):>8.4f}")
        lines.append("")
        lines.append("[Per Category]")
        lines.append(f"{'Category':<30} {'Count':>6} {'Prec':>8} {'Rec':>8} {'F1':>8} {'BLEU':>8}")
        lines.append("-" * 70)
        for cat, info in stats.get('by_category', {}).items():
            cnt = info.get('count', 0)
            if cnt == 0:
                lines.append(f"{self.category_display.get(cat, cat):<30} {cnt:>6} {'--':>8} {'--':>8} {'--':>8} {'--':>8}")
            else:
                lines.append(f"{self.category_display.get(cat, cat):<30} {cnt:>6} {info.get('p_mean',0):>8.4f} {info.get('r_mean',0):>8.4f} {info.get('f1_mean',0):>8.4f} {info.get('b_mean',0):>8.4f}")
        lines.append("")
        lines.append("=" * 80)
        report_file = out_dir / f"{method}_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        logger.info(f"Report saved to {report_file}")

    def generate_overall_comparison(self):
        if len(self.method_results) < 2:
            logger.info("Only one method, skipping comparison")
            return
        lines = []
        lines.append("=" * 80)
        lines.append("Multimodal Memory Evaluation - Method Comparison")
        lines.append("=" * 80)
        lines.append("[Overall Metrics Comparison]")
        lines.append(f"{'Method':<30} {'Count':>6} {'Prec':>8} {'Rec':>8} {'F1':>8} {'BLEU':>8}")
        lines.append("-" * 70)
        for method, data in sorted(self.method_results.items()):
            stats = self._calc_stats(data['results'])
            short = method[:28] + ".." if len(method) > 30 else method
            lines.append(f"{short:<30} {len(data['results']):>6} {stats['precision']['mean']:>8.4f} {stats['recall']['mean']:>8.4f} {stats['f1']['mean']:>8.4f} {stats['bleu1']['mean']:>8.4f}")
        lines.append("")
        lines.append("[Best Performance]")
        for metric in ['precision', 'recall', 'f1', 'bleu1']:
            best = max(self.method_results.items(), key=lambda x: self._calc_stats(x[1]['results'])[metric]['mean'])
            name = {'precision':'Precision','recall':'Recall','f1':'F1','bleu1':'BLEU-1'}[metric]
            lines.append(f"  {name}: {best[0]} ({self._calc_stats(best[1]['results'])[metric]['mean']:.4f})")
        lines.append("=" * 80)
        comp_file = self.output_dir / "method_comparison_report.txt"
        with open(comp_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        logger.info(f"Comparison report saved to {comp_file}")


def main():
    parser = argparse.ArgumentParser(description="Multimodal Memory Evaluation - P, R, F1, BLEU-1 (English)")
    parser.add_argument("--base_path", required=True, help="Root path containing 'dialogue*' or '对话*' folders")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--pattern", default="results_*.json", help="Result file pattern (default: results_*.json)")
    parser.add_argument("--stopwords_file", help="Custom stopwords file (optional)")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    text_processor = EnglishTextProcessor(stopwords_file=args.stopwords_file)
    metrics = EvaluationMetricsCalculator(text_processor)
    evaluator = MethodAggregatedEvaluator(metrics, args.output_dir)
    evaluator.scan_and_evaluate(args.base_path, args.pattern)
    print(f"Evaluation completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()