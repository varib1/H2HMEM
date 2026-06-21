# Evaluate Metrics

This directory contains evaluation scripts for scoring model predictions on H2HMem.

## Metrics

| Directory | Description |
|-----------|-------------|
| `Lexical_metrics/` | Lexical-based evaluation (e.g., exact match, F1, ROUGE, BLEU) |
| `LLM-as-judge/` | LLM-based evaluation using a judge model to assess response quality |

## Usage

```bash
# Lexical metrics
bash evaluate_metrics/Lexical_metrics/run_Lexical_metrics.sh

# LLM-as-a-judge
bash evaluate_metrics/LLM-as-judge/run_LLM_as_judge.sh
```

Both scripts automatically save evaluation outputs to their corresponding result directories.