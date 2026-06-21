# Baselines

This directory contains baseline memory method implementations for evaluation on H2HMem.

## Methods

### Text-based Methods
These methods first convert images into textual captions, then perform evaluation.

| Directory | Description |
|-----------|-------------|
| `FullText/` | Full-context text baseline using all dialogue history |
| `Naive-RAG/` | Naive retrieval-augmented generation baseline |
| `A-MEM/` | A-MEM agent-based memory construction and evaluation |

### Multimodal Methods
These methods directly process image-text inputs without caption generation.

| Directory | Description |
|-----------|-------------|
| `FullMM/` | Full-context multimodal baseline |
| `MuRAG/` | Multimodal retrieval-augmented generation |
| `NGM/` | Neural graph memory baseline |

## Usage

Each method provides a shell script for evaluation:

```bash
bash <method_dir>/evaluate.sh
```

For A-MEM, memory construction is required before evaluation:

```bash
bash baselines/A-MEM/memory_construct.sh
bash baselines/A-MEM/evaluate.sh
```