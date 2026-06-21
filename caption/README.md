# Caption

This directory contains scripts for generating image captions, required by text-based memory methods.

| File | Description |
|------|-------------|
| `gpt_4o_caption.py` | Python script using GPT-4o to generate image captions |
| `run_caption.sh` | Shell script to run caption generation |

## Usage

```bash
bash caption/run_caption.sh
```

Generated captions are used as input for text-based baseline methods (FullText, Naive-RAG, A-MEM).