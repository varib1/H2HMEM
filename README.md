# H2HMem: A Human-to-Human Multimodal Memory Benchmark

<div align="center">

[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-yellow)](https://www.python.org/)

**Evaluating LLM Memory in Realistic Human-to-Human Interactions**

</div>

---

## 📊 Dataset Overview

**H2HMem** is a benchmark for evaluating multimodal memory in LLM agents across dyadic and multi-party human-human conversations.

| Aspect | Dyadic | Multi-party | Total |
|:--|--:|--:|--:|
| Dialogues | 20 | 5 | **25** |
| Sessions | 284 | 25 | **309** |
| Dialogue Rounds | 5,316 | 1,762 | **7,078** |
| Images | 951 | 349 | **1,300** |
| QA Pairs | 2,046 | 190 | **2,236** |

### Task Types

- **UPR** — Unimodal Precise Recall  
- **CRR** — Cross-modal Related Retrieval  
- **KR** — Knowledge Resolution  
- **TR** — Temporal Reasoning  
- **MCR** — Multimodal Causal Reasoning  
- **RET** — Reference & Evolution Tracking  
- **TTL** — Test-Time Learning  
- **CD** — Conflict Detection  
- **AR** — Answer Refusal  

---

## ⚙️ Installation

```bash
# Clone repository
git clone https://github.com/varib1/H2HMEM.git
cd H2HMem

# Install dependencies
pip install -r requirements.txt
```

---

## 🧪 Usage

This repository supports both **text-based** and **multimodal** memory methods.

---

### 📝 Text-based Methods

Text-based methods first convert images into textual captions, and then perform evaluation using the corresponding baseline.

#### Step 1 — Generate Image Captions

```bash
bash ./caption/run_caption.sh
```

#### Step 2 — Run Evaluation

| Method | Command |
|:--|:--|
| **Full (Text)** | `bash ./baselines/Full (Text)/evaluate.sh` |
| **Naive RAG** | `bash ./baselines/Naive RAG/evaluate.sh` |
| **A-MEM** | `bash ./baselines/A-MEM/evaluate.sh` |

---

### 🖼️ Multimodal Methods

Multimodal methods directly process image-text inputs and therefore do **not** require caption generation.

| Method | Command |
|:--|:--|
| **Full (MM)** | `bash ./baselines/Full (MM)/evaluate.sh` |
| **MuRAG** | `bash ./baselines/MURAG/evaluate.sh` |
| **NGM** | `bash ./baselines/NGM/evaluate.sh` |

---

## 📈 Metrics Evaluation

After inference, evaluate the generated responses using the following metrics.

### 🔤 Lexical Metrics

```bash
bash ./evaluate_metrics/Lexical_metrics/run_Lexical_metrics.sh
```

### 🤖 LLM-as-a-Judge

```bash
bash ./evaluate_metrics/LLM-as-judge/run_LLM_as_judge.sh
```

---

## 📌 Notes

- Text-based methods require an additional caption generation stage.
- Multimodal methods directly consume image-text inputs.
- All evaluation scripts automatically save outputs to their corresponding result directories.


