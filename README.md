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

## 🧪 Inference

This repository supports both **text-based** and **multimodal** memory methods.


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

### 📌 Notes

- Text-based methods require an additional caption generation stage.
- Multimodal methods directly consume image-text inputs.
- All evaluation scripts automatically save outputs to their corresponding result directories.
  
---

## 🏆 Leaderboard Submission

We provide an online leaderboard for evaluating memory methods on **H2HMem**:

🌐 **Leaderboard Website:**  
[H2HMem Leaderboard](https://h2hmemleaderboard1.vercel.app/)

After running inference, you need to convert your generated predictions into the official leaderboard submission format and upload the resulting JSON files to the leaderboard website.

### 📦 Step 1 — Generate Prediction Files

After inference, use the provided scripts to generate the official submission files:

```bash
bash ./leaderboard/generate_submission.sh
```

This script will generate two files:

```text
prediction_dyadic.json
prediction_multi_party.json
```

These files follow the required leaderboard submission format.

### 📤 Step 2 — Submit to Leaderboard

1. Open the leaderboard website:

   [H2HMem Leaderboard](https://h2hmemleaderboard1.vercel.app/)

2. Upload:
   - `prediction_dyadic.json`
   - `prediction_multi_party.json`

3. Fill in:
   - Method name
   - Organization / affiliation (optional)
   - Additional method description (optional)

4. Submit your results.

After submission, your method will automatically appear on the public leaderboard.

### 📌 Submission Notes

- Please ensure the prediction files strictly follow the provided JSON format.
- Both dyadic and multi-party prediction files are required for complete evaluation.
- The leaderboard evaluates submissions using the official H2HMem evaluation pipeline.
- We recommend preserving original model outputs before post-processing for reproducibility.


