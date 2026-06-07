# H2HMem: A Human-to-Human Multimodal Memory Benchmark

<div align="center">

[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-yellow)](https://www.python.org/)

**Evaluating LLM Memory in Realistic Human-to-Human Interactions**

</div>

---
## 📥 Dataset Download

Please download the dataset from Hugging Face: [https://huggingface.co/datasets/varib/H2HMEM](https://huggingface.co/datasets/varib/H2HMEM)

Save the contents to the following local folders:

- `./dataset/dyadic`
- `./dataset/multi-party`

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

| main_type | sub_type | NASA |
|-----------|---------------------|------|
| Memory Recall | Unimodal Precise Recall | UPR |
| Memory Recall | Cross-modal Related Retrieval | CRR |
| Memory Recall | Knowledge Resolution | KR |
| Memory Reasoning | Temporal Reasoning | TR |
| Memory Reasoning | Multimodal Causal Reasoning | MCR |
| Memory Reasoning | Reference & Evolution Tracking | RET |
| Memory Application | Test-Time Learning | TTL |
| Memory Application | Conflict Detection | CD |
| Memory Application | Answer Refusal | AR |

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
| **Full (Text)** | `bash ./baselines/FullText/evaluate.sh` |
| **Naive RAG** | `bash ./baselines/Naive_RAG/evaluate.sh` |
| **A-MEM** | 1. `bash ./baselines/A-MEM/memory_construct.sh` (build memory)<br>2. `bash ./baselines/A-MEM/evaluate.sh` (run evaluation) |


### 🖼️ Multimodal Methods

Multimodal methods directly process image-text inputs and therefore do **not** require caption generation.

| Method | Command |
|:--|:--|
| **Full (MM)** | `bash ./baselines/FullMM/evaluate.sh` |
| **MuRAG** | `bash ./baselines/MuRAG/evaluate.sh` |
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

If you wish to participate in the leaderboard, after running inference, you need to format your predictions according to our submission template before uploading to the leaderboard website.

### 📦 Submission Template

We provide a JSON template file that defines the required submission format:

- `leaderboard/prediction_template.txt` - Template for submission format

Please follow this template exactly when preparing your prediction files.

Your final submission should include two JSON files formatted according to the template:

```text
prediction_dyadic.json
prediction_multiparty.json
```
### 📤Submit to Leaderboard

1. Open the leaderboard website:

   [H2HMem Leaderboard](https://h2hmemleaderboard1.vercel.app/)

2. Upload:
   - `prediction_dyadic.json`
   - `prediction_multiparty.json`

3. Fill in:
   - Method name
   - Organization / affiliation (optional)
   - Additional method description (optional)

4. Submit your results.

Your submission needs to be reviewed by an administrator before it appears on the public leaderboard.

### 📌 Submission Notes

- Please ensure the prediction files strictly follow the provided JSON format.
- Both dyadic and multi-party prediction files are required for complete evaluation.
- We recommend preserving original model outputs before post-processing for reproducibility.


