# SAKV: Semantic-Aware KV Cache Compression for Long-Context LLM Inference

Official implementation of **SAKV**, a semantic-aware KV cache compression framework for long-context large language model (LLM) inference.

SAKV leverages **sentence-level semantic redundancy** to guide:
- Layer-wise KV cache budget allocation
- Token-level semantic-aware eviction

The method is described in the paper:  
**“SAKV: Semantic-Aware KV Cache Compression for Long-Context Large Language Model Inference”**

---

## 🚀 Motivation

In long-context inference, the **KV cache** becomes the dominant GPU memory bottleneck.

Existing KV compression methods:
- Operate at the token level
- Ignore higher-level semantic redundancy
- Use uniform or heuristic layer-wise allocation

SAKV observes that:
- Long-context inputs contain **high sentence-level redundancy**
- Redundancy varies significantly across model layers
- Shallow layers exhibit higher semantic similarity than deeper layers

Therefore, SAKV:
1. Models **sentence-level semantic redundancy**
2. Dynamically allocates memory across layers
3. Performs semantic-aware token eviction

---

## 🧠 Method Overview

SAKV integrates into the **prefill stage** of LLM inference and contains three main components:

### 1️⃣ Sentence-Level Redundancy Analysis

- Split input into sentences
- Compute sentence embeddings via mean pooling of hidden states
- Measure pairwise cosine similarity
- Compute layer-wise average cosine similarity (ACS)

High ACS → high redundancy → lower memory allocation

---

### 2️⃣ Layer-Wise Memory Budget Allocation

Total KV cache budget \( B \) is dynamically partitioned:

- Priority per layer:
Pri(l) = 1 - ACS(l)

- Allocation:
B_l = β * (B / L) + (1 - β) * B * Pri(l) / sum(Pri)

This ensures:
- Redundant layers get smaller budgets
- Information-rich layers retain more KV states

---

### 3️⃣ Semantic-Aware KV Cache Eviction

Eviction occurs at token level but is guided by sentence redundancy:

- Recent tokens are always preserved (recency window)
- Tokens in redundant sentences receive a penalty factor γ
- Importance score:
Imp_t = Attn_t × γ_t

This hybrid strategy combines:
- Attention-based importance
- Sentence-level semantic redundancy

---

## 📂 Project Structure

```text
├── experiments/
│   └── LongBench/
│       ├── eval.py           # Evaluation script
│       ├── metrics.py        # Metric computation
│       └── pred_sakv.py      # SAKV inference entry
│
├── model/
│   └── modify_mistral.py     # Model modifications / patching
│
├── sakv_cache.py             # Core SAKV KV cache logic
├── monkeypatch.py            # Runtime patch for model attention
├── utils.py                  # Utility functions
├── __init__.py
└── README.md

🔧 Installation
git clone https://github.com/Nover11/SAKV.git
cd SAKV
pip install -r requirements.txt
Tested on:
Python 3.10+
PyTorch 2.x
FlashAttention-2 (recommended)

▶️ Running LongBench Evaluation
cd experiments/LongBench
python pred_cake.py --model mistral-0.3-7b-32k --compress --cascading --pred_name pred_result --device 0 --cache_size 1024 --tau 0.75 --beta 0.5  --gamma 0.3
Then evaluate:
python eval.py
