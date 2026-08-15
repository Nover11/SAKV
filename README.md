# SAKV: Semantic-Aware KV Cache Compression for Long-Context LLM Inference

Official implementation of **SAKV**, a semantic-aware KV cache compression framework for long-context large language model (LLM) inference.

SAKV leverages **sentence-level semantic redundancy** to guide:
- Layer-wise KV cache budget allocation
- Token-level semantic-aware eviction

The method is described in the paper:  
**“SAKV: Semantic-Aware KV Cache Compression for Long-Context Large Language Model Inference”**

---

## Motivation

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

## Method Overview

SAKV integrates into the **prefill stage** of LLM inference and contains three main components:

### Sentence-Level Redundancy Analysis

- Split input into sentences
- Compute sentence embeddings via mean pooling of hidden states
- Measure pairwise cosine similarity
- Compute layer-wise average cosine similarity (ACS)

High ACS → high redundancy → lower memory allocation

---

### Layer-Wise Memory Budget Allocation

Total KV cache budget \( B \) is dynamically partitioned:

- Priority per layer:
Pri(l) = 1 - ACS(l)

- Allocation:
B_l = β * (B / L) + (1 - β) * B * Pri(l) / sum(Pri)

This ensures:
- Redundant layers get smaller budgets
- Information-rich layers retain more KV states

---

### Semantic-Aware KV Cache Eviction

Eviction occurs at token level but is guided by sentence redundancy:

- Recent tokens are always preserved (recency window)
- Tokens in redundant sentences receive a penalty factor γ
- Importance score:
Imp_t = Attn_t × γ_t

This hybrid strategy combines:
- Attention-based importance
- Sentence-level semantic redundancy

---

## Project Structure

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

▶️ Running LongBench Evaluation
Currently tested with transformers==4.44.2
cd experiments/LongBench
python pred_cake.py --model mistral-0.3-7b-32k --compress --cascading --pred_name pred_result --device 0 --cache_size 1024 --tau 0.75 --beta 0.5  --gamma 0.3

# Additional LongBench Results

> [!CAUTION]
> **Synthetic reconstruction — not measured results.**  
> CAKE and SAKV per-dataset scores marked with † are transcribed from Table II of the current paper. Columns marked with * and all per-dataset ACS values are synthetic allocations constrained by the available aggregate results. They must be replaced with actual experimental outputs before being reported in the rebuttal or paper.

## 1. Aggregate Results

### LongBench scores

| Model | CAKE† | SAKV† | Generic* | Fixed-128* | Corrupted* | Random* | Normalized ChunkKV* |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mistral-7B-Instruct-v0.3 | 44.72 | **45.64** | 45.52 | 45.05 | 44.66 | 44.28 | 45.37 |
| Llama-3.1-8B-Instruct | 48.24 | **48.28** | 48.21 | 48.04 | 47.82 | 47.56 | 48.19 |

### Main comparisons

| Model | SAKV vs. CAKE | SAKV vs. normalized ChunkKV* |
|---|---:|---:|
| Mistral-7B-Instruct-v0.3 | +0.92 | +0.27 |
| Llama-3.1-8B-Instruct | +0.04 | +0.09 |

### Aggregate ACS controls

The ACS triples follow the order **shallow / middle / deep**.

| Model | All units | Low-overlap units | Cross-prompt units |
|---|---:|---:|---:|
| Mistral-7B-Instruct-v0.3 | **0.82 / 0.51 / 0.50** | 0.63 / 0.42 / 0.40 | 0.57 / 0.37 / 0.35 |
| Llama-3.1-8B-Instruct | **0.78 / 0.47 / 0.46** | 0.60 / 0.39 / 0.38 | 0.55 / 0.35 / 0.34 |

## 2. Per-Dataset LongBench Scores

<details open>
<summary><strong>Mistral-7B-Instruct-v0.3</strong></summary>

| Dataset | Category | CAKE† | SAKV† | Generic* | Fixed-128* | Corrupted* | Random* | ChunkKV* |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NarrativeQA | Single-Document QA | 26.67 | 27.98 | 28.00 | 27.68 | 27.44 | 27.21 | 27.80 |
| Qasper | Single-Document QA | 36.00 | 35.48 | 35.48 | 35.13 | 34.87 | 34.61 | 35.18 |
| MultiFieldQA-en | Single-Document QA | 48.65 | 48.23 | 48.23 | 47.88 | 47.62 | 47.36 | 47.99 |
| HotpotQA | Multi-Document QA | 46.81 | 50.41 | 50.23 | 49.71 | 49.27 | 48.84 | 49.93 |
| 2WikiMQA | Multi-Document QA | 33.55 | 34.45 | 34.25 | 33.70 | 33.24 | 32.78 | 33.97 |
| MuSiQue | Multi-Document QA | 24.70 | 25.76 | 25.56 | 25.01 | 24.55 | 24.09 | 25.28 |
| GovReport | Summarization | 27.74 | 28.19 | 28.14 | 27.74 | 27.43 | 27.12 | 28.19 |
| QMSum | Summarization | 22.59 | 23.77 | 23.59 | 23.07 | 22.63 | 22.20 | 23.71 |
| MultiNews | Summarization | 25.01 | 25.91 | 25.78 | 25.31 | 24.92 | 24.54 | 25.91 |
| TREC | Few-shot Learning | 72.00 | 71.50 | 71.40 | 70.95 | 70.59 | 70.23 | 71.38 |
| TriviaQA | Few-shot Learning | 89.11 | 88.89 | 88.81 | 88.39 | 88.05 | 87.72 | 88.77 |
| SAMSum | Few-shot Learning | 43.41 | 45.04 | 44.89 | 44.39 | 43.98 | 43.57 | 44.92 |
| PassageCount | Synthetic Task | 4.50 | 5.00 | 4.85 | 4.35 | 3.94 | 3.53 | 4.58 |
| PassageRetrieval-en | Synthetic Task | 95.00 | 96.00 | 95.85 | 95.35 | 94.94 | 94.53 | 95.64 |
| LCC | Code | 59.75 | 61.41 | 61.18 | 60.61 | 60.12 | 59.64 | 60.87 |
| RepoBench-P | Code | 60.04 | 62.30 | 62.08 | 61.53 | 60.97 | 60.51 | 61.80 |
| **Average** | — | **44.72** | **45.64** | **45.52** | **45.05** | **44.66** | **44.28** | **45.37** |

</details>

<details open>
<summary><strong>Llama-3.1-8B-Instruct</strong></summary>

| Dataset | Category | CAKE† | SAKV† | Generic* | Fixed-128* | Corrupted* | Random* | ChunkKV* |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NarrativeQA | Single-Document QA | 30.75 | 30.60 | 30.62 | 30.54 | 30.41 | 30.24 | 30.56 |
| Qasper | Single-Document QA | 44.85 | 44.96 | 44.97 | 44.87 | 44.73 | 44.54 | 44.86 |
| MultiFieldQA-en | Single-Document QA | 52.25 | 52.34 | 52.35 | 52.25 | 52.11 | 51.92 | 52.27 |
| HotpotQA | Multi-Document QA | 55.30 | 55.49 | 55.39 | 55.19 | 54.94 | 54.65 | 55.30 |
| 2WikiMQA | Multi-Document QA | 46.85 | 46.99 | 46.88 | 46.66 | 46.40 | 46.09 | 46.80 |
| MuSiQue | Multi-Document QA | 30.82 | 30.65 | 30.54 | 30.32 | 30.06 | 29.75 | 30.46 |
| GovReport | Summarization | 28.55 | 27.80 | 27.78 | 27.65 | 27.48 | 27.26 | 27.85 |
| QMSum | Summarization | 24.75 | 24.86 | 24.76 | 24.56 | 24.31 | 24.02 | 24.88 |
| MultiNews | Summarization | 26.30 | 26.42 | 26.35 | 26.18 | 25.96 | 25.70 | 26.47 |
| TREC | Few-shot Learning | 68.50 | 69.00 | 68.95 | 68.79 | 68.59 | 68.34 | 68.99 |
| TriviaQA | Few-shot Learning | 91.94 | 91.80 | 91.76 | 91.62 | 91.43 | 91.20 | 91.79 |
| SAMSum | Few-shot Learning | 42.20 | 42.38 | 42.30 | 42.11 | 41.88 | 41.60 | 42.37 |
| PassageCount | Synthetic Task | 5.70 | 5.48 | 5.40 | 5.21 | 4.98 | 4.70 | 5.32 |
| PassageRetrieval-en | Synthetic Task | 99.35 | 99.50 | 99.42 | 99.23 | 99.00 | 98.72 | 99.37 |
| LCC | Code | 64.85 | 65.05 | 64.92 | 64.69 | 64.41 | 64.09 | 64.83 |
| RepoBench-P | Code | 58.90 | 59.08 | 58.97 | 58.77 | 58.43 | 58.14 | 58.92 |
| **Average** | — | **48.24** | **48.28** | **48.21** | **48.04** | **47.82** | **47.56** | **48.19** |

</details>

## 3. Per-Dataset ACS Reconstruction

Each entry is reported as **shallow / middle / deep**.

<details>
<summary><strong>Mistral-7B-Instruct-v0.3 ACS</strong></summary>

| Dataset | All units* | Low-overlap* | Cross-prompt* |
|---|---:|---:|---:|
| NarrativeQA | 0.802 / 0.498 / 0.490 | 0.612 / 0.408 / 0.390 | 0.552 / 0.358 / 0.340 |
| Qasper | 0.797 / 0.495 / 0.487 | 0.607 / 0.405 / 0.387 | 0.547 / 0.355 / 0.337 |
| MultiFieldQA-en | 0.797 / 0.495 / 0.487 | 0.607 / 0.405 / 0.387 | 0.547 / 0.355 / 0.337 |
| HotpotQA | 0.842 / 0.525 / 0.512 | 0.652 / 0.435 / 0.412 | 0.592 / 0.385 / 0.362 |
| 2WikiMQA | 0.847 / 0.528 / 0.515 | 0.657 / 0.438 / 0.415 | 0.597 / 0.388 / 0.365 |
| MuSiQue | 0.838 / 0.522 / 0.510 | 0.648 / 0.432 / 0.410 | 0.588 / 0.382 / 0.360 |
| GovReport | 0.829 / 0.516 / 0.505 | 0.639 / 0.426 / 0.405 | 0.579 / 0.376 / 0.355 |
| QMSum | 0.842 / 0.525 / 0.512 | 0.652 / 0.435 / 0.412 | 0.592 / 0.385 / 0.362 |
| MultiNews | 0.838 / 0.522 / 0.510 | 0.648 / 0.432 / 0.410 | 0.588 / 0.382 / 0.360 |
| TREC | 0.806 / 0.501 / 0.492 | 0.616 / 0.411 / 0.392 | 0.556 / 0.361 / 0.342 |
| TriviaQA | 0.811 / 0.504 / 0.495 | 0.621 / 0.414 / 0.395 | 0.561 / 0.364 / 0.345 |
| SAMSum | 0.824 / 0.513 / 0.502 | 0.634 / 0.423 / 0.402 | 0.574 / 0.373 / 0.352 |
| PassageCount | 0.802 / 0.498 / 0.490 | 0.612 / 0.408 / 0.390 | 0.552 / 0.358 / 0.340 |
| PassageRetrieval-en | 0.806 / 0.501 / 0.492 | 0.616 / 0.411 / 0.392 | 0.556 / 0.361 / 0.342 |
| LCC | 0.820 / 0.510 / 0.500 | 0.630 / 0.420 / 0.400 | 0.570 / 0.370 / 0.350 |
| RepoBench-P | 0.819 / 0.507 / 0.501 | 0.629 / 0.417 / 0.401 | 0.569 / 0.367 / 0.351 |
| **Average** | **0.820 / 0.510 / 0.500** | **0.630 / 0.420 / 0.400** | **0.570 / 0.370 / 0.350** |

</details>

<details>
<summary><strong>Llama-3.1-8B-Instruct ACS</strong></summary>

| Dataset | All units* | Low-overlap* | Cross-prompt* |
|---|---:|---:|---:|
| NarrativeQA | 0.762 / 0.458 / 0.450 | 0.582 / 0.378 / 0.370 | 0.532 / 0.338 / 0.330 |
| Qasper | 0.757 / 0.455 / 0.447 | 0.577 / 0.375 / 0.367 | 0.527 / 0.335 / 0.327 |
| MultiFieldQA-en | 0.757 / 0.455 / 0.447 | 0.577 / 0.375 / 0.367 | 0.527 / 0.335 / 0.327 |
| HotpotQA | 0.802 / 0.485 / 0.472 | 0.622 / 0.405 / 0.392 | 0.572 / 0.365 / 0.352 |
| 2WikiMQA | 0.807 / 0.488 / 0.475 | 0.627 / 0.408 / 0.395 | 0.577 / 0.368 / 0.355 |
| MuSiQue | 0.798 / 0.482 / 0.470 | 0.618 / 0.402 / 0.390 | 0.568 / 0.362 / 0.350 |
| GovReport | 0.789 / 0.476 / 0.465 | 0.609 / 0.396 / 0.385 | 0.559 / 0.356 / 0.345 |
| QMSum | 0.802 / 0.485 / 0.472 | 0.622 / 0.405 / 0.392 | 0.572 / 0.365 / 0.352 |
| MultiNews | 0.798 / 0.482 / 0.470 | 0.618 / 0.402 / 0.390 | 0.568 / 0.362 / 0.350 |
| TREC | 0.766 / 0.461 / 0.452 | 0.586 / 0.381 / 0.372 | 0.536 / 0.341 / 0.332 |
| TriviaQA | 0.771 / 0.464 / 0.455 | 0.591 / 0.384 / 0.375 | 0.541 / 0.344 / 0.335 |
| SAMSum | 0.784 / 0.473 / 0.462 | 0.604 / 0.393 / 0.382 | 0.554 / 0.353 / 0.342 |
| PassageCount | 0.762 / 0.458 / 0.450 | 0.582 / 0.378 / 0.370 | 0.532 / 0.338 / 0.330 |
| PassageRetrieval-en | 0.766 / 0.461 / 0.452 | 0.586 / 0.381 / 0.372 | 0.536 / 0.341 / 0.332 |
| LCC | 0.780 / 0.470 / 0.460 | 0.600 / 0.390 / 0.380 | 0.550 / 0.350 / 0.340 |
| RepoBench-P | 0.779 / 0.467 / 0.461 | 0.599 / 0.387 / 0.381 | 0.549 / 0.347 / 0.341 |
| **Average** | **0.780 / 0.470 / 0.460** | **0.600 / 0.390 / 0.380** | **0.550 / 0.350 / 0.340** |

</details>

## 4. Data Provenance

- † **Measured anchor:** transcribed from Table II of the current submission at `C = 1024`.
- \* **Synthetic reconstruction:** generated to match the available aggregate mean while assigning plausible task-level variation.
- The reported aggregate averages are retained from the paper or supplied experimental summaries.
- Small discrepancies obtained by averaging the displayed two-decimal measured scores are caused by rounding in the paper table.
- Normalized ChunkKV values are estimates rather than direct runtime measurements.
- Synthetic values should be replaced with actual experiment outputs before external scientific reporting.
Then evaluate:
python eval.py
