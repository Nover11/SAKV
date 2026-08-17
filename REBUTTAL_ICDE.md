# 🟥 ICDE Rebuttal: Additional Experiments and Analyses

> [!IMPORTANT]
> **This document presents the additional experimental evidence provided specifically in response to the ICDE reviewers.** Each experiment below is explicitly mapped to the corresponding reviewer concerns and rebuttal section.

This page provides the additional experiments and clarifications referenced in our rebuttal. Each section identifies the reviewer concerns it addresses, describes the experimental setting, reports the corresponding results, and explains the conclusion supported by those results.

> **Scope.** SAKV uses semantic units only as transient prefill-time probes for estimating cross-unit redundancy. After compression, the semantic-unit representations are discarded, and decoding proceeds with a standard token-indexed KV cache.

## Rebuttal-to-Experiment Map

| Rebuttal section | Reviewer concerns | Evidence on this page |
|---|---|---|
| 1. ACS Validity | R2D2, R4W1D1 | Low-overlap and matched cross-prompt ACS controls |
| 3. Novelty and Baselines | R2W2, W3D3, R3W2, R4W3D3 | ChunkKV comparison under a matched cache budget |
| 4. Segmentation Robustness | R3W1, R4W2D2 | Five segmentation strategies on LongBench |
| 5. System Implementation and Latency | R3W4, R4W6D6 | Unified implementation setting and TTFT breakdown |
| 6. Performance Improvement | R2W1D1, R4W4D4 | Paired confidence intervals and larger-model results |
| 7. Ablation Study | R2D5, R3W3, R4W5D5 | Component definitions, semantic controls, pooling, and hyperparameter sensitivity |

---

## 🟥 Experiment 1: ACS Validity Controls

**For review comments:** R2D2 and R4W1D1  
**Refer to rebuttal section:** Section 1, *ACS Validity*

This experiment evaluates whether the observed layer-wise ACS trend captures meaningful cross-unit semantic redundancy rather than only the common geometry or anisotropy of hidden representations.

### Experimental setup

We reuse the semantic units and LongBench tasks used in the layer-wise analysis and compare three pair-construction strategies:

1. **Original pairs:** all eligible semantic-unit pairs used by SAKV.
2. **Non-repeating-unit control:** low-overlap pairs from the same prompt, selected using lexical-overlap and entity-overlap criteria.
3. **Shuffled-unit control:** unrelated cross-prompt pairs matched by dataset, layer, semantic-unit length bucket, and relative-position bucket.

We aggregate the results over layers `0-6`, `12-18`, and `24-30`, denoted below as shallow, middle, and deep layers.

### Results

| Pair construction | Shallow (0-6) | Middle (12-18) | Deep (24-30) |
|---|---:|---:|---:|
| Original semantic-unit pairs | **0.82** | **0.51** | **0.50** |
| Non-repeating / low-overlap control | 0.63 | 0.42 | 0.40 |
| Shuffled / matched cross-prompt control | 0.57 | 0.37 | 0.35 |

### Analysis

All three conditions exhibit a shallow-to-deep change because they share the same model, layers, and representation geometry. However, the original semantic-unit pairs remain consistently higher than both controls in every layer band. The gap is `0.19/0.09/0.10` relative to the within-prompt low-overlap control and `0.25/0.14/0.15` relative to the matched cross-prompt control. Therefore, the absolute layer-wise profile cannot be attributed solely to representation geometry: semantically related units contain additional, measurable cross-unit redundancy. This directly addresses R2D2 and supplies the cross-unit redundancy evidence requested in R4W1D1.

---

## 🟥 Experiment 2: ChunkKV Baseline Comparison

**For review comments:** R2W2, W3D3, R3W2, and R4W3D3  
**Refer to rebuttal section:** Section 3, *Novelty and Baselines*

This experiment compares SAKV with ChunkKV under the same constrained KV-cache budget to evaluate quality preservation and selection granularity.

### Matched comparison protocol

- The cache budget is held constant across methods.
- The same LongBench evaluation protocol and model checkpoints are used.
- SAKV retains token-level selection; ChunkKV performs atomic chunk-level selection.
- Quality is reported as the average over the 16 LongBench datasets.

### LongBench quality results

| Model | SAKV | ChunkKV | Difference |
|---|---:|---:|---:|
| Mistral-7B-Instruct-v0.3 | **45.64** | 45.37 | **+0.27** |
| Llama-3.1-8B-Instruct | **48.28** | 48.19 | **+0.09** |

### Analysis

The matched-budget results show that SAKV preserves slightly higher LongBench quality on both backbones while retaining token-level selection. The margin is larger on Mistral (`+0.27`) and smaller on Llama (`+0.09`). This comparison answers the baseline-coverage concern; it does not establish superiority over SentenceKV or KVzip, whose retrieval, reuse, memory-residency, and query-dependence assumptions differ from SAKV's single-prefill eviction setting.

---

## 🟥 Experiment 3: Segmentation Robustness

**For review comments:** R3W1 and R4W2D2  
**Refer to rebuttal section:** Section 4, *Segmentation Robustness*

This experiment evaluates whether SAKV remains effective when dataset-aware segmentation is replaced by generic, fixed-length, corrupted-boundary, or random-boundary alternatives.

### Experimental setup

We evaluate five segmentation strategies under the same model, cache budget, prompt, and LongBench evaluation settings:

1. **Dataset-aware:** the default deterministic structure-aware splitter.
2. **Generic:** one general-purpose splitter used across datasets.
3. **Fixed-128:** consecutive fixed-length chunks of 128 tokens.
4. **Corrupted:** deliberately perturbed semantic boundaries.
5. **Random:** length-matched random boundaries.

### Results

| Model | Dataset-aware | Generic | Fixed-128 | Corrupted | Random | Maximum range |
|---|---:|---:|---:|---:|---:|---:|
| Mistral-7B-Instruct-v0.3 | **45.64** | 45.52 | 45.05 | 44.66 | 44.28 | 1.36 |
| Llama-3.1-8B-Instruct | **48.28** | 48.21 | 48.04 | 47.82 | 47.56 | 0.72 |

### Analysis

The generic splitter nearly matches the dataset-aware splitter, with gaps of only `0.12` on Mistral and `0.07` on Llama. Performance degrades gradually as boundaries become less semantically aligned, while even the random-boundary condition remains within `1.36` points on Mistral and `0.72` points on Llama. These results show that dataset-aware segmentation is beneficial but not a prerequisite: SAKV can operate with a generic splitter, while better semantic alignment provides a stronger redundancy signal. The fixed-length and random-boundary controls also support the R3W3 semantic-redundancy discussion in Section 7 of the rebuttal.

---

## 🟥 Experiment 4: Unified Implementation and TTFT

**For review comments:** R3W4 and R4W6D6  
**Refer to rebuttal section:** Section 5, *System Implementation and Latency*

This experiment measures the additional TTFT introduced by SAKV relative to CAKE when semantic-unit pooling and pairwise comparison are fully included under the same inference configuration.

### Implementation setting

- **Model:** Mistral-7B-Instruct-v0.3
- **Hardware:** NVIDIA A100 40GB
- **Batch size:** 1
- **Attention backend:** FlashAttention-2
- **Cache setting:** `C = 1024`, `W = 32`
- **Generation length:** 1,024 tokens
- **Timing:** CUDA-synchronized wall-clock time
- **Samples:** three valid measurements after one warmup iteration
- **Disabled optimizations:** no `torch.compile`, CUDA Graphs, or custom CUDA/Triton kernels

SAKV's TTFT includes recent-window `QK^T` scoring, semantic-unit pooling, pairwise similarity, layer-budget allocation, and token eviction. Semantic processing is performed during prefill, so its cost is included rather than excluded from the reported latency.

### Results

| Prompt + decode length | Method | Total time (s) | TTFT (s) | Decode time (s) | ms/token | SAKV TTFT overhead vs. CAKE |
|---|---|---:|---:|---:|---:|---:|
| 7K + 1K | SAKV | 29.6871 | 0.7194 | 28.9677 | 28.2888 | +0.88% |
| 7K + 1K | CAKE | 29.4739 | 0.7131 | 28.7608 | 28.0867 | - |
| 15K + 1K | SAKV | 30.6286 | 1.6107 | 29.0179 | 28.3378 | +1.69% |
| 15K + 1K | CAKE | 30.3749 | 1.5839 | 28.7910 | 28.1162 | - |
| 31K + 1K | SAKV | 32.9959 | 3.9714 | 29.0244 | 28.3442 | +3.26% |
| 31K + 1K | CAKE | 33.1929 | 3.8459 | 29.3470 | 28.6591 | - |
| 64K + 1K | SAKV | 39.2811 | 10.6017 | 28.6793 | 28.0071 | +1.62% |
| 64K + 1K | CAKE | 39.5959 | 10.4323 | 29.1636 | 28.4801 | - |

### Analysis

The N-dependent semantic-unit processing cost appears in TTFT, as intended. Across the evaluated prompt lengths, SAKV's TTFT overhead relative to CAKE ranges from `0.88%` to `3.26%` and therefore remains below `3.3%`. Decode time and per-token latency remain effectively unchanged because semantic analysis is confined to prefill and decoding uses the standard token-indexed KV cache. The results support the limited claim that SAKV adds modest prefill overhead in the unified research runner; integrating it into a production serving engine would still require work on fused prefill kernels and paged-cache management.

---

## 🟥 Experiment 5: Statistical Reliability and Model Scaling

**For review comments:** R2W1D1 and R4W4D4  
**Refer to rebuttal section:** Section 6, *Performance Improvement*

This experiment evaluates whether SAKV's modest improvements on Llama3.1 are statistically distinguishable from run-to-run variation and whether the gains persist on larger model backbones.

### Paired repeated-run analysis

All reported runs use the same prompts and evaluation configuration. We perform 10 paired runs and compute confidence intervals for SAKV's improvement over CAKE on Llama3.1-8B.

| Model | Per-layer cap | Mean improvement over CAKE | Paired 95% confidence interval | Excludes zero? |
|---|---:|---:|---:|---:|
| Llama3.1-8B-Instruct | `C = 128` | +0.13 | **[+0.10, +0.16]** | Yes |
| Llama3.1-8B-Instruct | `C = 1024` | +0.04 | **[+0.02, +0.06]** | Yes |

### Larger-model results at `C = 1024`

| Model | CAKE | SAKV | Improvement |
|---|---:|---:|---:|
| Llama2-13B-Chat | 30.56 | **30.88** | **+0.32** |
| Llama3-70B-Instruct | 50.50 | **50.63** | **+0.13** |

### Analysis

Both paired confidence intervals exclude zero, showing that the Llama3.1 improvements are reproducible under the evaluated settings rather than being explained by run-to-run noise. We therefore characterize the gains as **modest but consistent**, rather than large. The larger-backbone experiment further shows that the improvement persists across model scales, reaching `+0.32` on Llama2-13B and `+0.13` on Llama3-70B at `C = 1024`.

---

## 🟥 Experiment 6: Ablations, Pooling Robustness, and Hyperparameters

**For review comments:** R2D5, R3W3, and R4W5D5  
**Refer to rebuttal section:** Section 7, *Ablation Study*

This experiment isolates the contribution of each SAKV component and evaluates robustness to the semantic-unit pooling operator and hyperparameter values.

### 6.1 Component definitions and corrected ablation labels

SAKV contains three distinct mechanisms:

1. **ACS-based priority:** allocates the global historical-token budget across layers.
2. **Uniform base:** prevents any layer from being starved of capacity.
3. **Semantic penalty:** adjusts within-layer token scores using cross-unit redundancy.

| Variant | ACS priority | Uniform base | Semantic penalty | LongBench average |
|---|:---:|:---:|:---:|---:|
| SAKV (full) | ✓ | ✓ | ✓ | **45.64** |
| Without uniform base | ✓ | - | ✓ | 43.21 |
| Without semantic penalty | ✓ | ✓ | - | 42.57 |
| **ACS only** (previously “w/o both components”) | ✓ | - | - | 41.71 |

### Ablation analysis

The last row retains ACS-based priority and removes only the uniform base and semantic penalty; it is therefore renamed **ACS only**. Removing the uniform base reduces the score by `2.43` points, while removing the semantic penalty reduces it by `3.07` points. Removing both produces the largest drop (`3.93` points). Thus, the semantic penalty has the larger individual contribution, while the full model confirms that stable layer capacity and redundancy-aware within-layer ranking are complementary.

### 6.2 Semantic-redundancy controls

The semantic-redundancy claim is supported jointly by:

- the low-overlap and matched cross-prompt ACS controls in **Experiment 1**;
- the fixed-128, corrupted-boundary, and random-boundary controls in **Experiment 3**; and
- the pooling robustness results below.

These controls distinguish the benefit of semantically aligned cross-unit similarity from a benefit caused merely by grouping tokens into arbitrary chunks.

### 6.3 Semantic-unit pooling robustness

The following results average Qasper, HotpotQA, QMSum, PassageRetrieval-en, LCC, and TREC.

| Aggregation variant | Average score |
|---|---:|
| Mean pooling | **56.43** |
| First-token pooling | 56.42 |
| Max pooling | 56.40 |

The three pooling operators differ by at most `0.03` points. The result therefore does not depend on a narrowly chosen pooling operation; mean pooling is used by default because it is parameter-free and incorporates all tokens in a unit.

### 6.4 One-at-a-time hyperparameter sensitivity

Each parameter is swept while all other hyperparameters remain at their default values. The reported average uses the same six representative datasets as the pooling experiment.

| τ | Average | γ | Average | β | Average |
|---:|---:|---:|---:|---:|---:|
| 0.60 | 56.05 | 0.1 | 56.21 | 0.3 | 53.94 |
| 0.65 | 55.99 | 0.2 | 56.35 | 0.4 | 55.95 |
| 0.70 | 55.82 | **0.3** | **56.43** | **0.5** | **56.43** |
| **0.75** | **56.43** | 0.4 | 55.76 | 0.6 | 55.98 |
| 0.80 | 56.33 | 0.5 | 55.97 | 0.7 | 55.77 |
| 0.85 | 56.24 | 0.6 | 55.85 | 0.8 | 55.84 |

### Hyperparameter analysis

The selected values are `τ = 0.75`, `γ = 0.3`, and `β = 0.5`. The one-at-a-time sweeps show a broad stable region rather than a single narrow optimum. The remaining constants are determined analytically or operationally: `α = 0.8` provides mild sublinear smoothing of layer priorities; `λ = 0.7 = 1 - γ` makes the maximum redundancy penalty reach the floor `γ`; `W = 32` is shared across compressed methods; and `ε = 10^-6` is a numerical safeguard. All values are fixed across datasets, models, and cache budgets, with no per-dataset tuning.

---

## Appendix: Measured Table-II Anchors at `C = 1024`

The following per-dataset CAKE and SAKV scores are transcribed from Table II of the submitted paper.

<details>
<summary><strong>Mistral-7B-Instruct-v0.3</strong></summary>

| Dataset | CAKE | SAKV |
|---|---:|---:|
| NarrativeQA | 26.67 | **27.98** |
| Qasper | **36.00** | 35.48 |
| MultiFieldQA-en | **48.65** | 48.23 |
| HotpotQA | 46.81 | **50.41** |
| 2WikiMQA | 33.55 | **34.45** |
| MuSiQue | 24.70 | **25.76** |
| GovReport | 27.74 | **28.19** |
| QMSum | 22.59 | **23.77** |
| MultiNews | 25.01 | **25.91** |
| TREC | **72.00** | 71.50 |
| TriviaQA | **89.11** | 88.89 |
| SAMSum | 43.41 | **45.04** |
| PassageCount | 4.50 | **5.00** |
| PassageRetrieval-en | 95.00 | **96.00** |
| LCC | 59.75 | **61.41** |
| RepoBench-P | 60.04 | **62.30** |
| **Reported average** | 44.72 | **45.64** |

</details>

<details>
<summary><strong>Llama3.1-8B-Instruct</strong></summary>

| Dataset | CAKE | SAKV |
|---|---:|---:|
| NarrativeQA | **30.75** | 30.60 |
| Qasper | 44.85 | **44.96** |
| MultiFieldQA-en | 52.25 | **52.34** |
| HotpotQA | 55.30 | **55.49** |
| 2WikiMQA | 46.85 | **46.99** |
| MuSiQue | **30.82** | 30.65 |
| GovReport | **28.55** | 27.80 |
| QMSum | 24.75 | **24.86** |
| MultiNews | 26.30 | **26.42** |
| TREC | 68.50 | **69.00** |
| TriviaQA | **91.94** | 91.80 |
| SAMSum | 42.20 | **42.38** |
| PassageCount | **5.70** | 5.48 |
| PassageRetrieval-en | 99.35 | **99.50** |
| LCC | 64.85 | **65.05** |
| RepoBench-P | 58.90 | **59.08** |
| **Reported average** | 48.24 | **48.28** |

</details>

## Data Provenance

- LongBench, runtime, larger-model, ablation, pooling, and hyperparameter values are taken from the submitted paper or the final rebuttal results supplied by the authors.
- The additional ACS, segmentation, ChunkKV, and confidence-interval values are obtained from the new rebuttal experiments supplied by the authors.
- Small differences obtained by averaging the displayed two-decimal Table-II entries are caused by rounding; the reported paper averages are retained.
