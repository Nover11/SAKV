# SAKV: Anonymous Source Artifact

SAKV uses semantic redundancy in the prompt to guide **layer-wise cache allocation** and **token-level KV eviction**. Semantic units are temporary prefill-time probes: the retained cache contains ordinary token-indexed key/value states.

[Method](#method) · [Installation](#installation) · [LongBench evaluation](#longbench-evaluation) · [Implementation and validation](#implementation-and-validation)

## Method

1. **Estimate redundancy.** Split the prompt into semantic units, mean-pool each unit's hidden states, and compute cosine similarities. Average pairwise cosine similarity (ACS) describes redundancy at each layer.
2. **Allocate capacity.** Give each layer a uniform base quota and distribute the remaining historical-token budget according to its priority, `max(1 - ACS, epsilon)^alpha`. The recent window is accounted for separately. With cascading enabled, allocation is updated over the layers observed so far.
3. **Select tokens.** Combine attention importance with a soft semantic penalty, retain the highest-scoring historical tokens, and preserve the recent window.

For unit `i`, let `s_i` be its maximum cosine similarity to a preceding unit. The soft penalty is:

```text
p_i     = clamp((s_i - tau) / (1 - tau), 0, 1)
gamma_i = max(gamma, 1 - lambda * p_i)
score_t = attention_t * gamma_unit(t)
```

The first unit is unpenalized. With `tau=0.75`, `gamma=0.3`, and `lambda=0.7`, penalties vary continuously from 1 to 0.3. The default semantic path compares all preceding units (`full_pairwise`); it has quadratic cost in the number of units.

## Installation

Inference requires **Linux, an NVIDIA CUDA GPU, and FlashAttention 2**. The model patches target **Transformers 4.44.2** and are not compatible with arbitrary newer Transformers versions. Use Python 3.10 or 3.11. CPU execution is intended for utility/cache tests only.

```bash
# Extract the source archive, then enter its top-level directory.
cd anonymous-source
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# CUDA 12.4 example; choose a PyTorch build compatible with your driver/toolkit.
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install packaging ninja
python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip install -e . --no-deps
```

FlashAttention may require a matching CUDA development toolkit and compiler. Download the desired checkpoint beforehand or allow Transformers to load its Hugging Face model ID. Access to gated models must be configured separately.

## LongBench evaluation

This runner targets **LongBench v1**, using the 16 English tasks listed in the script. Obtain the data from [LongBench](https://github.com/THUDM/LongBench/tree/main/LongBench) and extract the JSONL files into one directory, for example:

```text
data/LongBench/
├── qasper.jsonl
├── hotpotqa.jsonl
├── ...
└── repobench-p.jsonl
```

Prompt templates and generation limits are included under `experiments/LongBench/config/`. All commands below run from the repository root. `--model_path` accepts a local checkpoint path or a Hugging Face model ID; if omitted, the model alias selects a default model ID.

### Start with a smoke run

```bash
python experiments/LongBench/pred_sakv.py \
  --model mistral-0.3-7b-32k \
  --data_dir data/LongBench \
  --output_dir pred_result/mistral-smoke \
  --datasets qasper --max_samples 2 --max_length 4096 \
  --compress --cascading --cache_size 1024 --window_size 32 \
  --alpha 0.8 --beta 0.5 --tau 0.75 --gamma 0.3 \
  --semantic_penalty_lambda 0.7 --head_score_mode shared
```

### Run all 16 tasks

```bash
python experiments/LongBench/pred_sakv.py \
  --model mistral-0.3-7b-32k \
  --data_dir data/LongBench \
  --output_dir pred_result/mistral-sakv-1024 \
  --max_length 31500 --device 0 \
  --compress --cascading --cache_size 1024 --window_size 32 \
  --alpha 0.8 --beta 0.5 --tau 0.75 --gamma 0.3 \
  --semantic_penalty_lambda 0.7 --head_score_mode shared

python experiments/LongBench/eval.py \
  --dir_path pred_result/mistral-sakv-1024 --eval_avg
```

For full-cache inference, omit `--compress --cascading` and use a different output directory. Each run writes task JSONL files and `run_config.json`; evaluation writes `result.json`. The runner requires an empty output directory to prevent accidental mixing of models or settings. A smoke run evaluates only its selected examples and is not a full benchmark result.

For LongBench-E, provide the matching `TASK_e.jsonl` files, select the available tasks with `--datasets`, and pass `--e` to both prediction and evaluation. Do not use `--eval_avg` for LongBench-E's length-bucket scores.

### Parameters

| Option | Default | Meaning |
| --- | --- | --- |
| `--cache_size` | `1024` | Average per-layer capacity, including the recent window; individual layer allocations vary |
| `--window_size` | `32` | Recent tokens preserved per layer |
| `--alpha` | `0.8` | Priority exponent, passed explicitly to the allocator |
| `--beta` | `0.5` | Uniform base fraction of the historical-token budget |
| `--tau` / `--sim_threshold` | `0.75` | Semantic similarity threshold |
| `--gamma` / `--redundancy_scale` | `0.3` | Minimum semantic multiplier |
| `--semantic_penalty_lambda` | `0.7` | Strength of the soft semantic penalty |
| `--head_score_mode` | `shared` | Shared token ranking; `kv_group` computes an independent ranking per KV group |
| `--segmentation` | `sentence` | Sentence splitting; `task` enables task-aware semantic units |
| `--max_length` | `31500` | Prompt truncation limit before chat formatting; reduce it for smaller-context models |

The default `shared` score mode uses a common ranking across KV heads. Use `--head_score_mode kv_group` to run the group-specific formulation. These modes are **not interchangeable when comparing results**. Likewise, record segmentation and prompt length when reporting a run.

## Repository layout

```text
sakv/                         Public SAKV imports
cake/                         Core implementation (legacy internal naming)
  cake_cache.py               Cache allocation and eviction
  utils.py                    Segmentation, ACS, and semantic scoring
  model/                      Llama, Mistral, and Qwen2 attention patches
experiments/LongBench/
  config/                     Official LongBench v1 prompts and generation limits
  pred_sakv.py                 Prediction entrypoint
  eval.py                     Evaluation entrypoint
  metrics.py                  LongBench metrics
tests/                        CPU semantic/cache regression tests
docs/VALIDATION.md             Checks performed and validation limitations
```

The `cake` namespace is retained to avoid a broad rename of the implementation derived from CAKE. New user-facing imports use `sakv`, for example `from sakv import SAKVPrefillKVCache`. CAKE baseline names remain distinct from the proposed SAKV method.

## Implementation and validation

The source includes semantic scoring, layer-wise cache allocation, model patches,
and the installation/evaluation entrypoints. Run configurations must explicitly
record `alpha`, head-scoring mode, segmentation, and prompt length. Different
settings can produce different outputs; the source alone does not establish
the configuration or provenance of any previously reported result.

Run the CPU regression suite after installing PyTorch, Transformers, NumPy, and pytest:

```bash
python -m pytest -q
```

GPU generation, FlashAttention compatibility, quality scores, and latency must be
checked in the target inference environment. Model patches are included for
Llama, Mistral, and Qwen2; inclusion alone is not a claim that every checkpoint
has been tested. See [validation notes](docs/VALIDATION.md).

## Acknowledgments

This implementation builds on the CAKE code lineage and Hugging Face Transformers. Evaluation uses [LongBench](https://github.com/THUDM/LongBench); its vendored prompts and generation limits retain the [upstream license](experiments/LongBench/UPSTREAM_LICENSE).
