# Validation

- CPU regression suite: 30 tests passed.
- Python source syntax checks: passed.
- Editable package installation and prediction CLI help: passed.
- Tested environment: Python 3.10, PyTorch 2.6.0+cpu, Transformers 4.44.2,
  NumPy 2.2.6, pytest 8.4.2.

The tests cover semantic similarity, soft penalties, token-to-unit mapping,
shared and KV-group scoring, repeated cache eviction, recent-window retention,
bounded allocation, zero historical quota, CLI parameters, configuration
coverage, and evaluation input handling.

CUDA inference, FlashAttention installation, full LongBench evaluation,
latency, and peak memory have not been validated in this CPU environment.
The optional KV-group paths on Mistral and Qwen2 still require end-to-end GPU
checks. Inference is intended for one GPU and batch size one. Full-pairwise
semantic analysis has quadratic cost in the number of semantic units.

This source artifact contains no benchmark-result tables. Test success does
not establish the provenance or reproducibility of previously reported scores.

Third-party dependency links and the vendored LongBench license identify
upstream projects, not the contributors to this artifact.
