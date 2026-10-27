"""Regression checks for the public runner, model patches, and evaluator."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def runner():
    spec = importlib.util.spec_from_file_location('pred_sakv', ROOT / 'experiments/LongBench/pred_sakv.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_model_patches_import():
    import sakv.monkeypatch
    assert callable(sakv.monkeypatch.replace_flashmistral_attn_with_sakv_attn)


def test_cli_aliases_and_configs():
    module = runner()
    args = module.parse_args(['--data_dir', 'data', '--output_dir', 'result', '--tau', '.8', '--gamma', '.4'])
    assert args.sim_threshold == .8
    assert args.redundancy_scale == .4
    for name in ['dataset2prompt.json', 'dataset2maxlen.json']:
        config = json.loads((ROOT / 'experiments/LongBench/config' / name).read_text(encoding='utf-8'))
        assert set(module.DATASETS) <= set(config)


def test_invalid_cache_budget_rejected():
    with pytest.raises(SystemExit):
        runner().parse_args(['--data_dir', 'data', '--output_dir', 'result', '--cache_size', '32'])


def test_allocator_receives_cli_parameters(monkeypatch):
    module = runner()
    import sakv.monkeypatch as patches
    cfg = SimpleNamespace(num_hidden_layers=2)
    layers = [SimpleNamespace(self_attn=SimpleNamespace(config=cfg, num_heads=4)) for _ in range(2)]
    model = SimpleNamespace(model=SimpleNamespace(layers=layers))
    model.to = lambda device: model
    model.eval = lambda: model
    monkeypatch.setattr(patches, 'replace_flashmistral_attn_with_sakv_attn', lambda: None)
    monkeypatch.setattr(module.AutoTokenizer, 'from_pretrained', lambda *a, **k: SimpleNamespace(is_fast=True))
    monkeypatch.setattr(module.AutoModelForCausalLM, 'from_pretrained', lambda *a, **k: model)
    monkeypatch.setattr(module.AutoConfig, 'from_pretrained', lambda *a, **k: cfg)
    config = SimpleNamespace(compress=True, cascading=True, cache_size=64, window_size=8,
                             sim_threshold=.8, redundancy_scale=.4, alpha=.6, beta=.2,
                             semantic_penalty_lambda=.5, head_score_mode='kv_group')
    module.load_model_and_tokenizer('dummy', 'mistral-0.3-7b-32k', 'cpu', config)
    for allocator in cfg.prefill_cake_evict:
        assert allocator.alpha == .6 and allocator.beta == .2
    assert cfg.sakv_head_score_mode == 'kv_group'
    assert cfg.semantic_penalty_lambda == .5


def test_evaluator_scores_predictions(tmp_path):
    record = dict(pred='The answer is 3', answers=['3'], all_classes=[], length=100)
    (tmp_path / 'passage_count.jsonl').write_text(json.dumps(record) + '\n')
    proc = subprocess.run([sys.executable, str(ROOT / 'experiments/LongBench/eval.py'),
                           '--dir_path', str(tmp_path)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert json.loads((tmp_path / 'result.json').read_text())['passage_count'] == 100


def test_evaluator_rejects_empty_directory(tmp_path):
    proc = subprocess.run([sys.executable, str(ROOT / 'experiments/LongBench/eval.py'),
                           '--dir_path', str(tmp_path)], capture_output=True, text=True)
    assert proc.returncode != 0
    assert not (tmp_path / 'result.json').exists()
