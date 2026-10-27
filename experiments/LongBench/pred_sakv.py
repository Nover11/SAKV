import os
import torch
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoConfig, Gemma2ForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse

import torch.distributed as dist
import torch.multiprocessing as mp

import sys
from pathlib import Path

from sakv import SAKVPrefillKVCache as SAKV_pre_KVCache
from sakv.utils import CompressConfig, split_text_into_sentence_spans, split_text_into_semantic_unit_spans, build_sentence_ids_from_offsets



def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="mistral-0.3-7b-32k",
                        choices=["llama2-7b-chat-4k",
                                 "llama2-13b-chat-4k",
                                 "llama3.1-8b-128k",
                                 "mistral-0.3-7b-32k",
                                 "qwen2.5-7b-instruct"])
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--compress', action='store_true', help="Comrpess kv cache with sakv")
    parser.add_argument('--cascading', action='store_true', help="Using cascading cache mangement")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--cache_size', type=int, default=1024)
    parser.add_argument('--window_size', type=int, default=32)

    parser.add_argument('--sim_threshold', '--tau', type=float, default=0.75,
                        help="Similarity threshold for marking sentences as redundant")
    parser.add_argument('--redundancy_scale', '--gamma', type=float, default=0.3,
                        help="Scaling factor for redundant sentence tokens")

    parser.add_argument('--alpha', type=float, default=0.8, help='Exponent for layer importance reweighting')
    parser.add_argument('--beta', type=float, default=0.5, help='Base quota ratio for layer-wise KV allocation')

    parser.add_argument('--model_path', help='Local checkpoint or Hugging Face model ID')
    parser.add_argument('--data_dir', type=Path, required=True, help='LongBench v1 JSONL directory')
    parser.add_argument('--output_dir', type=Path, required=True, help='Empty directory for this run')
    parser.add_argument('--max_length', type=int, default=31500)
    parser.add_argument('--max_samples', type=int)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    parser.add_argument('--semantic_penalty_lambda', type=float, default=0.7)
    parser.add_argument('--head_score_mode', choices=['shared', 'kv_group'], default='shared')
    parser.add_argument('--segmentation', choices=['sentence', 'task'], default='sentence')
    parsed = parser.parse_args(args)
    if not 1 <= parsed.window_size < parsed.cache_size:
        parser.error('require 1 <= window_size < cache_size')
    if not 0 <= parsed.sim_threshold < 1 or not 0 < parsed.redundancy_scale <= 1:
        parser.error('require 0 <= tau < 1 and 0 < gamma <= 1')
    if not 0 <= parsed.beta <= 1 or not 0 < parsed.alpha <= 1 or not 0 <= parsed.semantic_penalty_lambda <= 1:
        parser.error('require 0 <= beta, lambda <= 1 and 0 < alpha <= 1')
    if parsed.max_length <= parsed.window_size or (parsed.max_samples is not None and parsed.max_samples < 1):
        parser.error('max_length must exceed window_size; max_samples must be positive')
    return parsed


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "llama3" in model_name:
        print("======== llama3 build chat ========")
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "mistral" in model_name:
        print("======== mistral build chat ========")
        prompt = f'<s>[INST] {prompt} [/INST]'
    elif "qwen" in model_name:
        print("======== qwen build chat ========")
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    return prompt


@torch.inference_mode()
def get_pred(model, tokenizer, compress, data, max_length, max_gen, prompt_format, dataset, model_name, model2path, out_path):

    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]

        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:  # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)

        encoded = tokenizer(
            prompt,
            truncation=False,
            return_tensors="pt",
            add_special_tokens=False,
            return_offsets_mapping=True
        )
        encoded = encoded.to(device)

        input_ids = encoded.input_ids  # [1, seq_len]
        context_length = input_ids.shape[-1]

        if compress and context_length < args.window_size:
            raise ValueError('Prompt shorter than window_size; reduce --window_size for this dataset')
        if context_length + max_gen > model.config.max_position_embeddings:
            raise ValueError('Prompt plus generation exceeds model context; reduce --max_length')
        sentence_spans = (split_text_into_semantic_unit_spans(prompt, dataset)
                          if args.segmentation == 'task' else split_text_into_sentence_spans(prompt))

        offset_mapping = encoded["offset_mapping"][0].tolist()  # List[(start, end)]
        sentence_ids = build_sentence_ids_from_offsets(
            offset_mapping,
            sentence_spans,
            device=device
        )  # [seq_len]
        for layer in model.model.layers:

            layer.self_attn.config.sentence_ids = sentence_ids


        input = {"input_ids": input_ids, "attention_mask": encoded.attention_mask}

        if dataset == "samsum":
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )[0]

        if compress:
            layers = len(model.model.layers)
            for i in range(layers):
                model.model.layers[i].self_attn.config.prefill = [True]*layers
                model.model.layers[i].self_attn.config.decoding_evict = [None]*layers

        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)

        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(path, model_name, device, compress_config):

    if compress_config.compress:
        if "llama" in model_name:
            from sakv.monkeypatch import replace_flashllama_attn_with_sakv_attn
            replace_flashllama_attn_with_sakv_attn()
        elif "mistral" in model_name:
            from sakv.monkeypatch import replace_flashmistral_attn_with_sakv_attn
            replace_flashmistral_attn_with_sakv_attn()
        elif "qwen2" in model_name:
            from sakv.monkeypatch import replace_flashqwen2_attn_with_sakv_attn
            replace_flashqwen2_attn_with_sakv_attn()

    if "qwen2" in model_name:
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError('A fast tokenizer is required for offset mappings')
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype,
        attn_implementation="flash_attention_2"
    ).to(device)
    config = AutoConfig.from_pretrained(path)
    if hasattr(config, 'num_hidden_layers'):
        layers = config.num_hidden_layers

    if compress_config.compress:
        for i in range(layers):
            model.model.layers[i].self_attn.config.key_size = [compress_config.cache_size - compress_config.window_size]*layers
            model.model.layers[i].self_attn.config.window_size = [compress_config.window_size]*layers
            model.model.layers[i].self_attn.config.prefill = [True]*layers
            model.model.layers[i].self_attn.config.decoding_evict = [None] * layers

            model.model.layers[i].self_attn.config.redundant_sim_threshold = getattr(compress_config, "sim_threshold", 0.75)
            model.model.layers[i].self_attn.config.redundancy_scale = getattr(compress_config, "redundancy_scale", 0.3)
            model.model.layers[i].self_attn.config.prefill_cake_evict = [SAKV_pre_KVCache(
                cache_size=compress_config.cache_size,
                window_size=compress_config.window_size,
                k_seq_dim=2,
                v_seq_dim=2,
                num_heads=model.model.layers[i].self_attn.num_heads,
                num_layers=layers,
                use_cascading=compress_config.cascading
            )]*layers
            cfg = model.model.layers[i].self_attn.config
            cfg.prefill_cake_evict[i].alpha = compress_config.alpha
            cfg.prefill_cake_evict[i].beta = compress_config.beta
            cfg.semantic_penalty_lambda = compress_config.semantic_penalty_lambda
            cfg.sakv_head_score_mode = compress_config.head_score_mode
            cfg.redundancy_mode = 'full_pairwise'

    model = model.eval()

    return model, tokenizer


def _extract_layer_sentence_sim_stats(cfg):
    stats = {}
    for i, (s, c) in enumerate(zip(cfg.layer_sim_sum, cfg.layer_sim_cnt)):
        if c > 0:
            avg = s / c
        else:
            avg = 0.0
        stats[f"layer_{i}"] = {
            "avg_sim": float(avg),
            "count": int(c),
        }
    return stats


def _reset_layer_sentence_sim_stats_inplace(cfg):
    # layer_sim_sum
    if torch.is_tensor(cfg.layer_sim_sum):
        cfg.layer_sim_sum.zero_()
    else:
        # assume list-like
        cfg.layer_sim_sum = [0.0 for _ in range(len(cfg.layer_sim_sum))]

    # layer_sim_cnt
    if torch.is_tensor(cfg.layer_sim_cnt):
        cfg.layer_sim_cnt.zero_()
    else:
        cfg.layer_sim_cnt = [0 for _ in range(len(cfg.layer_sim_cnt))]


DATASETS = ["qasper", "2wikimqa", "qmsum", "triviaqa", "passage_retrieval_en", "lcc",
            "gov_report", "multifieldqa_en", "multi_news", "trec", "narrativeqa", "samsum",
            "passage_count", "hotpotqa", "musique", "repobench-p"]
MODEL_PATHS = {
    "mistral-0.3-7b-32k": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama3.1-8b-128k": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama2-7b-chat-4k": "meta-llama/Llama-2-7b-chat-hf",
    "llama2-13b-chat-4k": "meta-llama/Llama-2-13b-chat-hf",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
}

if __name__ == '__main__':
    args = parse_args()
    import transformers
    if transformers.__version__ != '4.44.2':
        raise RuntimeError('Attention patches require transformers==4.44.2')
    if not torch.cuda.is_available():
        raise RuntimeError('Inference requires CUDA and FlashAttention 2; CPU is supported for unit tests only')
    data_paths = {task: args.data_dir / f"{task}{'_e' if args.e else ''}.jsonl" for task in args.datasets}
    for path in data_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('output_dir must be empty; choose a new directory to avoid mixing runs')
    config_dir = Path(__file__).resolve().parent / 'config'
    dataset2prompt = json.loads((config_dir / 'dataset2prompt.json').read_text(encoding='utf-8'))
    dataset2maxlen = json.loads((config_dir / 'dataset2maxlen.json').read_text(encoding='utf-8'))
    seed_everything(42)
    torch.cuda.set_device(args.device)
    device = torch.device(f'cuda:{args.device}')
    compress_config = CompressConfig(args.compress, args.cascading)
    for key in ('cache_size', 'window_size', 'sim_threshold', 'redundancy_scale', 'alpha', 'beta',
                'semantic_penalty_lambda', 'head_score_mode'):
        setattr(compress_config, key, getattr(args, key))
    model_path = args.model_path or MODEL_PATHS[args.model]
    model, tokenizer = load_model_and_tokenizer(model_path, args.model, device, compress_config)
    expected_type = 'llama' if args.model.startswith('llama') else 'mistral' if args.model.startswith('mistral') else 'qwen2'
    if model.config.model_type != expected_type:
        raise ValueError('--model and checkpoint architecture do not match')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    manifest.update(torch_version=torch.__version__, transformers_version=transformers.__version__, seed=42,
                    resolved_model=model_path, model_commit=getattr(model.config, '_commit_hash', None))
    (args.output_dir / 'run_config.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    for dataset, data_path in data_paths.items():
        with data_path.open(encoding='utf-8') as source:
            data = [json.loads(line) for line in source if line.strip()]
        if args.max_samples is not None:
            data = data[:args.max_samples]
        out_path = args.output_dir / f'{dataset}.jsonl'
        get_pred(model, tokenizer, args.compress, data, args.max_length, dataset2maxlen[dataset],
                 dataset2prompt[dataset], dataset, args.model, MODEL_PATHS, out_path)
