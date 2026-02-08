import os
from datasets import load_dataset
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

from sakv.sakv_cache import SAKV_pre_KVCache
from sakv.utils import CompressConfig, split_text_into_sentence_spans, build_sentence_ids_from_offsets



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
    parser.add_argument('--pred_name', type=str, default="pred", help="Pred Output Name")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--cache_size', type=int, default=1024)
    parser.add_argument('--window_size', type=int, default=32)

    parser.add_argument('--sim_threshold', type=float, default=0.75,
                        help="Similarity threshold for marking sentences as redundant")
    parser.add_argument('--redundancy_scale', type=float, default=0.3,
                        help="Scaling factor for redundant sentence tokens")

    parser.add_argument('--alpha', type=float, default=0.8, help='Exponent for layer importance reweighting')
    parser.add_argument('--beta', type=float, default=0.5, help='Base quota ratio for layer-wise KV allocation')

    return parser.parse_args(args)


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
            return_offsets_mapping=True
        )
        encoded = encoded.to(device)

        input_ids = encoded.input_ids  # [1, seq_len]
        context_length = input_ids.shape[-1]

        sentence_spans = split_text_into_sentence_spans(prompt)

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
            from sakv.monkeypatch import replace_flashllama_attn_with_sakvattn
            replace_flashllama_attn_with_sakvattn()
        elif "mistral" in model_name:
            from sakv.monkeypatch import replace_flashmistral_attn_with_sakeattn
            replace_flashmistral_attn_with_sakeattn()
        elif "qwen2" in model_name:
            from sakv.monkeypatch import replace_flashqwen2_attn_with_sakeattn
            replace_flashqwen2_attn_with_sakeattn()

    if "qwen2" in model_name:
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(path)
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
            model.model.layers[i].self_attn.config.prefill_sakv_evict = [SAKV_pre_KVCache(
                cache_size=compress_config.cache_size,
                window_size=compress_config.window_size,
                k_seq_dim=2,
                v_seq_dim=2,
                num_heads=model.model.layers[i].self_attn.num_heads,
                num_layers=layers,
                use_cascading=compress_config.cascading
            )]*layers

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


if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    pred_name = args.pred_name
    model_name = args.model
    compress = args.compress
    cascading = args.cascading
    compress_config = CompressConfig(compress, cascading)
    model2path = json.load(open("/root/autodl-tmp/sakv/experiments/LongBench/config/model2path.json", "r"))
    model2maxlen = json.load(open("/root/autodl-tmp/sakv/experiments/LongBench/config/model2maxlen.json", "r"))
    # define your model
    max_length = model2maxlen[model_name]
    if compress:
        compress_config.cache_size = args.cache_size
        compress_config.window_size = args.window_size
        cache_name = f"cache{args.cache_size}"

        compress_config.sim_threshold = args.sim_threshold
        compress_config.redundancy_scale = args.redundancy_scale

        compress_config.alpha = args.alpha
        compress_config.beta = args.beta

    else:
        cache_name = "cachefull"

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device, compress_config)

    datasets = ["qasper", "2wikimqa", "qmsum","triviaqa", "passage_retrieval_en","lcc",
                "gov_report","multifieldqa_en", "multi_news", "trec",  "narrativeqa", "samsum",
                "passage_count", "hotpotqa", "musique", "repobench-p"]

    dataset2prompt = json.load(open("/root/autodl-tmp/sakv/experiments/LongBench/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("/root/autodl-tmp/sakv/experiments/LongBench/config/dataset2maxlen.json", "r"))
    if not os.path.exists(f"./pred_result/{cache_name}/{pred_name}"):
        os.makedirs(f"./pred_result/{cache_name}/{pred_name}")

    per_dataset_layer_sentence_sim = {}  # dataset -> stats dict

    for dataset in datasets:
        # load offline
        data_files = {"test": f"{dataset}.jsonl"}
        data = load_dataset("json", data_dir='/root/autodl-tmp/datasets/LongBench', split='test', data_files=data_files)

        if not os.path.exists(f"./pred_result/{cache_name}/{pred_name}/{model_name}"):
            os.makedirs(f"./pred_result/{cache_name}/{pred_name}/{model_name}")
        out_path = f"./pred_result/{cache_name}/{pred_name}/{model_name}/{dataset}.jsonl"

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]

        if os.path.exists(out_path):
            with open(out_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            if len(data_all) == len(lines):
                continue
            else:
                data_all = data_all[len(lines):]

        if compress:
            cfg = model.model.layers[0].self_attn.config
            if hasattr(cfg, "layer_sim_sum") and hasattr(cfg, "layer_sim_cnt"):
                _reset_layer_sentence_sim_stats_inplace(cfg)

        get_pred(model, tokenizer, compress, data_all, max_length,
                 max_gen, prompt_format, dataset, model_name, model2path, out_path)

        if compress:
            cfg = model.model.layers[0].self_attn.config
            if hasattr(cfg, "layer_sim_sum") and hasattr(cfg, "layer_sim_cnt"):
                stats = _extract_layer_sentence_sim_stats(cfg)
                per_dataset_layer_sentence_sim[dataset] = stats

                out_dir = f"./pred_result/{cache_name}/{pred_name}/{model_name}"
                os.makedirs(out_dir, exist_ok=True)
                ds_out_path = os.path.join(out_dir, f"layer_sentence_sim_{model_name}_{dataset}.json")
                with open(ds_out_path, "w", encoding="utf-8") as f:
                    json.dump(stats, f, ensure_ascii=False, indent=2)
                print(f"[INFO] 已将 {dataset} 的各层平均句子相似度写入: {ds_out_path}")

    if compress:
        out_dir = f"./pred_result/{cache_name}/{pred_name}/{model_name}"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"layer_sentence_sim_by_dataset_{model_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(per_dataset_layer_sentence_sim, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 已将按数据集汇总的各层平均句子相似度写入: {out_path}")
