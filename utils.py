import torch
import numpy as np
import re
import torch
import torch.nn.functional as F
from typing import Tuple
import os


def split_text_into_sentence_spans(text: str):

    if not text:
        return []

    pattern = r'([^。！？\.!\?\n]*[。！？\.!\?\n])'
    matches = re.finditer(pattern, text, flags=re.MULTILINE)

    spans = []
    cursor = 0
    for m in matches:
        span_text = m.group(0)
        start = m.start()
        end = m.end()
        if start > cursor:
            raw = text[cursor:start].strip()
            if raw:
                spans.append((cursor, start))
        if span_text.strip():
            spans.append((start, end))
        cursor = end

    if cursor < len(text):
        tail = text[cursor:].strip()
        if tail:
            spans.append((cursor, len(text)))

    return spans  # list[(start_char, end_char)]

def build_sentence_ids_from_offsets(offset_mapping, sentence_spans, device=None):


    num_tokens = len(offset_mapping)
    sentence_ids = torch.zeros(num_tokens, dtype=torch.long)

    for tok_idx, (s, e) in enumerate(offset_mapping):
        if e == 0 and s == 0:
            sentence_ids[tok_idx] = 0
            continue
        for sent_id, (ss, se) in enumerate(sentence_spans):
            if s >= ss and s < se:
                sentence_ids[tok_idx] = sent_id
                break
        else:
            if tok_idx > 0:
                sentence_ids[tok_idx] = sentence_ids[tok_idx - 1]
            else:
                sentence_ids[tok_idx] = 0

    if device is not None:
        sentence_ids = sentence_ids.to(device)
    return sentence_ids  # shape: (seq_len,)
def compute_layer_prior_score_from_sentences(hidden_states: torch.Tensor,
                                            sentence_ids: torch.Tensor,
                                            min_pairs: int = 1,
                                            return_avg: bool = False) -> float:


    bsz, seq_len, hidden_dim = hidden_states.shape
    assert bsz == 1
    device = hidden_states.device
    sentence_ids = sentence_ids.to(device)

    num_sent = int(sentence_ids.max().item()) + 1
    # [num_sent, hidden_dim]
    sent_embeds = []
    for sid in range(num_sent):
        mask = (sentence_ids == sid)  # [seq_len]
        if not mask.any():
            continue
        # [1, seq_len, hidden_dim] -> [num_tokens_in_sent, hidden_dim]
        hs_sent = hidden_states[0][mask]
        sent_embeds.append(hs_sent.mean(dim=0, keepdim=True))  # [1, hidden_dim]

    if len(sent_embeds) <= 1:

        return 1.0

    sent_embeds = torch.cat(sent_embeds, dim=0)  # [num_sent_effective, hidden_dim]


    sent_embeds = F.normalize(sent_embeds, p=2, dim=-1)
    sim_matrix = torch.matmul(sent_embeds, sent_embeds.T)  # [num_sent, num_sent]


    num = sim_matrix.shape[0]
    mask = torch.ones_like(sim_matrix, dtype=torch.bool)
    mask.fill_(True)
    mask.fill_diagonal_(False)
    sims = sim_matrix[mask]

    if sims.numel() < min_pairs:
        avg_sim = 0.0
    else:
        avg_sim = sims.mean().item()

    prior_score = 1.0 - avg_sim
    if return_avg:
        return float(prior_score), float(avg_sim)
    else:
        return float(prior_score)



class CompressConfig:
    def __init__(self, compress=False, cascading=False, cache_size=1024, window_size=32, hyper=None):
        self.compress = compress
        self.cascading = cascading
        self.cache_size = cache_size
        self.window_size = window_size
        self.hyper = hyper
    
    def __str__(self):
        return f"Config(cache_size={self.cache_size}, window_size={self.window_size}, hyper={self.hyper})"



def adjust_budgets(budget_list, total_budget, seq_len, layer_nums):

    budget_list = np.array(budget_list, dtype=int)
    # Limit the budget of all layers to not exceed seq_len
    excess = np.maximum(budget_list - seq_len, 0)
    budget_list = np.minimum(budget_list, seq_len)

    # Adjust excess budget
    total_excess = np.sum(excess)

    if total_excess > 0:

        valid_indices = budget_list < seq_len
        num_valid = np.sum(valid_indices)

        if num_valid > 0:
            
            distribute_per_layer = total_excess // num_valid
            remainder = total_excess % num_valid

            budget_list[valid_indices] += distribute_per_layer
            budget_list[np.where(valid_indices)[0][:remainder]] += 1

    # Ensure total budget equals total_budget
    current_total_budget = np.sum(budget_list)
    budget_diff = total_budget - current_total_budget

    if budget_diff != 0:
        if budget_diff > 0:
            valid_indices = budget_list < seq_len  
        else:
            valid_indices = budget_list > 1  

        num_valid = np.sum(valid_indices)

        if num_valid > 0:
            adjust_per_layer = abs(budget_diff) // num_valid
            remainder = abs(budget_diff) % num_valid

            if budget_diff > 0:
                budget_list[valid_indices] += adjust_per_layer
                budget_list[np.where(valid_indices)[0][:remainder]] += 1
            else:
                budget_list[valid_indices] -= adjust_per_layer
                budget_list[np.where(valid_indices)[0][:remainder]] -= 1

    return budget_list.tolist()

def compute_sentence_embeddings(
    hidden_states: torch.Tensor,
    sentence_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = hidden_states.device
    bsz, seq_len, hidden_dim = hidden_states.shape

    sentence_ids = sentence_ids.to(device).long()
    num_sent = int(sentence_ids.max().item()) + 1

    sent_emb = hidden_states.new_zeros(bsz, num_sent, hidden_dim)
    counts = hidden_states.new_zeros(bsz, num_sent, 1)

    sid = sentence_ids.unsqueeze(0).expand(bsz, -1)  # [bsz, seq_len]

    index_emb = sid.unsqueeze(-1).expand(-1, -1, hidden_dim)  # [bsz, seq_len, hidden_dim]
    sent_emb.scatter_add_(1, index_emb, hidden_states)
    ones = torch.ones(bsz, seq_len, 1, device=device, dtype=hidden_states.dtype)
    index_cnt = sid.unsqueeze(-1)  # [bsz, seq_len, 1]
    counts.scatter_add_(1, index_cnt, ones)

    sent_emb = sent_emb / counts.clamp_min(1.0)
    return sent_emb, counts



def compute_redundant_sentence_mask(
    sent_emb: torch.Tensor,
    # sim_threshold: float = 0.75,    
    sim_threshold: float,
    return_sim: bool = False,  
) -> torch.Tensor:
    
    device = sent_emb.device
    bsz, num_sent, _ = sent_emb.shape

    if num_sent <= 1:
        return torch.zeros(bsz, num_sent, dtype=torch.bool, device=device)

    sent_norm = F.normalize(sent_emb, dim=-1)                 # [bsz, S, D]
    sim = torch.matmul(sent_norm, sent_norm.transpose(1, 2))  # [bsz, S, S]

    redundant_sent = torch.zeros(bsz, num_sent, dtype=torch.bool, device=device)

    for j in range(1, num_sent):
        redundant_sent[:, j] = (sim[:, :j, j] >= sim_threshold).any(dim=1)

    if return_sim:
        return redundant_sent, sim  

    return redundant_sent


def compute_sa_score_for_prefill(
    tmp_attn_weights: torch.Tensor,
    hidden_states: torch.Tensor,
    sentence_ids: torch.Tensor,
    window_size: int,
    num_key_value_heads: int,  
    sim_threshold: float,
    redundancy_scale: float,
    return_sim: bool = False,    
) -> torch.Tensor:
 
    if not getattr(compute_sa_score_for_prefill, "_printed", False):
        print("sim_threshold =", sim_threshold,"redundancy_scale =", redundancy_scale)
        compute_sa_score_for_prefill._printed = True

    
    device = hidden_states.device
    bsz, seq_len, _ = hidden_states.shape
    _, num_heads, win, seq_len_attn = tmp_attn_weights.shape

    assert seq_len_attn == seq_len, 
    assert win == window_size, 

    past_len = max(seq_len - window_size, 0)
    if past_len == 0:
        return hidden_states.new_zeros(bsz, num_key_value_heads, 0)

    attn_avg = tmp_attn_weights.mean(dim=2).mean(dim=1)     
    attn_avg_past = attn_avg[:, :past_len]                  

    sent_emb, _ = compute_sentence_embeddings(hidden_states, sentence_ids)
    redundant_sent, sim = compute_redundant_sentence_mask(
        sent_emb,
        sim_threshold=sim_threshold,
        return_sim=True, 
    )  # [bsz, num_sent]
  
    sid = sentence_ids.unsqueeze(0).expand(bsz, -1)                   # [bsz, seq_len]
    redundant_token = redundant_sent[torch.arange(bsz, device=device).unsqueeze(1), sid]
    redundant_token_past = redundant_token[:, :past_len]              # [bsz, past_len]

    scale = torch.where(
        redundant_token_past,
        torch.full_like(attn_avg_past, redundancy_scale),
        torch.ones_like(attn_avg_past),
    )
    token_score = attn_avg_past * scale                                # [bsz, past_len]

    sa_score = token_score.unsqueeze(1).expand(-1, num_key_value_heads, -1)

    if return_sim:
        return sa_score, sim       

    return sa_score

    
