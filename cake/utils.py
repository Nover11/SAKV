import torch
import numpy as np
###########切分句子新增库###########
import re
import time
import torch
import torch.nn.functional as F
from typing import Tuple
import os
#################################


#########新增基于字符的切分#########
# 1) 基于字符的切句：返回句子在原始字符串里的 char 范围
def split_text_into_sentence_spans(text: str):
    """
    按中英文句号/问号/感叹号/换行符切分，返回每个句子的 (start_char, end_char) 区间。
    - 中文：。！？ 
    - 英文：. ? ! 
    - 换行：\n
    """
    if not text:
        return []

    # 保留分隔符本身，这样句号所在的字符也算在句子里
    pattern = r'([^。！？\.!\?\n]*[。！？\.!\?\n])'
    matches = re.finditer(pattern, text, flags=re.MULTILINE)

    spans = []
    cursor = 0
    for m in matches:
        span_text = m.group(0)
        start = m.start()
        end = m.end()
        # 前面可能有没被匹配到的残余内容（比如最后一句没标点）
        if start > cursor:
            raw = text[cursor:start].strip()
            if raw:
                spans.append((cursor, start))
        # 当前匹配
        if span_text.strip():
            spans.append((start, end))
        cursor = end

    # 结尾还有残余
    if cursor < len(text):
        tail = text[cursor:].strip()
        if tail:
            spans.append((cursor, len(text)))

    return spans  # list[(start_char, end_char)]
#########新增基于字符的切分#########


def _nonempty_spans(text: str, spans):
    return [(start, end) for start, end in spans if end > start and text[start:end].strip()]


def _split_by_start_markers(text: str, marker_pattern: str):
    matches = list(re.finditer(marker_pattern, text, flags=re.MULTILINE))
    if not matches:
        return []

    spans = []
    first_start = matches[0].start()
    if text[:first_start].strip():
        spans.append((0, first_start))
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        spans.append((start, end))
    return _nonempty_spans(text, spans)


def _split_qmsum_turn_spans(text: str):
    # AMI-style transcripts use one speaker turn per line, e.g. "Project Manager: ...".
    marker = r"^[A-Za-z][A-Za-z ._'\-]{0,48}:\s"
    return _split_by_start_markers(text, marker)


def _split_code_unit_spans(text: str):
    if not text:
        return []

    lines = text.splitlines(keepends=True)
    spans = []
    cursor = 0
    block_start = None
    prev_indent = 0
    prev_stripped = ""

    def flush(end_pos):
        nonlocal block_start
        if block_start is not None and text[block_start:end_pos].strip():
            spans.append((block_start, end_pos))
        block_start = None

    openers = (
        "class ",
        "interface ",
        "enum ",
        "struct ",
        "def ",
        "async def ",
        "function ",
        "public ",
        "private ",
        "protected ",
        "static ",
        "if ",
        "for ",
        "while ",
        "switch ",
        "try",
        "catch",
        "except",
        "finally",
    )

    for line in lines:
        line_start = cursor
        line_end = cursor + len(line)
        stripped = line.strip()
        cursor = line_end

        if not stripped:
            flush(line_start)
            prev_indent = 0
            prev_stripped = ""
            continue

        indent = len(line) - len(line.lstrip(" \t"))
        starts_block = (
            block_start is None
            or indent < prev_indent
            or stripped.startswith(openers)
            or prev_stripped.endswith("{")
            or prev_stripped.endswith(":")
            or prev_stripped == "}"
        )

        if starts_block:
            flush(line_start)
            block_start = line_start

        # Keep simple statements/imports as complete code units. This avoids
        # natural-language period splitting inside namespaces/package names.
        if stripped.endswith(";") and block_start is not None:
            flush(line_end)

        prev_indent = indent
        prev_stripped = stripped

    flush(len(text))
    return _nonempty_spans(text, spans)


def split_text_into_semantic_unit_spans(text: str, dataset: str = ""):
    """
    Build task-aware semantic-unit spans for LongBench prompts.

    This does not change the prompt. It only changes the internal unit IDs used
    by SAKV semantic redundancy scoring.
    """
    if not text:
        return []

    dataset = (dataset or "").lower()
    spans = []

    if dataset in {"hotpotqa", "2wikimqa", "musique"}:
        spans = _split_by_start_markers(text, r"^Passage\s+\d+:\s*$")
    elif dataset in {"passage_count", "passage_retrieval_en"}:
        spans = _split_by_start_markers(text, r"^Paragraph\s+\d+:\s*")
    elif dataset == "multi_news":
        spans = _split_by_start_markers(text, r"^Passage\s+\d+:\s*$")
    elif dataset == "qmsum":
        spans = _split_qmsum_turn_spans(text)
    elif dataset == "trec":
        spans = _split_by_start_markers(text, r"^Question:\s")
    elif dataset == "triviaqa":
        spans = _split_by_start_markers(text, r"^Passage:\s")
    elif dataset == "samsum":
        spans = _split_by_start_markers(text, r"^Dialogue:\s")
    elif dataset in {"lcc", "repobench-p"}:
        spans = _split_code_unit_spans(text)

    if spans:
        return spans
    return split_text_into_sentence_spans(text)


#########新增用 tokenizer 的 offset 映射，把 token 分配到句子（得到 sentence_id）#########
def build_sentence_ids_from_offsets(offset_mapping, sentence_spans, device=None):
    """
    根据 tokenizer 的 offset_mapping 和句子 char span，给每个 token 一个 sentence_id。
    - offset_mapping: List[(start_char, end_char)]，来自 tokenizer(..., return_offsets_mapping=True)
    - sentence_spans: List[(sent_start_char, sent_end_char)]
    """
    # 假定 batch_size=1 的场景（LongBench 里确实是单条样本循环处理）
    num_tokens = len(offset_mapping)
    sentence_ids = torch.zeros(num_tokens, dtype=torch.long)

    if not sentence_spans:
        if device is not None:
            sentence_ids = sentence_ids.to(device)
        return sentence_ids

    # offset_mapping 和 sentence_spans 都按文本顺序排列。用单指针扫描，
    # 避免长上下文下 token 数 × 句子数 的双重循环。
    sent_id = 0
    last_valid_sent_id = 0
    num_sentences = len(sentence_spans)

    for tok_idx, (s, e) in enumerate(offset_mapping):
        if e == 0 and s == 0:
            # special token（如 BOS、EOS），先给它 sentence_id=0
            sentence_ids[tok_idx] = 0
            continue

        # 找这个 token 落在哪个句子 span 里（按起始 char 对齐即可）
        while sent_id + 1 < num_sentences and s >= sentence_spans[sent_id][1]:
            sent_id += 1

        ss, se = sentence_spans[sent_id]
        if s >= ss and s < se:
            sentence_ids[tok_idx] = sent_id
            last_valid_sent_id = sent_id
        else:
            # 没匹配到就归到前一个句子，或者 0
            if tok_idx > 0:
                sentence_ids[tok_idx] = last_valid_sent_id
            else:
                sentence_ids[tok_idx] = 0

    if device is not None:
        sentence_ids = sentence_ids.to(device)
    return sentence_ids  # shape: (seq_len,)
#########新增用 tokenizer 的 offset 映射，把 token 分配到句子（得到 sentence_id）#########

#########层级句子相似度 → 层重要程度#########
def compute_layer_pref_score_from_sentences(hidden_states: torch.Tensor,
                                            sentence_ids: torch.Tensor,
                                            min_pairs: int = 1,
                                            return_avg: bool = False) -> float:
    """
    给定当前层的 hidden_states（[bsz, seq_len, hidden_dim]）和 sentence_ids（[seq_len]），
    计算这一层的句子平均相似度，然后返回 1 - avg_sim 作为该层的 pref_score。
    只支持 bsz=1 的场景（LongBench 下 OK）。
    """
    # bsz, seq_len, hidden_dim
    bsz, seq_len, hidden_dim = hidden_states.shape
    assert bsz == 1, "目前实现假定 batch_size=1（LongBench 场景）"

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
        # 只有一个句子的话，就认为相似度为 0，pref_score = 1
        if return_avg:
            return 1.0, 0.0
        return 1.0

    sent_embeds = torch.cat(sent_embeds, dim=0)  # [num_sent_effective, hidden_dim]

    # 归一化后用 ACS 公式计算平均余弦相似度，避免构造完整 S x S similarity matrix。
    sent_embeds = F.normalize(sent_embeds, p=2, dim=-1)
    avg_sim = float(compute_acs_from_normalized_sentences(sent_embeds, min_pairs=min_pairs).item())

    pref_score = 1.0 - avg_sim
    if return_avg:
        return float(pref_score), float(avg_sim)
    else:
        return float(pref_score)

#########层级句子相似度 → 层重要程度#########


class CompressConfig:
    def __init__(self, compress=False, cascading=False, cache_size=1024, window_size=32, hyper=None):
        self.compress = compress
        self.cascading = cascading
        self.cache_size = cache_size
        self.window_size = window_size
        self.hyper = hyper
    
    def __str__(self):
        return f"Config(cache_size={self.cache_size}, window_size={self.window_size}, hyper={self.hyper})"

#计算给定注意力权重的熵
# def calculate_entropy(attention_scores):
#     attention_scores = attention_scores.to(torch.float32)
#     entropy = -torch.sum(attention_scores * torch.log(attention_scores + 1e-10))  
#     entropy= entropy.to(dtype=torch.float32)
#     return entropy

def adjust_budgets(budget_list, total_budget, seq_len, layer_nums):
    """Round and redistribute quotas within each observed layer's capacity.

    During cascading, fewer than layer_nums layers may have been observed.
    An infeasible total is therefore capped by the observed layers' capacity.
    """
    capacity = max(0, int(seq_len))
    budgets = np.clip(np.asarray(budget_list, dtype=np.int64), 0, capacity)
    target = min(max(0, int(total_budget)), len(budgets) * capacity)
    while int(budgets.sum()) != target:
        difference = target - int(budgets.sum())
        room = capacity - budgets if difference > 0 else budgets
        active = np.flatnonzero(room > 0)
        share, remainder = divmod(abs(difference), len(active))
        changes = np.minimum(room[active], share + (np.arange(len(active)) < remainder))
        budgets[active] += changes if difference > 0 else -changes
    return budgets.tolist()


def compute_sentence_embeddings(
    hidden_states: torch.Tensor,
    sentence_ids: torch.Tensor,
    num_sent: int = None,
    sentence_counts: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用当前层的 hidden_states 和 sentence_ids 计算句子级表示。
    这一层算出来的句子向量只用于这一层的冗余判断。

    Args:
        hidden_states: [bsz, seq_len, hidden_dim]
        sentence_ids : [seq_len]，同一个 batch 内 token 共享同一套 sentence_ids

    Returns:
        sent_emb : [bsz, num_sent, hidden_dim]
        counts   : [bsz, num_sent, 1] 每个句子的 token 数，用于调试/归一化
    """
    device = hidden_states.device
    bsz, seq_len, hidden_dim = hidden_states.shape

    sentence_ids = sentence_ids.to(device).long()
    if num_sent is None:
        num_sent = int(sentence_ids.max().item()) + 1

    sent_emb = hidden_states.new_zeros(bsz, num_sent, hidden_dim)

    if bsz == 1:
        # LongBench 当前是单样本推理。直接用 index_add_ 按句子编号聚合，
        # 避免构造 [bsz, seq_len, hidden_dim] 级别的 scatter index。
        sent_emb[0].index_add_(0, sentence_ids, hidden_states[0])
        sid = sentence_ids.unsqueeze(0)
    else:
        # 把 sentence_ids 扩展到 batch 维度
        sid = sentence_ids.unsqueeze(0).expand(bsz, -1)  # [bsz, seq_len]

        # 累加每个句子的 token 向量
        index_emb = sid.unsqueeze(-1).expand(-1, -1, hidden_dim)  # [bsz, seq_len, hidden_dim]
        sent_emb.scatter_add_(1, index_emb, hidden_states)

    if sentence_counts is None:
        # 统计每个句子 token 数。counts 只和 sentence_ids 有关，调用方可缓存后传入。
        counts = hidden_states.new_zeros(bsz, num_sent, 1)
        if bsz == 1:
            counts[0, :, 0].index_add_(0, sentence_ids, hidden_states.new_ones(seq_len))
        else:
            ones = torch.ones(bsz, seq_len, 1, device=device, dtype=hidden_states.dtype)
            index_cnt = sid.unsqueeze(-1)  # [bsz, seq_len, 1]
            counts.scatter_add_(1, index_cnt, ones)
    else:
        counts = sentence_counts.to(device=device, dtype=hidden_states.dtype)
        if counts.dim() == 1:
            counts = counts.view(1, -1, 1)
        elif counts.dim() == 2:
            counts = counts.unsqueeze(-1)
        if counts.shape[0] == 1 and bsz > 1:
            counts = counts.expand(bsz, -1, -1)

    sent_emb = sent_emb / counts.clamp_min(1.0)
    return sent_emb, counts


##########新增功能函数############
def _cuda_sync_for_profile(tensor_or_device):
    device = tensor_or_device.device if hasattr(tensor_or_device, "device") else tensor_or_device
    if torch.cuda.is_available() and getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)


def _profile_start(enabled: bool, tensor_or_device):
    if enabled:
        _cuda_sync_for_profile(tensor_or_device)
    return time.perf_counter()


def _profile_end(start_time: float, enabled: bool, tensor_or_device) -> float:
    if enabled:
        _cuda_sync_for_profile(tensor_or_device)
    return time.perf_counter() - start_time


def _normalize_sentence_embeddings(sent_emb: torch.Tensor) -> torch.Tensor:
    # fp16 下默认 eps=1e-12 会下溢，空句/空洞句子的 0 向量归一化可能产生 NaN。
    sent_norm = F.normalize(sent_emb, p=2, dim=-1, eps=1e-6)
    return torch.nan_to_num(sent_norm, nan=0.0, posinf=0.0, neginf=0.0)


def compute_acs_from_normalized_sentences(
    sent_norm: torch.Tensor,
    valid_mask: torch.Tensor = None,
    min_pairs: int = 1,
) -> torch.Tensor:
    """
    用等价公式计算 ACS，避免构造完整 S x S similarity matrix。

    假设 z_i 已经 L2 normalize：
        ACS = (||sum_i z_i||^2 - S) / (S * (S - 1))

    复杂度从 full pairwise 的 O(S^2 * D) 降为 O(S * D)。
    sent_norm 支持 [S, D] 或 [B, S, D]。
    """
    squeeze = sent_norm.dim() == 2
    if squeeze:
        sent_norm = sent_norm.unsqueeze(0)

    bsz, num_sent, _ = sent_norm.shape
    if valid_mask is None:
        valid_mask = torch.ones(bsz, num_sent, dtype=torch.bool, device=sent_norm.device)
    elif valid_mask.dim() == 1:
        valid_mask = valid_mask.unsqueeze(0)
    valid_mask = valid_mask.to(sent_norm.device).bool()

    # fp16 的最大有限值只有 65504，长上下文下 S * (S - 1) 很容易溢出；
    # 分母和最终范数固定用 fp32，避免 pref_score 被错误压到 1。
    mask = valid_mask.to(dtype=sent_norm.dtype).unsqueeze(-1)
    sent_count = valid_mask.sum(dim=-1).to(dtype=torch.float32)
    pair_count = sent_count * (sent_count - 1.0)
    summed = (sent_norm * mask).sum(dim=1).to(dtype=torch.float32)
    raw_acs = (summed.pow(2).sum(dim=-1) - sent_count) / pair_count.clamp_min(1.0)
    has_pairs = (sent_count > 1.0) & (pair_count >= float(min_pairs))
    acs = torch.where(has_pairs, raw_acs, torch.zeros_like(raw_acs))
    return acs[0] if squeeze else acs


def compute_cake_attention_prefill_scores(
    tmp_attn_weights: torch.Tensor,
    window_size: int,
    num_key_value_heads: int,
    num_key_value_groups: int,
    gamma: float = 200.0,
    tau1: float = 1.0,
    tau2: float = 1.0,
) -> Tuple[float, torch.Tensor]:
    """
    Original CAKE-style attention-only prefill scoring baseline.

    This does not use sentence embeddings or semantic redundancy. It is used by
    the CAKE baseline branch while SAKV remains the default semantic path.
    """
    past_attn = tmp_attn_weights[:, :, -window_size:, :-window_size]
    if past_attn.shape[-1] == 0:
        pref_score = 1.0
    else:
        disp = torch.mean(past_attn, dim=-2).sum()
        var = torch.var(past_attn, dim=-2).sum()
        pref_score = float(
            (
                disp.clamp_min(1e-12).pow(1.0 / tau1)
                * var.clamp_min(1e-12).pow(1.0 / tau2)
            ).item()
        )

    attention_score = tmp_attn_weights[:, :, -window_size:, :]
    attn_mean = attention_score.mean(dim=-2)
    attn_var = attention_score.var(dim=-2)
    attn_cache = attn_mean + gamma * attn_var
    attn_cache = attn_cache[:, :, :-window_size]
    if attn_cache.shape[-1] == 0:
        return pref_score, tmp_attn_weights.new_zeros(
            tmp_attn_weights.shape[0],
            num_key_value_heads,
            0,
        )

    attn_cache = F.avg_pool1d(attn_cache, kernel_size=5, padding=5 // 2, stride=1)
    attn_cache = attn_cache.reshape(
        attn_cache.shape[0],
        num_key_value_heads,
        num_key_value_groups,
        -1,
    )
    return pref_score, attn_cache.mean(dim=-2)


def compute_full_pairwise_acs_for_test(
    sent_norm: torch.Tensor,
    valid_mask: torch.Tensor = None,
    min_pairs: int = 1,
) -> torch.Tensor:
    """
    full pairwise ACS 实现，用于 SAKV full 场景和公式实现的数值对照。
    """
    squeeze = sent_norm.dim() == 2
    if squeeze:
        sent_norm = sent_norm.unsqueeze(0)

    bsz, num_sent, _ = sent_norm.shape
    if valid_mask is None:
        valid_mask = torch.ones(bsz, num_sent, dtype=torch.bool, device=sent_norm.device)
    elif valid_mask.dim() == 1:
        valid_mask = valid_mask.unsqueeze(0)
    valid_mask = valid_mask.to(sent_norm.device).bool()

    acs_values = []
    for batch_idx in range(bsz):
        cur = sent_norm[batch_idx][valid_mask[batch_idx]]
        sent_count = cur.shape[0]
        pair_count = sent_count * (sent_count - 1)
        if sent_count <= 1 or pair_count < min_pairs:
            acs_values.append(sent_norm.new_tensor(0.0))
            continue
        sim_matrix = torch.matmul(cur, cur.T)
        mask = ~torch.eye(sent_count, dtype=torch.bool, device=sent_norm.device)
        acs_values.append(sim_matrix[mask].mean())

    acs = torch.stack(acs_values)
    return acs[0] if squeeze else acs


def compute_full_pairwise_sentence_stats(
    sent_norm: torch.Tensor,
    sim_threshold: float,
    gamma_min: float,
    lambda_penalty: float = 0.7,
    valid_mask: torch.Tensor = None,
    previous_pair_mask: torch.Tensor = None,
    valid_pair_mask: torch.Tensor = None,
    min_pairs: int = 1,
    return_sim_matrix: bool = False,
) -> dict:
    """
    SAKV full sentence-pair path: construct one full S x S similarity matrix
    and reuse it for both ACS/Pri and redundant sentence detection.

    这是当前正式 full 场景使用的实现。相比先算 ACS 再算冗余句，这里避免
    对同一层 sentence embeddings 做两次 full pairwise matmul。
    """
    device = sent_norm.device
    dtype = sent_norm.dtype
    bsz, num_sent, _ = sent_norm.shape
    sent_norm = torch.nan_to_num(sent_norm, nan=0.0, posinf=0.0, neginf=0.0)

    if valid_mask is None:
        valid_mask = torch.ones(bsz, num_sent, dtype=torch.bool, device=device)
    elif valid_mask.dim() == 1:
        valid_mask = valid_mask.unsqueeze(0)
    valid_mask = valid_mask.to(device).bool()

    gamma_min = float(max(0.0, min(1.0, gamma_min)))
    lambda_penalty = float(max(0.0, lambda_penalty))
    tau = float(sim_threshold)
    denom = max(1.0 - tau, 1e-6)

    redundant_sent = torch.zeros(bsz, num_sent, dtype=torch.bool, device=device)
    redundancy_score = torch.zeros(bsz, num_sent, dtype=dtype, device=device)
    sentence_gamma = torch.ones(bsz, num_sent, dtype=dtype, device=device)

    if num_sent <= 1:
        avg_sim = sent_norm.new_zeros(bsz)
        result = {
            "avg_sim": avg_sim,
            "redundant_sentence_mask": redundant_sent,
            "sentence_redundancy_score": redundancy_score,
            "sentence_gamma": sentence_gamma,
        }
        if return_sim_matrix:
            result["sim_matrix"] = sent_norm.new_zeros(bsz, num_sent, num_sent)
        return result

    sim_matrix = torch.matmul(sent_norm, sent_norm.transpose(-1, -2))

    if valid_pair_mask is not None:
        valid_pair = valid_pair_mask.to(device=device).bool()
        if valid_pair.dim() == 2:
            valid_pair = valid_pair.unsqueeze(0)
    else:
        if previous_pair_mask is None:
            previous_mask = torch.tril(
                torch.ones(num_sent, num_sent, dtype=torch.bool, device=device),
                diagonal=-1,
            )
        else:
            previous_mask = previous_pair_mask.to(device=device).bool()
        valid_pair = (
            previous_mask.unsqueeze(0)
            & valid_mask.unsqueeze(2)
            & valid_mask.unsqueeze(1)
        )
    if valid_pair.shape[0] == 1 and bsz > 1:
        valid_pair = valid_pair.expand(bsz, -1, -1)

    # valid_pair 是 previous-sentence lower-triangle candidate set，每个无序句对只出现一次。
    # cosine similarity 对称，因此 lower-triangle mean 与 full off-diagonal ACS 等价；
    # 同时复用 redundancy mask，少构造一次完整 S x S acs_pair_mask。
    pair_count = valid_pair.sum(dim=(-1, -2)).to(dtype=torch.float32)
    acs_sum = sim_matrix.masked_fill(~valid_pair, 0.0).to(dtype=torch.float32).sum(dim=(-1, -2))
    avg_sim = torch.where(
        pair_count >= float(min_pairs),
        acs_sum / pair_count.clamp_min(1.0),
        torch.zeros_like(acs_sum),
    )

    masked_sim = sim_matrix.masked_fill(~valid_pair, -torch.inf)
    has_candidate = valid_pair.any(dim=-1)
    max_scores = masked_sim.max(dim=-1).values
    max_scores = torch.where(has_candidate, max_scores, torch.zeros_like(max_scores))

    redundancy_score = max_scores
    redundant_sent = has_candidate & (max_scores >= tau)
    redundant_sent[:, 0] = False

    p = ((max_scores - tau) / denom).clamp_min(0.0).clamp_max(1.0)
    soft_gamma = (1.0 - lambda_penalty * p).clamp(min=gamma_min, max=1.0)
    sentence_gamma = torch.where(redundant_sent, soft_gamma, sentence_gamma)

    result = {
        "avg_sim": avg_sim,
        "redundant_sentence_mask": redundant_sent,
        "sentence_redundancy_score": redundancy_score,
        "sentence_gamma": sentence_gamma.clamp(min=gamma_min, max=1.0),
    }
    if return_sim_matrix:
        result["sim_matrix"] = sim_matrix
    return result


def _center_sentence_embeddings(
    sent_emb: torch.Tensor,
    valid_mask: torch.Tensor = None,
) -> torch.Tensor:
    bsz, num_sent, _ = sent_emb.shape
    if valid_mask is None:
        valid_mask = torch.ones(bsz, num_sent, dtype=torch.bool, device=sent_emb.device)
    elif valid_mask.dim() == 1:
        valid_mask = valid_mask.unsqueeze(0)
    valid_mask = valid_mask.to(device=sent_emb.device, dtype=torch.bool)
    mask = valid_mask.to(dtype=sent_emb.dtype).unsqueeze(-1)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (sent_emb * mask).sum(dim=1, keepdim=True) / denom
    return (sent_emb - mean) * mask


def _remove_top_pcs_sentence_embeddings(
    sent_emb: torch.Tensor,
    k: int,
    valid_mask: torch.Tensor = None,
) -> torch.Tensor:
    bsz, num_sent, _ = sent_emb.shape
    if valid_mask is None:
        valid_mask = torch.ones(bsz, num_sent, dtype=torch.bool, device=sent_emb.device)
    elif valid_mask.dim() == 1:
        valid_mask = valid_mask.unsqueeze(0)
    valid_mask = valid_mask.to(device=sent_emb.device, dtype=torch.bool)

    out = torch.zeros_like(sent_emb)
    if k <= 0:
        return _center_sentence_embeddings(sent_emb, valid_mask)

    for batch_idx in range(bsz):
        cur_mask = valid_mask[batch_idx]
        cur = sent_emb[batch_idx][cur_mask]
        if cur.shape[0] <= 1:
            continue
        work = cur.float()
        centered = work - work.mean(dim=0, keepdim=True)
        q = min(int(k), centered.shape[0] - 1, centered.shape[1])
        if q <= 0:
            removed = centered
        else:
            try:
                _, _, v = torch.pca_lowrank(centered, q=q)
                pcs = v[:, :q]
            except Exception:
                _, _, vh = torch.linalg.svd(centered, full_matrices=False)
                pcs = vh[:q].T
            removed = centered - (centered @ pcs @ pcs.T)
        out[batch_idx][cur_mask] = removed.to(dtype=sent_emb.dtype)
    return out


def _get_global_acs_tensor(
    sent_emb: torch.Tensor,
    layer_idx: int,
    global_acs_by_layer,
) -> torch.Tensor:
    if global_acs_by_layer is None:
        raise ValueError("sakv_acs_mode='aniso_only' requires global_acs_by_layer.")
    if layer_idx is None:
        raise ValueError("sakv_acs_mode='aniso_only' requires layer_idx.")
    if isinstance(global_acs_by_layer, torch.Tensor):
        value = global_acs_by_layer[layer_idx].to(device=sent_emb.device, dtype=torch.float32)
    else:
        value = torch.tensor(
            float(global_acs_by_layer[int(layer_idx)]),
            device=sent_emb.device,
            dtype=torch.float32,
        )
    return value.reshape(1).expand(sent_emb.shape[0])


def compute_full_pairwise_redundancy_sentence_penalty_for_benchmark(
    sent_norm: torch.Tensor,
    sim_threshold: float,
    gamma_min: float,
    lambda_penalty: float = 0.7,
    valid_mask: torch.Tensor = None,
) -> dict:
    """
    Backward-compatible wrapper for callers that only need redundancy details.
    It still returns the full sim matrix like the previous implementation.
    """
    return compute_full_pairwise_sentence_stats(
        sent_norm=sent_norm,
        sim_threshold=sim_threshold,
        gamma_min=gamma_min,
        lambda_penalty=lambda_penalty,
        valid_mask=valid_mask,
        return_sim_matrix=True,
    )

def compute_redundant_sentence_mask(
    sent_emb: torch.Tensor,
    # sim_threshold: float = 0.75,    #暂时修改为0.5验证一下Idea是否在起作用
    sim_threshold: float,
    candidate_window: int = 16,
    gamma_min: float = 0.3,
    lambda_penalty: float = 0.7,
    return_sim: bool = False,   # ★ 新增
) -> torch.Tensor:
    """
    在“当前层的句子集合”里做 full pairwise 冗余检测，对“后出现”的高相似句子打冗余标记。

    定义：
        对句子 j (j > 0)，如果存在 i < j，使得 cos(sent_i, sent_j) >= threshold，
        则句子 j 为冗余句；前面的句子永远是“原版”，不会被标为冗余。

    Args:
        sent_emb: [bsz, num_sent, hidden_dim]，来自 compute_sentence_embeddings
        sim_threshold: 冗余阈值，默认 0.8

    Returns:
        redundant_sent: [bsz, num_sent]，bool，True 表示该句为冗余句
    """
    device = sent_emb.device
    bsz, num_sent, _ = sent_emb.shape

    sent_norm = _normalize_sentence_embeddings(sent_emb)      # [bsz, S, D]
    details = compute_full_pairwise_redundancy_sentence_penalty_for_benchmark(
        sent_norm=sent_norm,
        sim_threshold=sim_threshold,
        gamma_min=gamma_min,
        lambda_penalty=lambda_penalty,
    )
    redundant_sent = details["redundant_sentence_mask"]

    if return_sim:
        return redundant_sent, details.get("sim_matrix")

    return redundant_sent


def compute_semantic_scores_for_prefill(
    tmp_attn_weights: torch.Tensor,
    hidden_states: torch.Tensor,
    sentence_ids: torch.Tensor,
    window_size: int,
    num_key_value_heads: int,
    sim_threshold: float,
    redundancy_scale: float,
    min_pairs: int = 1,
    return_sim: bool = False,
    redundancy_mode: str = "full_pairwise",
    redundant_compare_window: int = 16,
    redundant_candidate_topk: int = 32,
    semantic_proj_dim: int = 128,
    semantic_penalty_lambda: float = 0.7,
    profile: bool = False,
    sentence_metadata: dict = None,
    acs_mode: str = "raw",
    acs_pc_k: int = 3,
    layer_idx: int = None,
    global_acs_by_layer=None,
    head_score_mode: str = "shared",
) -> dict:
    """
    一次性计算 prefill 阶段的层级 pref_score 和 token 级 hh_score。

    当前正式路径聚焦论文里的 full sentence-pair redundancy setting：
    - ACS/Pri 使用完整 pairwise ACS，和论文 full 场景严格一致。
    - redundant sentence detection 构造完整 S x S similarity matrix。
    - token penalty mapping 和 importance scoring：O(T)。

    head_score_mode="shared" preserves the original shared token ranking.
    "kv_group" averages only query heads belonging to the same KV head,
    so each KV head can retain different historical token positions.
    """
    if head_score_mode not in {"shared", "kv_group"}:
        raise ValueError(f"Unsupported head_score_mode: {head_score_mode}")
    redundancy_mode = (redundancy_mode or "full_pairwise").strip().lower()
    if redundancy_mode == "full":
        redundancy_mode = "full_pairwise"
    if redundancy_mode != "full_pairwise":
        raise ValueError(f"Unsupported redundancy_mode: {redundancy_mode}")

    if not getattr(compute_semantic_scores_for_prefill, "_printed", False):
        print(
            "sim_threshold =", sim_threshold,
            "redundancy_scale =", redundancy_scale,
            "redundancy_mode =", redundancy_mode,
            "semantic_proj_dim =", semantic_proj_dim,
            "semantic_penalty_lambda =", semantic_penalty_lambda,
            "head_score_mode =", head_score_mode,
        )
        compute_semantic_scores_for_prefill._printed = True

    semantic_profile = {
        "pooling_time": 0.0,
        "acs_time": 0.0,
        "pc_removal_time": 0.0,
        "redundant_detection_time": 0.0,
        "penalty_mapping_time": 0.0,
        "importance_scoring_time": 0.0,
        "semantic_total_time": 0.0,
    }
    semantic_total_t0 = _profile_start(profile, hidden_states)

    device = hidden_states.device
    bsz, seq_len, _ = hidden_states.shape
    _, num_heads, win, seq_len_attn = tmp_attn_weights.shape

    if num_key_value_heads < 1 or num_heads < 1 or num_heads % num_key_value_heads != 0:
        raise ValueError(
            "Attention head count must be a positive multiple of num_key_value_heads; "
            f"got {num_heads} query heads and {num_key_value_heads} KV heads."
        )

    assert bsz == 1, "目前实现假定 batch_size=1（LongBench 场景）"
    assert seq_len_attn == seq_len, "tmp_attn_weights 与 hidden_states 长度不一致"
    assert win == window_size, "tmp_attn_weights 的窗口长度应等于 window_size"

    sentence_ids = sentence_ids.to(device).long()
    assert sentence_ids.numel() == seq_len, "sentence_ids 与 hidden_states 长度不一致"
    sentence_metadata = sentence_metadata or {}
    metadata_num_sent = sentence_metadata.get("num_sent")
    metadata_counts = sentence_metadata.get("sentence_counts")
    metadata_valid_sent = sentence_metadata.get("valid_sent")
    metadata_previous_pair_mask = sentence_metadata.get("previous_pair_mask")
    metadata_valid_pair_mask = sentence_metadata.get("valid_pair_mask")

    t0 = _profile_start(profile, hidden_states)
    sent_emb, counts = compute_sentence_embeddings(
        hidden_states,
        sentence_ids,
        num_sent=metadata_num_sent,
        sentence_counts=metadata_counts,
    )
    semantic_profile["pooling_time"] = _profile_end(t0, profile, hidden_states)
    bsz, num_sent, _ = sent_emb.shape

    sent_norm = _normalize_sentence_embeddings(sent_emb)

    # pref_score 只统计实际有 token 的句子，避免 sentence_id 有空洞时空句子参与 ACS。
    if metadata_valid_sent is not None:
        valid_sent = metadata_valid_sent.to(device=device).bool()
        if valid_sent.dim() == 1:
            valid_sent = valid_sent.unsqueeze(0)
    else:
        valid_sent = counts[:, :, 0] > 0

    t0 = _profile_start(profile, hidden_states)
    sentence_stats = compute_full_pairwise_sentence_stats(
        sent_norm=sent_norm,
        sim_threshold=sim_threshold,
        gamma_min=redundancy_scale,
        lambda_penalty=semantic_penalty_lambda,
        valid_mask=valid_sent,
        previous_pair_mask=metadata_previous_pair_mask,
        valid_pair_mask=metadata_valid_pair_mask,
        min_pairs=min_pairs,
        return_sim_matrix=return_sim,
    )
    full_pairwise_time = _profile_end(t0, profile, hidden_states)
    semantic_profile["acs_time"] = full_pairwise_time
    semantic_profile["redundant_detection_time"] = 0.0

    raw_avg_sim_tensor = sentence_stats["avg_sim"]
    avg_sim_tensor = raw_avg_sim_tensor
    acs_mode = (acs_mode or "raw").strip().lower()
    if acs_mode in {"raw", "raw_full_pairwise"}:
        pass
    elif acs_mode == "centered":
        t0 = _profile_start(profile, hidden_states)
        centered = _center_sentence_embeddings(sent_emb, valid_sent)
        semantic_profile["pc_removal_time"] = _profile_end(t0, profile, hidden_states)
        t0 = _profile_start(profile, hidden_states)
        avg_sim_tensor = compute_acs_from_normalized_sentences(
            _normalize_sentence_embeddings(centered),
            valid_mask=valid_sent,
            min_pairs=min_pairs,
        )
        semantic_profile["acs_time"] += _profile_end(t0, profile, hidden_states)
    elif acs_mode in {"pc_removed", "pcremoved"}:
        t0 = _profile_start(profile, hidden_states)
        removed = _remove_top_pcs_sentence_embeddings(sent_emb, int(acs_pc_k), valid_sent)
        semantic_profile["pc_removal_time"] = _profile_end(t0, profile, hidden_states)
        t0 = _profile_start(profile, hidden_states)
        avg_sim_tensor = compute_acs_from_normalized_sentences(
            _normalize_sentence_embeddings(removed),
            valid_mask=valid_sent,
            min_pairs=min_pairs,
        )
        semantic_profile["acs_time"] += _profile_end(t0, profile, hidden_states)
    elif acs_mode == "aniso_only":
        avg_sim_tensor = _get_global_acs_tensor(
            sent_emb=sent_emb,
            layer_idx=layer_idx,
            global_acs_by_layer=global_acs_by_layer,
        )
    else:
        raise ValueError(f"Unsupported sakv acs_mode: {acs_mode}")

    avg_sim = torch.nan_to_num(
        avg_sim_tensor[0],
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    ).clamp(min=-1.0, max=1.0)
    pref_score = (1.0 - avg_sim).clamp_min(1e-6)
    sim = sentence_stats.get("sim_matrix")

    redundant_sent = sentence_stats["redundant_sentence_mask"]
    sentence_redundancy_score = sentence_stats["sentence_redundancy_score"]
    sentence_gamma = torch.nan_to_num(
        sentence_stats["sentence_gamma"],
        nan=1.0,
        posinf=1.0,
        neginf=redundancy_scale,
    ).clamp(min=redundancy_scale, max=1.0)

    past_len = max(seq_len - window_size, 0)
    if past_len == 0:
        hh_score = hidden_states.new_zeros(bsz, num_key_value_heads, 0)
        semantic_profile["semantic_total_time"] = _profile_end(semantic_total_t0, profile, hidden_states)
        return {
            "pref_score": pref_score,
            "avg_sim": avg_sim,
            "hh_score": hh_score,
            "sim": sim if return_sim else None,
            "redundant_sentence_mask": redundant_sent,
            "sentence_redundancy_score": sentence_redundancy_score,
            "sentence_gamma": sentence_gamma,
            "token_gamma": hidden_states.new_ones(bsz, 0),
            "profile": semantic_profile,
        }

    t0 = _profile_start(profile, hidden_states)
    sid = sentence_ids.unsqueeze(0).expand(bsz, -1)           # [bsz, seq_len]
    sid = sid.clamp(min=0, max=max(num_sent - 1, 0))
    token_gamma = sentence_gamma[torch.arange(bsz, device=device).unsqueeze(1), sid]
    token_gamma_past = token_gamma[:, :past_len]              # [bsz, past_len]
    semantic_profile["penalty_mapping_time"] = _profile_end(t0, profile, hidden_states)

    t0 = _profile_start(profile, hidden_states)
    # Attention remains the primary signal; the semantic factor is shared
    # across KV heads. repeat_kv orders query heads in contiguous KV groups.
    attn_by_query_head = tmp_attn_weights.mean(dim=2)        # [bsz, H_q, seq_len]
    if head_score_mode == "shared":
        # Keep the historical behavior and its shared-head top-k fast path.
        attn_avg = attn_by_query_head.mean(dim=1)
        token_score = attn_avg[:, :past_len] * token_gamma_past
        hh_score = token_score.unsqueeze(1).expand(-1, num_key_value_heads, -1)
    else:
        num_key_value_groups = num_heads // num_key_value_heads
        attn_by_kv_head = attn_by_query_head.reshape(
            bsz, num_key_value_heads, num_key_value_groups, seq_len
        ).mean(dim=2)
        hh_score = attn_by_kv_head[..., :past_len] * token_gamma_past.unsqueeze(1)
    semantic_profile["importance_scoring_time"] = _profile_end(t0, profile, hidden_states)
    semantic_profile["semantic_total_time"] = _profile_end(semantic_total_t0, profile, hidden_states)

    if profile and bool(getattr(compute_semantic_scores_for_prefill, "print_profile", False)):
        print(
            "[SAKV Profile] "
            f"pooling_time={semantic_profile['pooling_time']:.6f}s "
            f"acs_time={semantic_profile['acs_time']:.6f}s "
            f"pc_removal_time={semantic_profile['pc_removal_time']:.6f}s "
            f"redundant_detection_time={semantic_profile['redundant_detection_time']:.6f}s "
            f"penalty_mapping_time={semantic_profile['penalty_mapping_time']:.6f}s "
            f"token_importance_time={semantic_profile['importance_scoring_time']:.6f}s "
            f"semantic_total_time={semantic_profile['semantic_total_time']:.6f}s"
        )

    return {
        "pref_score": pref_score,
        "avg_sim": avg_sim,
        "hh_score": hh_score,
        "sim": sim if return_sim else None,
        "redundant_sentence_mask": redundant_sent,
        "sentence_redundancy_score": sentence_redundancy_score,
        "sentence_gamma": sentence_gamma,
        "token_gamma": token_gamma,
        "profile": semantic_profile,
    }


def compute_hh_score_for_prefill(
    tmp_attn_weights: torch.Tensor,
    hidden_states: torch.Tensor,
    sentence_ids: torch.Tensor,
    window_size: int,
    num_key_value_heads: int,
    # sim_threshold: float = 0.75,       #暂时修改为0.5
    # redundancy_scale: float = 0.3,      #极端一点，相似的大部分不保留看看实验结果
    sim_threshold: float,
    redundancy_scale: float,
    redundant_compare_window: int = 16,
    semantic_penalty_lambda: float = 0.7,
    return_sim: bool = False,    # 新增：要不要顺便把 sim 返回出去
) -> torch.Tensor:
    """
    基于“最近窗口平均注意力 + 冗余句子”生成 prefill 阶段的 hh_score。

    - 最近窗口平均注意力：表示这一层里最近 window 对各历史 token 的关注度。
    - 冗余句子：在这一层，用当前层 hidden_states 算句子向量，句子间做余弦相似度，
      对“后出现且相似度 >= sim_threshold”的句子标为冗余，并对其 token 降权。

    Args:
        tmp_attn_weights: [bsz, num_heads, win, seq_len]，已经 softmax 完成
        hidden_states   : [bsz, seq_len, hidden_dim]，当前层的 hidden_states
        sentence_ids    : [seq_len]
        window_size     : 最近窗口大小
        num_key_value_heads: kv heads 数量
        sim_threshold   : 冗余句相似度阈值（默认 0.8）
        redundancy_scale: 冗余句子 token 的缩放系数（0~1，越小越容易被删）

    Returns:
        hh_score: [bsz, num_key_value_heads, past_len]，past_len = seq_len - window_size
    """
    #打印相似度和冗余设置
    if not getattr(compute_hh_score_for_prefill, "_printed", False):
        print("sim_threshold =", sim_threshold,"redundancy_scale =", redundancy_scale)
        compute_hh_score_for_prefill._printed = True
    #打印相似度和冗余设置
    
    semantic_scores = compute_semantic_scores_for_prefill(
        tmp_attn_weights=tmp_attn_weights,
        hidden_states=hidden_states,
        sentence_ids=sentence_ids,
        window_size=window_size,
        num_key_value_heads=num_key_value_heads,
        sim_threshold=sim_threshold,
        redundancy_scale=redundancy_scale,
        return_sim=return_sim,
        redundancy_mode="full_pairwise",
        redundant_compare_window=redundant_compare_window,
        semantic_penalty_lambda=semantic_penalty_lambda,
    )
    hh_score = semantic_scores["hh_score"]

    if return_sim:
        return hh_score, semantic_scores["sim"]        # ★ 可选把 sim 带出去

    return hh_score

    
