import gc
import math
import pdb
import time
import torch
import torch.nn.functional as F
from torch import nn
import torch.utils.checkpoint
import transformers

from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.models.mistral.configuration_mistral import MistralConfig

    
from transformers.models.mistral.modeling_mistral import *

from transformers.modeling_flash_attention_utils import _flash_attention_forward

from typing import Optional, Tuple
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from ..cake_cache import CakeCache, CakeDecodingKVCache_LayerWise

logger = logging.get_logger(__name__)

######我们把语义分数计算写在了utils函数里
from ..utils import compute_semantic_scores_for_prefill, compute_cake_attention_prefill_scores


def _pool_recent_attention_to_kv_heads(
    tmp_attn_weights: torch.Tensor,
    window_size: int,
    num_key_value_heads: int,
    num_key_value_groups: int,
    kernel_size: int = 7,
    pooling: str = "maxpool",
) -> torch.Tensor:
    past_attn = tmp_attn_weights[:, :, -window_size:, :-window_size]
    if past_attn.shape[-1] == 0:
        return tmp_attn_weights.new_zeros(tmp_attn_weights.shape[0], num_key_value_heads, 0)

    attn_cache = past_attn.sum(dim=-2)
    if pooling == "avgpool":
        attn_cache = F.avg_pool1d(attn_cache, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
    elif pooling == "maxpool":
        attn_cache = F.max_pool1d(attn_cache, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
    else:
        raise ValueError(f"Unsupported pooling mode for unified KV baseline: {pooling}")

    attn_cache = attn_cache.reshape(
        attn_cache.shape[0],
        num_key_value_heads,
        num_key_value_groups,
        -1,
    )
    return attn_cache.mean(dim=-2)


def _compute_snapkv_prefill_scores(
    tmp_attn_weights: torch.Tensor,
    window_size: int,
    num_key_value_heads: int,
    num_key_value_groups: int,
    kernel_size: int = 7,
    pooling: str = "maxpool",
) -> Tuple[float, torch.Tensor]:
    hh_score = _pool_recent_attention_to_kv_heads(
        tmp_attn_weights=tmp_attn_weights,
        window_size=window_size,
        num_key_value_heads=num_key_value_heads,
        num_key_value_groups=num_key_value_groups,
        kernel_size=kernel_size,
        pooling=pooling,
    )
    return 1.0, hh_score


def _cuda_sync_if_needed(device):
    if isinstance(device, torch.Tensor):
        device = device.device
    if torch.cuda.is_available() and getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)


def _ensure_prefill_profile_buffers(cfg):
    num_layers = getattr(cfg, "num_hidden_layers", None)
    if num_layers is None:
        num_layers = len(getattr(cfg, "window_size"))

    if not hasattr(cfg, "prefill_profile") or not isinstance(cfg.prefill_profile, dict):
        cfg.prefill_profile = {
            "num_prefill_calls": 0,
            "tmp_attn_sec": 0.0,
            "pref_score_sec": 0.0,
            "hh_score_sec": 0.0,
            "sim_related_sec": 0.0,
            "pooling_time": 0.0,
            "acs_time": 0.0,
            "pc_removal_time": 0.0,
            "redundant_detection_time": 0.0,
            "penalty_mapping_time": 0.0,
            "importance_scoring_time": 0.0,
            "semantic_total_time": 0.0,
            "eviction_time": 0.0,
            "prefill_total_sec": 0.0,
            "q_len_total": 0,
            "num_sent_total": 0,
            "max_q_len": 0,
            "max_num_sent": 0,
            "per_layer": [
                {
                    "calls": 0,
                    "tmp_attn_sec": 0.0,
                    "pref_score_sec": 0.0,
                    "hh_score_sec": 0.0,
                    "sim_related_sec": 0.0,
                    "pooling_time": 0.0,
                    "acs_time": 0.0,
                    "pc_removal_time": 0.0,
                    "redundant_detection_time": 0.0,
                    "penalty_mapping_time": 0.0,
                    "importance_scoring_time": 0.0,
                    "semantic_total_time": 0.0,
                    "eviction_time": 0.0,
                    "prefill_total_sec": 0.0,
                    "max_q_len": 0,
                    "max_num_sent": 0,
                }
                for _ in range(num_layers)
            ],
        }
    else:
        for key in [
            "pooling_time",
            "acs_time",
            "pc_removal_time",
            "redundant_detection_time",
            "penalty_mapping_time",
            "importance_scoring_time",
            "semantic_total_time",
            "eviction_time",
        ]:
            cfg.prefill_profile.setdefault(key, 0.0)
        cfg.prefill_profile.setdefault("per_layer", [{} for _ in range(num_layers)])
        for layer_idx in range(num_layers):
            if layer_idx >= len(cfg.prefill_profile["per_layer"]):
                cfg.prefill_profile["per_layer"].append({})
            for key in [
                "calls",
                "tmp_attn_sec",
                "pref_score_sec",
                "hh_score_sec",
                "sim_related_sec",
                "pooling_time",
                "acs_time",
                "pc_removal_time",
                "redundant_detection_time",
                "penalty_mapping_time",
                "importance_scoring_time",
                "semantic_total_time",
                "eviction_time",
                "prefill_total_sec",
                "max_q_len",
                "max_num_sent",
            ]:
                cfg.prefill_profile["per_layer"][layer_idx].setdefault(key, 0.0)
    return cfg.prefill_profile


def _update_prefill_profile(
    cfg,
    layer_idx,
    q_len,
    num_sent,
    tmp_attn_sec,
    pref_score_sec,
    hh_score_sec,
    prefill_total_sec,
    semantic_profile=None,
    eviction_sec=0.0,
):
    prof = _ensure_prefill_profile_buffers(cfg)
    sim_related_sec = pref_score_sec + hh_score_sec
    semantic_profile = semantic_profile or {}

    prof["num_prefill_calls"] += 1
    prof["tmp_attn_sec"] += float(tmp_attn_sec)
    prof["pref_score_sec"] += float(pref_score_sec)
    prof["hh_score_sec"] += float(hh_score_sec)
    prof["sim_related_sec"] += float(sim_related_sec)
    for key in [
        "pooling_time",
        "acs_time",
        "pc_removal_time",
        "redundant_detection_time",
        "penalty_mapping_time",
        "importance_scoring_time",
        "semantic_total_time",
    ]:
        prof[key] += float(semantic_profile.get(key, 0.0))
    prof["eviction_time"] += float(eviction_sec)
    prof["prefill_total_sec"] += float(prefill_total_sec)
    prof["q_len_total"] += int(q_len)
    prof["num_sent_total"] += int(num_sent)
    prof["max_q_len"] = max(int(prof["max_q_len"]), int(q_len))
    prof["max_num_sent"] = max(int(prof["max_num_sent"]), int(num_sent))

    layer_prof = prof["per_layer"][layer_idx]
    layer_prof["calls"] += 1
    layer_prof["tmp_attn_sec"] += float(tmp_attn_sec)
    layer_prof["pref_score_sec"] += float(pref_score_sec)
    layer_prof["hh_score_sec"] += float(hh_score_sec)
    layer_prof["sim_related_sec"] += float(sim_related_sec)
    for key in [
        "pooling_time",
        "acs_time",
        "pc_removal_time",
        "redundant_detection_time",
        "penalty_mapping_time",
        "importance_scoring_time",
        "semantic_total_time",
    ]:
        layer_prof[key] += float(semantic_profile.get(key, 0.0))
    layer_prof["eviction_time"] += float(eviction_sec)
    layer_prof["prefill_total_sec"] += float(prefill_total_sec)
    layer_prof["max_q_len"] = max(int(layer_prof["max_q_len"]), int(q_len))
    layer_prof["max_num_sent"] = max(int(layer_prof["max_num_sent"]), int(num_sent))

def mistral_attn_forward_cake(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[CakeCache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
):

    if isinstance(past_key_value, StaticCache):
        raise ValueError(
            "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
            "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
        )
    if isinstance(past_key_value, DynamicCache):
        past_key_value = CakeCache.from_dynamic_cache(past_key_value)
 
    if self.config.decoding_evict[self.layer_idx] is None and len(past_key_value.layer_budget) == self.config.prefill_cake_evict[self.layer_idx].num_layers:
        self.config.decoding_evict[self.layer_idx] =CakeDecodingKVCache_LayerWise(
                hh_size =past_key_value.layer_budget[self.layer_idx],
                window_size=self.config.window_size[self.layer_idx],
                k_seq_dim=2,
                v_seq_dim=2
                )

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += cache_position[0]

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # Activate slicing cache only if the config has a value `sliding_windows` attribute
        cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
        if (
            getattr(self.config, "sliding_window", None) is not None
            and kv_seq_len > self.config.sliding_window
            and cache_has_contents
        ):
            slicing_tokens = 1 - self.config.sliding_window

            past_key = past_key_value[self.layer_idx][0]
            past_value = past_key_value[self.layer_idx][1]

            past_key = past_key[:, :, slicing_tokens:, :].contiguous()
            past_value = past_value[:, :, slicing_tokens:, :].contiguous()

            if past_key.shape[-2] != self.config.sliding_window - 1:
                raise ValueError(
                    f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                    f" {past_key.shape}"
                )

            if attention_mask is not None:
                attention_mask = attention_mask[:, slicing_tokens:]
                attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        ########第一次1出现
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = 0.0 if not self.training else self.attention_dropout
    ############修改偏好分数的计算方式 ############

    if self.config.prefill[self.layer_idx]:
        cfg = self.config
        profile_enabled = bool(getattr(cfg, "profile_prefill_time", False))
        layer_prefill_t0 = None
        tmp_attn_sec = 0.0
        pref_score_sec = 0.0
        hh_score_sec = 0.0

        if profile_enabled:
            _ensure_prefill_profile_buffers(cfg)
            _cuda_sync_if_needed(hidden_states)
            layer_prefill_t0 = time.perf_counter()

        if profile_enabled:
            _cuda_sync_if_needed(hidden_states)
            t_tmp0 = time.perf_counter()

        window_size = self.config.window_size[self.layer_idx]
        # 1. 不管后面走哪个分支，先统一算出 tmp_attn_weights
        tmp_attn_weights = torch.matmul(
            query_states[..., -window_size:, :],
            key_states.transpose(2, 3),
        ) / math.sqrt(self.head_dim)

        if q_len != 1:
            mask_cache = getattr(cfg, "_sakv_window_mask_cache", {})
            mask_key = (str(tmp_attn_weights.device), str(tmp_attn_weights.dtype), int(window_size))
            tmp_attention_mask = mask_cache.get(mask_key)
            if tmp_attention_mask is None:
                mask = torch.full(
                    (window_size, window_size),
                    torch.finfo(tmp_attn_weights.dtype).min,
                    device=tmp_attn_weights.device,
                )
                mask_cond = torch.arange(mask.size(-1), device=tmp_attn_weights.device)
                mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
                tmp_attention_mask = mask[None, None, :, :]
                mask_cache[mask_key] = tmp_attention_mask
                cfg._sakv_window_mask_cache = mask_cache
            tmp_attn_weights[
                :,
                :,
                -window_size:,
                -window_size:,
            ] += tmp_attention_mask

        if bool(getattr(cfg, "score_softmax_fp32", True)):
            tmp_attn_weights = nn.functional.softmax(
                tmp_attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)
        else:
            tmp_attn_weights = nn.functional.softmax(tmp_attn_weights, dim=-1)

        if profile_enabled:
            _cuda_sync_if_needed(tmp_attn_weights)
            tmp_attn_sec = time.perf_counter() - t_tmp0

        num_key_value_heads = past_key_value.key_cache[self.layer_idx].shape[1]
        method_variant = getattr(cfg, "sakv_method_variant", "sakv")
        semantic_profile = {}

        if method_variant == "cake_attention":
            num_sent = 0
            if profile_enabled:
                _cuda_sync_if_needed(hidden_states)
                t_cake_score0 = time.perf_counter()
            pref_score, hh_score = compute_cake_attention_prefill_scores(
                tmp_attn_weights=tmp_attn_weights,
                window_size=window_size,
                num_key_value_heads=num_key_value_heads,
                num_key_value_groups=self.num_key_value_groups,
                gamma=getattr(self.config, "gamma", 200.0),
                tau1=getattr(self.config, "tau1", 1.0),
                tau2=getattr(self.config, "tau2", 1.0),
            )
            if profile_enabled:
                _cuda_sync_if_needed(hh_score)
                hh_score_sec = time.perf_counter() - t_cake_score0
        elif method_variant in {"snapkv", "pyramidkv"}:
            num_sent = 0
            if profile_enabled:
                _cuda_sync_if_needed(hidden_states)
                t_snap_score0 = time.perf_counter()
            pref_score, hh_score = _compute_snapkv_prefill_scores(
                tmp_attn_weights=tmp_attn_weights,
                window_size=window_size,
                num_key_value_heads=num_key_value_heads,
                num_key_value_groups=self.num_key_value_groups,
                kernel_size=getattr(self.config, "baseline_kernel_size", 7),
                pooling=getattr(self.config, "baseline_pooling", "maxpool"),
            )
            if profile_enabled:
                _cuda_sync_if_needed(hh_score)
                hh_score_sec = time.perf_counter() - t_snap_score0
        else:
            # 2. 这里 sentence_ids 视为 *必需*，不再回退到旧方法
            sentence_ids = getattr(self.config, "sentence_ids", None)
            if sentence_ids is None:
                raise ValueError(
                    "sentence_ids is None, but the new sentence-level eviction strategy requires sentence_ids. "
                    "请检查数据预处理 / collator 是否正确传入 sentence_ids。"
                )

            sentence_metadata = getattr(self.config, "sakv_sentence_metadata", None)
            if sentence_metadata is not None and "num_sent" in sentence_metadata:
                num_sent = int(sentence_metadata["num_sent"])
            else:
                num_sent = int(sentence_ids.max().item()) + 1

            # 3. 新的 hh_score：最近窗口平均注意力 + 冗余句子
            if profile_enabled:
                _cuda_sync_if_needed(hidden_states)
                t_semantic0 = time.perf_counter()

            semantic_scores = compute_semantic_scores_for_prefill(
                tmp_attn_weights=tmp_attn_weights,
                hidden_states=hidden_states,
                sentence_ids=sentence_ids,
                window_size=window_size,
                num_key_value_heads=num_key_value_heads,
                sim_threshold=getattr(self.config, "redundant_sim_threshold", 0.75),
                redundancy_scale=getattr(self.config, "redundancy_scale", 0.3),
                redundancy_mode=getattr(self.config, "redundancy_mode", "full_pairwise"),
                redundant_compare_window=getattr(self.config, "redundant_compare_window", 16),
                redundant_candidate_topk=getattr(self.config, "redundant_candidate_topk", 32),
                semantic_proj_dim=getattr(self.config, "semantic_proj_dim", 128),
                semantic_penalty_lambda=getattr(self.config, "semantic_penalty_lambda", 0.7),
                head_score_mode=getattr(self.config, "sakv_head_score_mode", "shared"),
                profile=profile_enabled,
                sentence_metadata=sentence_metadata,
                acs_mode=getattr(self.config, "sakv_acs_mode", "raw"),
                acs_pc_k=getattr(self.config, "sakv_acs_pc_k", 3),
                layer_idx=self.layer_idx,
                global_acs_by_layer=getattr(self.config, "sakv_global_acs_by_layer", None),
            )
            pref_score = semantic_scores["pref_score"]
            hh_score = semantic_scores["hh_score"]
            semantic_profile = semantic_scores.get("profile", {})

            if profile_enabled:
                _cuda_sync_if_needed(hh_score)
                semantic_score_sec = time.perf_counter() - t_semantic0
                pref_score_sec = float(semantic_profile.get("acs_time", 0.0))
                hh_score_sec = max(0.0, semantic_score_sec - pref_score_sec)

        past_key_value.update_score(pref_score, hh_score)
        past_key_value.layer_budget.append(self.config.key_size[self.layer_idx])
        self.config.prefill[self.layer_idx] = False
        budget = self.config.key_size[self.layer_idx]

        eviction_sec = 0.0
        if profile_enabled:
            _cuda_sync_if_needed(hidden_states)
            t_evict0 = time.perf_counter()
        past_key_value = self.config.prefill_cake_evict[self.layer_idx](
            past_key_value, q_len, self.layer_idx, budget
        )
        if profile_enabled:
            _cuda_sync_if_needed(hidden_states)
            eviction_sec = time.perf_counter() - t_evict0

        if profile_enabled:
            _cuda_sync_if_needed(hidden_states)
            prefill_total_sec = time.perf_counter() - layer_prefill_t0
            _update_prefill_profile(
                cfg=cfg,
                layer_idx=self.layer_idx,
                q_len=q_len,
                num_sent=num_sent,
                tmp_attn_sec=tmp_attn_sec,
                pref_score_sec=pref_score_sec,
                hh_score_sec=hh_score_sec,
                prefill_total_sec=prefill_total_sec,
                semantic_profile=semantic_profile,
                eviction_sec=eviction_sec,
            )
            if bool(getattr(cfg, "profile_sakv_log", False)):
                profile_label = (
                    "[AnisoCtrl Profile]"
                    if getattr(cfg, "sakv_acs_mode", "raw") != "raw"
                    else "[SAKV Profile]"
                )
                print(
                    f"{profile_label} "
                    f"pooling_time={semantic_profile.get('pooling_time', 0.0):.6f}s "
                    f"acs_time={semantic_profile.get('acs_time', 0.0):.6f}s "
                    f"pc_removal_time={semantic_profile.get('pc_removal_time', 0.0):.6f}s "
                    f"redundant_detection_time={semantic_profile.get('redundant_detection_time', 0.0):.6f}s "
                    f"penalty_mapping_time={semantic_profile.get('penalty_mapping_time', 0.0):.6f}s "
                    f"token_importance_time={semantic_profile.get('importance_scoring_time', 0.0):.6f}s "
                    f"eviction_time={eviction_sec:.6f}s "
                    f"semantic_total_time={semantic_profile.get('semantic_total_time', 0.0):.6f}s "
                    f"prefill_total_time={prefill_total_sec:.6f}s"
                )

    if self.config.decoding_evict[self.layer_idx] is not None:


        tmp_attn_weights = torch.matmul(query_states[..., -self.config.window_size[self.layer_idx]:, :], key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        tmp_attn_weights = nn.functional.softmax(tmp_attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        past_key_value = self.config.decoding_evict[self.layer_idx](past_key_value, tmp_attn_weights, self.layer_idx)
    

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in float16 just to be sure everything works as expected.
    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    # Reashape to the expected shape for Flash Attention
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        dropout=dropout_rate,
        sliding_window=getattr(self.config, "sliding_window", None),
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
        is_causal=self.is_causal,
    )

    attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def mistral_model_forward_cake(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
        )
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    return_legacy_cache = False
    if use_cache and not isinstance(past_key_values, Cache) and not self.training:
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        return_legacy_cache = True
        logger.warning_once(
            "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
            "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
        )

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, use_cache, output_attentions
    )

    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            past_key_values = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None

    if return_legacy_cache:
        next_cache = next_cache.to_legacy_cache()

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
    

class CakeMistralForCausalLM(MistralForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = CakeMistralModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

def replace_mistral_attn_with_new_kvattn(model):
    print("replace_mistral_attn_with_new_kvattn")
    for name, module in model.named_modules():
        if isinstance(module, MistralAttention):
            module.__class__ = KVMistralAttention
    return model

def replace_mistral_attn_with_kvflashattn(model):
    print("replace_mistral_attn_with_kvflashattn")
    for name, module in model.named_modules():
        if isinstance(module, MistralFlashAttention2):
            module.__class__ = KVMistralFlashAttention2
    return model

def replace_mistral_attn_with_mattn(model):
    print("replace_mistral_attn_with_kvflashattn")
    for name, module in model.named_modules():
        if isinstance(module, MistralFlashAttention2):
            module.__class__ = MMistralFlashAttention2
    return model

def replace_mistral_attn_with_gpaattn(model):
    print("replace_mistral_attn_with_kvflashattn")
    for name, module in model.named_modules():
        if isinstance(module, MistralFlashAttention2):
            module.__class__ = GPAMistralFlashAttention2
    return model

def replace_mistral_attn_with_cakeattn(model):
    print("replace_mistral_attn_with_kvflashattn")
    for name, module in model.named_modules():
        if isinstance(module, MistralFlashAttention2):
            module.__class__ = CakeMistralFlashAttention2
    return model
