import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import DynamicCache, Cache, HybridCache
from typing import Any, Dict, List, Optional, Tuple, Union
from cake.utils import adjust_budgets

class CakeCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.pref_scores = []
        self.evict_scores = []
        self.layer_budget = []
    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds  
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `CakeCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def update_score(
        self,
        pref_score: torch.Tensor,
        evict_score: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    ):
        self.pref_scores.append(pref_score)
        self.evict_scores.append(evict_score)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. CakeCache does not have a maximum length."""
        return None

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `CakeCache` instance into the its equivalent in the legacy cache format. Used for
        backward compatibility."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "CakeCache":
        """Converts a cache in the legacy cache format into an equivalent `CakeCache`. Used for
        backward compatibility."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache
    @classmethod
    def from_dynamic_cache(cls, past_key_values: Optional[DynamicCache] = None) -> "CakeCache":
        cache = cls()
        if past_key_values is not None:
            cache.key_cache = past_key_values.key_cache
            cache.value_cache = past_key_values.value_cache

        return cache
    @classmethod
    def from_hybrid_cache(cls, past_key_values: Optional[HybridCache] = None) -> "CakeCache":
        cache = cls()
        if past_key_values is not None:
            cache.key_cache = past_key_values.key_cache
            cache.value_cache = past_key_values.value_cache

        return cache
    def crop(self, max_length: int):
        """Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be
        negative to remove `max_length` tokens. This is used in assisted decoding and contrastive search."""

        # In case it is negative
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        self._seen_tokens = max_length
        for idx in range(len(self.key_cache)):
            self.key_cache[idx] = self.key_cache[idx][..., :max_length, :]
            self.value_cache[idx] = self.value_cache[idx][..., :max_length, :]

    def batch_split(self, full_batch_size: int, split_size: int) -> List["CakeCache"]:
        """Split the current instance into a list of `DynamicCache` by the batch size. This will be used by
        `_split_model_inputs()` in `generation.utils`"""
        out = []
        for i in range(0, full_batch_size, split_size):
            current_split = CakeCache()
            current_split._seen_tokens = self._seen_tokens
            current_split.key_cache = [tensor[i : i + split_size] for tensor in self.key_cache]
            current_split.value_cache = [tensor[i : i + split_size] for tensor in self.value_cache]
            current_split.pref_scores = self.pref_scores
            current_split.evict_scores = self.evict_scores
            out.append(current_split)
        return out

    @classmethod
    def from_batch_splits(cls, splits: List["CakeCache"]) -> "CakeCache":
        """This is the opposite of the above `batch_split()` method. This will be used by `stack_model_outputs` in
        `generation.utils`"""
        cache = cls()
        for idx in range(len(splits[0])):
            layer_keys = torch.cat([current.key_cache[idx] for current in splits], dim=0)
            layer_values = torch.cat([current.value_cache[idx] for current in splits], dim=0)
            cache.update(layer_keys, layer_values, idx)
            cache.pref_scores = self.pref_scores
            cache.evict_scores = self.evict_scores
        return cache

    def batch_repeat_interleave(self, repeats: int):
        """Repeat the cache `repeats` times in the batch dimension. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].repeat_interleave(repeats, dim=0)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor):
        """Only keep the `indices` in the batch dimension of the cache. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][indices, ...]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][indices, ...]



class CakeprefillKVCache:
    def __init__(
        self,
        cache_size=512,
        window_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        num_heads = 32, 
        num_layers = 32,
        use_cascading = False
    ):

        self.window_size = window_size
        self.total_size = (cache_size-window_size) * num_layers
        self.cache_size = cache_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        # 每层的「相似度之和」和「统计次数」
        self.layer_sim_sum = [0.0 for _ in range(num_layers)]
        self.layer_sim_cnt = [0   for _ in range(num_layers)]
        self.use_cascading = use_cascading  # If true, ensure high attention precision
        # ⭐ 新增：层级 KV 分配超参数
        self.alpha = 1  
        self.beta = 0.5    
        self.fixed_layer_budgets = None
        self.defer_prefill_eviction = False
        self.last_applied_budget = [None for _ in range(num_layers)]

        # print(f"CakeprefillKVCache: {self.total_size}, {self.window_size}")

    def reset(self):
        # 每条样本开始前清空统计，避免跨样本累积
        self.layer_sim_sum = [0.0 for _ in range(self.num_layers)]
        self.layer_sim_cnt = [0   for _ in range(self.num_layers)]
        self.last_applied_budget = [None for _ in range(self.num_layers)]
        
    def __call__(self, past_key_values, seq_len, layer_idx, budget):
        if seq_len<=self.cache_size+self.window_size:
            return past_key_values

        pref_scores = past_key_values.pref_scores
        if self.defer_prefill_eviction and len(pref_scores) < self.num_layers:
            return past_key_values
        
  
  # #####layer_budgets计算每层的内存分配预算，pref_score就是优先级分数
  
  #       layer_budgets = [pref_score/sum(pref_scores)*self.total_size for pref_score in pref_scores]
    
  #       layer_budgets = adjust_budgets(layer_budgets, self.total_size, seq_len-self.window_size,  self.num_layers)

        
        if self.fixed_layer_budgets is not None:
            layer_budgets = self.fixed_layer_budgets[:len(pref_scores)]
        else:
            #测试新版内存分配
            alpha = self.alpha
            beta = self.beta
            
            L = len(pref_scores)
            B = self.total_size
            
            # 防止数值问题
            eps = 1e-6
            score_device = past_key_values.key_cache[layer_idx].device
            pref_score_tensors = []
            for score in pref_scores:
                if torch.is_tensor(score):
                    pref_score_tensors.append(score.to(device=score_device, dtype=torch.float32).reshape(()))
                else:
                    pref_score_tensors.append(torch.tensor(float(score), device=score_device, dtype=torch.float32))
            pref_scores_safe = torch.stack(pref_score_tensors)
            pref_scores_safe = torch.nan_to_num(
                pref_scores_safe,
                nan=eps,
                posinf=1.0,
                neginf=eps,
            ).clamp_min(eps)
            
            # 1) 基础配额部分：β * B / L
            base_budget = beta * B / L
            
            # 2) 幂指数重要度部分：(1-β) * B * I_n^α / sum(I_k^α)
            pow_scores = pref_scores_safe.pow(alpha)
            pow_sum = torch.nan_to_num(pow_scores.sum(), nan=eps, posinf=1.0, neginf=eps).clamp_min(eps)
            layer_budgets = (base_budget + (1 - beta) * B * (pow_scores / pow_sum)).detach().cpu().tolist()
            #测试新版内存分配
            layer_budgets = adjust_budgets(layer_budgets, self.total_size, seq_len-self.window_size,  self.num_layers)
        ####根据每一层的预算调整内存

        if self.use_cascading:
            layer_idx = 0 #初始化第一层，从第一层开始遍历每层的预算
            if getattr(self, "debug_budget", False):
                print(layer_budgets)
            for budget in layer_budgets:
                if budget>= seq_len-self.window_size:
                    budget = seq_len-self.window_size
                past_key_values = self._evict_layer_if_needed(past_key_values, layer_idx, budget)
                past_key_values.layer_budget[layer_idx]=budget #设置这一层的内存预算
                layer_idx +=1  #移动到下一层去设置内存预算
        else:
            layer_idx = 0
            if len(layer_budgets) ==self.num_layers:
                for budget in layer_budgets:
                    if budget>= seq_len-self.window_size:
                        budget = seq_len-self.window_size
                    past_key_values = self._evict_layer_if_needed(past_key_values, layer_idx, budget)
                    past_key_values.layer_budget[layer_idx]=budget
                    layer_idx +=1

        return past_key_values




    def _evict_layer_if_needed(self, past_key_values, layer_idx, budget):
        budget = int(budget)
        cur_len = past_key_values.key_cache[layer_idx].size(2)
        if cur_len <= budget + self.window_size:
            self.last_applied_budget[layer_idx] = budget
            return past_key_values

        past_key_values = self.evcit_layer_kvcache(past_key_values, layer_idx, budget)
        self.last_applied_budget[layer_idx] = budget
        return past_key_values


    def evcit_layer_kvcache(self, past_key_values, layer_idx, budget):
        # 当前层 KV 的形状
        k_cache = past_key_values.key_cache[layer_idx]
        v_cache = past_key_values.value_cache[layer_idx]
        bsz, num_key_value_heads, seq_len, head_dim = k_cache.shape
    
        # 固定保留的尾部窗口长度
        win = self.window_size
    
        # 序列太短，连窗口都不够，就不做压缩
        if seq_len <= win:
            return past_key_values
    
        # 历史部分长度（可被压缩的部分）
        past_len = seq_len - win
    
        # 这一层的 hh_score：在 attention forward 里已经按最近窗口+冗余句子算好了
        hh_score = past_key_values.evict_scores[layer_idx]          # [bsz, num_kv_heads, Lh]
        assert hh_score.shape[1] == num_key_value_heads, \
            f"hh_score num_heads {hh_score.shape[1]} != num_kv_heads {num_key_value_heads}"
    
        score_len = hh_score.shape[-1]
    
        # 理论上 score_len 就应该等于 past_len；为了健壮，取两者的 min 并截断
        if score_len != past_len:
            min_len = min(score_len, past_len)
            hh_score = hh_score[..., :min_len]
            past_len = min_len
    
        # 预算不能超过历史长度
        if budget > past_len:
            budget = past_len
    
        # A zero historical quota keeps only the protected recent window.
        if budget <= 0:
            past_key_values.key_cache[layer_idx] = k_cache[:, :, -win:, :].contiguous()
            past_key_values.value_cache[layer_idx] = v_cache[:, :, -win:, :].contiguous()
            past_key_values.evict_scores[layer_idx] = hh_score[..., :0]
            return past_key_values
    
        # 在历史部分上选出分数最高的 budget 个位置（越大越重要）。
        # Shared scores use an expanded head dimension and need only one top-k.
        # KV-group scores take the per-head branch below, preserving independent
        # token positions without changing each head's retained entry count.
        if num_key_value_heads > 1 and hh_score.stride(1) == 0:
            shared_hh_score = hh_score[:, :1, :]
            shared_indices = shared_hh_score.topk(budget, dim=-1).indices
            indices = shared_indices.expand(-1, num_key_value_heads, -1)
            hh_score_compress = shared_hh_score.gather(dim=2, index=shared_indices).expand(
                -1,
                num_key_value_heads,
                -1,
            )
        else:
            indices = hh_score.topk(budget, dim=-1).indices         # [bsz, num_kv_heads, budget]
            hh_score_compress = hh_score.gather(dim=2, index=indices)
        past_key_values.evict_scores[layer_idx] = hh_score_compress
    
        # 扩展成 gather 需要的形状
        indices_exp = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    
        # 只在历史部分 [0:past_len) 上做 gather
        k_past = k_cache[:, :, :past_len, :]                        # [bsz, num_kv_heads, past_len, hd]
        v_past = v_cache[:, :, :past_len, :]
        k_past_compress = k_past.gather(dim=2, index=indices_exp)   # [bsz, num_kv_heads, budget, hd]
        v_past_compress = v_past.gather(dim=2, index=indices_exp)
    
        # 固定保留尾部窗口
        k_cur = k_cache[:, :, -win:, :]                             # [bsz, num_kv_heads, win, hd]
        v_cur = v_cache[:, :, -win:, :]
    
        # 拼接：先压缩过的历史 token，再加上最近窗口
        key_states = torch.cat([k_past_compress, k_cur], dim=2)
        value_states = torch.cat([v_past_compress, v_cur], dim=2)
    
        past_key_values.key_cache[layer_idx] = key_states
        past_key_values.value_cache[layer_idx] = value_states
    
        return past_key_values


class CakeDecodingKVCache_LayerWise:
    def __init__(
        self,
        hh_size=128,
        window_size=32,
        k_seq_dim=2,
        v_seq_dim=2,

    ):
        # print(f"CakeDecodingKVCache_LayerWise: {hh_size}, {window_size}")
        self.hh_size = hh_size
        self.window_size = window_size
        self.cache_size = hh_size + window_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_score = None

    def __call__(self, past_key_values, attn_score_cache, layer_idx):
        # =========================
        # [1] 双 forward 安全 guard
        # =========================
        if attn_score_cache is None:
            return past_key_values
            
        cur_len = past_key_values.key_cache[layer_idx].size(self.k_seq_dim)
        if cur_len <= self.cache_size:
            return past_key_values
        
        num_heads = attn_score_cache.shape[1]
        bsz, num_key_value_heads, seq_len, head_dim = past_key_values.key_cache[layer_idx].shape
        num_key_value_groups = num_heads // num_key_value_heads

        seq_len = past_key_values.key_cache[layer_idx].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values
         ####原有代码
        # attn_cache = attn_score_cache[:, :, :, :-self.window_size].mean(dim = -2)
         ####原有代码
        old_kv_len = past_key_values.key_cache[layer_idx].size(2) - self.window_size  # 计算原始 key 维度长度
        attn_cache = attn_score_cache[..., :old_kv_len]  # 保证 attn_cache 的最后一维和 key 的长度一致
        attn_cache = attn_cache.mean(dim=2)  # 进行平均计算

        attn_cache = F.avg_pool1d(attn_cache, kernel_size = 5, padding=5//2, stride=1)
        ####原有代码
        # attn_cache = attn_cache.reshape(bsz, num_key_value_heads, num_key_value_groups, -1)

        # attn_cache = attn_cache.mean(dim=-2)
         ####原有代码
        attn_cache = attn_cache.reshape(bsz, num_key_value_heads, num_key_value_groups, old_kv_len)  # reshape 时使用 `old_kv_len`
        attn_cache = attn_cache.mean(dim=-2)  # 对于 head groups 求平均

        indices = attn_cache.topk(self.hh_size, dim=-1).indices
        # indices = indices.sort().values
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

        k_past_compress = past_key_values.key_cache[layer_idx][:, :, :-self.window_size, :].gather(dim=2, index=indices)
        v_past_compress = past_key_values.value_cache[layer_idx][:, :, :-self.window_size, :].gather(dim=2, index=indices)
        k_cur = past_key_values.key_cache[layer_idx][:, :, -self.window_size:, :]
        v_cur = past_key_values.value_cache[layer_idx][:, :, -self.window_size:, :]
        key_states = torch.cat([k_past_compress, k_cur], dim=2)
        value_states = torch.cat([v_past_compress, v_cur], dim=2)

        past_key_values.key_cache[layer_idx] = key_states
        past_key_values.value_cache[layer_idx] = value_states

        return past_key_values

    def _update_hh_score(self, attn_score_cache, num_key_value_heads):

        bsz,num_heads, num_new_tokens,_ = attn_score_cache.shape
        num_key_value_groups = num_heads //  num_key_value_heads
        if self.hh_score is None:
            self.hh_score = attn_score_cache.sum(2) #bsz, total num_heads, seq_len
            self.hh_score = self.hh_score.reshape(bsz, num_key_value_heads, num_key_value_groups, -1)
            self.hh_score = self.hh_score.mean(dim=-2)
        
        else:
            # print(self.hh_score.shape)
            attn_score_cache = attn_score_cache.sum(2)
            attn_score_cache = attn_score_cache.reshape(bsz, num_key_value_heads, num_key_value_groups, -1)
            attn_score_cache = attn_score_cache.mean(dim=-2)
            attn_score_cache[:, :, :-num_new_tokens] += self.hh_score
            self.hh_score = attn_score_cache

    def _clean_scores(self):
        self.hh_score = None
