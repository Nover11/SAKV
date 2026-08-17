import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import DynamicCache, Cache, HybridCache
from typing import Any, Dict, List, Optional, Tuple, Union
from cake.utils import adjust_budgets

class SAKV(Cache):


    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.prior_scores = []
        self.evict_scores = []
        self.layer_budget = []
    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:

        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):

        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):

        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

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
        prior_score: torch.Tensor,
        Imp_score: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    ):
        self.prior_scores.append(prior_score)
        self.evict_scores.append(Imp_score)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:


        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
   
        return None

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:

        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "SAKV":

        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache
    @classmethod
    def from_dynamic_cache(cls, past_key_values: Optional[DynamicCache] = None) -> "SAKV":
        cache = cls()
        if past_key_values is not None:
            cache.key_cache = past_key_values.key_cache
            cache.value_cache = past_key_values.value_cache

        return cache
    @classmethod
    def from_hybrid_cache(cls, past_key_values: Optional[HybridCache] = None) -> "SAKV":
        cache = cls()
        if past_key_values is not None:
            cache.key_cache = past_key_values.key_cache
            cache.value_cache = past_key_values.value_cache

        return cache
    def crop(self, max_length: int):

        # In case it is negative
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        self._seen_tokens = max_length
        for idx in range(len(self.key_cache)):
            self.key_cache[idx] = self.key_cache[idx][..., :max_length, :]
            self.value_cache[idx] = self.value_cache[idx][..., :max_length, :]

    def batch_split(self, full_batch_size: int, split_size: int) -> List["SAKV"]:

        out = []
        for i in range(0, full_batch_size, split_size):
            current_split = SAKV()
            current_split._seen_tokens = self._seen_tokens
            current_split.key_cache = [tensor[i : i + split_size] for tensor in self.key_cache]
            current_split.value_cache = [tensor[i : i + split_size] for tensor in self.value_cache]
            current_split.prior_scores = self.prior_scores
            current_split.evict_scores = self.evict_scores
            out.append(current_split)
        return out

    @classmethod
    def from_batch_splits(cls, splits: List["SAKV"]) -> "SAKV":

        cache = cls()
        for idx in range(len(splits[0])):
            layer_keys = torch.cat([current.key_cache[idx] for current in splits], dim=0)
            layer_values = torch.cat([current.value_cache[idx] for current in splits], dim=0)
            cache.update(layer_keys, layer_values, idx)
            cache.prior_scores = self.prior_scores
            cache.evict_scores = self.evict_scores
        return cache

    def batch_repeat_interleave(self, repeats: int):

        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].repeat_interleave(repeats, dim=0)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor):

        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][indices, ...]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][indices, ...]



class SAKV_pre_KVCache:
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
    
        self.layer_sim_sum = [0.0 for _ in range(num_layers)]
        self.layer_sim_cnt = [0   for _ in range(num_layers)]
        self.use_cascading = use_cascading  # If true, ensure high attention precision

        self.alpha = 1  
        self.beta = 0.5    


    def reset(self):
        self.layer_sim_sum = [0.0 for _ in range(self.num_layers)]
        self.layer_sim_cnt = [0   for _ in range(self.num_layers)]
        
    def __call__(self, past_key_values, seq_len, layer_idx, budget):

        cur_len = past_key_values.key_cache[layer_idx].size(2)
        if cur_len <= budget + self.window_size:
            return past_key_values

        if seq_len<=self.cache_size+self.window_size:
            return past_key_values

        prior_scores = past_key_values.prior_scores
        
        
        alpha = self.alpha
        beta = self.beta
        
        L = len(prior_scores)
        B = self.total_size
        
        eps = 1e-6
        prior_scores_safe = [max(s, eps) for s in prior_scores]
        
        base_budget = beta * B / L
        
        pow_scores = [s ** alpha for s in prior_scores_safe]
        pow_sum = sum(pow_scores)
        
        layer_budgets = [
            base_budget + (1 - beta) * B * (s_pow / pow_sum)
            for s_pow in pow_scores
        ]     
        layer_budgets = adjust_budgets(layer_budgets, self.total_size, seq_len-self.window_size,  self.num_layers)

        if self.use_cascading:
            layer_idx = 0
            print(layer_budgets)
            for budget in layer_budgets:
                if budget>= seq_len-self.window_size:
                    budget = seq_len-self.window_size
                past_key_values = self.evcit_layer_kvcache(past_key_values, layer_idx, budget)
                past_key_values.layer_budget[layer_idx]=budget 
                layer_idx +=1  
        else:
            layer_idx = 0
            if len(layer_budgets) ==self.num_layers:
                for budget in layer_budgets:
                    if budget>= seq_len-self.window_size:
                        budget = seq_len-self.window_size
                    past_key_values = self.evcit_layer_kvcache(past_key_values, layer_idx, budget)
                    past_key_values.layer_budget[layer_idx]=budget
                    layer_idx +=1

        return past_key_values




    def evcit_layer_kvcache(self, past_key_values, layer_idx, budget):

        k_cache = past_key_values.key_cache[layer_idx]
        v_cache = past_key_values.value_cache[layer_idx]
        bsz, num_key_value_heads, seq_len, head_dim = k_cache.shape
    
        win = self.window_size
    
        if seq_len <= win:
            return past_key_values
    
        past_len = seq_len - win
    
        sa_score = past_key_values.evict_scores[layer_idx]          
        assert sa_score.shape[1] == num_key_value_heads, \
            f"hh_score num_heads {sa_score.shape[1]} != num_kv_heads {num_key_value_heads}"
    
        score_len = sa_score.shape[-1]
    
        if score_len != past_len:
            min_len = min(score_len, past_len)
            sa_score = sa_score[..., :min_len]
            past_len = min_len
    

        if budget > past_len:
            budget = past_len
    

        if budget <= 0:
            return past_key_values
    

        indices = sa_score.topk(budget, dim=-1).indices             
        hh_score_compress = sa_score.gather(dim=2, index=indices)
        past_key_values.evict_scores[layer_idx] = hh_score_compress
    

        indices_exp = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

        k_past = k_cache[:, :, :past_len, :]                        
        v_past = v_cache[:, :, :past_len, :]
        k_past_compress = k_past.gather(dim=2, index=indices_exp)   
        v_past_compress = v_past.gather(dim=2, index=indices_exp)

        k_cur = k_cache[:, :, -win:, :]                            
        v_cur = v_cache[:, :, -win:, :]
    

        key_states = torch.cat([k_past_compress, k_cur], dim=2)
        value_states = torch.cat([v_past_compress, v_cur], dim=2)
    
        past_key_values.key_cache[layer_idx] = key_states
        past_key_values.value_cache[layer_idx] = value_states
    
        return past_key_values


class SAKV_Dec_KVCache:
    def __init__(
        self,
        hh_size=128,
        window_size=32,
        k_seq_dim=2,
        v_seq_dim=2,

    ):
        self.hh_size = hh_size
        self.window_size = window_size
        self.cache_size = hh_size + window_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_score = None

    def __call__(self, past_key_values, attn_score_cache, layer_idx):

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

        old_kv_len = past_key_values.key_cache[layer_idx].size(2) - self.window_size 
        attn_cache = attn_score_cache[..., :old_kv_len]  
        attn_cache = attn_cache.mean(dim=2)  

        attn_cache = F.avg_pool1d(attn_cache, kernel_size = 5, padding=5//2, stride=1)
        attn_cache = attn_cache.reshape(bsz, num_key_value_heads, num_key_value_groups, old_kv_len)  
        attn_cache = attn_cache.mean(dim=-2) 

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
            self.hh_score = attn_score_cache.sum(2) 
            self.hh_score = self.hh_score.reshape(bsz, num_key_value_heads, num_key_value_groups, -1)
            self.hh_score = self.hh_score.mean(dim=-2)
        
        else:
            attn_score_cache = attn_score_cache.sum(2)
            attn_score_cache = attn_score_cache.reshape(bsz, num_key_value_heads, num_key_value_groups, -1)
            attn_score_cache = attn_score_cache.mean(dim=-2)
            attn_score_cache[:, :, :-num_new_tokens] += self.hh_score
            self.hh_score = attn_score_cache

    def _clean_scores(self):
        self.hh_score = None
