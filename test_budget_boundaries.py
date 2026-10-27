import torch

from cake.cake_cache import CakeCache, CakeprefillKVCache
from cake.utils import adjust_budgets


def test_budget_redistribution_respects_capacity_and_total():
    for requested, total, capacity in [([100, 9, 0], 25, 10), ([1, 9, 9], 2, 10),
                                        ([3, 1], 100, 5), ([.9, .9, .9], 2, 4)]:
        actual = adjust_budgets(requested, total, capacity, len(requested))
        assert all(0 <= value <= capacity for value in actual)
        assert sum(actual) == min(total, len(requested) * capacity)


def test_zero_historical_budget_preserves_only_recent_window():
    cache = CakeCache()
    original = torch.arange(6.).reshape(1, 1, 6, 1)
    cache.key_cache = [original.clone()]
    cache.value_cache = [original.clone()]
    cache.evict_scores = [torch.ones(1, 1, 4)]
    allocator = CakeprefillKVCache(cache_size=3, window_size=2, num_layers=1)
    allocator.evcit_layer_kvcache(cache, 0, 0)
    assert torch.equal(cache.key_cache[0], original[:, :, -2:])
    assert cache.evict_scores[0].shape[-1] == 0
