"""CPU checks for scoring semantics and per-head cache gathering.

Run from the repository root with:
    python -m unittest discover -s tests -p test_sakv_kv_group_scoring.py -v
"""
import unittest

import torch

from cake.cake_cache import CakeCache, CakeprefillKVCache
from cake.utils import compute_semantic_scores_for_prefill


class KVGroupScoringTests(unittest.TestCase):
    def setUp(self):
        # The first KV group prefers historical token 0; the second prefers 2.
        heads = torch.tensor([
            [.60, .10, .05, .05, .10, .10],
            [.40, .30, .05, .05, .10, .10],
            [.02, .03, .70, .05, .10, .10],
            [.04, .01, .70, .05, .10, .10],
        ])
        self.attention = heads[None, :, None, :].expand(1, 4, 2, 6).contiguous()
        # Unit 1 repeats unit 0, so the optional semantic penalty is nontrivial.
        self.hidden = torch.tensor([[[1., 0.], [1., 0.], [1., 0.],
                                     [1., 0.], [0., 1.], [0., 1.]]])
        self.sentence_ids = torch.tensor([0, 0, 1, 1, 2, 2])

    def scores(self, mode=None, kv_heads=2, penalty=0., attention=None, window=2):
        kwargs = {} if mode is None else {"head_score_mode": mode}
        return compute_semantic_scores_for_prefill(
            tmp_attn_weights=self.attention if attention is None else attention,
            hidden_states=self.hidden,
            sentence_ids=self.sentence_ids,
            window_size=window,
            num_key_value_heads=kv_heads,
            sim_threshold=.75,
            redundancy_scale=.3,
            semantic_penalty_lambda=penalty,
            **kwargs,
        )

    def test_distinct_groups_have_distinct_token_preferences(self):
        result = self.scores("kv_group")
        expected = torch.tensor([[[.50, .20, .05, .05],
                                  [.03, .02, .70, .05]]])
        self.assertTrue(torch.allclose(result["hh_score"], expected))
        self.assertEqual(result["hh_score"].argmax(-1).tolist(), [[0, 2]])

    def test_default_preserves_shared_ranking_and_fast_path(self):
        default = self.scores()["hh_score"]
        explicit = self.scores("shared")["hh_score"]
        expected = torch.tensor([[[.265, .11, .375, .05],
                                  [.265, .11, .375, .05]]])
        self.assertTrue(torch.equal(default, explicit))
        self.assertTrue(torch.allclose(default, expected))
        self.assertEqual(default.argmax(-1).tolist(), [[2, 2]])
        self.assertEqual(default.stride(1), 0)

    def test_group_mode_preserves_acs_and_semantic_penalty(self):
        shared = self.scores("shared", penalty=.7)
        grouped = self.scores("kv_group", penalty=.7)
        for key in ("pref_score", "avg_sim", "sentence_gamma", "token_gamma"):
            self.assertTrue(torch.equal(shared[key], grouped[key]), key)
        self.assertLess(grouped["token_gamma"][0, 2].item(), 1.)
        expected = torch.tensor([[[.50, .20, .05, .05],
                                  [.03, .02, .70, .05]]])
        expected = expected * grouped["token_gamma"][:, :4].unsqueeze(1)
        self.assertTrue(torch.allclose(grouped["hh_score"], expected))

    def test_single_kv_head_matches_shared_mode(self):
        self.assertTrue(torch.allclose(
            self.scores("shared", kv_heads=1)["hh_score"],
            self.scores("kv_group", kv_heads=1)["hh_score"],
        ))

    def test_mha_keeps_query_heads_separate(self):
        result = self.scores("kv_group", kv_heads=4)["hh_score"]
        self.assertTrue(torch.allclose(result, self.attention[:, :, 0, :4]))

    def test_invalid_mode_and_head_counts_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "head_score_mode"):
            self.scores("typo")
        for count in (0, 3, 8):
            with self.subTest(kv_heads=count):
                with self.assertRaisesRegex(ValueError, "positive multiple"):
                    self.scores("kv_group", kv_heads=count)

    def test_no_history_has_empty_scores(self):
        attention = self.attention[:, :, :1, :].expand(1, 4, 6, 6)
        result = self.scores("kv_group", attention=attention, window=6)
        self.assertEqual(tuple(result["hh_score"].shape), (1, 2, 0))

    def test_eviction_keeps_different_positions_and_same_window_and_capacity(self):
        # Encode original position in the key and head identity in the value.
        original_keys = torch.arange(6, dtype=torch.float32).view(1, 1, 6, 1).expand(1, 2, 6, 1).clone()
        original_values = original_keys + torch.tensor([100., 200.]).view(1, 2, 1, 1)
        for mode, expected_positions in (("shared", [2., 2.]), ("kv_group", [0., 2.])):
            with self.subTest(mode=mode):
                cache = CakeCache()
                cache.key_cache = [original_keys.clone()]
                cache.value_cache = [original_values.clone()]
                cache.evict_scores = [self.scores(mode)["hh_score"]]
                evictor = CakeprefillKVCache(cache_size=4, window_size=2, num_layers=1)
                # Exercise repeated eviction, as used by progressive allocation.
                evictor.evcit_layer_kvcache(cache, layer_idx=0, budget=2)
                evictor.evcit_layer_kvcache(cache, layer_idx=0, budget=1)
                self.assertEqual(tuple(cache.key_cache[0].shape), (1, 2, 3, 1))
                self.assertEqual(cache.key_cache[0][0, :, 0, 0].tolist(), expected_positions)
                self.assertTrue(torch.equal(cache.key_cache[0][:, :, -2:], original_keys[:, :, -2:]))
                self.assertTrue(torch.equal(cache.value_cache[0][:, :, -2:], original_values[:, :, -2:]))
                self.assertEqual(cache.value_cache[0][0, :, 0, 0].tolist(),
                                 [expected_positions[0]+100., expected_positions[1]+200.])
                expected_scores = [.375, .375] if mode == "shared" else [.5, .7]
                self.assertTrue(torch.allclose(cache.evict_scores[0].reshape(-1), torch.tensor(expected_scores)))


if __name__ == "__main__":
    unittest.main()
