import torch
import torch.nn.functional as F

from cake.cake_cache import CakeCache, CakeprefillKVCache
from cake.utils import (
    compute_cake_attention_prefill_scores,
    compute_sentence_embeddings,
    compute_acs_from_normalized_sentences,
    compute_full_pairwise_acs_for_test,
    compute_full_pairwise_redundancy_sentence_penalty_for_benchmark,
    compute_full_pairwise_sentence_stats,
    compute_semantic_scores_for_prefill,
    split_text_into_semantic_unit_spans,
)


def test_acs_formula_matches_full_pairwise():
    torch.manual_seed(0)
    sent_emb = F.normalize(torch.randn(17, 32), dim=-1)

    full = compute_full_pairwise_acs_for_test(sent_emb)
    optimized = compute_acs_from_normalized_sentences(sent_emb)

    assert torch.allclose(full, optimized, atol=1e-5), (full.item(), optimized.item())


def test_cake_attention_prefill_scores_shapes():
    torch.manual_seed(0)
    tmp_attn_weights = torch.softmax(torch.randn(1, 4, 3, 8), dim=-1)

    pref_score, hh_score = compute_cake_attention_prefill_scores(
        tmp_attn_weights=tmp_attn_weights,
        window_size=3,
        num_key_value_heads=2,
        num_key_value_groups=2,
        gamma=200.0,
        tau1=1.0,
        tau2=1.0,
    )

    assert isinstance(pref_score, float)
    assert pref_score > 0
    assert hh_score.shape == (1, 2, 5)
    assert torch.isfinite(hh_score).all()


def test_acs_formula_uses_fp32_for_large_fp16_sentence_count():
    torch.manual_seed(0)
    sent_emb = F.normalize(torch.randn(1, 856, 64), dim=-1).half()

    full = compute_full_pairwise_acs_for_test(sent_emb.float())
    optimized = compute_acs_from_normalized_sentences(sent_emb)

    assert optimized.dtype == torch.float32
    assert torch.isfinite(optimized).all()
    assert torch.allclose(full, optimized, atol=2e-4), (full.item(), optimized.item())


def test_sentence_embedding_fast_path_matches_scatter_reference():
    torch.manual_seed(0)
    hidden_states = torch.randn(1, 7, 5)
    sentence_ids = torch.tensor([0, 0, 1, 2, 2, 2, 3])
    num_sent = 4

    sent_emb, counts = compute_sentence_embeddings(
        hidden_states,
        sentence_ids,
        num_sent=num_sent,
    )

    sid = sentence_ids.unsqueeze(0)
    ref_sum = hidden_states.new_zeros(1, num_sent, hidden_states.shape[-1])
    ref_sum.scatter_add_(
        1,
        sid.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1]),
        hidden_states,
    )
    ref_counts = hidden_states.new_zeros(1, num_sent, 1)
    ref_counts.scatter_add_(
        1,
        sid.unsqueeze(-1),
        torch.ones(1, hidden_states.shape[1], 1, dtype=hidden_states.dtype),
    )
    ref_emb = ref_sum / ref_counts.clamp_min(1.0)

    assert torch.allclose(sent_emb, ref_emb, atol=1e-6)
    assert torch.allclose(counts, ref_counts, atol=1e-6)


def test_full_pairwise_redundancy_detects_later_duplicate_only():
    sent_emb = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    sent_norm = F.normalize(sent_emb, dim=-1).unsqueeze(0)

    details = compute_full_pairwise_redundancy_sentence_penalty_for_benchmark(
        sent_norm,
        sim_threshold=0.75,
        gamma_min=0.3,
        lambda_penalty=0.7,
    )

    redundant = details["redundant_sentence_mask"][0].tolist()
    assert redundant == [False, True, False]


def test_fused_full_pairwise_stats_match_separate_acs_and_redundancy():
    torch.manual_seed(1)
    sent_norm = F.normalize(torch.randn(1, 9, 16), dim=-1)

    full_acs = compute_full_pairwise_acs_for_test(sent_norm)
    old_details = compute_full_pairwise_redundancy_sentence_penalty_for_benchmark(
        sent_norm,
        sim_threshold=0.5,
        gamma_min=0.3,
        lambda_penalty=0.7,
    )
    fused = compute_full_pairwise_sentence_stats(
        sent_norm,
        sim_threshold=0.5,
        gamma_min=0.3,
        lambda_penalty=0.7,
        return_sim_matrix=True,
    )

    assert torch.allclose(fused["avg_sim"], full_acs, atol=1e-5)
    assert torch.equal(
        fused["redundant_sentence_mask"],
        old_details["redundant_sentence_mask"],
    )
    assert torch.allclose(
        fused["sentence_gamma"],
        old_details["sentence_gamma"],
        atol=1e-6,
    )


def test_soft_gamma_range_and_non_redundant_gamma():
    sent_emb = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    sent_norm = F.normalize(sent_emb, dim=-1).unsqueeze(0)
    gamma_min = 0.3

    details = compute_full_pairwise_redundancy_sentence_penalty_for_benchmark(
        sent_norm,
        sim_threshold=0.75,
        gamma_min=gamma_min,
        lambda_penalty=0.7,
    )

    gamma = details["sentence_gamma"][0]
    assert torch.all(gamma >= gamma_min)
    assert torch.all(gamma <= 1.0)
    assert gamma[0].item() == 1.0
    assert gamma[2].item() == 1.0
    assert gamma[1].item() >= gamma_min
    assert gamma[1].item() < 1.0


def test_token_penalty_mapping_and_recent_window_exclusion():
    hidden_states = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ]
    )
    sentence_ids = torch.tensor([0, 0, 1, 1, 2, 2])
    tmp_attn_weights = torch.ones(1, 2, 2, 6)

    scores = compute_semantic_scores_for_prefill(
        tmp_attn_weights=tmp_attn_weights,
        hidden_states=hidden_states,
        sentence_ids=sentence_ids,
        window_size=2,
        num_key_value_heads=1,
        sim_threshold=0.75,
        redundancy_scale=0.3,
        redundancy_mode="full_pairwise",
        semantic_penalty_lambda=0.7,
    )

    token_gamma = scores["token_gamma"][0]
    sentence_gamma = scores["sentence_gamma"][0]
    assert torch.equal(token_gamma, sentence_gamma[sentence_ids])

    # prefill eviction score excludes the recent window, so the last two tokens
    # are preserved by the window and do not receive eviction scores here.
    hh_score = scores["hh_score"]
    assert hh_score.shape == (1, 1, 4)
    assert torch.allclose(hh_score[0, 0, :2], torch.ones(2))
    assert torch.allclose(hh_score[0, 0, 2:4], token_gamma[2:4])


def _span_texts(text, spans):
    return [text[start:end] for start, end in spans]


def test_semantic_unit_splitter_keeps_passage_blocks_together():
    text = (
        "Passage 1:\nAlpha title\nAlpha first sentence. Alpha second sentence.\n\n"
        "Passage 2:\nBeta title\nBeta first sentence. Beta second sentence."
    )

    units = _span_texts(text, split_text_into_semantic_unit_spans(text, "hotpotqa"))

    assert len(units) == 2
    assert units[0].startswith("Passage 1:")
    assert "Alpha second sentence." in units[0]
    assert units[1].startswith("Passage 2:")


def test_semantic_unit_splitter_keeps_numbered_paragraph_boundaries():
    text = (
        "Paragraph 1: first evidence sentence. second evidence sentence.\n"
        "Paragraph 2: another evidence sentence."
    )

    units = _span_texts(text, split_text_into_semantic_unit_spans(text, "passage_retrieval_en"))

    assert len(units) == 2
    assert units[0].startswith("Paragraph 1:")
    assert units[1].startswith("Paragraph 2:")


def test_semantic_unit_splitter_keeps_few_shot_examples_together():
    text = (
        "Question: Who wrote Hamlet?\nType: Individual\n"
        "Question: How many moons?\nType: Number\n"
        "Question: Current query?\nType:"
    )

    units = _span_texts(text, split_text_into_semantic_unit_spans(text, "trec"))

    assert len(units) == 3
    assert "Who wrote Hamlet?" in units[0]
    assert "Type: Individual" in units[0]
    assert units[-1].endswith("Type:")


def test_semantic_unit_splitter_uses_speaker_turns_for_meetings():
    text = (
        "Project Manager: We need a remote control design.\n"
        "Marketing: Users want speech recognition.\n"
        "Industrial Designer: Buttons are cheaper."
    )

    units = _span_texts(text, split_text_into_semantic_unit_spans(text, "qmsum"))

    assert units == [
        "Project Manager: We need a remote control design.\n",
        "Marketing: Users want speech recognition.\n",
        "Industrial Designer: Buttons are cheaper.",
    ]


def test_semantic_unit_splitter_does_not_split_code_namespaces_on_periods():
    text = (
        "using System.Collections.Generic;\n"
        "namespace Server.Engines.ConPVP\n"
        "{\n"
        "    public class DuelContext\n"
        "    {\n"
        "        private Mobile m_Initiator;\n"
        "    }\n"
        "}\n"
    )

    units = _span_texts(text, split_text_into_semantic_unit_spans(text, "lcc"))

    assert any("System.Collections.Generic;" in unit for unit in units)
    assert not any(unit == "using System." for unit in units)
    assert any("public class DuelContext" in unit for unit in units)


def test_shared_head_topk_eviction_matches_single_score_indices():
    bsz, kv_heads, seq_len, head_dim = 1, 4, 8, 2
    window_size = 2
    budget = 3
    past_len = seq_len - window_size

    k_cache = torch.arange(bsz * kv_heads * seq_len * head_dim, dtype=torch.float32).view(
        bsz,
        kv_heads,
        seq_len,
        head_dim,
    )
    v_cache = k_cache + 1000
    shared_score = torch.tensor([[[0.1, 0.9, 0.2, 0.8, 0.4, 0.7]]])
    hh_score = shared_score.expand(bsz, kv_heads, past_len)

    cache = CakeCache()
    cache.key_cache = [k_cache.clone()]
    cache.value_cache = [v_cache.clone()]
    cache.evict_scores = [hh_score]

    evictor = CakeprefillKVCache(
        cache_size=budget + window_size,
        window_size=window_size,
        num_layers=1,
    )
    evictor.evcit_layer_kvcache(cache, layer_idx=0, budget=budget)

    expected_idx = shared_score.topk(budget, dim=-1).indices
    expected_idx_exp = expected_idx.expand(bsz, kv_heads, budget).unsqueeze(-1).expand(
        -1,
        -1,
        -1,
        head_dim,
    )
    expected_k = torch.cat(
        [
            k_cache[:, :, :past_len, :].gather(dim=2, index=expected_idx_exp),
            k_cache[:, :, -window_size:, :],
        ],
        dim=2,
    )
    expected_v = torch.cat(
        [
            v_cache[:, :, :past_len, :].gather(dim=2, index=expected_idx_exp),
            v_cache[:, :, -window_size:, :],
        ],
        dim=2,
    )

    assert torch.equal(cache.key_cache[0], expected_k)
    assert torch.equal(cache.value_cache[0], expected_v)
    assert cache.evict_scores[0].stride(1) == 0


if __name__ == "__main__":
    test_acs_formula_matches_full_pairwise()
    test_acs_formula_uses_fp32_for_large_fp16_sentence_count()
    test_sentence_embedding_fast_path_matches_scatter_reference()
    test_full_pairwise_redundancy_detects_later_duplicate_only()
    test_fused_full_pairwise_stats_match_separate_acs_and_redundancy()
    test_soft_gamma_range_and_non_redundant_gamma()
    test_token_penalty_mapping_and_recent_window_exclusion()
    test_shared_head_topk_eviction_matches_single_score_indices()
    print("SAKV semantic optimization tests passed")
