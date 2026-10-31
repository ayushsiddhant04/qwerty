"""
End-to-End Five-Stage Integration Tests for FlashCache-KV.
"""

import pytest
import torch

from flashcache import FlashCacheRuntime, AllocationPolicy, RankingCriteria


@pytest.mark.parametrize("bits", [4, 2])
@pytest.mark.parametrize("policy", [AllocationPolicy.UNIFORM, AllocationPolicy.IMPORTANCE_WEIGHTED])
def test_full_pipeline_multi_layer(bits, policy):
    """
    Validates end-to-end execution of the complete 5-stage FlashCache-KV pipeline
    across multiple layers and precision modes.
    """
    runtime = FlashCacheRuntime(
        chunk_size=16,
        sink_tokens=4,
        recent_tokens=16,
        total_budget=80,
        bits=bits,
        allocation_policy=policy,
        enable_async=False,
    )

    batch_size = 1
    num_heads = 4
    seq_len = 160
    head_dim = 64
    num_layers = 2

    for l in range(num_layers):
        keys = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
        values = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)

        # Stage 1 -> 5 execution
        result = runtime.compact_layer(keys=keys, values=values, layer_idx=l)
        assert result.total_retained_tokens <= 80

        # Retrieve reconstructed cache for layer
        k_view, v_view = runtime.get_past_key_values(layer_idx=l, target_dtype=torch.bfloat16)
        assert k_view is not None
        assert v_view is not None
        assert k_view.shape[0] == batch_size
        assert k_view.shape[1] == num_heads

    stats = runtime.get_memory_stats()
    assert stats["compression_ratio"] > 1.5
    assert stats["memory_reduction_pct"] > 30.0
    assert stats["total_reclaimed_bytes"] > 0

    runtime.shutdown()


def test_consecutive_decoding_simulation():
    """
    Simulates consecutive autoregressive token decoding steps where new tokens
    are appended and compaction is triggered periodically.
    """
    runtime = FlashCacheRuntime(
        chunk_size=8,
        sink_tokens=4,
        recent_tokens=8,
        total_budget=32,
        bits=4,
    )

    batch_size = 1
    num_heads = 2
    head_dim = 32

    # Simulate accumulation of 48 tokens
    cur_keys = torch.randn(batch_size, num_heads, 48, head_dim, dtype=torch.float32)
    cur_values = torch.randn(batch_size, num_heads, 48, head_dim, dtype=torch.float32)

    runtime.compact_layer(cur_keys, cur_values, layer_idx=0)
    k_view, v_view = runtime.get_past_key_values(layer_idx=0)

    # Decode 5 additional tokens
    for _ in range(5):
        next_k = torch.randn(batch_size, num_heads, 1, head_dim, dtype=torch.float32)
        next_v = torch.randn(batch_size, num_heads, 1, head_dim, dtype=torch.float32)
        k_view = torch.cat([k_view, next_k], dim=-2)
        v_view = torch.cat([v_view, next_v], dim=-2)

    # Run compaction on extended cache
    res = runtime.compact_layer(k_view, v_view, layer_idx=0)
    assert res.total_retained_tokens <= 32

    final_k, final_v = runtime.get_past_key_values(layer_idx=0)
    assert final_k.shape[2] == res.total_retained_tokens // num_heads

    runtime.shutdown()
