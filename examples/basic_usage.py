"""
Minimal Usage Example for FlashCache-KV.
"""

import torch
from flashcache import FlashCacheRuntime, AllocationPolicy


def basic_example():
    # 1. Initialize FlashCache runtime with desired budget & precision
    runtime = FlashCacheRuntime(
        chunk_size=16,
        sink_tokens=4,
        recent_tokens=32,
        total_budget=128,
        bits=4,
        allocation_policy=AllocationPolicy.IMPORTANCE_WEIGHTED,
    )

    # 2. Simulate historical KV cache during autoregressive generation
    # [batch_size, num_heads, seq_len, head_dim]
    keys = torch.randn(1, 4, 256, 64, dtype=torch.bfloat16)
    values = torch.randn(1, 4, 256, 64, dtype=torch.bfloat16)

    # 3. Compact historical tokens across the five-stage pipeline
    ranking_summary = runtime.compact_layer(keys=keys, values=values, layer_idx=0)
    print(f"Compacted layer 0: {ranking_summary}")

    # 4. Access reconstructed cache views for the next decoding step
    k_cache, v_cache = runtime.get_past_key_values(layer_idx=0, target_dtype=torch.bfloat16)
    print(f"Ready for attention: Keys shape={k_cache.shape}, Values shape={v_cache.shape}")

    # 5. Check real-time physical memory savings
    stats = runtime.get_memory_stats()
    print(f"Compression Ratio: {stats['compression_ratio']:.2f}x "
          f"({stats['memory_reduction_pct']:.2f}% memory reduction)")

    runtime.shutdown()


if __name__ == "__main__":
    basic_example()

