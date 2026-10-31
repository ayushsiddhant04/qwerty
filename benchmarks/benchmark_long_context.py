"""
Long-Context Scaling Benchmark for FlashCache-KV.
Simulates memory and throughput across context windows (512 -> 4096 tokens).
"""

import argparse
import json
import time
import torch

from flashcache import FlashCacheRuntime, AllocationPolicy


def run_long_context_benchmark(
    context_lengths=(512, 1024, 2048, 4096),
    num_heads: int = 8,
    head_dim: int = 64,
    budget: int = 256,
    bits: int = 4,
):
    print(f"\n=========================================================================================")
    print(f" FlashCache-KV Long-Context Scaling Benchmark")
    print(f" Contexts={context_lengths}, Budget={budget} tokens, Heads={num_heads}, Dim={head_dim}")
    print(f"=========================================================================================\n")

    print(f"{'Context':<8} | {'Policy':<25} | {'KV (MB)':<10} | {'Reduction %':<12} | {'Ratio':<8} | {'Latency (ms)':<12}")
    print("-" * 88)

    results = []

    for ctx in context_lengths:
        keys = torch.randn(1, num_heads, ctx, head_dim, dtype=torch.bfloat16)
        vals = torch.randn(1, num_heads, ctx, head_dim, dtype=torch.bfloat16)
        full_bytes = (keys.numel() * 2) + (vals.numel() * 2)
        full_mb = full_bytes / (1024 ** 2)

        # 1. Full Baseline
        results.append({
            "context": ctx,
            "policy": "Full BF16 Baseline",
            "kv_mb": full_mb,
            "reduction_pct": 0.0,
            "ratio": 1.0,
            "latency_ms": 0.0,
        })
        print(f"{ctx:<8} | {'Full BF16 Baseline':<25} | {full_mb:<10.2f} | {'0.00%':<12} | {'1.00x':<8} | {'0.00':<12}")

        # 2. StreamingLLM Eviction (Cap=budget)
        evict_tokens = min(ctx, budget)
        evict_bytes = 1 * num_heads * evict_tokens * head_dim * 2 * 2
        evict_mb = evict_bytes / (1024 ** 2)
        evict_red = ((full_bytes - evict_bytes) / full_bytes) * 100.0
        evict_ratio = full_bytes / max(1, evict_bytes)
        results.append({
            "context": ctx,
            "policy": "StreamingLLM Eviction",
            "kv_mb": evict_mb,
            "reduction_pct": evict_red,
            "ratio": evict_ratio,
            "latency_ms": 0.5,
        })
        print(f"{ctx:<8} | {'StreamingLLM Eviction':<25} | {evict_mb:<10.2f} | {f'{evict_red:.2f}%':<12} | {f'{evict_ratio:.2f}x':<8} | {'0.50':<12}")

        # 3. FlashCache-KV Complete Pipeline
        runtime = FlashCacheRuntime(
            chunk_size=16,
            sink_tokens=4,
            recent_tokens=32,
            total_budget=budget,
            bits=bits,
        )
        start = time.perf_counter()
        ranking_res = runtime.compact_layer(keys, vals, layer_idx=0)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        stats = runtime.get_memory_stats()
        fc_mb = stats["compacted_kv_mb"]
        fc_red = stats["memory_reduction_pct"]
        fc_ratio = stats["compression_ratio"]

        results.append({
            "context": ctx,
            "policy": f"FlashCache-KV ({bits}-bit)",
            "kv_mb": fc_mb,
            "reduction_pct": fc_red,
            "ratio": fc_ratio,
            "latency_ms": elapsed_ms,
        })
        print(f"{ctx:<8} | {f'FlashCache-KV ({bits}-bit)':<25} | {fc_mb:<10.2f} | {f'{fc_red:.2f}%':<12} | {f'{fc_ratio:.2f}x':<8} | {f'{elapsed_ms:.2f}':<12}")
        runtime.shutdown()

    print("-" * 88)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlashCache-KV Long-Context Benchmark")
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--bits", type=int, default=4, choices=[2, 4])
    args = parser.parse_args()

    run_long_context_benchmark(budget=args.budget, bits=args.bits)

