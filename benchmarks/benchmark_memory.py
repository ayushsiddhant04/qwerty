"""
Comparative Memory Benchmark for FlashCache-KV.

Compares:
A. Full BF16 KV Cache (No eviction, unbounded)
B. Reference Token Eviction (StreamingLLM: sink + local window)
C. ChunkKV Without Compression (Adaptive ranking, BF16 retained)
D. ChunkKV With Compression (Adaptive ranking, 4-bit INT)
E. Complete FlashCache-KV Pipeline (Adaptive ranking + 4b/2b KIVI + safety barrier)
"""

import argparse
import json
import os
import sys
import torch

from flashcache import FlashCacheRuntime, AllocationPolicy
from flashcache.chunking import ChunkBuilder
from flashcache.allocator import AdaptiveBudgetAllocator
from flashcache.ranking import ChunkRanker
from flashcache.compression import KIVIQuantizer
from flashcache.runtime import MemoryMonitor


def run_memory_benchmark(
    seq_len: int = 2048,
    num_layers: int = 4,
    num_heads: int = 8,
    head_dim: int = 64,
    chunk_size: int = 16,
    sink_tokens: int = 4,
    recent_tokens: int = 32,
    bits: int = 4,
    target_capacity: int = 256,
    device: str = "cpu",
):
    print(f"\n=======================================================")
    print(f" FlashCache-KV Comparative Memory Benchmark")
    print(f" Config: SeqLen={seq_len}, Layers={num_layers}, Heads={num_heads}, Dim={head_dim}")
    print(f" Target Budget={target_capacity} tokens, ChunkSize={chunk_size}, Bits={bits}")
    print(f" Device={device}")
    print(f"=======================================================\n")

    dtype = torch.bfloat16
    batch_size = 1

    # Synthetic Key and Value generation across layers
    layer_keys = [torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device) for _ in range(num_layers)]
    layer_vals = [torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device) for _ in range(num_layers)]

    results = []

    # -----------------------------------------------------------------
    # Baseline A: Full BF16 KV Cache
    # -----------------------------------------------------------------
    full_bytes = sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in zip(layer_keys, layer_vals))
    results.append({
        "strategy": "Full BF16 Cache (Baseline A)",
        "retained_tokens": seq_len,
        "physical_bytes": full_bytes,
        "physical_mb": full_bytes / (1024 ** 2),
        "compression_ratio": 1.0,
        "reduction_pct": 0.0,
        "precision": "BF16 (16-bit)",
    })

    # -----------------------------------------------------------------
    # Baseline B: StreamingLLM Eviction (Cap=target_capacity, Sink=sink_tokens)
    # -----------------------------------------------------------------
    # Retains sink_tokens front + (target_capacity - sink_tokens) tail in BF16
    evicted_retained_tokens = min(seq_len, target_capacity)
    evicted_bytes = num_layers * 2 * (batch_size * num_heads * evicted_retained_tokens * head_dim * 2)
    evict_ratio = full_bytes / max(1, evicted_bytes)
    evict_reduct = ((full_bytes - evicted_bytes) / full_bytes) * 100.0
    results.append({
        "strategy": "StreamingLLM Eviction (Baseline B)",
        "retained_tokens": evicted_retained_tokens,
        "physical_bytes": evicted_bytes,
        "physical_mb": evicted_bytes / (1024 ** 2),
        "compression_ratio": evict_ratio,
        "reduction_pct": evict_reduct,
        "precision": "BF16 (16-bit)",
    })

    # -----------------------------------------------------------------
    # Approach C: ChunkKV Without Compression (Adaptive Ranking in BF16)
    # -----------------------------------------------------------------
    builder = ChunkBuilder(chunk_size=chunk_size, sink_tokens=sink_tokens, recent_tokens=recent_tokens)
    allocator = AdaptiveBudgetAllocator(policy=AllocationPolicy.IMPORTANCE_WEIGHTED)
    ranker = ChunkRanker(budget_includes_protected=True)

    c_bytes = 0
    c_tokens = 0
    for l in range(num_layers):
        cands = builder.partition_layer(layer_keys[l], layer_vals[l], layer_idx=l)
        h_budgets = allocator.allocate(total_budget=target_capacity, candidates=cands, num_layers=1, num_heads=num_heads)
        ranked = ranker.rank_and_select(cands, h_budgets)
        c_bytes += ranked.retained_bytes
        c_tokens = ranked.total_retained_tokens // num_heads

    results.append({
        "strategy": "ChunkKV Uncompressed (Approach C)",
        "retained_tokens": c_tokens,
        "physical_bytes": c_bytes,
        "physical_mb": c_bytes / (1024 ** 2),
        "compression_ratio": full_bytes / max(1, c_bytes),
        "reduction_pct": ((full_bytes - c_bytes) / full_bytes) * 100.0,
        "precision": "BF16 (16-bit)",
    })

    # -----------------------------------------------------------------
    # Approach D: ChunkKV With 4-bit Quantization (No Async Safety Barrier)
    # -----------------------------------------------------------------
    quantizer = KIVIQuantizer(bits=bits)
    d_bytes = 0
    for l in range(num_layers):
        cands = builder.partition_layer(layer_keys[l], layer_vals[l], layer_idx=l)
        h_budgets = allocator.allocate(total_budget=target_capacity, candidates=cands, num_layers=1, num_heads=num_heads)
        ranked = ranker.rank_and_select(cands, h_budgets)
        # Protected in BF16
        for p in ranked.protected_chunks:
            d_bytes += p.uncompressed_bytes
        # Selected in 4-bit
        for s in ranked.selected_chunks:
            ck = quantizer.quantize_key(s.key)
            cv = quantizer.quantize_value(s.value)
            d_bytes += ck.total_bytes + cv.total_bytes

    results.append({
        "strategy": f"ChunkKV + {bits}-bit Quantized (Approach D)",
        "retained_tokens": c_tokens,
        "physical_bytes": d_bytes,
        "physical_mb": d_bytes / (1024 ** 2),
        "compression_ratio": full_bytes / max(1, d_bytes),
        "reduction_pct": ((full_bytes - d_bytes) / full_bytes) * 100.0,
        "precision": f"{bits}-bit INT + FP16 Scales",
    })

    # -----------------------------------------------------------------
    # Approach E: Complete FlashCache-KV Pipeline
    # -----------------------------------------------------------------
    runtime = FlashCacheRuntime(
        chunk_size=chunk_size,
        sink_tokens=sink_tokens,
        recent_tokens=recent_tokens,
        total_budget=target_capacity,
        bits=bits,
        allocation_policy=AllocationPolicy.IMPORTANCE_WEIGHTED,
    )
    for l in range(num_layers):
        runtime.compact_layer(layer_keys[l], layer_vals[l], layer_idx=l)

    stats = runtime.get_memory_stats()
    e_bytes = stats["compacted_kv_bytes"]
    results.append({
        "strategy": "FlashCache-KV Full Pipeline (Approach E)",
        "retained_tokens": c_tokens,
        "physical_bytes": e_bytes,
        "physical_mb": e_bytes / (1024 ** 2),
        "compression_ratio": full_bytes / max(1, e_bytes),
        "reduction_pct": ((full_bytes - e_bytes) / full_bytes) * 100.0,
        "precision": f"Adaptive {bits}-bit / BF16 Sinks",
    })
    runtime.shutdown()

    # Format table output
    print(f"{'Strategy':<38} | {'Retained':<9} | {'Memory (MB)':<11} | {'Comp Ratio':<10} | {'Reduction %':<11}")
    print("-" * 92)
    for r in results:
        print(f"{r['strategy']:<38} | {r['retained_tokens']:<9} | {r['physical_mb']:<11.3f} | {r['compression_ratio']:<10.2f}x | {r['reduction_pct']:<11.2f}%")
    print("-" * 92)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlashCache-KV Memory Benchmark")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length in tokens")
    parser.add_argument("--num-layers", type=int, default=4, help="Layer count")
    parser.add_argument("--num-heads", type=int, default=8, help="KV head count")
    parser.add_argument("--head-dim", type=int, default=64, help="Head dimension")
    parser.add_argument("--chunk-size", type=int, default=16, help="Chunk size in tokens")
    parser.add_argument("--bits", type=int, default=4, choices=[2, 4], help="Quantization bits")
    parser.add_argument("--budget", type=int, default=256, help="Target token budget per layer")
    parser.add_argument("--json-out", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    res = run_memory_benchmark(
        seq_len=args.seq_len,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        chunk_size=args.chunk_size,
        bits=args.bits,
        target_capacity=args.budget,
    )

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nSaved benchmark results to {args.json_out}")

