"""
Latency and Execution Overhead Benchmark for FlashCache-KV Stages.
"""

import argparse
import time
import torch

from flashcache import FlashCacheRuntime, AllocationPolicy
from flashcache.chunking import ChunkBuilder
from flashcache.allocator import AdaptiveBudgetAllocator
from flashcache.ranking import ChunkRanker
from flashcache.compression import KIVIQuantizer, KIVIDequantizer


def run_latency_benchmark(
    seq_len: int = 1024,
    num_heads: int = 8,
    head_dim: int = 64,
    chunk_size: int = 16,
    bits: int = 4,
    trials: int = 5,
):
    print(f"\n=======================================================")
    print(f" FlashCache-KV Stage-by-Stage Latency Benchmark")
    print(f" Config: SeqLen={seq_len}, Heads={num_heads}, Dim={head_dim}, Bits={bits}")
    print(f" Trials={trials}")
    print(f"=======================================================\n")

    keys = torch.randn(1, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    values = torch.randn(1, num_heads, seq_len, head_dim, dtype=torch.bfloat16)

    # 1. Stage 1: ChunkKV Partitioning Latency
    builder = ChunkBuilder(chunk_size=chunk_size, sink_tokens=4, recent_tokens=32)
    t_stage1 = []
    for _ in range(trials):
        start = time.perf_counter()
        cands = builder.partition_layer(keys, values, layer_idx=0)
        t_stage1.append((time.perf_counter() - start) * 1000.0)
    avg_s1 = sum(t_stage1) / trials

    # 2. Stage 2: Budget Allocation Latency
    allocator = AdaptiveBudgetAllocator(policy=AllocationPolicy.IMPORTANCE_WEIGHTED)
    t_stage2 = []
    for _ in range(trials):
        start = time.perf_counter()
        budgets = allocator.allocate(total_budget=256, candidates=cands, num_layers=1, num_heads=num_heads)
        t_stage2.append((time.perf_counter() - start) * 1000.0)
    avg_s2 = sum(t_stage2) / trials

    # 3. Stage 3: Ranking Latency
    ranker = ChunkRanker()
    t_stage3 = []
    for _ in range(trials):
        start = time.perf_counter()
        ranked = ranker.rank_and_select(cands, budgets)
        t_stage3.append((time.perf_counter() - start) * 1000.0)
    avg_s3 = sum(t_stage3) / trials

    # 4. Stage 4: Quantization and Dequantization Latency per Chunk
    quantizer = KIVIQuantizer(bits=bits)
    dequantizer = KIVIDequantizer()
    sample_chunk = ranked.selected_chunks[0] if ranked.selected_chunks else cands[0]

    t_quant = []
    for _ in range(trials * 2):
        start = time.perf_counter()
        comp_k = quantizer.quantize_key(sample_chunk.key)
        comp_v = quantizer.quantize_value(sample_chunk.value)
        t_quant.append((time.perf_counter() - start) * 1000.0)
    avg_quant = sum(t_quant) / len(t_quant)

    t_dequant = []
    for _ in range(trials * 2):
        start = time.perf_counter()
        recon_k = dequantizer.dequantize(comp_k, target_dtype=torch.bfloat16)
        recon_v = dequantizer.dequantize(comp_v, target_dtype=torch.bfloat16)
        t_dequant.append((time.perf_counter() - start) * 1000.0)
    avg_dequant = sum(t_dequant) / len(t_dequant)

    # 5. Full End-to-End Compaction Latency
    runtime = FlashCacheRuntime(
        chunk_size=chunk_size,
        sink_tokens=4,
        recent_tokens=32,
        total_budget=256,
        bits=bits,
    )
    t_full = []
    for _ in range(trials):
        start = time.perf_counter()
        runtime.compact_layer(keys, values, layer_idx=0)
        t_full.append((time.perf_counter() - start) * 1000.0)
    avg_full = sum(t_full) / trials
    runtime.shutdown()

    print(f"{'Pipeline Stage / Operation':<42} | {'Latency (ms)':<15}")
    print("-" * 62)
    print(f"{'Stage 1: ChunkKV Partitioning':<42} | {avg_s1:<15.3f}")
    print(f"{'Stage 2: Adaptive Budget Allocator':<42} | {avg_s2:<15.3f}")
    print(f"{'Stage 3: Budget-Aware Chunk Ranking':<42} | {avg_s3:<15.3f}")
    print(f"{f'Stage 4: Quantization ({bits}-bit, per-chunk)':<42} | {avg_quant:<15.3f}")
    print(f"{f'Stage 4: Dequantization (per-chunk)':<42} | {avg_dequant:<15.3f}")
    print(f"{'Total End-to-End Compaction (per layer)':<42} | {avg_full:<15.3f}")
    print("-" * 62)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlashCache-KV Latency Benchmark")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--bits", type=int, default=4, choices=[2, 4])
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

    run_latency_benchmark(
        seq_len=args.seq_len,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        chunk_size=args.chunk_size,
        bits=args.bits,
        trials=args.trials,
    )

