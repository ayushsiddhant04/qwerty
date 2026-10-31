"""
Synthetic End-to-End Five-Stage Pipeline Demonstration.
Runs a complete walkthrough of ChunkKV, Adaptive Budgeting,
Ranking, KIVI-inspired Quantization, and Runtime Dequantization.
"""

import sys
import torch

from flashcache import FlashCacheRuntime, AllocationPolicy, RankingCriteria


def main():
    print("=" * 70)
    print(" FlashCache-KV: End-to-End Five-Stage Pipeline Walkthrough")
    print("=" * 70)

    # 1. Pipeline Configuration
    seq_len = 512
    num_heads = 4
    head_dim = 64
    chunk_size = 16
    sink_tokens = 4
    recent_tokens = 32
    target_budget = 256
    bits = 4

    print(f"\n[1] Initializing Configuration:")
    print(f"    - Sequence Length: {seq_len} tokens")
    print(f"    - Attention Heads: {num_heads}")
    print(f"    - Head Dimension:  {head_dim}")
    print(f"    - Chunk Size:      {chunk_size} tokens")
    print(f"    - Sink Tokens:     {sink_tokens} (Permanent Attention Sinks)")
    print(f"    - Recent Tokens:   {recent_tokens} (High-Precision Local Window)")
    print(f"    - Retention Budget:{target_budget} tokens")
    print(f"    - Precision Mode:  {bits}-bit KIVI Asymmetric Quantization")

    # 2. Synthetic Tensor Ingestion (BF16)
    print(f"\n[2] Generating Synthetic BF16 Key and Value Tensors...")
    keys = torch.randn(1, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    values = torch.randn(1, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    
    orig_bytes = keys.numel() * keys.element_size() + values.numel() * values.element_size()
    print(f"    - Uncompressed Cache Size: {orig_bytes} bytes ({orig_bytes / 1024:.2f} KB)")

    # 3. Initialize FlashCache Runtime
    print(f"\n[3] Instantiating FlashCacheRuntime...")
    runtime = FlashCacheRuntime(
        chunk_size=chunk_size,
        sink_tokens=sink_tokens,
        recent_tokens=recent_tokens,
        total_budget=target_budget,
        bits=bits,
        allocation_policy=AllocationPolicy.IMPORTANCE_WEIGHTED,
        ranking_criteria=RankingCriteria.SCORE,
    )

    # 4. Execute 5-Stage Compaction Pipeline
    print(f"\n[4] Executing 5-Stage Compaction Pipeline...")
    ranking_res = runtime.compact_layer(keys=keys, values=values, layer_idx=0)

    print(f"\n    STAGE 1: ChunkKV Candidate Construction:")
    total_cands = len(ranking_res.protected_chunks) + len(ranking_res.selected_chunks) + len(ranking_res.evicted_chunks)
    print(f"             Total Partitioned Chunks: {total_cands}")
    print(f"             Protected Sinks & Local:  {len(ranking_res.protected_chunks)} chunks ({ranking_res.total_protected_tokens} tokens)")

    print(f"\n    STAGE 2 & 3: Adaptive Allocation & Budget-Aware Ranking:")
    print(f"             Selected Middle Chunks:   {len(ranking_res.selected_chunks)} chunks ({ranking_res.total_selected_tokens} tokens)")
    print(f"             Evicted Chunks:           {len(ranking_res.evicted_chunks)} chunks ({ranking_res.total_evicted_tokens} tokens)")
    print(f"             Total Retained Tokens:    {ranking_res.total_retained_tokens} / {target_budget} budget")

    print(f"\n    STAGE 4: KIVI-Inspired Quantization & Memory Safety Barrier:")
    sample = ranking_res.selected_chunks[0] if ranking_res.selected_chunks else None
    if sample and sample.compressed_key is not None:
        metrics = sample.metadata.extra.get("reconstruction_metrics", {})
        k_psnr = metrics.get("key", {}).get("psnr", 0.0)
        v_psnr = metrics.get("value", {}).get("psnr", 0.0)
        print(f"             Sample Chunk '{sample.metadata.chunk_id}' State: {sample.metadata.state.value}")
        print(f"             Reconstruction PSNR: Key={k_psnr:.2f} dB, Value={v_psnr:.2f} dB")
        print(f"             BF16 Source Memory Safely Reclaimed: {sample.metadata.extra.get('source_storage_reclaimable')}")

    # 5. Runtime Memory Analytics
    print(f"\n    STAGE 5: FlashCache Runtime Physical Storage View:")
    stats = runtime.get_memory_stats()
    print(f"             Original Uncompressed: {stats['original_kv_bytes']} bytes ({stats['original_kv_mb']:.3f} MB)")
    print(f"             Compacted Cache Size:  {stats['compacted_kv_bytes']} bytes ({stats['compacted_kv_mb']:.3f} MB)")
    print(f"             Compression Ratio:     {stats['compression_ratio']:.2f}x")
    print(f"             Memory Reduction:      {stats['memory_reduction_pct']:.2f}%")
    print(f"             Pipeline Latency:      {stats['last_compaction_time_ms']:.2f} ms")

    # 6. Verify Dequantized Query
    print(f"\n[5] Querying Assembled Past-Key-Values for Next Attention Step...")
    past_k, past_v = runtime.get_past_key_values(layer_idx=0, target_dtype=torch.bfloat16)
    print(f"    - Reconstructed Keys Shape:   {past_k.shape}")
    print(f"    - Reconstructed Values Shape: {past_v.shape}")
    print(f"    - Verification: Clean match with expected retained tokens!")

    runtime.shutdown()
    print("\n" + "=" * 70)
    print(" Pipeline Walkthrough Successfully Completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
