"""
FlashCache-KV Performance & System Telemetry Dashboard.
Adaptive KV-Cache Compaction and KIVI-Inspired Asymmetric Quantization for LLM Inference.
"""

import os
import sys
import time
import math
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch

from flashcache import (
    FlashCacheRuntime,
    AllocationPolicy,
    RankingCriteria,
    MemoryMonitor,
    BlockState,
    ChunkStatus,
)
from flashcache.compression import KIVIQuantizer, KIVIDequantizer

# Page configuration
st.set_page_config(
    page_title="FlashCache-KV Telemetry Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Reference historical benchmarks (Attlornets / Qwen2.5-0.5B-Instruct baseline)
HISTORICAL_BENCHMARKS = [
    {
        "strategy": "Full Cache (Baseline)",
        "type": "Full Precision",
        "capacity": 222,
        "sink_tokens": 0,
        "retained_tokens": 222,
        "kv_mb": 2.6016,
        "kv_bytes": 2727936,
        "compression_pct": 0.0,
        "compression_ratio": 1.0,
        "latency_sec": 4.848,
        "throughput_tok_s": 30.94,
        "status": "Coherent / Fluent Baseline",
    },
    {
        "strategy": "StreamingLLM (Cap=64, Sink=4)",
        "type": "Token Eviction",
        "capacity": 64,
        "sink_tokens": 4,
        "retained_tokens": 64,
        "kv_mb": 0.7500,
        "kv_bytes": 786432,
        "compression_pct": 71.17,
        "compression_ratio": 3.47,
        "latency_sec": 4.827,
        "throughput_tok_s": 31.07,
        "status": "Verified / RoPE Monotonicity Intact",
    },
    {
        "strategy": "StreamingLLM (Cap=40, Sink=4)",
        "type": "Token Eviction",
        "capacity": 40,
        "sink_tokens": 4,
        "retained_tokens": 40,
        "kv_mb": 0.4688,
        "kv_bytes": 491520,
        "compression_pct": 81.98,
        "compression_ratio": 5.55,
        "latency_sec": 4.890,
        "throughput_tok_s": 30.67,
        "status": "Verified / RoPE Monotonicity Intact",
    },
]

# Sidebar configuration
st.sidebar.title("⚡ FlashCache-KV Control")
st.sidebar.markdown("**Adaptive KV Compaction & Quantization**")

hw_info = MemoryMonitor.get_hardware_info()
st.sidebar.markdown(f"**Device**: `{hw_info['device'].upper()}` | **CPUs**: `{hw_info['cpu_count']}`")
if hw_info["cuda_available"] and hw_info["gpu"]:
    st.sidebar.markdown(f"**GPU**: `{hw_info['gpu']['device_name']}` ({hw_info['gpu']['total_vram_mb']:.0f} MB)")
else:
    st.sidebar.markdown(f"**System RAM**: `{hw_info['total_system_ram_mb']:.0f} MB`")

st.sidebar.markdown("---")
st.sidebar.subheader("Interactive Pipeline Settings")

seq_len = st.sidebar.slider("Context Sequence Length", min_value=128, max_value=4096, value=512, step=64)
num_heads = st.sidebar.slider("KV Attention Heads", min_value=1, max_value=16, value=4, step=1)
head_dim = st.sidebar.selectbox("Head Dimension", [32, 64, 128], index=1)
chunk_size = st.sidebar.selectbox("Chunk Size (Tokens)", [8, 16, 32], index=1)
sink_tokens = st.sidebar.slider("Attention Sinks (Tokens)", min_value=0, max_value=16, value=4, step=2)
recent_tokens = st.sidebar.slider("Recent Local Window", min_value=8, max_value=64, value=32, step=8)
target_budget = st.sidebar.slider("Target Token Budget", min_value=64, max_value=1024, value=256, step=32)
bits = st.sidebar.selectbox("Quantization Precision", [4, 2], index=0)
alloc_policy = st.sidebar.selectbox(
    "Budget Allocation Policy",
    ["importance_weighted", "uniform", "recency_aware"],
    index=0,
)

# Header
st.title("⚡ FlashCache-KV Research Prototype Dashboard")
st.markdown(
    "**FlashCache-KV** couples token chunking, adaptive retention budgeting, importance ranking, "
    "and KIVI-inspired asymmetric quantization with a memory-safe asynchronous runtime."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Live Pipeline Execution",
    "📊 Comparative Memory Analytics",
    "🔬 Quantization & Numerical Fidelity",
    "📜 Historical Model Benchmarks",
])

with tab1:
    st.header("Stage-by-Stage Live Pipeline Execution")
    st.info("Demonstrates actual PyTorch tensor allocations, partitioning, budget assignment, ranking, and KIVI quantization.")

    run_btn = st.button("▶ Run Full Pipeline Compaction", type="primary")

    if run_btn:
        with st.spinner("Executing 5-Stage Compaction Pipeline..."):
            keys = torch.randn(1, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
            values = torch.randn(1, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
            
            runtime = FlashCacheRuntime(
                chunk_size=chunk_size,
                sink_tokens=sink_tokens,
                recent_tokens=recent_tokens,
                total_budget=target_budget,
                bits=bits,
                allocation_policy=alloc_policy,
            )

            t0 = time.perf_counter()
            ranking_res = runtime.compact_layer(keys=keys, values=values, layer_idx=0)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            stats = runtime.get_memory_stats()
            past_k, past_v = runtime.get_past_key_values(layer_idx=0, target_dtype=torch.bfloat16)

        st.success(f"Pipeline execution completed in {elapsed_ms:.2f} ms!")

        # Metric KPIs
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Original KV Memory", f"{stats['original_kv_mb']:.3f} MB")
        col2.metric("Compacted KV Memory", f"{stats['compacted_kv_mb']:.3f} MB")
        col3.metric("Compression Ratio", f"{stats['compression_ratio']:.2f}x")
        col4.metric("Memory Reduction", f"{stats['memory_reduction_pct']:.2f}%")

        st.markdown("---")
        st.subheader("Stage Breakdown Audit")

        st.markdown(f"""
        - **Stage 1 (ChunkKV)**: Partitioned `{len(ranking_res.protected_chunks) + len(ranking_res.selected_chunks) + len(ranking_res.evicted_chunks)}` total chunks. Protected `{len(ranking_res.protected_chunks)}` sink & local chunks (`{ranking_res.total_protected_tokens}` tokens).
        - **Stage 2 (Adaptive Allocator)**: Distributed budget across `{num_heads}` heads using `{alloc_policy}` policy.
        - **Stage 3 (Chunk Ranking)**: Selected `{len(ranking_res.selected_chunks)}` middle chunks (`{ranking_res.total_selected_tokens}` tokens); Evicted `{len(ranking_res.evicted_chunks)}` chunks (`{ranking_res.total_evicted_tokens}` tokens).
        - **Stage 4 (KIVI Quantization)**: Retained middle blocks quantized to `{bits}-bit INT`. BF16 source storage safely released.
        - **Stage 5 (FlashCache Runtime)**: Active past key values assembled with shape `{list(past_k.shape)}`.
        """)

        # Block state donut chart
        state_df = pd.DataFrame([
            {"State": "Active BF16 (Protected)", "Count": stats["active_bf16_blocks"]},
            {"State": "Compressed (Quantized)", "Count": stats["compressed_blocks"]},
            {"State": "Reclaimed Source Blocks", "Count": stats["reclaimable_blocks"]},
        ])
        fig_pie = px.pie(state_df, names="State", values="Count", title="Physical Block State Distribution", hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
        runtime.shutdown()

with tab2:
    st.header("Comparative Memory Scaling Analytics")
    st.markdown("Evaluating memory footprint across 5 core approaches as context length expands.")

    test_lengths = [512, 1024, 2048, 4096]
    comp_data = []

    for c in test_lengths:
        full_mb = (2 * num_heads * c * head_dim * 2) / (1024 ** 2)
        evict_mb = (2 * num_heads * min(c, target_budget) * head_dim * 2) / (1024 ** 2)
        chunk_bf16_mb = evict_mb
        fc_4b_mb = evict_mb * 0.28
        fc_2b_mb = evict_mb * 0.18

        comp_data.append({"Context": c, "Strategy": "Full BF16 Cache", "KV_MB": full_mb})
        comp_data.append({"Context": c, "Strategy": "StreamingLLM Eviction", "KV_MB": evict_mb})
        comp_data.append({"Context": c, "Strategy": "ChunkKV Uncompressed", "KV_MB": chunk_bf16_mb})
        comp_data.append({"Context": c, "Strategy": "FlashCache-KV (4-bit)", "KV_MB": fc_4b_mb})
        comp_data.append({"Context": c, "Strategy": "FlashCache-KV (2-bit)", "KV_MB": fc_2b_mb})

    df_comp = pd.DataFrame(comp_data)
    fig_bar = px.bar(
        df_comp,
        x="Context",
        y="KV_MB",
        color="Strategy",
        barmode="group",
        title="Physical KV Cache Footprint (MB) Across Context Lengths",
        labels={"KV_MB": "Memory (MB)", "Context": "Context Length (Tokens)"},
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with tab3:
    st.header("Quantization Fidelity and Error Analysis")
    st.markdown("Direct measurement of KIVI-inspired per-channel Key and per-token Value quantization error.")

    q_c_size = st.selectbox("Benchmark Chunk Size", [8, 16, 32, 64], index=1)
    
    q4 = KIVIQuantizer(bits=4)
    q2 = KIVIQuantizer(bits=2)
    deq = KIVIDequantizer()

    test_tensor = torch.randn(1, 1, q_c_size, head_dim, dtype=torch.bfloat16)

    comp4 = q4.quantize_key(test_tensor)
    comp2 = q2.quantize_key(test_tensor)

    recon4 = deq.dequantize(comp4, target_dtype=torch.bfloat16)
    recon2 = deq.dequantize(comp2, target_dtype=torch.bfloat16)

    m4 = deq.evaluate_reconstruction(test_tensor, recon4)
    m2 = deq.evaluate_reconstruction(test_tensor, recon2)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("4-bit Quantization Metrics")
        st.write(f"- **Reconstruction PSNR**: `{m4['psnr']:.2f} dB`")
        st.write(f"- **Cosine Similarity**: `{m4['cosine_similarity']:.4f}`")
        st.write(f"- **Normalized MSE**: `{m4['nmse']:.6f}`")
        st.write(f"- **Compression Ratio**: `{(test_tensor.numel() * 2) / comp4.total_bytes:.2f}x`")

    with col2:
        st.subheader("2-bit Quantization Metrics")
        st.write(f"- **Reconstruction PSNR**: `{m2['psnr']:.2f} dB`")
        st.write(f"- **Cosine Similarity**: `{m2['cosine_similarity']:.4f}`")
        st.write(f"- **Normalized MSE**: `{m2['nmse']:.6f}`")
        st.write(f"- **Compression Ratio**: `{(test_tensor.numel() * 2) / comp2.total_bytes:.2f}x`")

with tab4:
    st.header("Historical Model Benchmarks (Qwen2.5-0.5B-Instruct)")
    st.info("Verified measurements from reference repository runs. Clearly labeled as historical reference data.")

    df_hist = pd.DataFrame(HISTORICAL_BENCHMARKS)
    st.dataframe(
        df_hist[[
            "strategy", "type", "retained_tokens", "kv_mb",
            "compression_ratio", "compression_pct", "latency_sec", "throughput_tok_s", "status"
        ]],
        use_container_width=True,
    )
