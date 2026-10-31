# Changelog

All notable changes to the **FlashCache-KV** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-10-31

### Added
- **ChunkKV Candidate Construction (Stage 1)**:
  - Token chunking with configurable chunk sizes, position indexing, and attention-head tracking.
  - Multi-signal importance scoring (attention weight mass, exponential recency decay, Frobenius norm heuristics).
  - Explicit attention-sink token preservation ($N_{\text{sink}}$) and local sliding window protection.
- **Adaptive KV Budget Allocator (Stage 2)**:
  - Dynamic token retention budget allocation per attention head and layer.
  - Pluggable allocation policies: `uniform`, `importance_weighted`, and `recency_aware`.
- **Budget-Aware Chunk Ranking (Stage 3)**:
  - Priority ranking and selective retention constrained by head budgets.
  - Explicit classification into `SELECTED`, `EVICTED`, and `PROTECTED` chunks.
- **KIVI-Inspired Asymmetric Quantization (Stage 4)**:
  - 4-bit and 2-bit asymmetric affine quantization with uint8 bit packing.
  - Per-channel Key quantization and per-token Value quantization.
  - Full dequantization and reconstruction error evaluation (NMSE, PSNR).
  - **Memory Safety Barrier**: Enforced lifecycle states (`ACTIVE_BF16`, `COMPRESSION_PENDING`, `COMPRESSED`, `COMPRESSION_FAILED`, `RECLAIMABLE`) guaranteeing BF16 source storage is retained until quantization validation and atomic reference replacement succeed.
- **FlashCache Runtime (Stage 5)**:
  - Central cache orchestrator managing block states, asynchronous background compaction, thread safety, and memory accounting.
  - Hardware-aware execution with automated CUDA stream synchronization and CPU execution support.
- **Comprehensive Benchmarks & Test Suite**:
  - Memory, latency, compression, and end-to-end pipeline benchmarks.
  - Full pytest coverage across all five architecture stages and edge cases.
- **Interactive Streamlit Dashboard**:
  - Live pipeline visualization, memory reduction analytics, and multi-baseline comparative performance metrics.

## [0.1.0] - 2026-10-30
### Added
- Initial reference token eviction (StreamingLLM) and experimental similarity merging research scripts.
