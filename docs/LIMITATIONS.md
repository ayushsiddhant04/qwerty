# FlashCache-KV Known Limitations and Future Work

## 1. Experimental Research Prototype Scope

FlashCache-KV is designed as an experimental systems research prototype exploring adaptive KV-cache compaction. It is not an enterprise-grade drop-in replacement for production inference engines such as vLLM, TensorRT-LLM, or SGLang.

---

## 2. Documented Technical Limitations

### A. Quantization Precision & Reconstruction Loss
- **Lossy Compression**: While 4-bit and 2-bit quantization dramatically reduce memory consumption, they introduce quantization noise into historical Key and Value representations. Sensitive retrieval tasks (e.g., precise code generation or long-distance cryptographic key recall) may experience accuracy degradation compared to uncompressed BF16 baselines.
- **2-bit Quantization Artifacts**: Under aggressive 2-bit quantization, attention patterns on deeply historical tokens may show elevated perplexity if outlier channels exceed the dynamic range of uniform scaling.

### B. Custom Quantizer vs. Fused CUDA Kernels
- **PyTorch Emulation**: The KIVI-inspired quantizer and bit-packing mechanisms are implemented using vectorized PyTorch operations and CPU/CUDA tensors rather than custom handwritten Triton or CUDA C++ fused attention kernels. 
- **Latency Trade-off**: In the present implementation, on-the-fly dequantization incurs Python and tensor allocation overhead. Peak throughput gains require custom fused GEMM/FlashAttention kernels that consume packed 4-bit/2-bit keys and values directly without full dequantization into temporary BF16 buffers.

### C. Hardware and Precision Support
- **BFloat16 on Older GPUs**: BF16 tensor operations require NVIDIA Ampere (compute capability $\ge 8.0$) or newer architectures. On older GPUs (Turing, Pascal) or CPUs lacking native AVX-512 BF16 instructions, operations fall back to standard FP16 or FP32.
- **Host Memory Swapping**: FlashCache-KV does not currently implement hierarchical CPU offloading or NVMe page swapping. All retained blocks reside in main system memory (or GPU VRAM when CUDA is enabled).

### D. Attention-Score Availability
- When using black-box model inference APIs or optimized attention kernels that do not return attention probability matrices, Stage 1 falls back to exponential recency decay and Frobenius norm heuristic scoring.

---

## 3. Roadmap for Future Development

1. **Custom Triton Fused Kernels**: Implement fused 2-bit and 4-bit dequantizing attention kernels to eliminate intermediate reconstruction allocations.
2. **Learned Head Budget Allocator**: Train a lightweight MLP policy model that predicts per-head retention budgets directly from hidden activation statistics.
3. **PagedKV Integration**: Marry FlashCache-KV chunking with paged virtual memory block allocators (similar to vLLM's BlockManager).
4. **Speculative Dequantization**: Prefetch and dequantize candidate chunks predicted by draft tokens in speculative decoding workflows.

