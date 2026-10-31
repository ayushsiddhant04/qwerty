# FlashCache-KV Pipeline Specification

This document details the exact input and output data contracts, mathematical formulations, and operational behaviors for all five pipeline stages.

---

## Stage 1: ChunkKV — Candidate Construction

### Inputs
- `keys`: `torch.Tensor` of shape `[batch_size, num_heads, seq_len, head_dim]`.
- `values`: `torch.Tensor` of shape `[batch_size, num_heads, seq_len, head_dim]`.
- `chunk_size`: Integer (e.g., 8, 16, or 32 tokens per chunk).
- `sink_tokens`: Number of initial prompt tokens to permanently protect ($N_{\text{sink}} \ge 0$).
- `recent_tokens`: Number of tail tokens to retain uncompressed ($N_{\text{recent}} \ge 0$).
- `raw_attention_scores`: Optional attention matrix of shape `[batch_size, num_heads, seq_len]`.

### Processing Logic
1. **Partitioning**: Historical tokens outside $N_{\text{sink}}$ and $N_{\text{recent}}$ are partitioned into chunks of length $C = \text{chunk\_size}$. Residual trailing tokens form a final variable-sized chunk.
2. **Scoring Formulation**:
   For each chunk $i$ spanning token indices $[t_{\text{start}}, t_{\text{end}})$:
   $$\text{score}(i) = w_{\text{attn}} \cdot \bar{A}_i + w_{\text{recency}} \cdot \exp\left(-\lambda \cdot \frac{L - t_{\text{end}}}{L}\right) + w_{\text{norm}} \cdot \frac{\|K_i\|_F + \|V_i\|_F}{2 C}$$
   where $\bar{A}_i$ is the mean attention score over the chunk (or 0.0 if attention is unavailable), $\lambda$ is a decay coefficient, and $\| \cdot \|_F$ is the Frobenius norm of the key/value sub-tensors.
3. **Metadata Construction**: Each chunk produces a `KVChunk` object holding tensor slices, chunk ID, head ID, position span, element byte size, and lifecycle state.

### Outputs
- `List[KVChunk]`: Candidate chunk objects ready for budget allocation and ranking.

---

## Stage 2: Adaptive KV Budget Allocator

### Inputs
- `candidates`: `List[KVChunk]` from Stage 1.
- `total_token_budget`: Target retention capacity in tokens.
- `policy`: Allocation strategy (`"uniform"`, `"importance"`, `"recency"`).
- `num_heads`: Total active KV heads.
- `num_layers`: Total model transformer layers.

### Processing Logic
- **Uniform Policy**:
  $$B_h = \left\lfloor \frac{B_{\text{total}}}{H} \right\rfloor$$
- **Importance Policy**:
  Computes aggregate chunk score per head $S_h = \sum_{c \in \text{head } h} \text{score}(c)$.
  $$B_h = \text{round}\left( B_{\text{total}} \cdot \frac{S_h + \epsilon}{\sum_{j} (S_j + \epsilon)} \right)$$
  Subject to a minimum retention guarantee $B_{\min} = C$ to prevent head starvation.
- **Recency Policy**:
  Weights head budgets using a depth and temporal bias factor.

### Outputs
- `HeadBudgets`: Dictionary mapping `(layer_idx, head_idx)` to assigned token budget.

---

## Stage 3: Budget-Aware Chunk Ranking

### Inputs
- `candidates`: `List[KVChunk]` from Stage 1.
- `head_budgets`: `HeadBudgets` from Stage 2.
- `sink_tokens`: Protected count.
- `recent_tokens`: Protected tail count.

### Processing Logic
1. Sinks and recent tokens are assigned `PROTECTED` status.
2. For each head budget $B_h$, sort non-protected chunks in descending order of importance score.
3. Greedily accumulate chunks until $\sum \text{len}(c) \le B_h$.
4. Mark selected chunks as `SELECTED` and unselected candidates as `EVICTED`.

### Outputs
- `RankingResult`:
  - `selected_chunks`: Chunks slated for retention (and subsequent compression).
  - `evicted_chunks`: Chunks marked for discarding.
  - `protected_chunks`: High-priority chunks retained in high precision.
  - `budget_utilization`: Statistics on budget vs. actual consumption.

---

## Stage 4: KIVI-Inspired Quantization

### Inputs
- `selected_chunks`: Retained chunks from Stage 3.
- `bits`: Bit-width (2 or 4).
- `group_size`: Quantization block size (default: 32 or full chunk).

### Processing Logic
- **Key Quantization (Per-Channel)**:
  Quantizes across sequence length $S$ per channel $d$:
  $$s_k = \frac{\max(K_{:, d}) - \min(K_{:, d})}{2^b - 1}, \quad z_k = \text{round}\left(-\frac{\min(K_{:, d})}{s_k}\right)$$
  $$Q_K = \text{clamp}\left(\text{round}\left(\frac{K_{:, d}}{s_k}\right) + z_k, 0, 2^b - 1\right)$$
- **Value Quantization (Per-Token)**:
  Quantizes across hidden dimension $D$ per token $t$:
  $$s_v = \frac{\max(V_{t, :}) - \min(V_{t, :})}{2^b - 1}, \quad z_v = \text{round}\left(-\frac{\min(V_{t, :})}{s_v}\right)$$
  $$Q_V = \text{clamp}\left(\text{round}\left(\frac{V_{t, :}}{s_v}\right) + z_v, 0, 2^b - 1\right)$$
- **Bit Packing**:
  Packs four 2-bit values or two 4-bit values per `uint8` element.

### Outputs
- `CompressedChunk`: Holds packed `torch.uint8` tensors, scale tensors, zero-point tensors, and original metadata.

---

## Stage 5: FlashCache Runtime

### Inputs
- Incoming KV updates during decoding step.
- Storage queries for attention forward passes.

### Processing Logic
- Orchestrates concurrent compression worker pool.
- Validates dequantization reconstruction error against target SNR threshold.
- Safely manages block lifecycle state transitions.
- Assembles contiguous virtual cache tensor views on-the-fly for PyTorch attention kernels.

### Outputs
- Decoded past-key-value cache ready for generation.
- Real-time telemetry: physical bytes, compression ratio, latencies, and cache health.
