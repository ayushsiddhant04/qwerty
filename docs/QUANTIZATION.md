# FlashCache-KV Quantization Specification

## 1. Asymmetric Affine Quantization Principles

FlashCache-KV adopts an asymmetric uniform quantization strategy inspired by the **KIVI** (KV-Cache In-situ Quantization) algorithm. Keys and Values exhibit fundamentally different outlier distributions during self-attention:

- **Keys ($K$)**: Feature channels across tokens exhibit significant dynamic range variances, but values within a specific channel across tokens are relatively stable. Therefore, Keys are quantized **per-channel** along the token sequence length dimension ($S$).
- **Values ($V$)**: Individual tokens exhibit distinct magnitude spikes across all hidden channels. Therefore, Values are quantized **per-token** along the hidden channel dimension ($D$).

---

## 2. Mathematical Formulation

Let $X$ represent an input tensor slice in BF16 or FP16 precision. For a given target bit-width $b \in \{2, 4\}$ with $Q_{\max} = 2^b - 1$:

### Scale and Zero-Point Calculation:
$$s = \frac{\max(X) - \min(X)}{Q_{\max}}$$
$$z = \text{round}\left( -\frac{\min(X)}{s} \right)$$
$$z = \text{clamp}(z, 0, Q_{\max})$$

Where $s \in \mathbb{R}$ is stored in FP16/BF16, and $z \in \mathbb{Z}$ is stored as an integer.

### Quantization:
$$Q = \text{clamp}\left( \left\lfloor \frac{X}{s} + 0.5 \right\rfloor + z, 0, Q_{\max} \right)$$

### Dequantization (Reconstruction):
$$\hat{X} = s \cdot (Q - z)$$

---

## 3. Bit Packing Layout

Because PyTorch does not natively provide 2-bit (`uint2`) or 4-bit (`uint4`) standard dtypes, values are packed into standard `torch.uint8` bytes:

### 4-bit Packing (2 values per byte):
$$\text{byte} = (q_0 \ \& \ 0\text{x0F}) \ | \ ((q_1 \ \& \ 0\text{x0F}) \ll 4)$$

### 2-bit Packing (4 values per byte):
$$\text{byte} = (q_0 \ \& \ 0\text{x03}) \ | \ ((q_1 \ \& \ 0\text{x03}) \ll 2) \ | \ ((q_2 \ \& \ 0\text{x03}) \ll 4) \ | \ ((q_3 \ \& \ 0\text{x03}) \ll 6)$$

---

## 4. Metadata Accounting and Storage Formulas

For a chunk of shape $[B, H, C, D]$ where $C = \text{chunk\_size}$, $D = \text{head\_dim}$:

### Uncompressed BF16 Size:
$$\text{Bytes}_{\text{BF16}} = B \times H \times C \times D \times 2 \text{ bytes}$$

### Compressed Key Size (Per-Channel):
- Packed quantized key payload: $B \times H \times \left\lceil \frac{C \times b}{8} \right\rceil \times D \text{ bytes}$
- Scales ($s_k$): $B \times H \times 1 \times D \times 2 \text{ bytes}$ (FP16/BF16)
- Zero-points ($z_k$): $B \times H \times 1 \times D \times 1 \text{ byte}$ (uint8)

### Compressed Value Size (Per-Token):
- Packed quantized value payload: $B \times H \times C \times \left\lceil \frac{D \times b}{8} \right\rceil \text{ bytes}$
- Scales ($s_v$): $B \times H \times C \times 1 \times 2 \text{ bytes}$ (FP16/BF16)
- Zero-points ($z_v$): $B \times H \times C \times 1 \times 1 \text{ byte}$ (uint8)

### Net Compression Ratio:
$$\text{Ratio} = \frac{\text{Bytes}_{\text{BF16}}(K) + \text{Bytes}_{\text{BF16}}(V)}{\text{Bytes}_{\text{Packed}}(K, V) + \text{Bytes}_{\text{Scales}}(K, V) + \text{Bytes}_{\text{Zeros}}(K, V)}$$

---

## 5. Reconstruction Error Evaluation

Reconstruction accuracy is monitored via Normalized Mean Squared Error (NMSE) and Peak Signal-to-Noise Ratio (PSNR):

$$\text{NMSE} = \frac{\|X - \hat{X}\|_F^2}{\|X\|_F^2 + \epsilon}$$
$$\text{PSNR} = 10 \cdot \log_{10} \left( \frac{\max(X)^2}{\text{MSE}(X, \hat{X}) + \epsilon} \right)$$
