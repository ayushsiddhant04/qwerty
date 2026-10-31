"""
KIVI-Inspired Asymmetric Quantizer Implementation (Stage 4).
Note: This is an independent, clean-room KIVI-inspired systems implementation,
not the official KIVI CUDA library.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import torch

from flashcache.compression.quantization_utils import (
    pack_int4,
    unpack_int4,
    pack_int2,
    unpack_int2,
    compute_asymmetric_scales_and_zeros,
)


@dataclass
class CompressedTensor:
    """
    Self-contained compressed tensor with packed payloads, scales, and zero points.
    """
    packed_data: torch.Tensor        # uint8 packed bytes
    scales: torch.Tensor             # FP16/BF16 scale factors
    zeros: torch.Tensor              # uint8 zero-point offsets
    original_shape: torch.Size       # Shape prior to quantization
    bits: int                        # 2 or 4 bits
    axis: int                        # Axis along which min/max was computed
    dtype: torch.dtype               # Original data type (e.g. bfloat16, float16)
    
    @property
    def payload_bytes(self) -> int:
        return self.packed_data.numel() * self.packed_data.element_size()

    @property
    def scale_bytes(self) -> int:
        return self.scales.numel() * self.scales.element_size()

    @property
    def zero_bytes(self) -> int:
        return self.zeros.numel() * self.zeros.element_size()

    @property
    def total_bytes(self) -> int:
        """Physical storage including packed values and all quantization metadata."""
        return self.payload_bytes + self.scale_bytes + self.zero_bytes


class KIVIQuantizer:
    """
    Asymmetric affine quantizer implementing KIVI principles:
    - Keys are quantized per-channel (along sequence length axis).
    - Values are quantized per-token (along hidden channel axis).
    Supports 4-bit and 2-bit precision.
    """

    def __init__(self, bits: int = 4):
        if bits not in (2, 4):
            raise ValueError(f"KIVIQuantizer supports 2-bit or 4-bit quantization, got {bits}")
        self.bits = bits
        self.q_max = (1 << bits) - 1

    def quantize_key(self, key: torch.Tensor) -> CompressedTensor:
        """
        Quantizes key tensor per-channel along the sequence axis (dim=-2).
        
        Args:
            key: Shape [..., seq_len, head_dim]
        """
        original_shape = key.shape
        orig_dtype = key.dtype
        # Per-channel along sequence axis: axis = -2
        axis = -2
        scales, zeros = compute_asymmetric_scales_and_zeros(key.float(), axis=axis, bits=self.bits)

        # Quantize: clamp(round((X / scale) + zero), 0, q_max)
        scaled = key.float() / scales
        shifted = torch.round(scaled) + zeros.float()
        quantized = torch.clamp(shifted, 0, self.q_max).to(torch.uint8)

        # Pack along head_dim (last dimension)
        if self.bits == 4:
            packed = pack_int4(quantized)
        else:
            packed = pack_int2(quantized)

        # Retain scales in original precision (e.g. BF16/FP16)
        scale_dtype = orig_dtype if orig_dtype in (torch.float16, torch.bfloat16) else torch.float16
        stored_scales = scales.to(scale_dtype)

        return CompressedTensor(
            packed_data=packed,
            scales=stored_scales,
            zeros=zeros,
            original_shape=original_shape,
            bits=self.bits,
            axis=axis,
            dtype=orig_dtype,
        )

    def quantize_value(self, value: torch.Tensor) -> CompressedTensor:
        """
        Quantizes value tensor per-token along the hidden channel axis (dim=-1).
        
        Args:
            value: Shape [..., seq_len, head_dim]
        """
        original_shape = value.shape
        orig_dtype = value.dtype
        # Per-token along channel axis: axis = -1
        axis = -1
        scales, zeros = compute_asymmetric_scales_and_zeros(value.float(), axis=axis, bits=self.bits)

        # Quantize
        scaled = value.float() / scales
        shifted = torch.round(scaled) + zeros.float()
        quantized = torch.clamp(shifted, 0, self.q_max).to(torch.uint8)

        # Pack along head_dim (last dimension)
        if self.bits == 4:
            packed = pack_int4(quantized)
        else:
            packed = pack_int2(quantized)

        scale_dtype = orig_dtype if orig_dtype in (torch.float16, torch.bfloat16) else torch.float16
        stored_scales = scales.to(scale_dtype)

        return CompressedTensor(
            packed_data=packed,
            scales=stored_scales,
            zeros=zeros,
            original_shape=original_shape,
            bits=self.bits,
            axis=axis,
            dtype=orig_dtype,
        )

