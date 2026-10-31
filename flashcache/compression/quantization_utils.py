"""
Low-Level Bit Packing and Quantization Math Utilities (Stage 4).
"""

from typing import Tuple
import torch


def pack_int4(quantized: torch.Tensor) -> torch.Tensor:
    """
    Packs two 4-bit unsigned integers (0-15) into a single uint8 byte.
    
    Args:
        quantized: Tensor of unsigned integers with last dimension even.
    
    Returns:
        torch.Tensor of dtype torch.uint8 with last dimension halved.
    """
    shape = list(quantized.shape)
    last_dim = shape[-1]
    if last_dim % 2 != 0:
        # Pad with 0 to make even
        pad = torch.zeros(*shape[:-1], 1, dtype=quantized.dtype, device=quantized.device)
        quantized = torch.cat([quantized, pad], dim=-1)
        shape[-1] += 1

    q_flat = quantized.view(*shape[:-1], -1, 2).to(torch.uint8)
    low = q_flat[..., 0] & 0x0F
    high = (q_flat[..., 1] & 0x0F) << 4
    packed = low | high
    return packed


def unpack_int4(packed: torch.Tensor, target_length: int) -> torch.Tensor:
    """
    Unpacks uint8 bytes into two 4-bit unsigned integers.
    """
    low = (packed & 0x0F).to(torch.uint8)
    high = ((packed >> 4) & 0x0F).to(torch.uint8)
    unpacked = torch.stack([low, high], dim=-1).view(*packed.shape[:-1], -1)
    if unpacked.shape[-1] > target_length:
        unpacked = unpacked[..., :target_length]
    return unpacked


def pack_int2(quantized: torch.Tensor) -> torch.Tensor:
    """
    Packs four 2-bit unsigned integers (0-3) into a single uint8 byte.
    """
    shape = list(quantized.shape)
    last_dim = shape[-1]
    remainder = last_dim % 4
    if remainder != 0:
        pad_size = 4 - remainder
        pad = torch.zeros(*shape[:-1], pad_size, dtype=quantized.dtype, device=quantized.device)
        quantized = torch.cat([quantized, pad], dim=-1)
        shape[-1] += pad_size

    q_flat = quantized.view(*shape[:-1], -1, 4).to(torch.uint8)
    b0 = q_flat[..., 0] & 0x03
    b1 = (q_flat[..., 1] & 0x03) << 2
    b2 = (q_flat[..., 2] & 0x03) << 4
    b3 = (q_flat[..., 3] & 0x03) << 6
    packed = b0 | b1 | b2 | b3
    return packed


def unpack_int2(packed: torch.Tensor, target_length: int) -> torch.Tensor:
    """
    Unpacks uint8 bytes into four 2-bit unsigned integers.
    """
    b0 = (packed & 0x03).to(torch.uint8)
    b1 = ((packed >> 2) & 0x03).to(torch.uint8)
    b2 = ((packed >> 4) & 0x03).to(torch.uint8)
    b3 = ((packed >> 6) & 0x03).to(torch.uint8)
    unpacked = torch.stack([b0, b1, b2, b3], dim=-1).view(*packed.shape[:-1], -1)
    if unpacked.shape[-1] > target_length:
        unpacked = unpacked[..., :target_length]
    return unpacked


def compute_asymmetric_scales_and_zeros(
    tensor: torch.Tensor,
    axis: int,
    bits: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes min/max scales and integer zero-points along a specified axis.
    
    Returns:
        (scales, zero_points)
    """
    q_max = (1 << bits) - 1
    t_min = torch.amin(tensor, dim=axis, keepdim=True)
    t_max = torch.amax(tensor, dim=axis, keepdim=True)

    diff = t_max - t_min
    # Avoid zero division
    scales = torch.where(diff > 1e-8, diff / float(q_max), torch.ones_like(diff))
    
    zeros = torch.round(-t_min / scales)
    zeros = torch.clamp(zeros, 0, q_max).to(torch.uint8)
    return scales, zeros

