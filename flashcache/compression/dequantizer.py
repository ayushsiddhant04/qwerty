"""
KIVI-Inspired Dequantizer and Quality Evaluation (Stage 4).
"""

import math
from typing import Optional, Dict
import torch

from flashcache.compression.quantization_utils import unpack_int4, unpack_int2
from flashcache.compression.kivi_quantizer import CompressedTensor


class KIVIDequantizer:
    """
    Reconstructs floating-point Key and Value tensors from compressed representations
    and calculates quantitative reconstruction error metrics.
    """

    @staticmethod
    def dequantize(
        compressed: CompressedTensor,
        target_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Dequantizes a CompressedTensor back to floating-point representation.
        
        Formula:
            X_hat = scales * (Q - zeros)
        """
        target_dtype = target_dtype or compressed.dtype
        target_last_dim = compressed.original_shape[-1]

        # 1. Unpack uint8 data into integer range [0, 2^b - 1]
        if compressed.bits == 4:
            unpacked = unpack_int4(compressed.packed_data, target_length=target_last_dim)
        elif compressed.bits == 2:
            unpacked = unpack_int2(compressed.packed_data, target_length=target_last_dim)
        else:
            raise ValueError(f"Unsupported bits: {compressed.bits}")

        # 2. Dequantize: scales * (Q - zeros)
        scales = compressed.scales.float()
        zeros = compressed.zeros.float()
        q_float = unpacked.float()

        reconstructed = scales * (q_float - zeros)
        reconstructed = reconstructed.view(compressed.original_shape)
        return reconstructed.to(target_dtype)

    @classmethod
    def evaluate_reconstruction(
        cls,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Computes numerical fidelity metrics between original and reconstructed tensors.
        
        Returns:
            - 'mse': Mean squared error
            - 'nmse': Normalized mean squared error
            - 'psnr': Peak signal-to-noise ratio in dB
            - 'cosine_similarity': Mean cosine similarity
        """
        orig_f = original.float()
        recon_f = reconstructed.float()

        mse = torch.mean((orig_f - recon_f) ** 2).item()
        orig_norm_sq = torch.sum(orig_f ** 2).item()
        error_norm_sq = torch.sum((orig_f - recon_f) ** 2).item()

        nmse = error_norm_sq / max(1e-9, orig_norm_sq)

        # Dynamic range for PSNR
        dynamic_range = (torch.amax(orig_f) - torch.amin(orig_f)).item()
        if mse > 1e-12 and dynamic_range > 1e-12:
            psnr = 10.0 * math.log10((dynamic_range ** 2) / mse)
        else:
            psnr = 100.0 # Near-perfect reconstruction

        # Cosine similarity along last dimension (head_dim)
        cos_sim = torch.nn.functional.cosine_similarity(
            orig_f.view(-1, orig_f.shape[-1]),
            recon_f.view(-1, recon_f.shape[-1]),
            dim=-1
        ).mean().item()

        return {
            "mse": float(mse),
            "nmse": float(nmse),
            "psnr": float(psnr),
            "cosine_similarity": float(cos_sim),
        }
