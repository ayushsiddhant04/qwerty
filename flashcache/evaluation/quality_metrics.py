"""
Quality Metrics and Reconstruction Evaluation Utilities.
"""

import math
from typing import Dict
import torch


def compute_reconstruction_metrics(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> Dict[str, float]:
    """
    Computes NMSE, PSNR, and Cosine Similarity between original and reconstructed tensors.
    """
    orig_f = original.float()
    recon_f = reconstructed.float()

    mse = torch.mean((orig_f - recon_f) ** 2).item()
    orig_norm_sq = torch.sum(orig_f ** 2).item()
    error_norm_sq = torch.sum((orig_f - recon_f) ** 2).item()

    nmse = error_norm_sq / max(1e-9, orig_norm_sq)

    dynamic_range = (torch.amax(orig_f) - torch.amin(orig_f)).item()
    if mse > 1e-12 and dynamic_range > 1e-12:
        psnr = 10.0 * math.log10((dynamic_range ** 2) / mse)
    else:
        psnr = 100.0

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
