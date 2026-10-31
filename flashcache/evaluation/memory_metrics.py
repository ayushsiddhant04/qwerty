"""
Memory Metrics Evaluation Utilities.
"""

from typing import Dict, Any
import torch


def calculate_tensor_bytes(tensor: Optional[torch.Tensor]) -> int:
    if tensor is None or not isinstance(tensor, torch.Tensor):
        return 0
    return tensor.numel() * tensor.element_size()


def compute_compression_metrics(
    uncompressed_bytes: int,
    compressed_bytes: int,
) -> Dict[str, float]:
    """
    Computes exact compression ratio and percentage reduction.
    """
    u_bytes = max(0, int(uncompressed_bytes))
    c_bytes = max(0, int(compressed_bytes))

    reduction_pct = 0.0
    ratio = 1.0

    if u_bytes > 0:
        reduction_pct = ((u_bytes - c_bytes) / u_bytes) * 100.0
        ratio = u_bytes / max(1, c_bytes)

    return {
        "uncompressed_bytes": u_bytes,
        "compressed_bytes": c_bytes,
        "compression_ratio": float(ratio),
        "memory_reduction_pct": float(reduction_pct),
        "uncompressed_mb": u_bytes / (1024 ** 2),
        "compressed_mb": c_bytes / (1024 ** 2),
    }

