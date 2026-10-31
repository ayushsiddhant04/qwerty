"""
FlashCache Evaluation module.
"""

from flashcache.evaluation.memory_metrics import compute_compression_metrics, calculate_tensor_bytes
from flashcache.evaluation.quality_metrics import compute_reconstruction_metrics

__all__ = [
    "compute_compression_metrics",
    "calculate_tensor_bytes",
    "compute_reconstruction_metrics",
]

