"""
Hardware and Memory Telemetry Monitor (Stage 5).
"""

from typing import Dict, Any, Optional
import psutil
import torch


class MemoryMonitor:
    """
    Monitors process RAM, CUDA VRAM allocations, and cache memory compaction metrics.
    Clearly distinguishes logical tensor sizes from physical hardware reservations.
    """

    @staticmethod
    def get_hardware_info() -> Dict[str, Any]:
        """Inspects and returns current host and accelerator hardware configuration."""
        cuda_avail = torch.cuda.is_available()
        gpu_info = None
        if cuda_avail:
            gpu_info = {
                "device_name": torch.cuda.get_device_name(0),
                "device_count": torch.cuda.device_count(),
                "total_vram_mb": torch.cuda.get_device_properties(0).total_memory / (1024 ** 2),
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }

        ram = psutil.virtual_memory()
        return {
            "device": "cuda" if cuda_avail else "cpu",
            "cuda_available": cuda_avail,
            "gpu": gpu_info,
            "cpu_count": psutil.cpu_count(logical=True),
            "total_system_ram_mb": ram.total / (1024 ** 2),
            "available_system_ram_mb": ram.available / (1024 ** 2),
        }

    @staticmethod
    def get_current_memory_stats(
        original_kv_bytes: int = 0,
        compacted_kv_bytes: int = 0,
    ) -> Dict[str, Any]:
        """
        Calculates and returns precise memory usage metrics.
        """
        process = psutil.Process()
        cpu_ram_mb = process.memory_info().rss / (1024 ** 2)

        cuda_stats = {}
        if torch.cuda.is_available():
            cuda_stats = {
                "cuda_allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
                "cuda_reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
                "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
            }

        # Compression calculations
        orig_bytes = max(0, original_kv_bytes)
        comp_bytes = max(0, compacted_kv_bytes)
        
        reduction_pct = 0.0
        ratio = 1.0
        if orig_bytes > 0:
            reduction_pct = ((orig_bytes - comp_bytes) / orig_bytes) * 100.0
            ratio = orig_bytes / max(1, comp_bytes)

        return {
            "process_ram_mb": cpu_ram_mb,
            "cuda": cuda_stats,
            "original_kv_bytes": orig_bytes,
            "compacted_kv_bytes": comp_bytes,
            "original_kv_mb": orig_bytes / (1024 ** 2),
            "compacted_kv_mb": comp_bytes / (1024 ** 2),
            "memory_reduction_pct": float(reduction_pct),
            "compression_ratio": float(ratio),
        }

