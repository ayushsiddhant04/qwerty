"""
Compression Manager Enforcing the BF16 Source-Storage Lifecycle Barrier (Stage 4).
"""

from typing import List, Dict, Optional, Tuple, Any
import logging
import torch

from flashcache.core.cache_state import BlockState, ChunkStatus
from flashcache.chunking.chunk_metadata import KVChunk
from flashcache.compression.kivi_quantizer import KIVIQuantizer, CompressedTensor
from flashcache.compression.dequantizer import KIVIDequantizer

logger = logging.getLogger(__name__)


class CompressionManager:
    """
    Orchestrates the safe, atomic compression of historical KV blocks.
    Strictly guarantees that source BF16/FP16 memory is never reclaimed
    until compression has been numerically validated, published, and synchronized.
    """

    def __init__(
        self,
        bits: int = 4,
        min_psnr_threshold: float = 15.0, # Minimum acceptable reconstruction PSNR (dB)
        cuda_stream: Optional[Any] = None,
    ):
        """
        Args:
            bits: Quantization bit-width (2 or 4).
            min_psnr_threshold: Numerical validation gate threshold.
            cuda_stream: Optional torch.cuda.Stream for GPU operations.
        """
        self.bits = bits
        self.quantizer = KIVIQuantizer(bits=bits)
        self.dequantizer = KIVIDequantizer()
        self.min_psnr_threshold = min_psnr_threshold
        self.cuda_stream = cuda_stream

    def compress_chunk_safe(
        self,
        chunk: KVChunk,
        simulate_failure: bool = False,
    ) -> bool:
        """
        Executes the mandatory 9-step memory-safe compression lifecycle:
        
        1. Mark original BF16 block as COMPRESSION_PENDING.
        2. Retain valid reference to source tensor.
        3. Execute compression on Key and Value.
        4. Produce compressed representations.
        5. Validate numerical integrity and reconstruction quality.
        6. Confirm compression success.
        7. Atomically update chunk active representation references.
        8. Synchronize GPU stream/events or CPU barrier.
        9. Mark source tensor as RECLAIMABLE.

        Returns:
            True if compression succeeded and is active; False if failed and rolled back.
        """
        # Step 1: Mark as COMPRESSION_PENDING
        chunk.metadata.state = BlockState.COMPRESSION_PENDING

        # Step 2: Retain valid source references
        source_key_ref = chunk.key
        source_value_ref = chunk.value

        try:
            if simulate_failure:
                raise RuntimeError("Simulated compression hardware/kernel failure.")

            # Step 3 & 4: Produce compressed representation
            # Optional CUDA stream context
            if self.cuda_stream is not None and torch.cuda.is_available():
                with torch.cuda.stream(self.cuda_stream):
                    comp_k = self.quantizer.quantize_key(source_key_ref)
                    comp_v = self.quantizer.quantize_value(source_value_ref)
            else:
                comp_k = self.quantizer.quantize_key(source_key_ref)
                comp_v = self.quantizer.quantize_value(source_value_ref)

            # Step 5: Validate compressed data and metadata
            self._validate_compressed_tensor(comp_k, "Key")
            self._validate_compressed_tensor(comp_v, "Value")

            # Check reconstruction fidelity
            recon_k = self.dequantizer.dequantize(comp_k, target_dtype=chunk.metadata.dtype)
            recon_v = self.dequantizer.dequantize(comp_v, target_dtype=chunk.metadata.dtype)
            metrics_k = self.dequantizer.evaluate_reconstruction(source_key_ref, recon_k)
            metrics_v = self.dequantizer.evaluate_reconstruction(source_value_ref, recon_v)

            if metrics_k["psnr"] < self.min_psnr_threshold or metrics_v["psnr"] < self.min_psnr_threshold:
                raise ValueError(
                    f"Reconstruction PSNR below threshold {self.min_psnr_threshold} dB: "
                    f"Key={metrics_k['psnr']:.2f} dB, Value={metrics_v['psnr']:.2f} dB"
                )

            # Step 6 & 7: Confirm success & Atomically update block references
            chunk.compressed_key = comp_k
            chunk.compressed_value = comp_v
            chunk.metadata.extra["reconstruction_metrics"] = {
                "key": metrics_k,
                "value": metrics_v,
            }

            # Step 8: GPU Stream / CPU Synchronization
            if self.cuda_stream is not None and torch.cuda.is_available():
                self.cuda_stream.synchronize()

            chunk.metadata.state = BlockState.COMPRESSED
            # Step 9: Source storage can now be marked RECLAIMABLE
            chunk.metadata.extra["source_storage_reclaimable"] = True
            return True

        except Exception as e:
            # Failure Handling: Retain original BF16 block and record error
            chunk.metadata.state = BlockState.COMPRESSION_FAILED
            chunk.metadata.error_log = str(e)
            chunk.compressed_key = None
            chunk.compressed_value = None
            chunk.metadata.extra["source_storage_reclaimable"] = False
            
            # Safe recovery back to ACTIVE_BF16 for uninterrupted inference
            chunk.key = source_key_ref
            chunk.value = source_value_ref
            logger.warning(
                f"Compression failed for chunk {chunk.metadata.chunk_id}: {e}. "
                f"Retaining original BF16 tensor."
            )
            return False

    def _validate_compressed_tensor(self, comp: CompressedTensor, name: str):
        """Validates that scales, zeros, and packed payloads contain no corrupted values."""
        if torch.isnan(comp.scales).any() or torch.isinf(comp.scales).any():
            raise ValueError(f"{name} scales contain NaN or Inf values")
        if (comp.scales <= 0.0).any():
            raise ValueError(f"{name} scales contain non-positive values")
        if comp.packed_data.numel() == 0:
            raise ValueError(f"{name} packed data buffer is empty")

