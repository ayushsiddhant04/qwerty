"""
Unit Tests for Stage 4: KIVI-Inspired Asymmetric Quantization and Memory Safety.
"""

import pytest
import torch

from flashcache.core.cache_state import BlockState
from flashcache.chunking import ChunkMetadata, KVChunk
from flashcache.compression import (
    KIVIQuantizer,
    KIVIDequantizer,
    CompressionManager,
    pack_int4,
    unpack_int4,
    pack_int2,
    unpack_int2,
)


def test_bit_packing_int4():
    """Verify packing and unpacking of 4-bit integers."""
    original = torch.tensor([0, 15, 7, 3, 12, 1, 8, 4], dtype=torch.uint8)
    packed = pack_int4(original)
    assert packed.shape[-1] == 4
    unpacked = unpack_int4(packed, target_length=8)
    assert torch.equal(original, unpacked)


def test_bit_packing_int2():
    """Verify packing and unpacking of 2-bit integers."""
    original = torch.tensor([0, 1, 2, 3, 3, 2, 1, 0], dtype=torch.uint8)
    packed = pack_int2(original)
    assert packed.shape[-1] == 2
    unpacked = unpack_int2(packed, target_length=8)
    assert torch.equal(original, unpacked)


def test_kivi_quantizer_4bit_accuracy_and_compression():
    """Verify 4-bit quantization reduces memory and reconstructs with high fidelity."""
    key = torch.randn(1, 1, 16, 64, dtype=torch.bfloat16)
    value = torch.randn(1, 1, 16, 64, dtype=torch.bfloat16)

    quantizer = KIVIQuantizer(bits=4)
    dequantizer = KIVIDequantizer()

    comp_k = quantizer.quantize_key(key)
    comp_v = quantizer.quantize_value(value)

    # Check that total bytes is significantly smaller than uncompressed BF16
    uncompressed_bytes = key.numel() * key.element_size() # 16 * 64 * 2 = 2048 bytes
    assert comp_k.total_bytes < uncompressed_bytes
    # 4-bit payload is ~1/4 size of BF16 (512 bytes) + scales (128 bytes) + zeros (64 bytes) = 704 bytes
    assert comp_k.payload_bytes == 512

    # Verify dequantization reconstruction fidelity
    recon_k = dequantizer.dequantize(comp_k, target_dtype=torch.bfloat16)
    recon_v = dequantizer.dequantize(comp_v, target_dtype=torch.bfloat16)

    metrics_k = dequantizer.evaluate_reconstruction(key, recon_k)
    metrics_v = dequantizer.evaluate_reconstruction(value, recon_v)

    # 4-bit reconstruction should have high cosine similarity (>0.95) and healthy PSNR (>20 dB)
    assert metrics_k["cosine_similarity"] > 0.95
    assert metrics_v["cosine_similarity"] > 0.95
    assert metrics_k["psnr"] > 20.0
    assert metrics_v["psnr"] > 20.0


def test_kivi_quantizer_2bit_mode():
    """Verify 2-bit quantization mode functions and further compresses tensors."""
    key = torch.randn(1, 1, 16, 64, dtype=torch.bfloat16)
    value = torch.randn(1, 1, 16, 64, dtype=torch.bfloat16)

    q2 = KIVIQuantizer(bits=2)
    dequantizer = KIVIDequantizer()

    comp_k = q2.quantize_key(key)
    # 2-bit payload: 16 * 64 * 2 / 8 = 256 bytes
    assert comp_k.payload_bytes == 256

    recon_k = dequantizer.dequantize(comp_k, target_dtype=torch.bfloat16)
    metrics = dequantizer.evaluate_reconstruction(key, recon_k)
    assert metrics["cosine_similarity"] > 0.85


def test_memory_safety_lifecycle_on_successful_compression():
    """Verify the 9-step lifecycle on successful compression."""
    key = torch.randn(1, 1, 16, 64, dtype=torch.bfloat16)
    value = torch.randn(1, 1, 16, 64, dtype=torch.bfloat16)
    meta = ChunkMetadata(
        chunk_id="test_safe_chunk",
        layer_idx=0,
        head_idx=0,
        start_pos=0,
        end_pos=16,
        dtype=torch.bfloat16,
        state=BlockState.ACTIVE_BF16,
    )
    chunk = KVChunk(key, value, meta)

    manager = CompressionManager(bits=4)
    success = manager.compress_chunk_safe(chunk)

    assert success is True
    assert chunk.metadata.state == BlockState.COMPRESSED
    assert chunk.metadata.extra.get("source_storage_reclaimable") is True
    assert chunk.compressed_key is not None
    assert chunk.compressed_value is not None


def test_memory_safety_failure_handling_and_rollback():
    """
    CRITICAL SAFETY TEST:
    Simulates a compression failure and confirms that:
    1. The original BF16 block remains available and completely valid.
    2. Reclamation is strictly forbidden (reclaimable is False).
    3. Block transitions to COMPRESSION_FAILED.
    4. Runtime retains valid BF16 tensor references.
    """
    key = torch.randn(1, 1, 16, 64, dtype=torch.bfloat16)
    value = torch.randn(1, 1, 16, 64, dtype=torch.bfloat16)
    original_key_clone = key.clone()
    original_value_clone = value.clone()

    meta = ChunkMetadata(
        chunk_id="test_fail_chunk",
        layer_idx=0,
        head_idx=0,
        start_pos=0,
        end_pos=16,
        dtype=torch.bfloat16,
        state=BlockState.ACTIVE_BF16,
    )
    chunk = KVChunk(key, value, meta)

    manager = CompressionManager(bits=4)
    # Trigger simulated failure
    success = manager.compress_chunk_safe(chunk, simulate_failure=True)

    assert success is False
    assert chunk.metadata.state == BlockState.COMPRESSION_FAILED
    # Crucial safety invariants:
    assert chunk.metadata.extra.get("source_storage_reclaimable") is False
    assert chunk.compressed_key is None
    assert chunk.compressed_value is None
    # Source tensors must be intact and match original values
    assert torch.equal(chunk.key, original_key_clone)
    assert torch.equal(chunk.value, original_value_clone)
    assert "Simulated compression" in chunk.metadata.error_log
