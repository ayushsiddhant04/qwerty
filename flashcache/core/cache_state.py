"""
Cache State Definitions and Block Lifecycle Enums.
"""

from enum import Enum, auto


class BlockState(Enum):
    """
    Lifecycle states for individual memory blocks in FlashCache-KV.
    Enforces strict memory safety and asynchronous hand-off guarantees.
    """
    ACTIVE_BF16 = "ACTIVE_BF16"                 # Original uncompressed tensor active
    COMPRESSION_PENDING = "COMPRESSION_PENDING" # In compression pipeline; source retained
    COMPRESSED = "COMPRESSED"                   # Quantized & verified; source retained until sync
    COMPRESSION_FAILED = "COMPRESSION_FAILED"   # Fallback state; original BF16 preserved
    RECLAIMABLE = "RECLAIMABLE"                 # Source BF16 safely scheduled for memory release


class ChunkStatus(Enum):
    """
    Retention designation determined by ranking and budget allocation.
    """
    CANDIDATE = "CANDIDATE"     # Initial unranked candidate
    PROTECTED = "PROTECTED"     # Attention sink or recent local token (never evicted)
    SELECTED = "SELECTED"       # Retained under allocated budget (eligible for compression)
    EVICTED = "EVICTED"         # Slated for eviction/discarding

