"""
Asynchronous Compression Worker Engine (Stage 5).
"""

from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Optional, Callable
import threading
import torch

from flashcache.chunking.chunk_metadata import KVChunk
from flashcache.compression.compression_manager import CompressionManager


class AsyncCompressor:
    """
    Manages background thread execution for non-blocking KV block compression.
    Guarantees that readers continue accessing BF16 blocks until compression completes.
    """

    def __init__(
        self,
        compression_manager: CompressionManager,
        max_workers: int = 2,
    ):
        self.manager = compression_manager
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="FlashCacheCompressor")
        self._futures: List[Future] = []
        self._lock = threading.Lock()

    def submit_compression(
        self,
        chunk: KVChunk,
        on_complete_callback: Optional[Callable[[KVChunk, bool], None]] = None,
    ) -> Future:
        """Submits a chunk to the background worker pool for compression."""
        def _task():
            success = self.manager.compress_chunk_safe(chunk)
            if on_complete_callback is not None:
                on_complete_callback(chunk, success)
            return success

        future = self.executor.submit(_task)
        with self._lock:
            self._futures.append(future)
        return future

    def wait_all(self, timeout: Optional[float] = None):
        """Blocks until all active compression jobs finish."""
        with self._lock:
            pending = list(self._futures)
            self._futures.clear()

        for f in pending:
            f.result(timeout=timeout)

    def shutdown(self, wait: bool = True):
        self.executor.shutdown(wait=wait)

