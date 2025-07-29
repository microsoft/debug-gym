"""
Shared cache manager for RAG agent representations.
This allows multiple agents to share the same cached representations without
loading multiple copies into memory.
"""

import os
import pickle
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from debug_gym.gym.utils import filter_non_utf8
from debug_gym.logger import DebugGymLogger


class SharedCacheManager:
    """Thread-safe cache manager for sharing embeddings across multiple RAG agents."""

    def __init__(self, cache_dir: str = ".rag_cache"):
        self.cache_dir = cache_dir
        self.cache_data: Dict[str, Dict] = {}
        self.lock = threading.RLock()
        self.access_times: Dict[str, float] = {}
        self.max_cache_size = 5  # Maximum number of different caches to keep in memory
        self.logger = DebugGymLogger(__name__)

        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_path(self, cache_key: str) -> str:
        """Get the full path for the cache file."""
        return os.path.join(self.cache_dir, f"rag_cache_{cache_key}.pkl")

    def _evict_oldest_cache(self):
        """Evict the oldest accessed cache to free memory."""
        if len(self.cache_data) < self.max_cache_size:
            return

        # Find the oldest accessed cache
        oldest_key = min(self.access_times, key=self.access_times.get)
        del self.cache_data[oldest_key]
        del self.access_times[oldest_key]
        self.logger.info(f"Evicted cache {oldest_key} from memory")

    def load_or_create_cache(
        self,
        cache_key: str,
        indexing_method: List,
        encoder_model: str,
        data_input: Optional[List[str]] = None,
        compute_callback: Optional[callable] = None,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Load cache if exists, otherwise create it.

        Args:
            cache_key: Unique identifier for the cache
            indexing_method: RAG indexing method for validation
            encoder_model: Encoder model name for validation
            data_input: Input data to cache (if creating new cache)
            compute_callback: Function to compute embeddings if cache doesn't exist

        Returns:
            Tuple of (data_input, input_representations)
        """
        with self.lock:
            # Check if already loaded in memory
            if cache_key in self.cache_data:
                self.access_times[cache_key] = time.time()
                cache_data = self.cache_data[cache_key]
                self.logger.info(f"Using in-memory cache for {cache_key}")
                return cache_data["data_input"], cache_data["input_representations"]

            # Try to load from disk
            cache_path = self._get_cache_path(cache_key)
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "rb") as f:
                        cache_data = pickle.load(f)

                    # Verify cache consistency
                    if (
                        cache_data.get("indexing_method") != indexing_method
                        or cache_data.get("encoder_model") != encoder_model
                    ):
                        self.logger.warning(
                            f"Cache configuration mismatch for {cache_key}, ignoring cache"
                        )
                    else:
                        # Load into memory
                        self._evict_oldest_cache()
                        self.cache_data[cache_key] = cache_data
                        self.access_times[cache_key] = time.time()
                        self.logger.info(
                            f"Loaded cache {cache_key} from disk into memory"
                        )
                        return (
                            cache_data["data_input"],
                            cache_data["input_representations"],
                        )

                except Exception as e:
                    self.logger.warning(f"Failed to load cache {cache_key}: {e}")

            # Cache doesn't exist or is invalid, create new one
            if data_input is None or compute_callback is None:
                raise ValueError(
                    "data_input and compute_callback must be provided to create new cache"
                )

            self.logger.info(
                f"Computing embeddings for cache {cache_key} (this may take time)..."
            )
            input_representations = compute_callback(data_input)

            # Save to disk
            self._save_cache_to_disk(
                cache_key,
                data_input,
                input_representations,
                indexing_method,
                encoder_model,
            )

            # Load into memory
            self._evict_oldest_cache()
            cache_data = {
                "data_input": data_input,
                "input_representations": input_representations,
                "indexing_method": indexing_method,
                "encoder_model": encoder_model,
            }
            self.cache_data[cache_key] = cache_data
            self.access_times[cache_key] = time.time()

            return data_input, input_representations

    def _save_cache_to_disk(
        self,
        cache_key: str,
        data_input: List[str],
        input_representations: np.ndarray,
        indexing_method: List,
        encoder_model: str,
    ):
        """Save cache to disk."""
        cache_path = self._get_cache_path(cache_key)
        try:
            cache_data = {
                "data_input": data_input,
                "input_representations": input_representations,
                "indexing_method": indexing_method,
                "encoder_model": encoder_model,
            }
            with open(cache_path, "wb") as f:
                pickle.dump(cache_data, f)
            self.logger.info(f"Saved cache {cache_key} to disk")
        except Exception as e:
            self.logger.warning(f"Failed to save cache {cache_key}: {e}")

    def clear_memory_cache(self):
        """Clear all caches from memory (but keep on disk)."""
        with self.lock:
            self.cache_data.clear()
            self.access_times.clear()
            self.logger.info("Cleared all caches from memory")

    def get_cache_info(self) -> Dict:
        """Get information about current cache state."""
        with self.lock:
            return {
                "in_memory_caches": list(self.cache_data.keys()),
                "memory_usage_mb": sum(
                    cache["input_representations"].nbytes / (1024 * 1024)
                    for cache in self.cache_data.values()
                ),
                "disk_caches": [
                    f.replace("rag_cache_", "").replace(".pkl", "")
                    for f in os.listdir(self.cache_dir)
                    if f.startswith("rag_cache_") and f.endswith(".pkl")
                ],
            }


# Global shared cache manager instances by cache directory
_shared_cache_managers = {}
_cache_manager_lock = threading.Lock()


def get_shared_cache_manager(cache_dir: str = ".rag_cache") -> SharedCacheManager:
    """Get the global shared cache manager instance for the specified cache directory."""
    global _shared_cache_managers
    with _cache_manager_lock:
        if cache_dir not in _shared_cache_managers:
            _shared_cache_managers[cache_dir] = SharedCacheManager(cache_dir)
        return _shared_cache_managers[cache_dir]


class BatchProcessor:
    """Process multiple encoding requests in batches for efficiency."""

    def __init__(
        self, encoder_client, max_batch_size: int = 64, max_wait_time: float = 0.1
    ):
        self.encoder_client = encoder_client
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.pending_requests = []
        self.lock = threading.Lock()
        self.processing_thread = None
        self.stop_event = threading.Event()
        self.logger = DebugGymLogger(__name__)

    def start(self):
        """Start the batch processing thread."""
        self.processing_thread = threading.Thread(target=self._process_batches)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def stop(self):
        """Stop the batch processing."""
        self.stop_event.set()
        if self.processing_thread:
            self.processing_thread.join()

    def _process_batches(self):
        """Main batch processing loop."""
        while not self.stop_event.is_set():
            with self.lock:
                if not self.pending_requests:
                    continue

                # Take a batch of requests
                batch = self.pending_requests[: self.max_batch_size]
                self.pending_requests = self.pending_requests[self.max_batch_size :]

            if batch:
                self._process_batch(batch)

            time.sleep(self.max_wait_time)

    def _process_batch(self, batch):
        """Process a batch of requests."""
        try:
            # Separate texts and callbacks
            texts = [req["text"] for req in batch]
            is_query = batch[0]["is_query"]  # Assume all in batch have same type

            # Encode all texts at once
            if is_query:
                embeddings = self.encoder_client.encode_sentence_querying(texts)
            else:
                embeddings = self.encoder_client.encode_sentence(texts)

            # Return results to callbacks
            for i, req in enumerate(batch):
                try:
                    req["callback"](embeddings[i])
                except Exception as e:
                    self.logger.error(f"Error in callback: {e}")

        except Exception as e:
            self.logger.error(f"Error processing batch: {e}")
            # Return errors to callbacks
            for req in batch:
                try:
                    req["callback"](None, error=str(e))
                except:
                    pass

    def encode_async(self, text: str, callback: callable, is_query: bool = False):
        """Add an encoding request to the batch queue."""
        with self.lock:
            self.pending_requests.append(
                {"text": text, "callback": callback, "is_query": is_query}
            )
