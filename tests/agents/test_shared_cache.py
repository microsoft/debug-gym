"""
Test cases for the shared cache manager functionality.
"""

import os
import tempfile
import threading
import time
from unittest.mock import Mock

import numpy as np
import pytest

from debug_gym.agents.shared_cache import SharedCacheManager, get_shared_cache_manager


class TestSharedCacheManager:
    """Test cases for SharedCacheManager."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_manager = SharedCacheManager(cache_dir=self.temp_dir)

    def teardown_method(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test that cache manager initializes correctly."""
        assert self.cache_manager.cache_dir == self.temp_dir
        assert os.path.exists(self.temp_dir)
        assert len(self.cache_manager.cache_data) == 0
        assert self.cache_manager.max_cache_size == 5

    def test_get_cache_path(self):
        """Test cache path generation."""
        cache_key = "test_key"
        expected_path = os.path.join(self.temp_dir, f"rag_cache_{cache_key}.pkl")
        actual_path = self.cache_manager._get_cache_path(cache_key)
        assert actual_path == expected_path

    def test_load_or_create_cache_new_cache(self):
        """Test creating new cache when it doesn't exist."""
        cache_key = "test_cache"
        data_input = ["test sentence 1", "test sentence 2"]
        indexing_method = ["tfidf"]
        encoder_model = "test_model"
        mock_embeddings = np.array([[1, 2, 3], [4, 5, 6]])

        def mock_compute(texts):
            return mock_embeddings

        result_data, result_embeddings = self.cache_manager.load_or_create_cache(
            cache_key=cache_key,
            indexing_method=indexing_method,
            encoder_model=encoder_model,
            data_input=data_input,
            compute_callback=mock_compute,
        )

        assert result_data == data_input
        np.testing.assert_array_equal(result_embeddings, mock_embeddings)
        assert cache_key in self.cache_manager.cache_data

    def test_load_or_create_cache_from_memory(self):
        """Test loading cache from memory."""
        cache_key = "test_cache"
        data_input = ["test sentence 1", "test sentence 2"]
        indexing_method = ["tfidf"]
        encoder_model = "test_model"
        mock_embeddings = np.array([[1, 2, 3], [4, 5, 6]])

        def mock_compute(texts):
            return mock_embeddings

        # Create cache first
        self.cache_manager.load_or_create_cache(
            cache_key=cache_key,
            indexing_method=indexing_method,
            encoder_model=encoder_model,
            data_input=data_input,
            compute_callback=mock_compute,
        )

        # Mock compute function should not be called for cached data
        def mock_compute_not_called(texts):
            pytest.fail("Compute function should not be called for cached data")

        result_data, result_embeddings = self.cache_manager.load_or_create_cache(
            cache_key=cache_key,
            indexing_method=indexing_method,
            encoder_model=encoder_model,
            compute_callback=mock_compute_not_called,
        )

        assert result_data == data_input
        np.testing.assert_array_equal(result_embeddings, mock_embeddings)

    def test_cache_config_validation(self):
        """Test that cache is invalidated when configuration doesn't match."""
        cache_key = "test_cache"
        data_input = ["test sentence"]
        indexing_method = ["tfidf"]
        encoder_model = "model1"
        mock_embeddings = np.array([[1, 2, 3]])

        def mock_compute(texts):
            return mock_embeddings

        # Create cache with initial config
        self.cache_manager.load_or_create_cache(
            cache_key=cache_key,
            indexing_method=indexing_method,
            encoder_model=encoder_model,
            data_input=data_input,
            compute_callback=mock_compute,
        )

        # Save to disk to test loading logic
        self.cache_manager.clear_memory_cache()

        # Try to load with different encoder model
        called = False

        def mock_compute_called(texts):
            nonlocal called
            called = True
            return np.array([[4, 5, 6]])

        result_data, result_embeddings = self.cache_manager.load_or_create_cache(
            cache_key=cache_key,
            indexing_method=indexing_method,
            encoder_model="different_model",
            data_input=data_input,
            compute_callback=mock_compute_called,
        )

        assert called  # Should recompute due to model mismatch

    def test_memory_eviction(self):
        """Test memory eviction when max cache size is reached."""
        # Create more caches than max_cache_size
        for i in range(self.cache_manager.max_cache_size + 2):
            cache_key = f"test_cache_{i}"
            data_input = [f"test sentence {i}"]
            indexing_method = ["tfidf"]
            encoder_model = "test_model"
            mock_embeddings = np.array([[i, i + 1, i + 2]])

            def mock_compute(texts):
                return mock_embeddings

            self.cache_manager.load_or_create_cache(
                cache_key=cache_key,
                indexing_method=indexing_method,
                encoder_model=encoder_model,
                data_input=data_input,
                compute_callback=mock_compute,
            )

        # Should have evicted some caches
        assert len(self.cache_manager.cache_data) <= self.cache_manager.max_cache_size

    def test_thread_safety(self):
        """Test that cache manager is thread-safe."""
        cache_key = "test_cache"
        data_input = ["test sentence"]
        indexing_method = ["tfidf"]
        encoder_model = "test_model"
        mock_embeddings = np.array([[1, 2, 3]])
        results = []
        errors = []

        def mock_compute(texts):
            time.sleep(0.01)  # Simulate some processing time
            return mock_embeddings

        def worker():
            try:
                result = self.cache_manager.load_or_create_cache(
                    cache_key=cache_key,
                    indexing_method=indexing_method,
                    encoder_model=encoder_model,
                    data_input=data_input,
                    compute_callback=mock_compute,
                )
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should succeed
        assert len(errors) == 0
        assert len(results) == 5
        # All results should be the same
        for result in results:
            assert result[0] == data_input
            np.testing.assert_array_equal(result[1], mock_embeddings)

    def test_clear_memory_cache(self):
        """Test memory cache clearing functionality."""
        cache_key = "test_cache"
        data_input = ["test sentence"]
        indexing_method = ["tfidf"]
        encoder_model = "test_model"
        mock_embeddings = np.array([[1, 2, 3]])

        def mock_compute(texts):
            return mock_embeddings

        # Create cache
        self.cache_manager.load_or_create_cache(
            cache_key=cache_key,
            indexing_method=indexing_method,
            encoder_model=encoder_model,
            data_input=data_input,
            compute_callback=mock_compute,
        )
        assert len(self.cache_manager.cache_data) > 0

        # Clear memory cache
        self.cache_manager.clear_memory_cache()
        assert len(self.cache_manager.cache_data) == 0

    def test_get_cache_info(self):
        """Test cache information retrieval."""
        cache_key = "test_cache"
        data_input = ["test sentence"]
        indexing_method = ["tfidf"]
        encoder_model = "test_model"
        mock_embeddings = np.array([[1, 2, 3]])

        def mock_compute(texts):
            return mock_embeddings

        # Create cache
        self.cache_manager.load_or_create_cache(
            cache_key=cache_key,
            indexing_method=indexing_method,
            encoder_model=encoder_model,
            data_input=data_input,
            compute_callback=mock_compute,
        )

        info = self.cache_manager.get_cache_info()
        assert "memory_usage_mb" in info
        assert "in_memory_caches" in info
        assert "disk_caches" in info
        assert len(info["in_memory_caches"]) > 0

    def test_missing_compute_callback_error(self):
        """Test error when compute_callback is missing for new cache."""
        with pytest.raises(
            ValueError, match="data_input and compute_callback must be provided"
        ):
            self.cache_manager.load_or_create_cache(
                cache_key="test_cache",
                indexing_method=["tfidf"],
                encoder_model="test_model",
            )


class TestGetSharedCacheManager:
    """Test cases for get_shared_cache_manager function."""

    def test_singleton_behavior(self):
        """Test that the same cache manager is returned for the same cache_dir."""
        cache_dir1 = "/tmp/test_cache1"
        cache_dir2 = "/tmp/test_cache2"

        manager1a = get_shared_cache_manager(cache_dir1)
        manager1b = get_shared_cache_manager(cache_dir1)
        manager2 = get_shared_cache_manager(cache_dir2)

        # Same cache_dir should return same instance
        assert manager1a is manager1b
        # Different cache_dir should return different instance
        assert manager1a is not manager2

    def test_default_cache_dir(self):
        """Test default cache directory behavior."""
        manager1 = get_shared_cache_manager()
        manager2 = get_shared_cache_manager()

        assert manager1 is manager2
        assert manager1.cache_dir == ".rag_cache"
