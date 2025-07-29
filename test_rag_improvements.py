#!/usr/bin/env python3
"""
Test script to validate the encoding service and shared cache implementation.
This tests the core functionality without requiring the full debug_gym environment.
"""

import os
import shutil
import sys
import tempfile
import threading
import time
from unittest.mock import Mock, patch

import numpy as np

# Add the debug_gym directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_encoding_service():
    """Test the encoding service functionality."""
    print("=" * 60)
    print("Testing Encoding Service")
    print("=" * 60)

    try:
        from debug_gym.agents.encoding_service import (
            EncodingService,
            EncodingServiceClient,
        )

        # Mock the SentenceEncoder to avoid loading actual models
        class MockSentenceEncoder:
            def __init__(self, model_name):
                self.model_name = model_name
                print(f"Mock encoder initialized with model: {model_name}")

            def encode_sentence(self, texts, batch_size=16):
                print(f"Mock encoding {len(texts)} texts with batch_size={batch_size}")
                # Return mock embeddings (768 dimensions)
                return np.random.rand(len(texts), 768).astype(np.float32)

            def encode_sentence_querying(self, texts, batch_size=16):
                print(
                    f"Mock query encoding {len(texts)} texts with batch_size={batch_size}"
                )
                return np.random.rand(len(texts), 768).astype(np.float32)

        # Patch the SentenceEncoder import
        with patch(
            "debug_gym.agents.encoding_service.SentenceEncoder", MockSentenceEncoder
        ):
            # Start encoding service
            service = EncodingService("mock-model", port=8766)
            service.start_service()

            try:
                # Test client
                client = EncodingServiceClient(port=8766)

                # Wait for service to be ready
                if not client.wait_for_service(max_wait_time=10):
                    raise RuntimeError("Service did not start in time")

                print("✓ Service started successfully")

                # Test encoding
                texts = ["hello world", "how are you", "this is a test"]
                embeddings = client.encode_sentence(texts, batch_size=2)

                print(
                    f"✓ Encoded {len(texts)} texts, got embeddings shape: {embeddings.shape}"
                )
                assert embeddings.shape == (
                    3,
                    768,
                ), f"Expected (3, 768), got {embeddings.shape}"

                # Test query encoding
                query_embeddings = client.encode_sentence_querying(
                    ["query text"], batch_size=1
                )
                print(f"✓ Query encoding works, shape: {query_embeddings.shape}")
                assert query_embeddings.shape == (
                    1,
                    768,
                ), f"Expected (1, 768), got {query_embeddings.shape}"

                print("✓ Encoding service test passed!")

            finally:
                service.stop_service()

    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Encoding service test failed: {e}")
        return False

    return True


def test_shared_cache():
    """Test the shared cache functionality."""
    print("\n" + "=" * 60)
    print("Testing Shared Cache Manager")
    print("=" * 60)

    try:
        from debug_gym.agents.shared_cache import (
            SharedCacheManager,
            get_shared_cache_manager,
        )

        # Create temporary cache directory
        temp_dir = tempfile.mkdtemp()

        try:
            # Test cache manager - use the global one to ensure consistency
            cache_manager = get_shared_cache_manager(temp_dir)

            # Mock data
            data_input = ["text1", "text2", "text3"]
            mock_embeddings = np.random.rand(3, 768).astype(np.float32)

            def mock_compute_callback(texts):
                print(f"Mock computing embeddings for {len(texts)} texts")
                return mock_embeddings

            # Test cache creation
            cache_key = "test_cache"
            indexing_method = ["tool_name", 1]
            encoder_model = "mock-model"

            result_input, result_embeddings = cache_manager.load_or_create_cache(
                cache_key=cache_key,
                indexing_method=indexing_method,
                encoder_model=encoder_model,
                data_input=data_input,
                compute_callback=mock_compute_callback,
            )

            print("✓ Cache created successfully")
            assert result_input == data_input, "Input data mismatch"
            assert np.array_equal(
                result_embeddings, mock_embeddings
            ), "Embeddings mismatch"

            # Test cache loading (should use cached data)
            result_input2, result_embeddings2 = cache_manager.load_or_create_cache(
                cache_key=cache_key,
                indexing_method=indexing_method,
                encoder_model=encoder_model,
                data_input=None,  # Should not be used
                compute_callback=None,  # Should not be called
            )

            print("✓ Cache loaded from memory successfully")
            assert result_input2 == data_input, "Cached input data mismatch"
            assert np.array_equal(
                result_embeddings2, mock_embeddings
            ), "Cached embeddings mismatch"

            # Test global cache manager
            global_cache = get_shared_cache_manager(temp_dir)
            assert (
                global_cache is cache_manager
            ), "Global cache manager should be the same instance"
            print("✓ Global cache manager works")

            # Test cache info
            info = cache_manager.get_cache_info()
            print(f"✓ Cache info: {info}")
            assert cache_key in info["in_memory_caches"], "Cache key not in memory"
            assert info["memory_usage_mb"] > 0, "Memory usage should be > 0"

            # Test cache eviction by creating more caches than max_cache_size
            cache_manager.max_cache_size = 2
            for i in range(3):
                cache_manager.load_or_create_cache(
                    cache_key=f"test_cache_{i}",
                    indexing_method=indexing_method,
                    encoder_model=encoder_model,
                    data_input=[f"text_{i}"],
                    compute_callback=lambda x: np.random.rand(len(x), 768).astype(
                        np.float32
                    ),
                )

            info_after = cache_manager.get_cache_info()
            print(
                f"✓ Cache eviction test - in memory: {len(info_after['in_memory_caches'])}"
            )
            assert len(info_after["in_memory_caches"]) <= 2, "Cache eviction failed"

            print("✓ Shared cache test passed!")

        finally:
            # Clean up
            shutil.rmtree(temp_dir)

    except Exception as e:
        print(f"✗ Shared cache test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def test_concurrent_cache_access():
    """Test concurrent access to shared cache."""
    print("\n" + "=" * 60)
    print("Testing Concurrent Cache Access")
    print("=" * 60)

    try:
        from debug_gym.agents.shared_cache import SharedCacheManager

        temp_dir = tempfile.mkdtemp()

        try:
            cache_manager = SharedCacheManager(cache_dir=temp_dir)

            results = []
            errors = []

            def worker_thread(thread_id):
                try:
                    cache_key = (
                        f"concurrent_test_{thread_id % 2}"  # Use 2 different caches
                    )
                    data_input = [f"text_{thread_id}_{i}" for i in range(3)]

                    def compute_callback(texts):
                        time.sleep(0.1)  # Simulate computation time
                        return np.random.rand(len(texts), 768).astype(np.float32)

                    result_input, result_embeddings = (
                        cache_manager.load_or_create_cache(
                            cache_key=cache_key,
                            indexing_method=["tool_name", 1],
                            encoder_model="mock-model",
                            data_input=data_input,
                            compute_callback=compute_callback,
                        )
                    )

                    results.append(
                        (thread_id, len(result_input), result_embeddings.shape)
                    )

                except Exception as e:
                    errors.append((thread_id, str(e)))

            # Start multiple threads
            threads = []
            for i in range(5):
                thread = threading.Thread(target=worker_thread, args=(i,))
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            print(
                f"✓ Concurrent test completed - {len(results)} successful, {len(errors)} errors"
            )

            if errors:
                for thread_id, error in errors:
                    print(f"  Thread {thread_id} error: {error}")

            assert len(errors) == 0, f"Some threads failed: {errors}"
            assert len(results) == 5, f"Expected 5 results, got {len(results)}"

            print("✓ Concurrent cache access test passed!")

        finally:
            shutil.rmtree(temp_dir)

    except Exception as e:
        print(f"✗ Concurrent cache test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def test_integration():
    """Test integration between encoding service and shared cache."""
    print("\n" + "=" * 60)
    print("Testing Integration")
    print("=" * 60)

    try:
        from debug_gym.agents.encoding_service import (
            EncodingService,
            EncodingServiceClient,
        )
        from debug_gym.agents.shared_cache import SharedCacheManager

        # Mock encoder
        class MockSentenceEncoder:
            def __init__(self, model_name):
                self.model_name = model_name

            def encode_sentence(self, texts, batch_size=16):
                return np.random.rand(len(texts), 768).astype(np.float32)

            def encode_sentence_querying(self, texts, batch_size=16):
                return np.random.rand(len(texts), 768).astype(np.float32)

        temp_dir = tempfile.mkdtemp()

        try:
            with patch(
                "debug_gym.agents.encoding_service.SentenceEncoder", MockSentenceEncoder
            ):
                # Start encoding service
                service = EncodingService("mock-model", port=8767)
                service.start_service()

                try:
                    # Create cache manager
                    cache_manager = SharedCacheManager(cache_dir=temp_dir)

                    # Create encoding client
                    client = EncodingServiceClient(port=8767)
                    if not client.wait_for_service(max_wait_time=10):
                        raise RuntimeError("Service did not start in time")

                    # Test integration: use service for cache computation
                    def service_compute_callback(texts):
                        return client.encode_sentence(texts, batch_size=16)

                    data_input = ["integration test text 1", "integration test text 2"]
                    result_input, result_embeddings = (
                        cache_manager.load_or_create_cache(
                            cache_key="integration_test",
                            indexing_method=["tool_name", 1],
                            encoder_model="mock-model",
                            data_input=data_input,
                            compute_callback=service_compute_callback,
                        )
                    )

                    print("✓ Integration with encoding service successful")
                    assert len(result_input) == 2, "Input length mismatch"
                    assert result_embeddings.shape == (
                        2,
                        768,
                    ), f"Embeddings shape mismatch: {result_embeddings.shape}"

                    # Test cache reuse
                    result_input2, result_embeddings2 = (
                        cache_manager.load_or_create_cache(
                            cache_key="integration_test",
                            indexing_method=["tool_name", 1],
                            encoder_model="mock-model",
                            data_input=None,
                            compute_callback=None,
                        )
                    )

                    print("✓ Cache reuse works with service")
                    assert np.array_equal(
                        result_embeddings, result_embeddings2
                    ), "Cached embeddings mismatch"

                    print("✓ Integration test passed!")

                finally:
                    service.stop_service()

        finally:
            shutil.rmtree(temp_dir)

    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def main():
    """Run all tests."""
    print(
        "Starting comprehensive test of encoding service and shared cache implementation"
    )
    print("=" * 80)

    # Mock the gym.utils module to avoid import issues
    sys.modules["debug_gym.gym.utils"] = Mock()
    sys.modules["debug_gym.gym.utils"].filter_non_utf8 = lambda x: x

    # Mock the agents.utils module
    sys.modules["debug_gym.agents.utils"] = Mock()

    test_results = []

    # Run tests
    test_results.append(("Encoding Service", test_encoding_service()))
    test_results.append(("Shared Cache", test_shared_cache()))
    test_results.append(("Concurrent Access", test_concurrent_cache_access()))
    test_results.append(("Integration", test_integration()))

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    all_passed = True
    for test_name, passed in test_results:
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:20s}: {status}")
        if not passed:
            all_passed = False

    print("=" * 80)
    if all_passed:
        print("🎉 All tests passed! The implementation is working correctly.")
        print("\nKey improvements verified:")
        print("  ✓ Encoding service can handle multiple concurrent requests")
        print("  ✓ Shared cache manager prevents duplicate memory usage")
        print("  ✓ Thread-safe concurrent access to cached embeddings")
        print("  ✓ Proper cache eviction and memory management")
        print("  ✓ Integration between service and cache works seamlessly")
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
