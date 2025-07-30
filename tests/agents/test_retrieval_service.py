import json
import os
import tempfile
import threading
import time
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import requests

from debug_gym.agents.retrieval_service import (
    RetrievalManager,
    RetrievalService,
    RetrievalServiceClient,
    RetrievalServiceHandler,
)


class TestRetrievalManager:
    """Test cases for the RetrievalManager class."""

    def create_sample_trajectory_file(self, content):
        """Helper to create a temporary trajectory file."""
        temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl")
        for line in content:
            temp_file.write(json.dumps(line) + "\n")
        temp_file.close()
        return temp_file.name

    def create_sample_trajectory_data(self):
        """Create sample trajectory data for testing."""
        return [
            {
                "satisfied_criteria": [
                    "follows_proper_debugging_workflow",
                    "has_successful_outcome",
                ],
                "messages": [
                    {"role": "system", "content": "System message"},
                    {"role": "user", "content": "Test observation 1"},
                    {
                        "role": "assistant",
                        "content": "Let me use a tool",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "test_tool",
                                    "arguments": {"arg": "value1"},
                                }
                            }
                        ],
                    },
                    {"role": "tool", "content": "Tool response 1"},
                    {
                        "role": "assistant",
                        "content": "Another tool call",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "another_tool",
                                    "arguments": {"arg": "value2"},
                                }
                            }
                        ],
                    },
                ],
            },
            {
                "satisfied_criteria": [
                    "follows_proper_debugging_workflow",
                    "has_successful_outcome",
                ],
                "messages": [
                    {"role": "system", "content": "System message"},
                    {"role": "user", "content": "Test observation 2"},
                    {
                        "role": "assistant",
                        "content": "Using tool with reasoning",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "debug_tool",
                                    "arguments": {"breakpoint": "line 10"},
                                }
                            }
                        ],
                    },
                ],
            },
        ]

    @patch("debug_gym.agents.retrieval_service.SentenceEncoder")
    @patch("debug_gym.agents.retrieval_service.get_shared_cache_manager")
    def test_init(self, mock_cache_manager, mock_sentence_encoder):
        """Test RetrievalManager initialization."""
        config = {
            "rag_cache_dir": ".test_cache",
            "rag_use_cache": True,
            "sentence_encoder_model": "test-model",
        }

        mock_encoder_instance = MagicMock()
        mock_sentence_encoder.return_value = mock_encoder_instance
        mock_cache_manager_instance = MagicMock()
        mock_cache_manager.return_value = mock_cache_manager_instance

        manager = RetrievalManager(config)

        assert manager.config == config
        assert manager.cache_dir == ".test_cache"
        assert manager.use_cache is True
        assert manager.sentence_encoder_model == "test-model"
        assert manager.encoder == mock_encoder_instance
        mock_sentence_encoder.assert_called_once_with(model_name="test-model")

    def test_parse_indexing_method(self):
        """Test parsing of indexing methods."""
        config = {"rag_use_cache": False}

        with patch("debug_gym.agents.retrieval_service.SentenceEncoder"):
            manager = RetrievalManager(config)

            # Test valid methods
            assert manager.parse_indexing_method("tool_call-1") == ["tool_call", 1]
            assert manager.parse_indexing_method("tool_call_with_reasoning-3") == [
                "tool_call_with_reasoning",
                3,
            ]
            assert manager.parse_indexing_method("observation-5") == ["observation", 5]
            assert manager.parse_indexing_method("tool_name") == ["tool_name", 1]

            # Test invalid methods
            with pytest.raises(AssertionError, match="Invalid rag_indexing_method"):
                manager.parse_indexing_method("invalid_method-1")

            with pytest.raises(AssertionError, match="Invalid step value"):
                manager.parse_indexing_method("tool_call-abc")

            with pytest.raises(AssertionError, match="Step must be a positive integer"):
                manager.parse_indexing_method("tool_call-0")

    @patch("debug_gym.agents.retrieval_service.SentenceEncoder")
    def test_load_experience_trajectory_from_file(self, mock_sentence_encoder):
        """Test loading experience trajectories from file."""
        config = {"rag_use_cache": False}
        manager = RetrievalManager(config)

        trajectory_data = self.create_sample_trajectory_data()
        trajectory_file = self.create_sample_trajectory_file(trajectory_data)

        try:
            trajectories = manager.load_experience_trajectory_from_file(trajectory_file)

            assert len(trajectories) == 2
            assert len(trajectories[0]) == 5  # 5 messages in first trajectory
            assert len(trajectories[1]) == 3  # 3 messages in second trajectory
        finally:
            os.unlink(trajectory_file)

    @patch("debug_gym.agents.retrieval_service.SentenceEncoder")
    def test_load_experience_trajectory_filters_unsatisfied(
        self, mock_sentence_encoder
    ):
        """Test that unsatisfied trajectories are filtered out."""
        config = {"rag_use_cache": False}
        manager = RetrievalManager(config)

        # Create data with one unsatisfied trajectory
        trajectory_data = [
            {
                "satisfied_criteria": [
                    "has_successful_outcome"
                ],  # Missing workflow criteria
                "messages": [{"role": "user", "content": "Should be filtered"}],
            },
            {
                "satisfied_criteria": [
                    "follows_proper_debugging_workflow",
                    "has_successful_outcome",
                ],
                "messages": [{"role": "user", "content": "Should be included"}],
            },
        ]

        trajectory_file = self.create_sample_trajectory_file(trajectory_data)

        try:
            trajectories = manager.load_experience_trajectory_from_file(trajectory_file)

            assert len(trajectories) == 1  # Only one trajectory should remain
            assert trajectories[0][0]["content"] == "Should be included"
        finally:
            os.unlink(trajectory_file)

    @patch("debug_gym.agents.retrieval_service.SentenceEncoder")
    def test_build_retrieval_dataset_tool_call_method(self, mock_sentence_encoder):
        """Test building retrieval dataset with tool_call method."""
        config = {"rag_use_cache": False}
        manager = RetrievalManager(config)

        trajectory_data = self.create_sample_trajectory_data()
        trajectory_file = self.create_sample_trajectory_file(trajectory_data)

        try:
            trajectories = manager.load_experience_trajectory_from_file(trajectory_file)
            data_input, data_label = manager.build_retrieval_dataset(
                trajectories, ["tool_call", 1]
            )

            assert len(data_input) > 0
            assert len(data_input) == len(data_label)

            # Check that labels contain tool call information
            for label in data_label:
                label_dict = json.loads(label)
                assert "tool_calls" in label_dict
                assert "name" in label_dict["tool_calls"]
                assert "arguments" in label_dict["tool_calls"]
        finally:
            os.unlink(trajectory_file)

    @patch("debug_gym.agents.retrieval_service.FaissRetriever")
    @patch("debug_gym.agents.retrieval_service.SentenceEncoder")
    def test_build_index(self, mock_sentence_encoder, mock_faiss_retriever):
        """Test building an index."""
        config = {"rag_use_cache": False}

        mock_encoder_instance = MagicMock()
        mock_sentence_encoder.return_value = mock_encoder_instance
        mock_encoder_instance.encode_sentence.return_value = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )

        mock_retriever_instance = MagicMock()
        mock_faiss_retriever.return_value = mock_retriever_instance

        manager = RetrievalManager(config)

        trajectory_data = self.create_sample_trajectory_data()
        trajectory_file = self.create_sample_trajectory_file(trajectory_data)

        try:
            success = manager.build_index(
                index_key="test_index",
                experience_trajectory_path=trajectory_file,
                rag_indexing_method="tool_call-1",
                sentence_encoder_model="test-model",
                rag_indexing_batch_size=16,
                use_cache=False,
            )

            assert success is True
            assert "test_index" in manager.indexes

            index_data = manager.indexes["test_index"]
            assert "retriever" in index_data
            assert "data_input" in index_data
            assert "data_label" in index_data

            mock_retriever_instance.add.assert_called_once()
        finally:
            os.unlink(trajectory_file)

    @patch("debug_gym.agents.retrieval_service.FaissRetriever")
    @patch("debug_gym.agents.retrieval_service.SentenceEncoder")
    def test_retrieve(self, mock_sentence_encoder, mock_faiss_retriever):
        """Test retrieving examples from an index."""
        config = {"rag_use_cache": False}

        mock_encoder_instance = MagicMock()
        mock_sentence_encoder.return_value = mock_encoder_instance
        mock_encoder_instance.encode_sentence.return_value = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )

        mock_retriever_instance = MagicMock()
        mock_faiss_retriever.return_value = mock_retriever_instance
        mock_retriever_instance.retrieve.return_value = (
            np.array([[0.1, 0.2]]),  # distances
            np.array([[0, 1]]),  # indices
        )

        manager = RetrievalManager(config)

        trajectory_data = self.create_sample_trajectory_data()
        trajectory_file = self.create_sample_trajectory_file(trajectory_data)

        try:
            # Build index first
            manager.build_index(
                index_key="test_index",
                experience_trajectory_path=trajectory_file,
                rag_indexing_method="tool_call-1",
                sentence_encoder_model="test-model",
                use_cache=False,
            )

            # Mock the query encoding
            mock_encoder_instance.encode_sentence.return_value = np.array(
                [[0.7, 0.8, 0.9]]
            )

            # Test retrieval
            results = manager.retrieve("test_index", "test query", num_retrievals=2)

            assert len(results) <= 2
            mock_retriever_instance.retrieve.assert_called_once()
        finally:
            os.unlink(trajectory_file)

    @patch("debug_gym.agents.retrieval_service.SentenceEncoder")
    def test_retrieve_nonexistent_index(self, mock_sentence_encoder):
        """Test retrieving from a nonexistent index raises error."""
        config = {"rag_use_cache": False}
        manager = RetrievalManager(config)

        with pytest.raises(ValueError, match="Index 'nonexistent' not found"):
            manager.retrieve("nonexistent", "test query")


class TestRetrievalService:
    """Test cases for the RetrievalService class."""

    @patch("debug_gym.agents.retrieval_service.RetrievalManager")
    @patch("debug_gym.agents.retrieval_service.ThreadedHTTPServer")
    def test_start_service(self, mock_server_class, mock_manager_class):
        """Test starting the retrieval service."""
        config = {"test": "config"}
        mock_manager_instance = MagicMock()
        mock_manager_class.return_value = mock_manager_instance

        mock_server_instance = MagicMock()
        mock_server_class.return_value = mock_server_instance

        service = RetrievalService(config, port=8766, host="localhost")
        service.start_service()

        assert service.retrieval_manager == mock_manager_instance
        mock_manager_class.assert_called_once_with(config)
        mock_server_class.assert_called_once()
        assert service.server_thread is not None

    @patch("debug_gym.agents.retrieval_service.RetrievalManager")
    def test_stop_service(self, mock_manager_class):
        """Test stopping the retrieval service."""
        config = {}
        service = RetrievalService(config)

        mock_server = MagicMock()
        mock_thread = MagicMock()

        service.server = mock_server
        service.server_thread = mock_thread

        service.stop_service()

        mock_server.shutdown.assert_called_once()
        mock_server.server_close.assert_called_once()
        mock_thread.join.assert_called_once()


class TestRetrievalServiceClient:
    """Test cases for the RetrievalServiceClient class."""

    def test_init(self):
        """Test client initialization."""
        client = RetrievalServiceClient(host="test-host", port=9999, timeout=60)

        assert client.base_url == "http://test-host:9999"
        assert client.timeout == 60

    @patch("requests.get")
    def test_is_service_available_true(self, mock_get):
        """Test service availability check when service is available."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        client = RetrievalServiceClient()
        assert client.is_service_available() is True
        mock_get.assert_called_once_with("http://localhost:8766/health", timeout=5)

    @patch("requests.get")
    def test_is_service_available_false(self, mock_get):
        """Test service availability check when service is not available."""
        mock_get.side_effect = requests.ConnectionError("Connection failed")

        client = RetrievalServiceClient()
        assert client.is_service_available() is False

    @patch("requests.post")
    def test_build_index_success(self, mock_post):
        """Test successful index building."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True, "index_key": "test_index"}
        mock_post.return_value = mock_response

        client = RetrievalServiceClient()
        result = client.build_index(
            index_key="test_index",
            experience_trajectory_path="/path/to/file.jsonl",
            rag_indexing_method="tool_call-1",
            sentence_encoder_model="test-model",
        )

        assert result is True
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_build_index_failure(self, mock_post):
        """Test index building failure."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        mock_post.return_value = mock_response

        client = RetrievalServiceClient()

        with pytest.raises(RuntimeError, match="Retrieval service error: 500"):
            client.build_index(
                index_key="test_index",
                experience_trajectory_path="/path/to/file.jsonl",
                rag_indexing_method="tool_call-1",
                sentence_encoder_model="test-model",
            )

    @patch("requests.post")
    def test_retrieve_success(self, mock_post):
        """Test successful retrieval."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "relevant_examples": [
                '{"tool_calls": {"name": "test_tool", "arguments": {"arg": "value"}}}',
                '{"tool_calls": {"name": "another_tool", "arguments": {"arg": "value2"}}}',
            ]
        }
        mock_post.return_value = mock_response

        client = RetrievalServiceClient()
        results = client.retrieve("test_index", "test query", num_retrievals=2)

        assert len(results) == 2
        assert "test_tool" in results[0]
        assert "another_tool" in results[1]
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_retrieve_connection_error(self, mock_post):
        """Test retrieval with connection error."""
        mock_post.side_effect = requests.ConnectionError("Connection failed")

        client = RetrievalServiceClient()

        with pytest.raises(
            RuntimeError, match="Failed to connect to retrieval service"
        ):
            client.retrieve("test_index", "test query")

    @patch("requests.get")
    def test_list_indexes(self, mock_get):
        """Test listing indexes."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"indexes": ["index1", "index2", "index3"]}
        mock_get.return_value = mock_response

        client = RetrievalServiceClient()
        indexes = client.list_indexes()

        assert indexes == ["index1", "index2", "index3"]
        mock_get.assert_called_once_with("http://localhost:8766/indexes", timeout=10)


class TestRetrievalServiceIntegration:
    """Integration tests for the retrieval service."""

    def create_sample_trajectory_file(self, content):
        """Helper to create a temporary trajectory file."""
        temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl")
        for line in content:
            temp_file.write(json.dumps(line) + "\n")
        temp_file.close()
        return temp_file.name

    def create_sample_trajectory_data(self):
        """Create sample trajectory data for testing."""
        return [
            {
                "satisfied_criteria": [
                    "follows_proper_debugging_workflow",
                    "has_successful_outcome",
                ],
                "messages": [
                    {"role": "system", "content": "System message"},
                    {"role": "user", "content": "Test observation"},
                    {
                        "role": "assistant",
                        "content": "Using debug tool",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "pdb",
                                    "arguments": {"command": "l"},
                                }
                            }
                        ],
                    },
                    {"role": "tool", "content": "Tool output"},
                    {
                        "role": "assistant",
                        "content": "Analysis complete",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "view",
                                    "arguments": {"path": "test.py"},
                                }
                            }
                        ],
                    },
                ],
            }
        ]

    @patch("debug_gym.agents.retrieval_service.SentenceEncoder")
    @patch("debug_gym.agents.retrieval_service.FaissRetriever")
    def test_end_to_end_workflow(self, mock_faiss_retriever, mock_sentence_encoder):
        """Test end-to-end workflow with mocked dependencies."""
        # Setup mocks
        mock_encoder_instance = MagicMock()
        mock_sentence_encoder.return_value = mock_encoder_instance
        mock_encoder_instance.encode_sentence.return_value = np.array([[0.1, 0.2, 0.3]])

        mock_retriever_instance = MagicMock()
        mock_faiss_retriever.return_value = mock_retriever_instance
        mock_retriever_instance.retrieve.return_value = (
            np.array([[0.1]]),  # distances
            np.array([[0]]),  # indices
        )

        # Create test data
        trajectory_data = self.create_sample_trajectory_data()
        trajectory_file = self.create_sample_trajectory_file(trajectory_data)

        try:
            # Test with RetrievalManager directly
            config = {
                "rag_cache_dir": ".test_cache",
                "rag_use_cache": False,
                "sentence_encoder_model": "test-model",
            }

            manager = RetrievalManager(config)

            # Build index
            success = manager.build_index(
                index_key="test_integration",
                experience_trajectory_path=trajectory_file,
                rag_indexing_method="tool_call-1",
                sentence_encoder_model="test-model",
            )

            assert success is True

            # Retrieve examples
            results = manager.retrieve(
                "test_integration", "test query", num_retrievals=1
            )
            assert len(results) <= 1

        finally:
            os.unlink(trajectory_file)
