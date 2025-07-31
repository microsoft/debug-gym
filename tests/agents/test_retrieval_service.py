import json
import os
import socket
import tempfile
import threading
import time
from http.server import HTTPServer
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import requests
import yaml

from debug_gym.agents.retrieval_service import (
    RetrievalManager,
    RetrievalService,
    RetrievalServiceClient,
    RetrievalServiceHandler,
    ThreadedHTTPServer,
    start_retrieval_service_standalone,
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
        """Test retrieval with connection error returns empty list."""
        mock_post.side_effect = requests.ConnectionError("Connection failed")

        client = RetrievalServiceClient()

        # Should return empty list instead of raising exception
        results = client.retrieve("test_index", "test query")
        assert results == []

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


class TestThreadedHTTPServer:
    """Test cases for the ThreadedHTTPServer class."""

    def test_server_bind_socket_options(self):
        """Test that server_bind sets the correct socket options."""
        with patch.object(HTTPServer, "server_bind") as mock_super_bind:
            with patch("socket.socket") as mock_socket:
                mock_socket_instance = MagicMock()

                # Create a server instance (this will call server_bind once)
                server = ThreadedHTTPServer(("localhost", 0), MagicMock)
                server.socket = mock_socket_instance

                # Reset the mock to clear the call from initialization
                mock_super_bind.reset_mock()
                mock_socket_instance.reset_mock()

                # Call server_bind explicitly
                server.server_bind()

                # Verify HTTPServer.server_bind was called once after reset
                mock_super_bind.assert_called_once()

                # Verify socket options were set (using actual socket constant values)
                expected_calls = [
                    (65535, 4, 1),  # SOL_SOCKET, SO_REUSEADDR, 1
                    (6, 1, 1),  # IPPROTO_TCP, TCP_NODELAY, 1
                    (65535, 8, 1),  # SOL_SOCKET, SO_KEEPALIVE, 1
                ]

                actual_calls = [
                    call[0] for call in mock_socket_instance.setsockopt.call_args_list
                ]
                for expected_call in expected_calls:
                    assert expected_call in actual_calls

                # Verify timeout was set
                mock_socket_instance.settimeout.assert_called_once_with(30)

    def test_server_attributes(self):
        """Test that ThreadedHTTPServer has the correct attributes."""
        server = ThreadedHTTPServer(("localhost", 0), MagicMock)

        assert server.daemon_threads is True
        assert server.timeout == 60
        assert server.allow_reuse_address is True
        assert server.request_queue_size == 128


class TestRetrievalServiceHandler:
    """Comprehensive test cases for the RetrievalServiceHandler class."""

    def create_mock_handler(self, retrieval_manager=None):
        """Helper to create a mock handler with necessary attributes."""
        if retrieval_manager is None:
            retrieval_manager = MagicMock()

        # Create handler without triggering __init__ to avoid HTTP parsing
        handler = RetrievalServiceHandler.__new__(RetrievalServiceHandler)
        handler.retrieval_manager = retrieval_manager
        handler.logger = MagicMock()
        handler.send_response = MagicMock()
        handler.send_error = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = MagicMock()
        handler.connection = MagicMock()
        handler.rfile = MagicMock()
        handler.headers = {}
        handler.path = "/"

        return handler

    def test_handler_init(self):
        """Test RetrievalServiceHandler initialization."""
        retrieval_manager = MagicMock()

        # Test that handler stores retrieval_manager correctly
        handler = self.create_mock_handler(retrieval_manager)

        assert handler.retrieval_manager == retrieval_manager

    def test_log_request_does_nothing(self):
        """Test that log_request method does nothing (overridden to reduce noise)."""
        handler = self.create_mock_handler()

        # Should not raise any exceptions and do nothing
        handler.log_request(200, 1024)
        handler.log_request()

    def test_safe_send_response_success(self):
        """Test safe_send_response when successful."""
        handler = self.create_mock_handler()

        result = handler.safe_send_response(200, "OK")

        assert result is True
        handler.send_response.assert_called_once_with(200, "OK")

    def test_safe_send_response_broken_pipe(self):
        """Test safe_send_response handles BrokenPipeError."""
        handler = self.create_mock_handler()
        handler.send_response.side_effect = BrokenPipeError("Broken pipe")

        result = handler.safe_send_response(200)

        assert result is False

    def test_safe_send_response_connection_reset(self):
        """Test safe_send_response handles ConnectionResetError."""
        handler = self.create_mock_handler()
        handler.send_response.side_effect = ConnectionResetError("Connection reset")

        result = handler.safe_send_response(200)

        assert result is False

    def test_safe_send_response_generic_exception(self):
        """Test safe_send_response handles generic exceptions."""
        handler = self.create_mock_handler()
        handler.send_response.side_effect = Exception("Generic error")

        result = handler.safe_send_response(200)

        assert result is False

    def test_safe_send_error_success(self):
        """Test safe_send_error when successful."""
        handler = self.create_mock_handler()

        handler.safe_send_error(404, "Not found")

        handler.send_error.assert_called_once_with(404, "Not found")

    def test_safe_send_error_broken_pipe(self):
        """Test safe_send_error handles BrokenPipeError."""
        handler = self.create_mock_handler()
        handler.send_error.side_effect = BrokenPipeError("Broken pipe")

        # Should not raise exception
        handler.safe_send_error(500)

    def test_safe_send_error_connection_reset(self):
        """Test safe_send_error handles ConnectionResetError."""
        handler = self.create_mock_handler()
        handler.send_error.side_effect = ConnectionResetError("Connection reset")

        # Should not raise exception
        handler.safe_send_error(500)

    def test_safe_send_error_generic_exception(self):
        """Test safe_send_error handles generic exceptions."""
        handler = self.create_mock_handler()
        handler.send_error.side_effect = Exception("Generic error")

        # Should not raise exception
        handler.safe_send_error(500)

    def test_safe_write_response_success(self):
        """Test safe_write_response when successful."""
        handler = self.create_mock_handler()
        test_data = {"test": "data"}

        result = handler.safe_write_response(test_data)

        assert result is True
        handler.send_header.assert_any_call("Content-Type", "application/json")
        handler.send_header.assert_any_call("Connection", "close")
        handler.end_headers.assert_called_once()
        handler.wfile.write.assert_called_once()
        handler.wfile.flush.assert_called_once()

    def test_safe_write_response_broken_pipe(self):
        """Test safe_write_response handles BrokenPipeError."""
        handler = self.create_mock_handler()
        handler.wfile.write.side_effect = BrokenPipeError("Broken pipe")

        result = handler.safe_write_response({"test": "data"})

        assert result is False

    def test_safe_write_response_connection_reset(self):
        """Test safe_write_response handles ConnectionResetError."""
        handler = self.create_mock_handler()
        handler.wfile.flush.side_effect = ConnectionResetError("Connection reset")

        result = handler.safe_write_response({"test": "data"})

        assert result is False

    def test_safe_write_response_generic_exception(self):
        """Test safe_write_response handles generic exceptions."""
        handler = self.create_mock_handler()
        handler.send_header.side_effect = Exception("Generic error")

        result = handler.safe_write_response({"test": "data"})

        assert result is False

    def test_do_get_health_check(self):
        """Test GET /health endpoint."""
        handler = self.create_mock_handler()
        handler.path = "/health"

        with patch.object(handler, "safe_send_response", return_value=True):
            with patch.object(handler, "safe_write_response") as mock_write:
                handler.do_GET()

                mock_write.assert_called_once_with({"status": "healthy"})

    def test_do_get_indexes(self):
        """Test GET /indexes endpoint."""
        handler = self.create_mock_handler()
        handler.path = "/indexes"
        handler.retrieval_manager.indexes = {"index1": {}, "index2": {}}

        with patch.object(handler, "safe_send_response", return_value=True):
            with patch.object(handler, "safe_write_response") as mock_write:
                handler.do_GET()

                mock_write.assert_called_once_with({"indexes": ["index1", "index2"]})

    def test_do_get_not_found(self):
        """Test GET to unknown endpoint returns 404."""
        handler = self.create_mock_handler()
        handler.path = "/unknown"

        with patch.object(handler, "safe_send_error") as mock_error:
            handler.do_GET()

            mock_error.assert_called_once_with(404, "Endpoint not found")

    def test_do_get_broken_pipe_error(self):
        """Test GET handles BrokenPipeError gracefully."""
        handler = self.create_mock_handler()
        handler.path = "/health"

        with patch.object(
            handler, "safe_send_response", side_effect=BrokenPipeError("Broken pipe")
        ):
            # Should not raise exception
            handler.do_GET()

    def test_do_get_connection_reset_error(self):
        """Test GET handles ConnectionResetError gracefully."""
        handler = self.create_mock_handler()
        handler.path = "/health"

        with patch.object(
            handler,
            "safe_send_response",
            side_effect=ConnectionResetError("Connection reset"),
        ):
            # Should not raise exception
            handler.do_GET()

    def test_do_get_generic_exception(self):
        """Test GET handles generic exceptions."""
        handler = self.create_mock_handler()
        handler.path = "/health"

        with patch.object(
            handler, "safe_send_response", side_effect=Exception("Generic error")
        ):
            with patch.object(handler, "safe_send_error") as mock_error:
                handler.do_GET()

                mock_error.assert_called_once_with(
                    500, "Internal server error: Generic error"
                )

    def test_do_post_retrieve_success(self):
        """Test POST /retrieve endpoint success."""
        handler = self.create_mock_handler()
        handler.path = "/retrieve"
        handler.headers = {"Content-Length": "50"}

        post_data = json.dumps(
            {"index_key": "test_index", "query_text": "test query", "num_retrievals": 2}
        ).encode("utf-8")

        handler.rfile.read.return_value = post_data
        handler.retrieval_manager.retrieve.return_value = ["result1", "result2"]

        with patch.object(handler, "safe_send_response", return_value=True):
            with patch.object(
                handler, "safe_write_response", return_value=True
            ) as mock_write:
                handler.do_POST()

                handler.retrieval_manager.retrieve.assert_called_once_with(
                    "test_index", "test query", 2
                )
                mock_write.assert_called_once_with(
                    {"relevant_examples": ["result1", "result2"]}
                )

    def test_do_post_retrieve_missing_params(self):
        """Test POST /retrieve with missing parameters."""
        handler = self.create_mock_handler()
        handler.path = "/retrieve"
        handler.headers = {"Content-Length": "20"}

        post_data = json.dumps({"index_key": "test"}).encode("utf-8")
        handler.rfile.read.return_value = post_data

        with patch.object(handler, "safe_send_error") as mock_error:
            handler.do_POST()

            mock_error.assert_called_once_with(
                400, "index_key and query_text are required"
            )

    def test_do_post_retrieve_retrieval_exception(self):
        """Test POST /retrieve handles retrieval exceptions."""
        handler = self.create_mock_handler()
        handler.path = "/retrieve"
        handler.headers = {"Content-Length": "50"}

        post_data = json.dumps(
            {"index_key": "test_index", "query_text": "test query"}
        ).encode("utf-8")

        handler.rfile.read.return_value = post_data
        handler.retrieval_manager.retrieve.side_effect = Exception("Retrieval failed")

        with patch.object(handler, "safe_send_error") as mock_error:
            handler.do_POST()

            mock_error.assert_called_once_with(500, "Retrieval error: Retrieval failed")

    def test_do_post_retrieve_broken_pipe_during_retrieval(self):
        """Test POST /retrieve handles BrokenPipeError during retrieval."""
        handler = self.create_mock_handler()
        handler.path = "/retrieve"
        handler.headers = {"Content-Length": "50"}

        post_data = json.dumps(
            {"index_key": "test_index", "query_text": "test query"}
        ).encode("utf-8")

        handler.rfile.read.return_value = post_data
        handler.retrieval_manager.retrieve.side_effect = BrokenPipeError("Broken pipe")

        # Should not raise exception
        handler.do_POST()

    def test_do_post_build_index_success(self):
        """Test POST /build_index endpoint success."""
        handler = self.create_mock_handler()
        handler.path = "/build_index"
        handler.headers = {"Content-Length": "100"}

        post_data = json.dumps(
            {
                "index_key": "test_index",
                "experience_trajectory_path": "/path/to/file.jsonl",
                "rag_indexing_method": "tool_call-1",
                "sentence_encoder_model": "test-model",
            }
        ).encode("utf-8")

        handler.rfile.read.return_value = post_data
        handler.retrieval_manager.build_index.return_value = True

        with patch.object(handler, "safe_send_response", return_value=True):
            with patch.object(
                handler, "safe_write_response", return_value=True
            ) as mock_write:
                handler.do_POST()

                mock_write.assert_called_once_with(
                    {"success": True, "index_key": "test_index"}
                )

    def test_do_post_build_index_missing_params(self):
        """Test POST /build_index with missing parameters."""
        handler = self.create_mock_handler()
        handler.path = "/build_index"
        handler.headers = {"Content-Length": "30"}

        post_data = json.dumps({"index_key": "test"}).encode("utf-8")
        handler.rfile.read.return_value = post_data

        with patch.object(handler, "safe_send_error") as mock_error:
            handler.do_POST()

            mock_error.assert_called_once_with(
                400, "Missing required parameters for index building"
            )

    def test_do_post_build_index_exception(self):
        """Test POST /build_index handles exceptions."""
        handler = self.create_mock_handler()
        handler.path = "/build_index"
        handler.headers = {"Content-Length": "100"}

        post_data = json.dumps(
            {
                "index_key": "test_index",
                "experience_trajectory_path": "/path/to/file.jsonl",
                "rag_indexing_method": "tool_call-1",
                "sentence_encoder_model": "test-model",
            }
        ).encode("utf-8")

        handler.rfile.read.return_value = post_data
        handler.retrieval_manager.build_index.side_effect = Exception("Build failed")

        with patch.object(handler, "safe_send_error") as mock_error:
            handler.do_POST()

            mock_error.assert_called_once_with(
                500, "Index building error: Build failed"
            )

    def test_do_post_check_index_success(self):
        """Test POST /check_index endpoint success."""
        handler = self.create_mock_handler()
        handler.path = "/check_index"
        handler.headers = {"Content-Length": "30"}

        post_data = json.dumps({"index_key": "test_index"}).encode("utf-8")
        handler.rfile.read.return_value = post_data
        handler.retrieval_manager.has_index.return_value = True

        with patch.object(handler, "safe_send_response", return_value=True):
            with patch.object(
                handler, "safe_write_response", return_value=True
            ) as mock_write:
                handler.do_POST()

                mock_write.assert_called_once_with(
                    {"exists": True, "index_key": "test_index"}
                )

    def test_do_post_check_index_missing_key(self):
        """Test POST /check_index with missing index_key."""
        handler = self.create_mock_handler()
        handler.path = "/check_index"
        handler.headers = {"Content-Length": "10"}

        post_data = json.dumps({}).encode("utf-8")
        handler.rfile.read.return_value = post_data

        with patch.object(handler, "safe_send_error") as mock_error:
            handler.do_POST()

            mock_error.assert_called_once_with(400, "index_key is required")

    def test_do_post_check_index_exception(self):
        """Test POST /check_index handles exceptions."""
        handler = self.create_mock_handler()
        handler.path = "/check_index"
        handler.headers = {"Content-Length": "30"}

        post_data = json.dumps({"index_key": "test_index"}).encode("utf-8")
        handler.rfile.read.return_value = post_data
        handler.retrieval_manager.has_index.side_effect = Exception("Check failed")

        with patch.object(handler, "safe_send_error") as mock_error:
            handler.do_POST()

            mock_error.assert_called_once_with(500, "Index check error: Check failed")

    def test_do_post_unknown_endpoint(self):
        """Test POST to unknown endpoint returns 404."""
        handler = self.create_mock_handler()
        handler.path = "/unknown"
        handler.headers = {"Content-Length": "10"}
        handler.rfile.read.return_value = b'{"test": 1}'

        with patch.object(handler, "safe_send_error") as mock_error:
            handler.do_POST()

            mock_error.assert_called_once_with(404, "Endpoint not found")

    def test_do_post_broken_pipe_error(self):
        """Test POST handles BrokenPipeError gracefully."""
        handler = self.create_mock_handler()
        handler.path = "/retrieve"
        handler.headers = {"Content-Length": "10"}
        handler.rfile.read.side_effect = BrokenPipeError("Broken pipe")

        # Should not raise exception
        handler.do_POST()

    def test_do_post_connection_reset_error(self):
        """Test POST handles ConnectionResetError gracefully."""
        handler = self.create_mock_handler()
        handler.path = "/retrieve"
        handler.headers = {"Content-Length": "10"}
        handler.rfile.read.side_effect = ConnectionResetError("Connection reset")

        # Should not raise exception
        handler.do_POST()

    def test_do_post_generic_exception(self):
        """Test POST handles generic exceptions."""
        handler = self.create_mock_handler()
        handler.path = "/retrieve"
        handler.headers = {"Content-Length": "invalid"}  # This will cause int() to fail

        with patch.object(handler, "safe_send_error") as mock_error:
            handler.do_POST()

            # Should call safe_send_error with 500 status
            assert mock_error.called
            args = mock_error.call_args[0]
            assert args[0] == 500
            assert "Internal server error" in args[1]

    def test_connection_shutdown_exception_handling(self):
        """Test that connection.shutdown exceptions are handled gracefully."""
        handler = self.create_mock_handler()
        handler.path = "/retrieve"
        handler.headers = {"Content-Length": "50"}

        post_data = json.dumps(
            {"index_key": "test_index", "query_text": "test query"}
        ).encode("utf-8")

        handler.rfile.read.return_value = post_data
        handler.retrieval_manager.retrieve.return_value = ["result"]
        handler.connection.shutdown.side_effect = Exception("Shutdown failed")

        with patch.object(handler, "safe_send_response", return_value=True):
            with patch.object(handler, "safe_write_response", return_value=True):
                # Should not raise exception despite connection.shutdown failing
                handler.do_POST()

                # Verify the operation completed
                handler.retrieval_manager.retrieve.assert_called_once()
