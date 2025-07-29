from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import requests

from debug_gym.agents.encoding_service import EncodingService, EncodingServiceClient


class TestEncodingService:
    """Test cases for the encoding service."""

    def create_mock_encoder(self):
        """Create a mock encoder for testing."""
        mock_encoder = MagicMock()
        mock_encoder.encode_sentence.return_value = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32
        )
        mock_encoder.encode_sentence_querying.return_value = np.array(
            [[0.7, 0.8, 0.9]], dtype=np.float32
        )
        return mock_encoder

    def test_encoding_service_initialization(self):
        """Test encoding service initialization."""
        service = EncodingService(model_name="test-model", host="localhost", port=8765)

        assert service.model_name == "test-model"
        assert service.host == "localhost"
        assert service.port == 8765
        assert service.encoder is None  # Encoder is initialized when service starts

    def test_encoding_service_start_stop(self):
        """Test starting and stopping the encoding service."""
        mock_encoder = self.create_mock_encoder()

        with patch(
            "debug_gym.agents.encoding_service.SentenceEncoder",
            return_value=mock_encoder,
        ):
            service = EncodingService(
                model_name="test-model", host="localhost", port=0
            )  # Use port 0 for auto-assignment

            # Start service
            service.start_service()

            assert service.encoder is not None
            assert service.server is not None
            assert service.server_thread is not None
            assert service.server_thread.is_alive()

            # Stop service
            service.stop_service()
            service.server_thread.join(timeout=5)

    def test_encoding_service_health_check(self):
        """Test health check endpoint."""
        mock_encoder = self.create_mock_encoder()

        with patch(
            "debug_gym.agents.encoding_service.SentenceEncoder",
            return_value=mock_encoder,
        ):
            service = EncodingService(model_name="test-model", host="localhost", port=0)
            service.start_service()

            try:
                # Get the actual port assigned
                actual_port = service.server.server_address[1]

                # Test health check
                response = requests.get(
                    f"http://localhost:{actual_port}/health", timeout=5
                )

                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "healthy"

            finally:
                service.stop_service()

    def test_encoding_service_encode_endpoint(self):
        """Test the encode endpoint."""
        mock_encoder = self.create_mock_encoder()
        expected_embeddings = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32
        )
        mock_encoder.encode_sentence.return_value = expected_embeddings

        with patch(
            "debug_gym.agents.encoding_service.SentenceEncoder",
            return_value=mock_encoder,
        ):
            service = EncodingService(model_name="test-model", host="localhost", port=0)
            service.start_service()

            try:
                # Get the actual port assigned
                actual_port = service.server.server_address[1]

                # Test encoding endpoint
                data = {"texts": ["Hello", "World"], "batch_size": 2, "is_query": False}

                response = requests.post(
                    f"http://localhost:{actual_port}/encode", json=data, timeout=5
                )

                assert response.status_code == 200
                result = response.json()

                # Check structure
                assert "embeddings" in result
                assert "shape" in result

                # Check embeddings
                embeddings = np.array(result["embeddings"], dtype=np.float32)
                np.testing.assert_array_equal(embeddings, expected_embeddings)

                # Verify mock was called correctly
                mock_encoder.encode_sentence.assert_called_once_with(
                    ["Hello", "World"], batch_size=2
                )

            finally:
                service.stop_service()

    def test_encoding_service_encode_querying_endpoint(self):
        """Test the encode_querying endpoint."""
        mock_encoder = self.create_mock_encoder()
        expected_embeddings = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32
        )
        mock_encoder.encode_sentence_querying.return_value = expected_embeddings

        with patch(
            "debug_gym.agents.encoding_service.SentenceEncoder",
            return_value=mock_encoder,
        ):
            service = EncodingService(model_name="test-model", host="localhost", port=0)
            service.start_service()

            try:
                # Get the actual port assigned
                actual_port = service.server.server_address[1]

                # Test encoding endpoint with is_query=True
                data = {"texts": ["Query text"], "batch_size": 1, "is_query": True}

                response = requests.post(
                    f"http://localhost:{actual_port}/encode", json=data, timeout=5
                )

                assert response.status_code == 200
                result = response.json()

                # Check structure
                assert "embeddings" in result
                assert "shape" in result

                # Check embeddings
                embeddings = np.array(result["embeddings"], dtype=np.float32)
                np.testing.assert_array_equal(embeddings, expected_embeddings)

                # Verify mock was called correctly
                mock_encoder.encode_sentence_querying.assert_called_once_with(
                    ["Query text"], batch_size=1
                )

            finally:
                service.stop_service()

    def test_encoding_service_error_handling(self):
        """Test error handling in encoding service."""
        mock_encoder = self.create_mock_encoder()
        mock_encoder.encode_sentence.side_effect = Exception("Encoding failed")

        with patch(
            "debug_gym.agents.encoding_service.SentenceEncoder",
            return_value=mock_encoder,
        ):
            service = EncodingService(model_name="test-model", host="localhost", port=0)
            service.start_service()

            try:
                # Get the actual port assigned
                actual_port = service.server.server_address[1]

                # Test error handling
                data = {"texts": ["Hello"], "batch_size": 1, "is_query": False}

                response = requests.post(
                    f"http://localhost:{actual_port}/encode", json=data, timeout=5
                )

                assert response.status_code == 500

            finally:
                service.stop_service()


class TestEncodingServiceClient:
    """Test cases for the encoding service client."""

    def test_client_initialization(self):
        """Test client initialization."""
        client = EncodingServiceClient(host="localhost", port=8765)
        assert client.base_url == "http://localhost:8765"
        assert client.timeout == 30

    @patch("requests.get")
    def test_is_service_available_success(self, mock_get):
        """Test successful service availability check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        client = EncodingServiceClient(host="localhost", port=8765)
        result = client.is_service_available()

        assert result is True
        mock_get.assert_called_once_with("http://localhost:8765/health", timeout=5)

    @patch("requests.get")
    def test_is_service_available_failure(self, mock_get):
        """Test service availability check failure."""
        mock_get.side_effect = requests.exceptions.RequestException("Connection failed")

        client = EncodingServiceClient(host="localhost", port=8765)
        result = client.is_service_available()

        assert result is False

    @patch("requests.post")
    def test_encode_sentence_success(self, mock_post):
        """Test successful sentence encoding."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        }
        mock_post.return_value = mock_response

        client = EncodingServiceClient(host="localhost", port=8765)
        result = client.encode_sentence(["Hello", "World"], batch_size=2)

        expected = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        np.testing.assert_array_equal(result, expected)

        mock_post.assert_called_once_with(
            "http://localhost:8765/encode",
            json={"texts": ["Hello", "World"], "batch_size": 2, "is_query": False},
            timeout=30,
        )

    @patch("requests.post")
    def test_encode_sentence_querying_success(self, mock_post):
        """Test successful query encoding."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embeddings": [[0.7, 0.8, 0.9]]}
        mock_post.return_value = mock_response

        client = EncodingServiceClient(host="localhost", port=8765)
        result = client.encode_sentence_querying(["Query"], batch_size=1)

        expected = np.array([[0.7, 0.8, 0.9]])
        np.testing.assert_array_equal(result, expected)

        mock_post.assert_called_once_with(
            "http://localhost:8765/encode",
            json={"texts": ["Query"], "batch_size": 1, "is_query": True},
            timeout=30,
        )

    @patch("requests.post")
    def test_encode_sentence_failure(self, mock_post):
        """Test encoding failure handling."""
        mock_post.side_effect = requests.exceptions.RequestException("Request failed")

        client = EncodingServiceClient(host="localhost", port=8765)

        with pytest.raises(requests.exceptions.RequestException):
            client.encode_sentence(["Hello"], batch_size=1)

    @patch("requests.post")
    def test_encode_sentence_server_error(self, mock_post):
        """Test handling of server errors."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        mock_post.return_value = mock_response

        client = EncodingServiceClient(host="localhost", port=8765)

        with pytest.raises(RuntimeError, match="Encoding service error"):
            client.encode_sentence(["Hello"], batch_size=1)


class TestEncodingServiceIntegration:
    """Integration tests for encoding service with RAG agent."""

    @patch("debug_gym.agents.rag_agent.EncodingServiceClient")
    def test_rag_agent_with_encoding_service(self, mock_client_class):
        """Test RAG agent integration with encoding service."""
        # Mock the client
        mock_client = MagicMock()
        mock_client.is_service_available.return_value = True
        mock_client.encode_sentence.return_value = np.random.rand(2, 768).astype(
            np.float32
        )
        mock_client_class.return_value = mock_client

        # Create config for RAG agent with all required parameters
        config = {
            "rag_use_encoding_service": True,
            "rag_encoding_service_host": "localhost",
            "rag_encoding_service_port": 8765,
            "experience_trajectory_path": "test_path.jsonl",
            "output_path": "/tmp/test_output",  # Required by base agent
            "rag_indexing_method": "tool_call-1",  # Required for RAG agent
            "random_seed": 42,  # Required by base agent
            "memory_size": 100,  # Required by base agent
        }

        # Mock other dependencies to avoid file system and environment dependencies
        with patch(
            "debug_gym.agents.rag_agent.get_shared_cache_manager"
        ) as mock_cache_manager:
            mock_cache_manager.return_value = MagicMock()

            # Import and create RAG agent
            from debug_gym.agents.rag_agent import RAGAgent

            # Mock the file loading and dataset building methods to avoid file dependencies
            with (
                patch.object(RAGAgent, "load_experience_trajectory_from_file"),
                patch.object(RAGAgent, "build_retrieval_dataset"),
                patch.object(RAGAgent, "_build_index"),
            ):

                agent = RAGAgent(config=config, env=None, llm=None, logger=MagicMock())

                # Verify encoding service client was created and configured
                assert agent.use_encoding_service == True
                assert agent.encoding_service_host == "localhost"
                assert agent.encoding_service_port == 8765

    @patch("debug_gym.agents.rag_agent.EncodingServiceClient")
    @patch("debug_gym.agents.rag_agent.SentenceEncoder")
    def test_rag_agent_fallback_to_local_encoder(
        self, mock_sentence_encoder, mock_client_class
    ):
        """Test RAG agent fallback to local encoder when service unavailable."""
        # Mock the client to be unavailable
        mock_client = MagicMock()
        mock_client.is_service_available.return_value = False
        mock_client_class.return_value = mock_client

        # Mock local encoder
        mock_local_encoder = MagicMock()
        mock_sentence_encoder.return_value = mock_local_encoder

        # Create config for RAG agent with all required parameters
        config = {
            "rag_use_encoding_service": True,
            "rag_encoding_service_host": "localhost",
            "rag_encoding_service_port": 8765,
            "sentence_encoder_model": "test-model",
            "experience_trajectory_path": "test_path.jsonl",
            "output_path": "/tmp/test_output",  # Required by base agent
            "rag_indexing_method": "tool_call-1",  # Required for RAG agent
            "random_seed": 42,  # Required by base agent
            "memory_size": 100,  # Required by base agent
        }

        # Mock other dependencies
        with patch(
            "debug_gym.agents.rag_agent.get_shared_cache_manager"
        ) as mock_cache_manager:
            mock_cache_manager.return_value = MagicMock()

            # Import and create RAG agent
            from debug_gym.agents.rag_agent import RAGAgent

            # Mock the file loading and dataset building methods
            with (
                patch.object(RAGAgent, "load_experience_trajectory_from_file"),
                patch.object(RAGAgent, "build_retrieval_dataset"),
                patch.object(RAGAgent, "_build_index"),
            ):

                agent = RAGAgent(config=config, env=None, llm=None, logger=MagicMock())

                # Verify fallback to local encoder
                assert agent.use_encoding_service == False
                assert agent.encoder == mock_local_encoder
                mock_sentence_encoder.assert_called_once_with(model_name="test-model")
