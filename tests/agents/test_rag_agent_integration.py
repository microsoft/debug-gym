import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

try:
    from debug_gym.agents.rag_agent import RAGAgent

    RETRIEVAL_SERVICE_AVAILABLE = True
except ImportError:
    RAGAgent = None
    RETRIEVAL_SERVICE_AVAILABLE = False


# Unit tests that always run with mocked dependencies
class TestRAGAgentMocked:
    """Unit tests for RAGAgent using mocked retrieval service."""

    def create_sample_trajectory_file(self, content):
        """Helper to create a temporary trajectory file."""
        temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl")
        for line in content:
            temp_file.write(json.dumps(line) + "\n")
        temp_file.close()
        return temp_file.name

    def create_mock_config(self, trajectory_file_path):
        """Helper to create mock configuration."""
        return {
            "rag_num_retrievals": 2,
            "rag_indexing_method": "tool_call-1",
            "sentence_encoder_model": "test-model",
            "experience_trajectory_path": trajectory_file_path,
            "rag_retrieval_service_host": "localhost",
            "rag_retrieval_service_port": 8766,
            "rag_retrieval_service_timeout": 120,
            "rag_cache_dir": ".test_cache",
            "rag_use_cache": True,
            "rag_indexing_batch_size": 16,
        }

    @pytest.mark.skipif(
        not RETRIEVAL_SERVICE_AVAILABLE, reason="Retrieval service not available"
    )
    @patch("debug_gym.agents.rag_agent.RetrievalServiceClient")
    @patch("debug_gym.agents.debug_agent.DebugAgent.__init__")
    @patch.object(RAGAgent, "_is_retrieval_service_available", return_value=True)
    def test_rag_agent_with_mocked_service(
        self, mock_availability_check, mock_debug_agent_init, mock_client_class
    ):
        """Test RAGAgent with fully mocked retrieval service."""
        # Create temporary trajectory file
        trajectory_data = [
            {
                "messages": [
                    {"role": "user", "content": "Test"},
                    {"role": "assistant", "content": "Response"},
                ]
            }
        ]
        trajectory_file = self.create_sample_trajectory_file(trajectory_data)
        config = self.create_mock_config(trajectory_file)

        try:
            # Completely replace RAGAgent.__init__ with a custom implementation for testing
            def patched_rag_init(self, config, env, llm=None, logger=None):
                # Set the base attributes that would normally be set by DebugAgent.__init__
                self.config = config
                self.env = env
                self.llm = llm
                self.logger = logger

                # Initialize RAG-specific configuration parameters (copied from original __init__)
                self.rag_num_retrievals = self.config.get("rag_num_retrievals", 1)
                self.rag_indexing_method = self.parse_indexing_method(
                    self.config.get("rag_indexing_method", None)
                )
                self.rag_indexing_batch_size = self.config.get(
                    "rag_indexing_batch_size", 16
                )
                self.sentence_encoder_model = self.config.get(
                    "sentence_encoder_model", "Qwen/Qwen3-Embedding-0.6B"
                )

                # Cache directory for storing computed representations
                self.cache_dir = self.config.get("rag_cache_dir", ".rag_cache")
                self.use_cache = self.config.get("rag_use_cache", True)

                # Retrieval service configuration
                self.retrieval_service_host = self.config.get(
                    "rag_retrieval_service_host", "localhost"
                )
                self.retrieval_service_port = self.config.get(
                    "rag_retrieval_service_port", 8766
                )
                self.retrieval_service_timeout = self.config.get(
                    "rag_retrieval_service_timeout", 120
                )

                self.experience_trajectory_path = self.config.get(
                    "experience_trajectory_path", None
                )
                assert (
                    self.experience_trajectory_path is not None
                ), "Experience path must be provided in the config"

                # Initialize retrieval service client (mocked)
                self._initialize_retrieval_service()

            # Temporarily replace the __init__ method
            original_init = RAGAgent.__init__
            RAGAgent.__init__ = patched_rag_init

            # Mock retrieval service client
            mock_client_instance = MagicMock()
            mock_client_class.return_value = mock_client_instance
            mock_client_instance.is_service_available.return_value = True
            mock_client_instance.check_index.return_value = True  # Index already exists
            mock_client_instance.build_index.return_value = True

            # Create mock environment and logger
            mock_env = MagicMock()
            mock_logger = MagicMock()

            # Initialize RAGAgent
            agent = RAGAgent(config, mock_env, logger=mock_logger)

            # Restore original __init__ method
            RAGAgent.__init__ = original_init

            # Verify basic attributes
            assert agent.rag_num_retrievals == 2
            assert agent.rag_indexing_method == ["tool_call", 1]
            assert hasattr(agent, "retrieval_client")

            # Test that service was called
            mock_client_instance.is_service_available.assert_called_once()

        finally:
            # Restore original __init__ method if it was replaced
            if "original_init" in locals():
                RAGAgent.__init__ = original_init
            os.unlink(trajectory_file)

    @pytest.mark.skipif(
        not RETRIEVAL_SERVICE_AVAILABLE, reason="Retrieval service not available"
    )
    @patch("debug_gym.agents.rag_agent.RetrievalServiceClient")
    def test_extract_query_text_tool_call_method(self, mock_client_class):
        """Test query text extraction with tool_call method."""
        # Create agent without full initialization
        agent = RAGAgent.__new__(RAGAgent)
        agent.rag_indexing_method = ["tool_call", 1]
        agent.delimiter = " <STEP_DELIMITER> "

        # Create mock history
        mock_env_info = MagicMock()
        mock_action = MagicMock()
        mock_action.name = "pdb"
        mock_action.arguments = {"command": "list"}
        mock_env_info.action = mock_action

        mock_history_manager = MagicMock()
        mock_history_manager.get.return_value = ([mock_env_info], None)
        agent.history = mock_history_manager

        # Test extraction
        query_text = agent.extract_query_text_from_history()

        expected = '{"name": "pdb", "arguments": {"command": "list"}}'
        assert query_text == expected

    @pytest.mark.skipif(
        not RETRIEVAL_SERVICE_AVAILABLE, reason="Retrieval service not available"
    )
    @patch("debug_gym.agents.rag_agent.RetrievalServiceClient")
    def test_build_question_prompt_with_mocked_retrieval(self, mock_client_class):
        """Test building question prompt with mocked retrieval results."""
        # Create agent
        agent = RAGAgent.__new__(RAGAgent)
        agent.rag_indexing_method = ["tool_call", 1]
        agent.delimiter = " <STEP_DELIMITER> "
        agent.rag_num_retrievals = 2
        agent.logger = MagicMock()

        # Mock history
        mock_env_info = MagicMock()
        mock_action = MagicMock()
        mock_action.name = "pdb"
        mock_action.arguments = {"command": "list"}
        mock_env_info.action = mock_action

        mock_history_manager = MagicMock()
        mock_history_manager.get.return_value = ([mock_env_info], None)
        agent.history = mock_history_manager

        # Mock retrieval client
        mock_client_instance = MagicMock()
        mock_client_instance.retrieve.return_value = [
            '{"tool_calls": {"name": "pdb", "arguments": {"command": "l"}}, "content": "List code"}',
            '{"tool_calls": {"name": "view", "arguments": {"path": "test.py"}}}',
        ]
        agent.retrieval_client = mock_client_instance
        agent.index_key = "test_index"

        # Test prompt building
        messages = agent.build_question_prompt()

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "debug_gym_ignore" in messages[0]
        assert "retrieved some relevant examples" in messages[0]["content"]
        assert "Example 1" in messages[0]["content"]

    @pytest.mark.skipif(
        not RETRIEVAL_SERVICE_AVAILABLE, reason="Retrieval service not available"
    )
    def test_parse_indexing_method_static(self):
        """Test parsing indexing methods without full initialization."""
        # Create an instance without calling __init__
        agent = RAGAgent.__new__(RAGAgent)

        # Test valid methods
        assert agent.parse_indexing_method("tool_call-1") == ["tool_call", 1]
        assert agent.parse_indexing_method("tool_call_with_reasoning-3") == [
            "tool_call_with_reasoning",
            3,
        ]
        assert agent.parse_indexing_method("observation-5") == ["observation", 5]
        assert agent.parse_indexing_method("tool_name") == ["tool_name", 1]

        # Test invalid methods
        with pytest.raises(AssertionError, match="Invalid rag_indexing_method"):
            agent.parse_indexing_method("invalid_method-1")


# Integration tests that require actual running service
@pytest.mark.skipif(
    not RETRIEVAL_SERVICE_AVAILABLE, reason="Retrieval service not available"
)
class TestRAGAgentIntegration:
    """Simplified integration tests for the RAGAgent class using retrieval service."""

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

    def create_mock_config(self, trajectory_file_path):
        """Helper to create mock configuration for retrieval service."""
        return {
            "rag_num_retrievals": 2,
            "rag_indexing_method": "tool_call-1",
            "sentence_encoder_model": "test-model",
            "experience_trajectory_path": trajectory_file_path,
            "rag_retrieval_service_host": "localhost",
            "rag_retrieval_service_port": 8766,
            "rag_retrieval_service_timeout": 120,
            "rag_cache_dir": ".test_cache",
            "rag_use_cache": True,
            "rag_indexing_batch_size": 16,
        }

    @patch("debug_gym.agents.rag_agent.RetrievalServiceClient")
    @patch("debug_gym.agents.debug_agent.DebugAgent.__init__")
    @patch.object(RAGAgent, "_is_retrieval_service_available", return_value=True)
    def test_rag_agent_initialization_with_service(
        self, mock_availability_check, mock_debug_agent_init, mock_client_class
    ):
        """Test RAGAgent initialization with retrieval service."""
        trajectory_data = self.create_sample_trajectory_data()
        trajectory_file = self.create_sample_trajectory_file(trajectory_data)
        config = self.create_mock_config(trajectory_file)

        try:
            # Create agent instance
            mock_env = MagicMock()
            mock_llm = MagicMock()
            mock_logger = MagicMock()

            # Completely replace RAGAgent.__init__ with a custom implementation for testing
            def patched_rag_init(self, config, env, llm=None, logger=None):
                # Set the base attributes that would normally be set by DebugAgent.__init__
                self.config = config
                self.env = env
                self.llm = llm
                self.logger = logger

                # Initialize RAG-specific configuration parameters (copied from original __init__)
                self.rag_num_retrievals = self.config.get("rag_num_retrievals", 1)
                self.rag_indexing_method = self.parse_indexing_method(
                    self.config.get("rag_indexing_method", None)
                )
                self.rag_indexing_batch_size = self.config.get(
                    "rag_indexing_batch_size", 16
                )
                self.sentence_encoder_model = self.config.get(
                    "sentence_encoder_model", "Qwen/Qwen3-Embedding-0.6B"
                )

                # Cache directory for storing computed representations
                self.cache_dir = self.config.get("rag_cache_dir", ".rag_cache")
                self.use_cache = self.config.get("rag_use_cache", True)

                # Retrieval service configuration
                self.retrieval_service_host = self.config.get(
                    "rag_retrieval_service_host", "localhost"
                )
                self.retrieval_service_port = self.config.get(
                    "rag_retrieval_service_port", 8766
                )
                self.retrieval_service_timeout = self.config.get(
                    "rag_retrieval_service_timeout", 120
                )

                self.experience_trajectory_path = self.config.get(
                    "experience_trajectory_path", None
                )
                assert (
                    self.experience_trajectory_path is not None
                ), "Experience path must be provided in the config"

                # Initialize retrieval service client (mocked)
                self._initialize_retrieval_service()

            # Temporarily replace the __init__ method
            original_init = RAGAgent.__init__
            RAGAgent.__init__ = patched_rag_init

            # Mock the retrieval service client
            mock_client_instance = MagicMock()
            mock_client_class.return_value = mock_client_instance
            mock_client_instance.is_service_available.return_value = True
            mock_client_instance.build_index.return_value = True

            # Initialize RAGAgent normally
            agent = RAGAgent(config, mock_env, mock_llm, mock_logger)

            # Restore original __init__ method
            RAGAgent.__init__ = original_init

            # Verify initialization
            assert agent.config == config
            assert hasattr(agent, "retrieval_client")

        finally:
            # Restore original __init__ method if it was replaced
            if "original_init" in locals():
                RAGAgent.__init__ = original_init
            os.unlink(trajectory_file)

    @patch("debug_gym.agents.rag_agent.RetrievalServiceClient")
    @patch("debug_gym.agents.debug_agent.DebugAgent.__init__")
    @patch.object(RAGAgent, "_is_retrieval_service_available", return_value=True)
    def test_rag_agent_service_unavailable(
        self, mock_availability_check, mock_debug_agent_init, mock_client_class
    ):
        """Test RAGAgent initialization when retrieval service is unavailable."""
        trajectory_data = self.create_sample_trajectory_data()
        trajectory_file = self.create_sample_trajectory_file(trajectory_data)
        config = self.create_mock_config(trajectory_file)

        try:
            # Create mocks
            mock_env = MagicMock()
            mock_llm = MagicMock()
            mock_logger = MagicMock()

            # Completely replace RAGAgent.__init__ with a custom implementation for testing
            def patched_rag_init(self, config, env, llm=None, logger=None):
                # Set the base attributes that would normally be set by DebugAgent.__init__
                self.config = config
                self.env = env
                self.llm = llm
                self.logger = logger

                # Initialize RAG-specific configuration parameters (copied from original __init__)
                self.rag_num_retrievals = self.config.get("rag_num_retrievals", 1)
                self.rag_indexing_method = self.parse_indexing_method(
                    self.config.get("rag_indexing_method", None)
                )
                self.rag_indexing_batch_size = self.config.get(
                    "rag_indexing_batch_size", 16
                )
                self.sentence_encoder_model = self.config.get(
                    "sentence_encoder_model", "Qwen/Qwen3-Embedding-0.6B"
                )

                # Cache directory for storing computed representations
                self.cache_dir = self.config.get("rag_cache_dir", ".rag_cache")
                self.use_cache = self.config.get("rag_use_cache", True)

                # Retrieval service configuration
                self.retrieval_service_host = self.config.get(
                    "rag_retrieval_service_host", "localhost"
                )
                self.retrieval_service_port = self.config.get(
                    "rag_retrieval_service_port", 8766
                )
                self.retrieval_service_timeout = self.config.get(
                    "rag_retrieval_service_timeout", 120
                )

                self.experience_trajectory_path = self.config.get(
                    "experience_trajectory_path", None
                )
                assert (
                    self.experience_trajectory_path is not None
                ), "Experience path must be provided in the config"

                # Initialize retrieval service client (mocked)
                self._initialize_retrieval_service()

            # Temporarily replace the __init__ method
            original_init = RAGAgent.__init__
            RAGAgent.__init__ = patched_rag_init

            # Mock the retrieval service client as unavailable
            mock_client_instance = MagicMock()
            mock_client_class.return_value = mock_client_instance
            mock_client_instance.is_service_available.return_value = False

            # Test that RuntimeError is raised when service is unavailable
            with pytest.raises(RuntimeError, match="Retrieval service not available"):
                agent = RAGAgent(config, mock_env, mock_llm, mock_logger)

            # Restore original __init__ method
            RAGAgent.__init__ = original_init

        finally:
            # Restore original __init__ method if it was replaced
            if "original_init" in locals():
                RAGAgent.__init__ = original_init
            os.unlink(trajectory_file)

    @patch("debug_gym.agents.rag_agent.RetrievalServiceClient")
    def test_retrieve_relevant_examples_method(self, mock_client_class):
        """Test retrieving relevant examples method."""
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        mock_client_instance.retrieve.return_value = [
            '{"tool_calls": {"name": "pdb", "arguments": {"command": "l"}}, "content": "Let me list the code"}',
            '{"tool_calls": {"name": "view", "arguments": {"path": "test.py"}}, "content": "Viewing file"}',
        ]

        # Create agent without full initialization
        agent = RAGAgent.__new__(RAGAgent)
        agent.retrieval_client = mock_client_instance
        agent.index_key = "test_index"
        agent.rag_num_retrievals = 2

        results = agent._retrieve_relevant_examples("test query")

        assert len(results) == 2
        assert "pdb" in results[0]
        assert "view" in results[1]
        mock_client_instance.retrieve.assert_called_once_with(
            index_key="test_index",
            query_text="test query",
            num_retrievals=2,
        )

    @patch("debug_gym.agents.rag_agent.RetrievalServiceClient")
    def test_build_question_prompt_basic(self, mock_client_class):
        """Test building question prompt with retrieved examples."""
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        mock_client_instance.retrieve.return_value = [
            '{"tool_calls": {"name": "pdb", "arguments": {"command": "l"}}, "content": "List code"}',
            '{"tool_calls": {"name": "view", "arguments": {"path": "test.py"}}}',
        ]

        # Create agent without full initialization
        agent = RAGAgent.__new__(RAGAgent)
        agent.retrieval_client = mock_client_instance
        agent.index_key = "test_index"
        agent.rag_num_retrievals = 2
        agent.logger = MagicMock()
        agent.rag_indexing_method = ["tool_call", 1]
        agent.delimiter = " <STEP_DELIMITER> "

        # Mock history
        mock_history_manager = MagicMock()
        mock_env_info = MagicMock()
        mock_env_info.action.name = "test_tool"
        mock_env_info.action.arguments = {"arg": "value"}
        mock_history_manager.get.return_value = ([mock_env_info], None)
        agent.history = mock_history_manager

        messages = agent.build_question_prompt()

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "debug_gym_ignore" in messages[0]
        assert messages[0]["debug_gym_ignore"] is True
        assert "retrieved some relevant examples" in messages[0]["content"]
        assert "Example 1" in messages[0]["content"]
        assert "Example 2" in messages[0]["content"]
