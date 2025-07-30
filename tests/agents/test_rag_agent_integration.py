import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from debug_gym.agents.rag_agent import RAGAgent


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
            "rag_use_retrieval_service": True,
            "rag_retrieval_service_host": "localhost",
            "rag_retrieval_service_port": 8766,
            "rag_retrieval_service_timeout": 120,
            "rag_cache_dir": ".test_cache",
            "rag_use_cache": True,
            "rag_indexing_batch_size": 16,
        }

    @patch("debug_gym.agents.rag_agent.RetrievalServiceClient")
    @patch("debug_gym.agents.debug_agent.DebugAgent.__init__")
    def test_rag_agent_initialization_with_service(
        self, mock_debug_agent_init, mock_client_class
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

            # Mock the base class initialization to set essential attributes
            def mock_init(
                instance_config, instance_env, instance_llm=None, instance_logger=None
            ):
                # Find the instance that's being initialized and set attributes
                # This will work because RAGAgent.__init__ calls super().__init__
                pass

            mock_debug_agent_init.side_effect = mock_init

            # Mock the retrieval service client
            mock_client_instance = MagicMock()
            mock_client_class.return_value = mock_client_instance
            mock_client_instance.is_service_available.return_value = True
            mock_client_instance.build_index.return_value = True

            # Pre-create instance and set attributes manually to avoid the initialization issue
            agent = RAGAgent.__new__(RAGAgent)
            agent.config = config
            agent.env = mock_env
            agent.llm = mock_llm
            agent.logger = mock_logger

            # Now call __init__ to test the rest of the initialization
            RAGAgent.__init__(agent, config, mock_env, mock_llm, mock_logger)

            # Verify initialization
            assert agent.config == config
            assert hasattr(agent, "retrieval_client")
            assert agent.use_retrieval_service is True

        finally:
            os.unlink(trajectory_file)

    @patch("debug_gym.agents.rag_agent.RetrievalServiceClient")
    @patch("debug_gym.agents.debug_agent.DebugAgent.__init__")
    def test_rag_agent_service_unavailable(
        self, mock_debug_agent_init, mock_client_class
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

            # Mock the base class initialization
            def mock_init(
                instance_config, instance_env, instance_llm=None, instance_logger=None
            ):
                pass

            mock_debug_agent_init.side_effect = mock_init

            # Mock the retrieval service client as unavailable
            mock_client_instance = MagicMock()
            mock_client_class.return_value = mock_client_instance
            mock_client_instance.is_service_available.return_value = False

            # Pre-create instance and set attributes manually
            agent = RAGAgent.__new__(RAGAgent)
            agent.config = config
            agent.env = mock_env
            agent.llm = mock_llm
            agent.logger = mock_logger

            # Test that RuntimeError is raised when service is unavailable
            with pytest.raises(RuntimeError, match="Retrieval service not available"):
                RAGAgent.__init__(agent, config, mock_env, mock_llm, mock_logger)

        finally:
            os.unlink(trajectory_file)

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

    @patch("debug_gym.agents.debug_agent.DebugAgent.__init__")
    def test_local_retrieval_not_supported(self, mock_debug_agent_init):
        """Test that local retrieval raises NotImplementedError."""
        trajectory_data = self.create_sample_trajectory_data()
        trajectory_file = self.create_sample_trajectory_file(trajectory_data)
        config = self.create_mock_config(trajectory_file)
        config["rag_use_retrieval_service"] = False  # Disable retrieval service

        try:
            # Create mocks
            mock_env = MagicMock()
            mock_llm = MagicMock()
            mock_logger = MagicMock()

            # Mock the base class initialization
            def mock_init(
                instance_config, instance_env, instance_llm=None, instance_logger=None
            ):
                pass

            mock_debug_agent_init.side_effect = mock_init

            # Pre-create instance and set attributes manually
            agent = RAGAgent.__new__(RAGAgent)
            agent.config = config
            agent.env = mock_env
            agent.llm = mock_llm
            agent.logger = mock_logger

            with pytest.raises(
                NotImplementedError, match="Local retrieval is no longer supported"
            ):
                RAGAgent.__init__(agent, config, mock_env, mock_llm, mock_logger)

        finally:
            os.unlink(trajectory_file)

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
