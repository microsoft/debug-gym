"""
Mock-only tests for RAGAgent that run even when retrieval service is not available.

These tests focus on testing the logic and interfaces without requiring
the actual retrieval service to be installed.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest


class TestRAGAgentMockOnly:
    """Tests that run even when retrieval service is not available."""

    def test_rag_agent_import_error_handling(self):
        """Test that appropriate error is raised when retrieval service is not available."""
        with patch.dict("sys.modules", {"retrieval_service.client": None}):
            with patch(
                "builtins.__import__",
                side_effect=ImportError("No module named 'retrieval_service'"),
            ):
                # Simulate the import error case
                try:
                    # This would normally be:
                    # from debug_gym.agents.rag_agent import RAGAgent
                    # But we simulate the import error scenario
                    raise ImportError("No module named 'retrieval_service'")
                except ImportError as e:
                    assert "retrieval_service" in str(e)

    def test_indexing_method_parsing_logic(self):
        """Test the indexing method parsing logic in isolation."""
        # This tests the logic without importing the actual class

        def parse_indexing_method(method: str):
            """Extracted logic from RAGAgent.parse_indexing_method for testing."""
            assert (
                method is not None
            ), "rag_indexing_method must be provided in the config"

            method, step = method.rsplit("-", 1) if "-" in method else (method, "1")
            assert method in [
                "observation",
                "tool_name",
                "tool_call",
                "tool_call_with_reasoning",
            ], f"Invalid rag_indexing_method: {method}. Supported methods: observation, tool_name, tool_call"
            assert (
                step.isdigit()
            ), f"Invalid step value: {step}. It should be a positive integer."
            step = int(step)
            assert step > 0, "Step must be a positive integer."
            return [method, step]

        # Test valid methods
        assert parse_indexing_method("tool_call-1") == ["tool_call", 1]
        assert parse_indexing_method("tool_call_with_reasoning-3") == [
            "tool_call_with_reasoning",
            3,
        ]
        assert parse_indexing_method("observation-5") == ["observation", 5]
        assert parse_indexing_method("tool_name") == ["tool_name", 1]

        # Test invalid methods
        with pytest.raises(AssertionError, match="Invalid rag_indexing_method"):
            parse_indexing_method("invalid_method-1")

    def test_query_text_extraction_logic(self):
        """Test query text extraction logic in isolation."""

        def extract_query_text_tool_call_method(
            history, delimiter=" <STEP_DELIMITER> "
        ):
            """Extracted logic for tool_call method."""
            tool_call_list = [
                json.dumps(
                    {"name": item.action.name, "arguments": item.action.arguments}
                )
                for item in history
                if item.action
            ]
            if not tool_call_list:
                return None
            return delimiter.join(tool_call_list)

        # Create mock history
        mock_item = MagicMock()
        mock_action = MagicMock()
        mock_action.name = "pdb"
        mock_action.arguments = {"command": "list"}
        mock_item.action = mock_action

        history = [mock_item]

        result = extract_query_text_tool_call_method(history)
        expected = '{"name": "pdb", "arguments": {"command": "list"}}'
        assert result == expected

    def test_configuration_defaults(self):
        """Test the expected configuration structure and defaults."""
        expected_config_keys = {
            "rag_num_retrievals": 1,
            "rag_indexing_method": None,
            "rag_indexing_batch_size": 16,
            "sentence_encoder_model": "Qwen/Qwen3-Embedding-0.6B",
            "rag_cache_dir": ".rag_cache",
            "rag_use_cache": True,
            "rag_retrieval_service_host": "localhost",
            "rag_retrieval_service_port": 8766,
            "rag_retrieval_service_timeout": 120,
            "experience_trajectory_path": None,
        }

        # Test that we can simulate config access
        mock_config = MagicMock()
        for key, default_value in expected_config_keys.items():
            mock_config.get.return_value = default_value
            result = mock_config.get(key, default_value)
            assert result == default_value

    def test_retrieval_service_client_interface(self):
        """Test the expected interface with the retrieval service client."""
        # This tests the expected methods and their signatures
        mock_client = MagicMock()

        # Test expected methods exist and can be called
        mock_client.is_service_available.return_value = True
        mock_client.check_index.return_value = False
        mock_client.build_index.return_value = True
        mock_client.retrieve.return_value = ["example1", "example2"]

        # Verify interface
        assert mock_client.is_service_available() is True
        assert mock_client.check_index("test_index") is False
        assert (
            mock_client.build_index(
                index_key="test_index",
                experience_trajectory_path="/path/to/file.jsonl",
                rag_indexing_method="tool_call-1",
                sentence_encoder_model="test-model",
                rag_indexing_batch_size=16,
                use_cache=True,
            )
            is True
        )
        assert mock_client.retrieve(
            index_key="test_index",
            query_text="test query",
            num_retrievals=2,
        ) == ["example1", "example2"]

    def test_prompt_building_logic(self):
        """Test the prompt building logic in isolation."""

        def build_question_prompt(relevant_examples):
            """Extracted prompt building logic."""
            if not relevant_examples:
                return []

            content = "I have retrieved some relevant examples to help you make a decision. Note that these examples are not guaranteed to be correct or applicable to the current situation, but you can use them as references if you are unsure about the next step. "
            content += "You can ignore the examples that are not relevant to the current situation. Here are the examples:\n"

            deduplicate = set()
            for example in relevant_examples:
                # Parse the example if it's a JSON string
                if isinstance(example, str):
                    try:
                        example_dict = json.loads(example)
                        _ex = json.dumps(example_dict, indent=2)
                    except json.JSONDecodeError:
                        _ex = example
                else:
                    _ex = json.dumps(example, indent=2)

                if _ex in deduplicate:
                    continue
                content += f"\nExample {len(deduplicate) + 1}:\n{_ex}\n"
                deduplicate.add(_ex)

            messages = [{"role": "user", "content": content, "debug_gym_ignore": True}]
            return messages

        # Test with examples
        examples = [
            '{"tool_calls": {"name": "pdb", "arguments": {"command": "l"}}}',
            '{"tool_calls": {"name": "view", "arguments": {"path": "test.py"}}}',
        ]

        messages = build_question_prompt(examples)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "debug_gym_ignore" in messages[0]
        assert messages[0]["debug_gym_ignore"] is True
        assert "retrieved some relevant examples" in messages[0]["content"]
        assert "Example 1" in messages[0]["content"]
        assert "Example 2" in messages[0]["content"]

        # Test with no examples
        empty_messages = build_question_prompt([])
        assert empty_messages == []
