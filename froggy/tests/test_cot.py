import unittest
from unittest.mock import MagicMock, mock_open, patch

from froggy.agents.cot import AgentCoT, AgentCoT_NoPDB
from froggy.agents.utils import (
    HistoryTracker,
    build_history_prompt,
    trim_prompt_messages,
)


class TestAgentCoT(unittest.TestCase):

    @patch("tiktoken.encoding_for_model")
    @patch("os.path.exists", return_value=True)
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"test-model": {"model": "test-model", "max_tokens": 100, "tokenizer": "gpt-4o", "context_limit": 4, "api_key": "test-api-key", "endpoint": "https://test-endpoint", "api_version": "v1", "tags": ["azure openai"]}}',
    )
    def setUp(self, mock_open, mock_exists, mock_encoding_for_model):
        mock_encoding = MagicMock()
        mock_encoding.encode = lambda x: x.split()
        mock_encoding_for_model.return_value = mock_encoding

        self.config_dict = {
            "llm_name": "test-model",
            "max_steps": 10,
            "max_rewrite_steps": 5,
            "llm_temperature": [0.5, 0.7],
            "use_conversational_prompt": True,
            "reset_prompt_history_after_rewrite": False,
            "memory_size": 10,
            "output_path": "",
            "random_seed": 42,
        }
        self.env = MagicMock()
        self.llm = MagicMock()
        self.history = MagicMock()
        self.agent = AgentCoT(self.config_dict, self.env)
        self.agent.llm = self.llm
        self.agent.history = self.history

    def test_build_cot_prompt(self):
        messages = self.agent.build_cot_prompt()
        self.assertEqual(len(messages), 1)
        self.assertIn(
            "Let's think step by step using the following questions:",
            messages[0]["content"],
        )

    def test_build_prompt_step_1(self):
        info = {
            "instructions": "Test instructions",
            "dir_tree": "Test dir tree",
            "editable_files": "Test editable files",
            "current_code_with_line_number": "Test code",
            "current_breakpoints": "Test breakpoints",
            "last_run_obs": "Test last run obs",
        }
        messages = self.agent.build_prompt_step_1(info)
        self.assertGreater(len(messages), 0)

    def test_fill_in_cot_response(self):
        response = "Test response"
        messages = self.agent.fill_in_cot_response(response)
        self.assertEqual(len(messages), 1)
        self.assertIn("assistant", messages[0]["role"])

    def test_build_question_prompt(self):
        messages = self.agent.build_question_prompt()
        self.assertEqual(len(messages), 1)
        self.assertIn("what is the best next command?", messages[0]["content"])

    def test_build_prompt_step_2(self):
        info = {
            "instructions": "Test instructions",
            "dir_tree": "Test dir tree",
            "editable_files": "Test editable files",
            "current_code_with_line_number": "Test code",
            "current_breakpoints": "Test breakpoints",
            "last_run_obs": "Test last run obs",
        }
        response = "Test response"
        messages = self.agent.build_prompt_step_2(info, response)
        self.assertGreater(len(messages), 0)

    def test_run(self):
        self.env.reset.return_value = (
            None,
            {
                "done": False,
                "score": 0,
                "max_score": 10,
                "instructions": "Test instructions",
                "dir_tree": "Test dir tree",
                "editable_files": "Test editable files",
                "current_code_with_line_number": "Test code",
                "current_breakpoints": "Test breakpoints",
                "last_run_obs": "Test last run obs",
            },
        )
        self.env.step.return_value = (
            None,
            None,
            True,
            {
                "done": True,
                "score": 10,
                "max_score": 10,
                "instructions": "Test instructions",
                "dir_tree": "Test dir tree",
                "editable_files": "Test editable files",
                "current_code_with_line_number": "Test code",
                "current_breakpoints": "Test breakpoints",
                "last_run_obs": "Test last run obs",
            },
        )
        self.llm.return_value = ("Expected answer", "Expected token usage")
        result = self.agent.run(task_name="test_task", debug=False)
        self.assertTrue(result)


class TestAgentCoT_NoPDB(unittest.TestCase):

    @patch("tiktoken.encoding_for_model")
    @patch("os.path.exists", return_value=True)
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"test-model": {"model": "test-model", "max_tokens": 100, "tokenizer": "gpt-4o", "context_limit": 4, "api_key": "test-api-key", "endpoint": "https://test-endpoint", "api_version": "v1", "tags": ["azure openai"]}}',
    )
    def setUp(self, mock_open, mock_exists, mock_encoding_for_model):
        mock_encoding = MagicMock()
        mock_encoding.encode = lambda x: x.split()
        mock_encoding_for_model.return_value = mock_encoding
        self.config_dict = {
            "llm_name": "test-model",
            "max_steps": 10,
            "max_rewrite_steps": 5,
            "llm_temperature": [0.5, 0.7],
            "use_conversational_prompt": True,
            "memory_size": 10,
            "output_path": "",
            "random_seed": 42,
        }
        self.env = MagicMock()
        self.llm = MagicMock()
        self.history = MagicMock()
        self.agent = AgentCoT_NoPDB(self.config_dict, self.env)
        self.agent.llm = self.llm
        self.agent.history = self.history

    def test_build_system_prompt(self):
        info = {
            "instructions": "Test instructions",
            "dir_tree": "Test dir tree",
            "editable_files": "Test editable files",
            "current_code_with_line_number": "Test code",
            "current_breakpoints": "Test breakpoints",
            "last_run_obs": "Test last run obs",
        }
        messages = self.agent.build_system_prompt(info)
        self.assertEqual(len(messages), 1)
        self.assertIn("Overall task", messages[0]["content"])

    def test_build_cot_prompt(self):
        messages = self.agent.build_cot_prompt()
        self.assertEqual(len(messages), 1)
        self.assertIn(
            "Let's think step by step using the following questions:",
            messages[0]["content"],
        )


if __name__ == "__main__":
    unittest.main()
