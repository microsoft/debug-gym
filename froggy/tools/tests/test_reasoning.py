import subprocess
import unittest
from unittest.mock import MagicMock, patch

import pytest

from froggy.envs.env import RepoEnv
from froggy.tools.reasoning import ReasoningTool
from froggy.utils import TimeoutException


class TestReasoningTool(unittest.TestCase):

    def setUp(self):
        self.env = MagicMock(spec=RepoEnv)
        self.environment = MagicMock()
        self.reasoning_tool = ReasoningTool()
        self.reasoning_tool.environment = self.environment

    def test_register(self):
        self.reasoning_tool.register(self.env)
        self.assertEqual(self.reasoning_tool.environment, self.env)

    def test_register_invalid_environment(self):
        with self.assertRaises(ValueError):
            self.reasoning_tool.register(MagicMock())

    def test_split_reasoning(self):
        action = "```reasoning\nreasoning text\n</reasoning>\nnext_action ```"
        self.assertEqual(
            self.reasoning_tool.split_reasoning(action),
            ("reasoning text", "next_action"),
        )

        # this will raise exception because failing the split
        action = "```reasoning\nreasoning text```"
        with self.assertRaises(ValueError):
            self.reasoning_tool.split_reasoning(action)

        # this will raise exception because failing the rsplit
        action = "reasoning\nreasoning text\n</reasoning>\nnext_action"
        with self.assertRaises(IndexError):
            self.reasoning_tool.split_reasoning(action)

    def test_use_without_chaining(self):
        action = "```reasoning\nreasoning text```"
        self.assertEqual(
            self.reasoning_tool.use_without_chaining(action),
            "Reasoning:\nreasoning text",
        )

        action = "```reasoning\nreasoning text\n</reasoning>\nnext_action```"
        self.assertEqual(
            self.reasoning_tool.use_without_chaining(action),
            "Reasoning:\nreasoning text\n</reasoning>\nnext_action",
        )

        # missing ``` at the beginning
        action = "reasoning\nreasoning text```"
        self.assertEqual(
            self.reasoning_tool.use_without_chaining(action),
            "SyntaxError: invalid syntax.",
        )

    def test_use_with_chaining(self):
        self.reasoning_tool.split_reasoning = MagicMock(
            return_value=("reasoning", "next_action")
        )
        self.environment.step = MagicMock(
            return_value=("obs", 0.0, False, {"key": "value"})
        )
        self.assertEqual(
            self.reasoning_tool.use_with_chaining("action"),
            "Reasoning:\nreasoning\nExecuting action:\nnext_action\nNext observation:\nobs",
        )

        # invalid syntax
        self.reasoning_tool.split_reasoning = MagicMock(side_effect=ValueError)
        self.assertEqual(
            self.reasoning_tool.use_with_chaining("action"),
            "SyntaxError: invalid syntax.",
        )

        # invalid syntax
        self.reasoning_tool.split_reasoning = MagicMock(
            return_value=("reasoning", "next_action")
        )
        self.environment.step = MagicMock(side_effect=ValueError)
        self.assertEqual(
            self.reasoning_tool.use_with_chaining("action"),
            "Error while executing the action after reasoning.\nSyntaxError: invalid syntax.",
        )

        # invalid syntax: chaining reasoning actions
        self.reasoning_tool.split_reasoning = MagicMock(
            return_value=("reasoning", "```reasoning something else```")
        )
        self.environment.step = MagicMock(
            return_value=("obs", 0.0, False, {"key": "value"})
        )
        self.assertEqual(
            self.reasoning_tool.use_with_chaining("action"),
            "SyntaxError: invalid syntax. You cannot chain reasoning actions.",
        )

        # invalid syntax: fail to execute the next action
        self.reasoning_tool.split_reasoning = MagicMock(
            return_value=("reasoning", "next_action")
        )
        self.environment.step = MagicMock(
            return_value=("Invalid action: action.", 0.0, False, {"key": "value"})
        )
        self.assertEqual(
            self.reasoning_tool.use_with_chaining("action"),
            "Error while executing the action after reasoning.\nInvalid action: action.",
        )

        # invalid syntax: fail to execute the next action
        self.reasoning_tool.split_reasoning = MagicMock(
            return_value=("reasoning", "next_action")
        )
        self.environment.step = MagicMock(
            return_value=("Error while using tool: cot", 0.0, False, {"key": "value"})
        )
        self.assertEqual(
            self.reasoning_tool.use_with_chaining("action"),
            "Error while executing the action after reasoning.\nError while using tool: cot",
        )


if __name__ == "__main__":
    unittest.main()
