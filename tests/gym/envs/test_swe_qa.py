from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from debug_gym.gym.entities import EvalOutput
from debug_gym.gym.envs.swe_qa import SWEQA_REPOS, SWEQAEnv


class TestSWEQAEnv:
    @pytest.fixture
    def task_data(self):
        return {
            "instance_id": "scikit_learn-42",
            "question": "What is the return type of fit()?",
            "answer": "self",
        }

    @pytest.fixture
    def task_data_django(self):
        return {
            "instance_id": "django-10",
            "question": "How does QuerySet work?",
            "answer": "It is lazy.",
        }

    def test_init_with_unsupported_terminal_raises(self, task_data):
        with pytest.raises(ValueError, match="only supports DockerTerminal"):
            SWEQAEnv(task_data=task_data, terminal=object())

    def test_setup_task_scikit_learn(self, task_data):
        env = SWEQAEnv.__new__(SWEQAEnv)
        env.task_data = task_data

        env.setup_task()

        assert env.repo_name == "scikit-learn"
        assert env.problem_idx == 42
        assert env.base_image == "python:3.12"
        assert env.answer == "self"
        assert env.instructions == task_data["question"]
        assert env.task_name == task_data["instance_id"]

    def test_setup_task_django(self, task_data_django):
        env = SWEQAEnv.__new__(SWEQAEnv)
        env.task_data = task_data_django

        env.setup_task()

        assert env.repo_name == "django"
        assert env.problem_idx == 10

    def test_eval_and_calculate_resolved(self, task_data):
        env = SWEQAEnv.__new__(SWEQAEnv)
        env.task_data = task_data

        eval_output = env.eval()
        assert eval_output == EvalOutput(
            success=True, output="Agent has submitted an answer."
        )
        assert env.last_eval == eval_output

        eval_output_success = EvalOutput(success=True, output="ok")
        eval_output_fail = EvalOutput(success=False, output="fail")
        assert env.calculate_resolved(eval_output_success) is True
        assert env.calculate_resolved(eval_output_fail) is False

    def test_setup_workspace(self, task_data, tmp_path):
        env = SWEQAEnv.__new__(SWEQAEnv)
        env.task_data = task_data
        env.setup_task()
        env.CACHE = tmp_path
        env.terminal = Mock()
        env.workspace = Mock(working_dir="/testbed")
        env.logger = Mock()

        env.setup_workspace()

        assert env.terminal.task_name == task_data["instance_id"]
        assert env.terminal.base_image == "python:3.12"
        env.workspace.reset.assert_called_once_with()
        env.workspace.copy_content.assert_called_once_with(
            src=tmp_path / "scikit-learn", target="/testbed"
        )

    def test_setup_terminal(self, task_data):
        env = SWEQAEnv.__new__(SWEQAEnv)
        env.task_data = task_data
        env.terminal = Mock(env_vars={}, session_commands=[])
        env.workspace = SimpleNamespace(working_dir="/testbed")
        env.logger = Mock()

        env.setup_terminal()

        assert env.terminal.env_vars["PATH"] == "/root/.local/bin:/bin"
        assert env.terminal.run.call_args_list == [
            call("chown -R root:root /testbed"),
            call("curl -LsSf https://astral.sh/uv/install.sh | sh"),
            call("uv venv && source .venv/bin/activate"),
            call("uv pip install pip"),
        ]
        assert env.terminal.session_commands == ["source /testbed/.venv/bin/activate"]


class TestSWEQARepos:
    def test_sweqa_repos_format(self):
        """Test that SWEQA_REPOS has the expected structure."""
        assert len(SWEQA_REPOS) == 15
        for repo in SWEQA_REPOS:
            assert "url" in repo
            assert "commit" in repo
            assert repo["url"].startswith("https://github.com/")
            assert len(repo["commit"]) == 7  # Short commit hash

    def test_sweqa_repos_contains_expected_repos(self):
        """Test that SWEQA_REPOS contains expected repositories."""
        repo_names = [r["url"].split("/")[-1] for r in SWEQA_REPOS]
        assert "django" in repo_names
        assert "pytest" in repo_names
        assert "scikit-learn" in repo_names
        assert "sympy" in repo_names
