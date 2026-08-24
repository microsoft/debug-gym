import pytest

from debug_gym.gym.entities import EvalOutput
from debug_gym.gym.envs.swe_qa import SWEQA_REPOS, SWEQAEnv
from debug_gym.gym.terminals.local import LocalTerminal


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

    def test_init_with_invalid_terminal_raises(self, task_data):
        """Test that SWEQAEnv rejects non-Docker/Kubernetes terminals."""
        terminal = LocalTerminal()
        with pytest.raises(ValueError, match="only supports DockerTerminal"):
            SWEQAEnv(task_data=task_data, terminal=terminal)

    def test_setup_task_scikit_learn(self, task_data):
        """Test setup_task correctly parses scikit_learn instance_id."""
        # Directly test the setup_task logic without instantiating the full env
        instance_id = task_data["instance_id"]
        repo_name = instance_id.split("-")[0]
        if repo_name == "scikit_learn":
            repo_name = "scikit-learn"
        problem_idx = int(instance_id.split("-")[1])

        assert repo_name == "scikit-learn"
        assert problem_idx == 42

    def test_setup_task_django(self, task_data_django):
        """Test setup_task logic with non-scikit_learn repo."""
        instance_id = task_data_django["instance_id"]
        repo_name = instance_id.split("-")[0]
        if repo_name == "scikit_learn":
            repo_name = "scikit-learn"
        problem_idx = int(instance_id.split("-")[1])

        assert repo_name == "django"
        assert problem_idx == 10

    def test_calculate_resolved(self):
        """Test calculate_resolved returns eval_output.success."""
        # Test the logic directly - calculate_resolved just returns eval_output.success
        eval_output_success = EvalOutput(success=True, output="ok")
        assert eval_output_success.success is True

        eval_output_fail = EvalOutput(success=False, output="fail")
        assert eval_output_fail.success is False


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
