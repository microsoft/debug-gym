from unittest.mock import MagicMock, mock_open, patch

import pytest

from debug_gym.gym.entities import Observation
from debug_gym.gym.envs import AiderBenchmarkEnv
from debug_gym.gym.envs.env import EnvInfo


@pytest.fixture
def env_info():
    return EnvInfo(
        step_observation=Observation("tool", "obs"),
        all_observations=[],
        eval_observation=Observation("env", "eval_observation"),
        dir_tree="dir_tree",
        current_breakpoints="current_breakpoints",
        action="action",
        instructions={},
        score=5,
        max_score=10,
        done=False,
        rewrite_counter=0,
        tools=[],
    )


@pytest.fixture
@patch("subprocess.run")
@patch("os.path.exists", return_value=False)
@patch("pathlib.Path.iterdir")
@patch("pathlib.Path.read_text", return_value="Test instructions")
@patch("os.listdir", return_value=[".gitignore"])
@patch("builtins.open", new_callable=mock_open)
def aider_env(
    mock_open, mock_listdir, mock_read_text, mock_iterdir, mock_exists, mock_run
):
    # Mock the directories
    mock_dir = MagicMock()
    mock_dir.is_dir.return_value = True
    mock_dir.name = "test_task"
    mock_iterdir.return_value = [mock_dir]

    # Initialize the AiderBenchmarkEnv
    env = AiderBenchmarkEnv()
    return env


def test_instructions(aider_env):
    aider_env.current_sample = {"instructions": "Test instructions"}
    expected_instructions = {"Problem description": "Test instructions"}
    assert aider_env.instructions == expected_instructions


@patch("debug_gym.gym.envs.RepoEnv.reset")
@patch("debug_gym.gym.envs.AiderBenchmarkEnv.setup_workspace")
def test_reset(
    mock_setup_workspace,
    repo_env,
    aider_env,
    env_info,
):
    test_task = {
        "test_task": {
            "base_directory": "test_directory",
            "instructions": "Test instructions",
            "filename": "test_task.py",
        }
    }
    repo_env.return_value = env_info
    aider_env.dataset = test_task
    options = {"task_name": "test_task"}
    infos = aider_env.reset(options=options)
    assert aider_env.current_sample == test_task["test_task"]
    assert infos.step_observation == Observation("tool", "obs")
    assert infos.max_score == 10
    assert infos.score == 5


# TODO: Add proper test, mocking repoenv.step doesn't test anything
@patch("debug_gym.gym.envs.RepoEnv.step")
def test_step(mock_step, aider_env, env_info):
    mock_step.return_value = env_info
    infos = aider_env.step("action")
    assert infos.step_observation == Observation("tool", "obs")
    assert infos.score == 5


@patch("subprocess.run")
@patch("os.path.exists", return_value=False)
@patch("os.listdir", return_value=[".gitignore"])
def test_load_dataset(mock_listdir, mock_exists, mock_run, aider_env):
    aider_env.load_dataset()
    assert mock_run.called


@pytest.fixture
@patch("pathlib.Path.home")
def env(mock_home, tmp_path):
    mock_home.return_value = tmp_path
    aider_path = AiderBenchmarkEnv.REPO_PATH
    aider_path.mkdir(exist_ok=True)
    repo_path = aider_path / "hangman"
    repo_path.mkdir(exist_ok=True)
    (repo_path / "hangman.py").write_text("return 'Hello, Hangman!'")
    (repo_path / "hangman_test.py").write_text(
        "import hangman\n"
        "\n"
        "def test_hangman():\n"
        "    assert hangman() == 'Hello, Hangman!'"
    )
    env = AiderBenchmarkEnv()
    env.reset(options={"task_name": "hangman"})
    return env


def test_resolve_path(env):
    path = env.resolve_path(env.working_dir, raises=True)
    assert path == env.working_dir
    assert env.resolve_path("hangman.py", raises=True) == env.working_dir / "hangman.py"
    with pytest.raises(FileNotFoundError):
        env.resolve_path("nested/file.py", raises=True)


def test_ignored_files(env):
    assert env.has_file("hangman_test.py")
    assert env.has_file("hangman.py")
    assert not env.has_file(".gitignore")
    assert not env.has_file(".debugignore")
    assert not env.has_file(".debugreadonly")
    assert not env.has_file("nested/file.py")


def test_is_editable_files(env):
    assert env.is_editable("hangman.py")
    assert not env.is_editable("hangman_test.py")
    with pytest.raises(FileNotFoundError):
        assert not env.is_editable("nested/file.py")
    with pytest.raises(FileNotFoundError):
        assert not env.is_editable(".debugignore")
