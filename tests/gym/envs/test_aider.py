import pytest

from debug_gym.gym.envs import AiderBenchmarkEnv
from debug_gym.gym.tools.tool import ToolCall
from debug_gym.gym.tools.toolbox import Toolbox


@pytest.fixture(scope="session")
def setup_aider_repo(tmp_path_factory):
    """Set up a minimal Aider repository for testing,
    avoiding cloning Aider repo for each test."""
    aider_path = tmp_path_factory.mktemp("aider_repo")
    # aider_path.mkdir(parents=True, exist_ok=True)
    repo_path = aider_path / "exercises" / "practice" / "clock"
    repo_path.mkdir(parents=True, exist_ok=True)
    (repo_path / "clock.py").write_text(
        "def current_time():  # returns a message with the current time\n"
        "    return 'It is 10:00 a.m.'"
    )
    (repo_path / "clock_test.py").write_text(
        "import clock\n"
        "\n"
        "def test_clock():\n"
        "    assert clock.current_time() == 'It is 12:00 a.m.'"
    )
    (repo_path / ".docs").mkdir(parents=True, exist_ok=True)
    (repo_path / ".docs" / "instructions.md").write_text("What time is it?")
    # Patch the REPO_PATH in AiderBenchmarkEnv
    AiderBenchmarkEnv.REPO_PATH = aider_path
    return aider_path


@pytest.fixture
def env(setup_aider_repo):
    env = AiderBenchmarkEnv()
    env.reset(options={"task_name": "clock"})
    return env


def test_resolve_path(env):
    path = env.resolve_path(env.working_dir, raises=True)
    assert path == env.working_dir
    assert env.resolve_path("clock.py", raises=True) == env.working_dir / "clock.py"
    with pytest.raises(FileNotFoundError):
        env.resolve_path("nested/file.py", raises=True)


def test_ignored_files(env):
    assert env.has_file("clock_test.py")
    assert env.has_file("clock.py")
    assert not env.has_file(".gitignore")
    assert not env.has_file(".debugignore")
    assert not env.has_file(".debugreadonly")
    assert not env.has_file("nested/file.py")


def test_is_editable_files(env):
    assert env.is_editable("clock.py")
    assert not env.is_editable("clock_test.py")
    with pytest.raises(FileNotFoundError):
        assert not env.is_editable("nested/file.py")
    with pytest.raises(FileNotFoundError):
        assert not env.is_editable(".debugignore")


def test_steps(env):
    eval_tool = Toolbox.get_tool("eval")
    env.add_tool(eval_tool)
    eval_call = ToolCall(id="eval_id", name="eval", arguments={})
    infos = env.step(eval_call)
    assert infos.step_observation.source == "eval"
    assert "clock_test.py F" in infos.eval_observation.observation
    assert "1 failed" in infos.eval_observation.observation
    assert infos.score == 0
    rewrite_tool = Toolbox.get_tool("rewrite")
    env.add_tool(rewrite_tool)
    infos = env.step(
        ToolCall(
            id="rewrite_id",
            name="rewrite",
            arguments={
                "path": "clock.py",
                "start": 2,
                "new_code": "    return 'It is 12:00 a.m.'",
            },
        )
    )
    assert infos.step_observation.source == "rewrite"
    assert infos.step_observation.observation.startswith(
        "The file `clock.py` has been updated successfully."
    )
    assert env.auto_eval_on_rewrite is True
    assert infos.score == 1

    infos = env.step(eval_call)
    assert infos.step_observation.source == "eval"
    assert "clock_test.py ." in infos.eval_observation.observation
    assert "1 passed" in infos.eval_observation.observation
    assert infos.score == 1


def test_instructions(env):
    assert env.instructions == "What time is it?"
