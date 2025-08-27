import pytest

from debug_gym.gym.envs.env import RepoEnv
from debug_gym.gym.tools.listdir import ListdirTool


@pytest.fixture
def setup_listdir_repo_env(setup_test_repo):
    def _setup_listdir_repo_env(base_dir):
        test_repo = setup_test_repo(base_dir)
        env = RepoEnv(path=str(test_repo))
        listdir_tool = ListdirTool()
        listdir_tool.register(env)
        env.reset()
        return listdir_tool, env

    return _setup_listdir_repo_env


def test_listdir_default(tmp_path, setup_listdir_repo_env):
    listdir, env = setup_listdir_repo_env(tmp_path)
    obs = listdir.use(env)
    assert obs.source == "listdir"
    assert obs.observation == (
        f"{env.working_dir}/\n"
        # "|-- .pytest_cache/\n"
        "|-- __pycache__/\n"
        "|-- file1.py\n"
        "|-- file2.py\n"
        "|-- test_fail.py\n"
        "|-- test_pass.py"
    )


def test_listdir_invalid_depth(tmp_path, setup_listdir_repo_env):
    listdir, env = setup_listdir_repo_env(tmp_path)
    obs = listdir.use(env, depth=0)
    assert obs.source == "listdir"
    assert obs.observation == "Depth must be 1 or greater, got `depth=0`"

    obs = listdir.use(env, depth=-1)
    assert obs.source == "listdir"
    assert obs.observation == "Depth must be 1 or greater, got `depth=-1`"


def test_listdir_with_invalid_path(tmp_path, setup_listdir_repo_env):
    listdir, env = setup_listdir_repo_env(tmp_path)
    obs = listdir.use(env, path="non_existent_path")
    assert obs.source == "listdir"
    assert obs.observation == (
        f"Error listing directory 'non_existent_path': `non_existent_path` "
        f"does not exist or is not in the working directory `{env.working_dir}`."
    )
