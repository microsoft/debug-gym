import pytest

from froggy.envs.env import RepoEnv
from froggy.tools.tool import EnvironmentTool


class FakeTool(EnvironmentTool):
    pass


def test_register_valid_environment():
    tool = EnvironmentTool()
    env = RepoEnv()
    tool.register(env)
    assert tool.environment == env


def test_register_invalid_environment():
    tool = EnvironmentTool()
    with pytest.raises(ValueError):
        tool.register(object())


def test_is_triggered():
    tool = EnvironmentTool()
    tool.action = "pdb"
    assert tool.is_triggered("pdb breakpoints")
    assert not tool.is_triggered("otherstuff")


def test_use_not_implemented():
    tool = FakeTool()
    with pytest.raises(NotImplementedError):
        tool.use("something", None)
