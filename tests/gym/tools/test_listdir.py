from unittest.mock import MagicMock

import pytest

from debug_gym.gym.tools.listdir import ListdirTool


def test_parse_args():
    _tool = ListdirTool()
    _tool.environment = MagicMock()

    # Test with no arguments
    listdir_path, depth = _tool.parse_args("")
    assert listdir_path == "."
    assert depth is None

    # Test with one argument
    listdir_path, depth = _tool.parse_args("code_dump")
    assert listdir_path == "code_dump"
    assert depth is None

    # Test with two arguments
    listdir_path, depth = _tool.parse_args("code_dump 2")
    assert listdir_path == "code_dump"
    assert depth == 2

    # Test with invalid depth
    with pytest.raises(ValueError):
        _tool.parse_args("code_dump 0")
    with pytest.raises(ValueError):
        _tool.parse_args("code_dump -1")

    # Test with too many arguments
    with pytest.raises(ValueError):
        _tool.parse_args("code_dump 2 extra_arg")
