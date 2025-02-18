from unittest.mock import MagicMock

import pytest

from froggy.tools.rewrite import RewriteTool


@pytest.fixture
def mock_environment():
    env = MagicMock()
    env.current_file_content = "def greet():\n    print('Hello, world!')\n".splitlines()
    env.current_file = "test.py"
    env.all_files = ["test.py"]
    env.editable_files = ["test.py"]
    return env


def test_substitution_patcher(mock_environment):
    patcher = RewriteTool()
    patcher.environment = mock_environment
    result = patcher.use(start=2, new_code="    print(f'Hello, {name}!')")

    assert result == "Rewriting done."
    assert patcher.rewrite_success
    mock_environment.overwrite_file.assert_called_once_with(
        filepath="test.py", content=""
    )


def test_substitution_patcher_with_file_path(mock_environment):
    patcher = RewriteTool()
    patcher.environment = mock_environment
    result = patcher.use(
        path="test.py", start=2, new_code="    print(f'Hello, {name}!')"
    )

    assert result == "Rewriting done."
    assert patcher.rewrite_success
    mock_environment.overwrite_file.assert_called_once_with(
        filepath="test.py", content=""
    )


def test_substitution_patcher_invalid_file(mock_environment):
    patcher = RewriteTool()
    patcher.environment = mock_environment
    mock_environment.all_files = ["another_file.py"]

    result = patcher.use(
        path="test.py", start=2, new_code="    print(f'Hello, {name}!')"
    )

    assert (
        result
        == "Error while rewriting the file: File test.py does not exist or is not in the current repository.\nRewrite failed."
    )
    assert not patcher.rewrite_success


def test_substitution_patcher_invalid_line_number(mock_environment):
    patcher = RewriteTool()
    patcher.environment = mock_environment

    result = patcher.use(
        path="test.py", start=0, new_code="    print(f'Hello, {name}!')"
    )

    assert result == "Invalid line number, line numbers are 1-based.\nRewrite failed."
    assert not patcher.rewrite_success


def test_substitution_patcher_invalid_line_number_2(mock_environment):
    patcher = RewriteTool()
    patcher.environment = mock_environment

    result = patcher.use(
        path="test.py", start=12, end=4, new_code="    print(f'Hello, {name}!')"
    )

    assert (
        result
        == "Invalid line number range, start should be less than or equal to end.\nRewrite failed."
    )
    assert not patcher.rewrite_success
