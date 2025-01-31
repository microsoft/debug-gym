from unittest.mock import MagicMock

import pytest

from froggy.tools.patchers import SubstitutionPatcher


@pytest.fixture
def mock_environment():
    env = MagicMock()
    env.current_file_content = "def greet():\n    print('Hello, world!')\n".splitlines()
    env.current_file = "test.py"
    env.all_files = ["test.py"]
    env.editable_files = ["test.py"]
    return env


def test_substitution_patcher(mock_environment):
    patcher = SubstitutionPatcher()
    patcher.environment = mock_environment

    patch = "```rewrite 2 <c>    print(f'Hello, {name}!')</c>```"
    result = patcher.use(patch)

    assert result == "Rewriting done."
    assert patcher.rewrite_success
    mock_environment.overwrite_file.assert_called_once_with(
        filepath="test.py", content=""
    )


def test_substitution_patcher_with_file_path(mock_environment):
    patcher = SubstitutionPatcher()
    patcher.environment = mock_environment

    patch = "```rewrite test.py 2 <c>    print(f'Hello, {name}!')</c>```"
    result = patcher.use(patch)

    assert result == "Rewriting done."
    assert patcher.rewrite_success
    mock_environment.overwrite_file.assert_called_once_with(
        filepath="test.py", content=""
    )


def test_substitution_patcher_invalid_content(mock_environment):
    patcher = SubstitutionPatcher()
    patcher.environment = mock_environment

    patch = "```rewrite invalid content```"
    result = patcher.use(patch)

    assert result == "SyntaxError: invalid syntax.\nRewrite failed."
    assert not patcher.rewrite_success


def test_substitution_patcher_invalid_file(mock_environment):
    patcher = SubstitutionPatcher()
    patcher.environment = mock_environment
    mock_environment.all_files = ["another_file.py"]

    patch = "```rewrite test.py 2 <c>    print(f'Hello, {name}!')</c>```"
    result = patcher.use(patch)

    assert (
        result
        == "File test.py does not exist or is not in the current repository.\nRewrite failed."
    )
    assert not patcher.rewrite_success


def test_substitution_patcher_invalid_line_number(mock_environment):
    patcher = SubstitutionPatcher()
    patcher.environment = mock_environment

    patch = "```rewrite test.py 0 <c>    print(f'Hello, {name}!')</c>```"
    result = patcher.use(patch)

    assert result == "Invalid line number, line numbers are 1-based.\nRewrite failed."
    assert not patcher.rewrite_success


def test_substitution_patcher_invalid_line_number_2(mock_environment):
    patcher = SubstitutionPatcher()
    patcher.environment = mock_environment

    patch = "```rewrite test.py 12:4 <c>    print(f'Hello, {name}!')</c>```"
    result = patcher.use(patch)

    assert (
        result
        == "Invalid line number range, head should be less than or equal to tail.\nRewrite failed."
    )
    assert not patcher.rewrite_success
