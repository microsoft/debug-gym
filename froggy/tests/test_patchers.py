import pytest
from unittest.mock import MagicMock, patch
from froggy.tools.patchers import CodePatcher, UDiffPatcher, WholePatcher, SubstitutionPatcher

@pytest.fixture
def mock_environment():
    env = MagicMock()
    env.current_file_content = "def greet():\n    print('Hello, world!')\n"
    env.current_file = "test.py"
    env.all_files = ["test.py"]
    env.editable_files = ["test.py"]
    return env

def test_udiff_patcher(mock_environment):
    patcher = UDiffPatcher()
    patcher.environment = mock_environment

    patch = "```rewrite\n@@ -1,2 +1,2 @@\n-def greet():\n-    print('Hello, world!')\n+def greet(name):\n+    print(f'Hello, {name}!')\n```"
    # result = patcher.use(patch)

    # assert result == "Rewrite successful."
    # assert patcher.rewrite_success
    # mock_environment.overwrite_file.assert_called_once_with(filepath="test.py", content="def greet(name):\n    print(f'Hello, {name}!')\n")

def test_whole_patcher(mock_environment):
    patcher = WholePatcher()
    patcher.environment = mock_environment

    patch = "```rewrite\ndef greet(name):\n    print(f'Hello, {name}!')\n```"
    result = patcher.use(patch)

    assert result == "Rewrite successful."
    assert patcher.rewrite_success
    # mock_environment.overwrite_file.assert_called_once_with(filepath="test.py", content="def greet(name):\n    print(f'Hello, {name}!')\n")

def test_substitution_patcher(mock_environment):
    patcher = SubstitutionPatcher()
    patcher.environment = mock_environment

    patch = "```rewrite 2 <c>    print(f'Hello, {name}!')</c>```"
    # result = patcher.use(patch)

    # assert result == "Rewriting done."
    # assert patcher.rewrite_success
    # mock_environment.overwrite_file.assert_called_once_with(filepath="test.py", content="def greet():\n    print(f'Hello, {name}!')\n")

def test_substitution_patcher_with_file_path(mock_environment):
    patcher = SubstitutionPatcher()
    patcher.environment = mock_environment

    patch = "```rewrite test.py 2 <c>    print(f'Hello, {name}!')</c>```"
    result = patcher.use(patch)

    assert result == "Rewriting done."
    assert patcher.rewrite_success
    # mock_environment.overwrite_file.assert_called_once_with(filepath="test.py", content="def greet():\n    print(f'Hello, {name}!')\n")

def test_substitution_patcher_invalid_content(mock_environment):
    patcher = SubstitutionPatcher()
    patcher.environment = mock_environment

    patch = "```rewrite invalid content```"
    result = patcher.use(patch)

    assert result == "Rewrite failed."
    assert not patcher.rewrite_success

def test_substitution_patcher_invalid_file(mock_environment):
    patcher = SubstitutionPatcher()
    patcher.environment = mock_environment
    mock_environment.all_files = ["another_file.py"]

    patch = "```rewrite test.py 2 <c>    print(f'Hello, {name}!')</c>```"
    result = patcher.use(patch)

    assert result == "File test.py does not exist or is not in the current repository.\nRewrite failed."
    assert not patcher.rewrite_success

if __name__ == '__main__':
    pytest.main()