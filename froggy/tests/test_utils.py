import pytest
from froggy.utils import clean_code

@pytest.mark.parametrize("code, expected", [
    ("def foo():    \n    return 42    \n", "def foo():\n    return 42\n"),
    ("", ""),
    ("def foo():\n    return 42", "def foo():\n    return 42"),
    ("def foo():    \n    return 42    \n\n", "def foo():\n    return 42\n\n"),
    ("def foo():\\n    return 42\\n", "def foo():\n    return 42\n")
])
def test_clean_code(code, expected):
    assert clean_code(code) == expected
