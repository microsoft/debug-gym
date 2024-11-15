import pytest

from froggy.utils import clean_code, trim_prompt_messages


@pytest.mark.parametrize(
    "code, expected",
    [
        ("def foo():    \n    return 42    \n", "def foo():\n    return 42\n"),
        ("", ""),
        ("def foo():\n    return 42", "def foo():\n    return 42"),
        ("def foo():    \n    return 42    \n\n", "def foo():\n    return 42\n\n"),
        ("def foo():\\n    return 42\\n", "def foo():\n    return 42\n"),
    ],
)
def test_clean_code(code, expected):
    assert clean_code(code) == expected


def test_trim_prompt_messages():
    def token_counter(text):
        return len(text)

    with pytest.raises(ValueError):
        trim_prompt_messages([], -1, token_counter)

    assert trim_prompt_messages([], 0, token_counter) == []

    messages = [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "User message"},
    ]
    expected = [{"role": "user", "content": "User message"}]
    assert trim_prompt_messages(messages, 20, token_counter) == expected

    messages = [
        {"role": "user", "content": "User message"},
        {"role": "assistant", "content": "Assistant message"},
    ]
    expected = [{"role": "user", "content": "User message"}]
    assert trim_prompt_messages(messages, 20, token_counter) == expected

    messages = [
        {"role": "user", "content": "User message 1"},
        {"role": "assistant", "content": "Assistant message"},
        {"role": "user", "content": "User message 2"},
    ]
    expected = [
        {"role": "assistant", "content": "Assistant message"},
        {"role": "user", "content": "User message 2"},
    ]
    assert trim_prompt_messages(messages, 35, token_counter) == expected

    messages = [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "User message"},
        {"role": "assistant", "content": "Assistant message"},
    ]
    expected = [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "User message"},
    ]
    assert trim_prompt_messages(messages, 35, token_counter) == expected

    messages = [
        {"role": "system", "content": "System message"},
        {"role": "assistant", "content": "Assistant message"},
        {"role": "user", "content": "User message"},
    ]
    expected = [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "User message"},
    ]
    assert trim_prompt_messages(messages, 35, token_counter) == expected
