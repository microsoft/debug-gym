import pytest

from froggy.utils import clean_code, trim_prompt_messages, show_line_number, make_is_readonly, HistoryTracker


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

    with pytest.raises(Exception, match="messages should not be empty"):
        trim_prompt_messages([], 5, token_counter)

    with pytest.raises(
        Exception,
        match='all messages should be dictionaries with keys "content" and "role"',
    ):
        messages = [{"role": "system", "key": "System message"}]
        trim_prompt_messages(messages, 20, token_counter)

    with pytest.raises(Exception, match="the last message should be from the user"):
        messages = [
            {"role": "system", "content": "System message"},
            {"role": "assistant", "content": "Assistant message"},
        ]
        trim_prompt_messages(messages, 20, token_counter)

    with pytest.raises(
        Exception,
        match="if two consecutive messages are from the same role, they should be merged first",
    ):
        messages = [
            {"role": "system", "content": "System message 1"},
            {"role": "system", "content": "System message 2"},
            {"role": "user", "content": "User message"},
        ]
        trim_prompt_messages(messages, 20, token_counter)

    with pytest.raises(Exception, match="context_length should be non-negative"):
        messages = [{"role": "user", "content": "User message"}]
        trim_prompt_messages(messages, -1, token_counter)

    messages = [{"role": "user", "content": "User message"}]
    assert trim_prompt_messages(messages, 0, token_counter) == messages

    messages = [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "User message"},
    ]
    expected = [{"role": "user", "content": "User message"}]
    assert trim_prompt_messages(messages, 20, token_counter) == expected

    messages = [
        {"role": "user", "content": "User message"},
        {"role": "assistant", "content": "Assistant message"},
        {"role": "user", "content": "User message 2"},
    ]
    expected = messages
    assert trim_prompt_messages(messages, 200, token_counter) == expected

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
        {"role": "user", "content": "User message 2"},
    ]
    expected = [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "User message 2"},
    ]
    assert trim_prompt_messages(messages, 35, token_counter) == expected


def test_show_line_number():
    s4 = "    "
    s2 = "  "

    # code_string is empty
    with pytest.raises(
        Exception,
        match="code_string should not be None",
    ):
        show_line_number(None)

    # code_string is a list
    code_string = ["def foo():", "    return 42"]
    with pytest.raises(
        Exception,
        match=f"code_string should be a string, but got {type(code_string)}",
    ):
        show_line_number(code_string)

    # no code_path, no breakpoints
    code_string = f"def foo():\n{s4}return 42\n"
    expected = f"{s2}   1 def foo():\n{s2}   2 {s4}return 42\n{s2}   3 "
    assert show_line_number(code_string) == expected

    # with code_path
    code_path = "path/to/code.py"
    breakpoints_state = {"path/to/code.py|||2": "b 2"}
    code_string = f"def foo():\n{s4}return 42\n"
    expected = f"{s2}   1 def foo():\nB    2 {s4}return 42\n{s2}   3 "
    assert show_line_number(code_string, code_path, breakpoints_state) == expected

    # multiple breakpoints
    code_path = "path/to/code.py"
    breakpoints_state = {"path/to/code.py|||2": "b 2", "path/to/code.py|||3": "b 3, bar > 4"}
    code_string = f"def foo():\n"
    code_string += f"{s4}bar = 20\n"
    code_string += f"{s4}foobar = 42\n"
    code_string += f"{s4}print('frog')\n"
    code_string += f"{s4}return foobar\n"
    expected = f"{s2}   1 def foo():\n"
    expected += f"B    2 {s4}bar = 20\n"
    expected += f"B    3 {s4}foobar = 42\n"
    expected += f"{s2}   4 {s4}print('frog')\n"
    expected += f"{s2}   5 {s4}return foobar\n"
    expected += f"{s2}   6 "
    assert show_line_number(code_string, code_path, breakpoints_state) == expected

    # 10000 lines, so line numbers will take 8 digits
    code_string = f"def foo():\n" 
    for i in range(9997):
        code_string += f"{s4}print({i})\n"
    code_string += f"{s4}return 42\n"
    expected = f"{s2}{s4}   1 def foo():\n"
    for i in range(9997):
        expected += "{}{:>8} {}print({})\n".format(s2, i + 2, s4, i)
    expected += f"{s2}{s4}9999 {s4}return 42\n"
    expected += f"{s2}   10000 "
    assert show_line_number(code_string) == expected


def test_make_is_readonly():
    import os
    import tempfile
    from pathlib import Path
    import atexit
    # do the test in a tmp folder
    tempdir = tempfile.TemporaryDirectory(prefix="TestFroggyignore-")
    working_dir = Path(tempdir.name)
    ignore_file = working_dir / ".froggyignore"
    atexit.register(
        tempdir.cleanup
    )  # Make sure to cleanup that folder once done.

    froggyignore_contents = "\n".join(
                    [
                        ".DS_Store",
                        "__pycache__/",
                        ".approaches/",
                        ".docs/",
                        ".meta/",
                        ".pytest_cache/",
                        "*test*.py",
                        "*.pyc",
                        "*.md",
                        ".froggyignore",
                        "log/",
                        "data/",
                    ]
                )
    
    with open(ignore_file, "w") as f:
            f.write(froggyignore_contents)

    is_readonly = make_is_readonly(ignore_file, patterns=["source/*.frog"])

    assert is_readonly(working_dir / "foo.py") is False
    assert is_readonly(working_dir / "source/source.py") is False
    assert is_readonly(working_dir / "source/__init__.py") is False
    assert is_readonly(working_dir / "source/main.frog") is True
    assert is_readonly(working_dir / "utils/main.frog") is False
    assert is_readonly(working_dir / ".DS_Store") is True
    assert is_readonly(working_dir / "foo.pyc") is True
    assert is_readonly(working_dir / "foo_test.py") is True
    assert is_readonly(working_dir / "testy.py") is True
    assert is_readonly(working_dir / "data/foo.py") is True
    assert is_readonly(working_dir / "docs/source_code.py") is False
    assert is_readonly(working_dir / ".docs/source_code.py") is True
    assert is_readonly(working_dir / "this_is_code.md") is True
    assert is_readonly(working_dir / ".froggyignore") is True
    assert is_readonly(working_dir / "log/foo.py") is True
    assert is_readonly(working_dir / "source/fotesto.py") is True
    assert is_readonly(working_dir / ".meta/important.cc") is True


def test_history_tracker():
    ht = HistoryTracker(history_steps=3)

    # should start empty
    assert len(ht) == 0
    assert ht.get() == []
    assert ht.get_all() == []
    assert ht.score() == 0
    assert ht.prompt_response_pairs == [[]]  # at 0-th step, there is no prompt-response pair

    # json should return an empty dict
    assert ht.json() == {}

    # push some steps
    ht.step({"obs": "obs1", "action": None, "score": 1})
    ht.step({"obs": "obs2", "action": "action2", "score": 2})
    ht.step({"obs": "obs3", "action": "action3", "score": 3})
    ht.step({"obs": "obs4", "action": "action4", "score": 4, "token_usage": 12345})
    ht.step({"obs": "obs5", "action": "action5", "score": 5})
    # push some prompt-response pairs
    ht.save_prompt_response_pairs([("prompt_2_1", "response_2_1")])
    ht.save_prompt_response_pairs([("prompt_3_1", "response_3_1"), ("prompt_3_2", "response_3_2")])
    ht.save_prompt_response_pairs([("prompt_4_1", "response_4_1")])
    ht.save_prompt_response_pairs([("prompt_5_1", "response_5_1"), ("prompt_5_2", "response_5_2")])

    # get_all should return all steps
    assert ht.get_all() == [
        {"obs": "obs1", "action": None, "score": 1},
        {"obs": "obs2", "action": "action2", "score": 2},
        {"obs": "obs3", "action": "action3", "score": 3},
        {"obs": "obs4", "action": "action4", "score": 4, "token_usage": 12345},
        {"obs": "obs5", "action": "action5", "score": 5},
    ]

    # get should return the last 3 steps
    assert ht.get() == [
        {"obs": "obs3", "action": "action3", "score": 3},
        {"obs": "obs4", "action": "action4", "score": 4, "token_usage": 12345},
        {"obs": "obs5", "action": "action5", "score": 5},
    ]

    # json should return the last step by default
    assert ht.json() == {
        "step_id": 4,
        "action": "action5",
        "obs": "obs5",
    }

    # json should return the speficied step
    assert ht.json(2) == {
        "step_id": 2,
        "action": "action3",
        "obs": "obs3",
    }

    # json should return also the prompt-response pairs if include_prompt_response_pairs is True
    assert ht.json(2, include_prompt_response_pairs=True) == {
        "step_id": 2,
        "action": "action3",
        "obs": "obs3",
        "prompt_response_pairs": {
            "prompt_0": "prompt_3_1",
            "response_0": "response_3_1",
            "prompt_1": "prompt_3_2",
            "response_1": "response_3_2",
        },
    }

    # for 0-th step, prompt-response pairs should be None
    assert ht.json(0, include_prompt_response_pairs=True) == {
        "step_id": 0,
        "action": None,
        "obs": "obs1",
        "prompt_response_pairs": None,
    }

    # score should return the sum of the scores
    assert ht.score() == 15

    # len should return the number of steps
    assert len(ht) == 5

    # should reset properly
    ht.reset()
    assert len(ht) == 0
    assert ht.get() == []
    assert ht.get_all() == []
    assert ht.score() == 0
    assert ht.prompt_response_pairs == [[]]

    # json should return an empty dict
    assert ht.json() == {}
