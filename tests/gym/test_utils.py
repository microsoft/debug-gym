from pathlib import Path

import pytest

from debug_gym.gym.envs.local import LocalEnv
from debug_gym.gym.utils import (
    cleanup_pytest_output,
    extract_max_score_from_pytest_output,
    extract_reward_from_pytest_output,
    filter_non_utf8,
    filter_problems,
    show_line_number,
)


def test_show_line_number_empty_code_string():
    # code_string is empty
    with pytest.raises(
        Exception,
        match="code_string should not be empty",
    ):
        show_line_number(None)


def test_show_line_number_code_string_is_list():
    # code_string is a list
    code_string = ["def foo():", "    return 42"]
    with pytest.raises(
        Exception,
        match=f"code_string should be a string, but got {type(code_string)}",
    ):
        show_line_number(code_string)


def test_show_line_number_no_code_path_no_breakpoints():
    s4 = "    "
    s2 = "  "
    code_string = f"def foo():\n{s4}return 42\n"
    expected = f"{s2}   1 def foo():\n{s2}   2 {s4}return 42\n{s2}   3 "
    assert show_line_number(code_string) == expected


def test_show_line_number_with_code_path(tmp_path):
    env = LocalEnv(path=tmp_path)
    env.reset()
    code_path = f"{env.working_dir}/code.py"
    breakpoints_state = {f"{code_path}|||2": "b 2"}
    env.current_breakpoints_state = breakpoints_state
    # fmt: off
    code_string = (
        "def foo():\n"
        "    return 42\n"
    )
    expected = (
        "     1 def foo():\n"
        "B    2     return 42\n"
        "     3 "
    )
    # fmt: on
    assert show_line_number(code_string, code_path, env) == expected


def test_show_line_number_multiple_breakpoints(tmp_path):
    env = LocalEnv(path=tmp_path)
    env.reset()
    code_path = f"{env.working_dir}/code.py"
    breakpoints_state = {
        f"{code_path}|||2": "b 2",
        f"{code_path}|||3": "b 3, bar > 4",
    }
    env.current_breakpoints_state = breakpoints_state
    code_string = (
        "def foo():\n"
        "    bar = 20\n"
        "    foobar = 42\n"
        "    print('frog')\n"
        "    return foobar\n"
    )
    expected = (
        "     1 def foo():\n"
        "B    2     bar = 20\n"
        "B    3     foobar = 42\n"
        "     4     print('frog')\n"
        "     5     return foobar\n"
        "     6 "
    )
    assert show_line_number(code_string, code_path, env) == expected


def test_show_line_number_multiple_breakpoints_with_start_index(tmp_path):
    env = LocalEnv(path=tmp_path)
    env.reset()
    code_path = f"{env.working_dir}/code.py"
    breakpoints_state = {
        f"{code_path}|||102": "b 102",
        f"{code_path}|||103": "b 103, bar > 4",
    }
    env.current_breakpoints_state = breakpoints_state
    code_string = (
        "def foo():\n"
        "    bar = 20\n"
        "    foobar = 42\n"
        "    print('frog')\n"
        "    return foobar\n"
    )
    start_index = 101
    annotated_code = show_line_number(code_string, code_path, env, start_index)
    expected = (
        "   101 def foo():\n"
        "B  102     bar = 20\n"
        "B  103     foobar = 42\n"
        "   104     print('frog')\n"
        "   105     return foobar\n"
        "   106 "
    )
    assert annotated_code == expected


def test_show_line_number_large_number_of_lines():
    s4 = "    "
    s2 = "  "
    code_string = "def foo():\n"
    for i in range(9997):
        code_string += f"{s4}print({i})\n"
    code_string += f"{s4}return 42\n"
    annotated_code = show_line_number(code_string)

    expected = "         1 def foo():\n"
    for i in range(9997):
        expected += "{}{:>8} {}print({})\n".format(s2, i + 2, s4, i)
    expected += f"      9999 {s4}return 42\n"
    expected += "     10000 "

    # Check full match, but only report the first and last 100 characters
    # If the test fails and the output is too large, pytest may hang
    assert annotated_code[:100] == expected[:100]
    assert annotated_code[-100:] == expected[-100:]
    match = annotated_code == expected
    assert match, "Annotated code does not match expected output"


def test_show_line_number_large_number_of_lines_with_start_index():
    s4 = "    "
    s2 = "  "
    code_string = "def foo():\n"
    for i in range(9997):
        code_string += f"{s4}print({i})\n"
    code_string += f"{s4}return 42\n"
    start_index = 101
    annotated_code = show_line_number(code_string, start_index=start_index)

    expected = "       101 def foo():\n"
    for i in range(9997):
        expected += "{}{:>8} {}print({})\n".format(s2, i + start_index + 1, s4, i)
    expected += f"     10099 {s4}return 42\n"
    expected += "     10100 "

    # Check full match, but only report the first and last 100 characters
    # If the test fails and the output is too large, pytest may hang
    assert annotated_code[:100] == expected[:100]
    assert annotated_code[-100:] == expected[-100:]
    match = annotated_code == expected
    assert match, "Annotated code does not match expected output"


def test_extract_max_score_from_pytest_output():
    message_15 = "============================= test session starts ==============================\ncollecting ... collected 15 items\n\ntwelve_days_test.py::TwelveDaysTest::test_eighth_day_eight_maids_a_milking FAILED\ntwelve_days_test.py::TwelveDaysTest::test_eleventh_day_eleven_pipers_piping FAILED\ntwelve_days_test.py::TwelveDaysTest::test_fifth_day_five_gold_rings FAILED\ntwelve_days_test.py::TwelveDaysTest::test_first_day_a_partridge_in_a_pear_tree PASSED\ntwelve_days_test.py::TwelveDaysTest::test_fourth_day_four_calling_birds FAILED\ntwelve_days_test.py::TwelveDaysTest::test_ninth_day_nine_ladies_dancing FAILED\ntwelve_days_test.py::TwelveDaysTest::test_recites_first_three_verses_of_the_song PASSED\ntwelve_days_test.py::TwelveDaysTest::test_recites_the_whole_song PASSED\ntwelve_days_test.py::TwelveDaysTest::test_recites_three_verses_from_the_middle_of_the_song PASSED\ntwelve_days_test.py::TwelveDaysTest::test_second_day_two_turtle_doves FAILED\ntwelve_days_test.py::TwelveDaysTest::test_seventh_day_seven_swans_a_swimming FAILED\ntwelve_days_test.py::TwelveDaysTest::test_sixth_day_six_geese_a_laying FAILED\ntwelve_days_test.py::TwelveDaysTest::test_tenth_day_ten_lords_a_leaping FAILED\ntwelve_days_test.py::TwelveDaysTest::test_third_day_three_french_hens FAILED\ntwelve_days_test.py::TwelveDaysTest::test_twelfth_day_twelve_drummers_drumming FAILED\n\n=================================== FAILURES ===================================\n"

    assert extract_max_score_from_pytest_output(message_15) == 15

    message_1 = "============================= test session starts ==============================\ncollecting ... collected 1 item\n \nhello_world_test.py::HelloWorldTest::test_say_hi FAILED\n \n=================================== FAILURES ===================================\n__________________________ HelloWorldTest.test_say_hi __________________________\n \nself = <hello_world_test.HelloWorldTest testMethod=test_say_hi>\n \n    def test_say_hi(self):\n        msg = \"\n\nThis test expects a return of the string 'Hello, World!' \nDid you use print('Hello, World!') by mistake?\"\n>       self.assertEqual(hello(), \"Hello, World!\", msg=msg)\nE       AssertionError: 'Goodbye, Mars!' != 'Hello, World!'\nE       - Goodbye, Mars!\nE       + Hello, World!\nE        : \nE       \nE       This test expects a return of the string 'Hello, World!' \nE       Did you use print('Hello, World!') by mistake?\n \nhello_world_test.py:30: AssertionError\n=========================== short test summary info ============================\nFAILED hello_world_test.py::HelloWorldTest::test_say_hi - AssertionError: 'Go...\n============================== 1 failed in 0.01s ==============================="

    assert extract_max_score_from_pytest_output(message_1) == 1

    message_0 = "============================= test session starts ==============================\ncollecting ... collected 0 items\n\n============================== no tests ran in 0.01s ==============================="

    assert extract_max_score_from_pytest_output(message_0) == 1

    message_rand = "============================= here are some random text ==============================="
    with pytest.raises(
        ValueError, match="Cannot extract max score from pytest output."
    ):
        extract_max_score_from_pytest_output(message_rand)


def test_extract_reward_from_pytest_output():
    message_15 = "========================= 11 failed, 4 passed in 0.05s =========================\n"

    assert extract_reward_from_pytest_output(message_15) == 4

    message_1 = "============================= test session starts ==============================\ncollecting ... collected 1 item\n \nhello_world_test.py::HelloWorldTest::test_say_hi FAILED\n \n=================================== FAILURES ===================================\n__________________________ HelloWorldTest.test_say_hi __________________________\n \nself = <hello_world_test.HelloWorldTest testMethod=test_say_hi>\n \n    def test_say_hi(self):\n        msg = \"\n\nThis test expects a return of the string 'Hello, World!' \nDid you use print('Hello, World!') by mistake?\"\n>       self.assertEqual(hello(), \"Hello, World!\", msg=msg)\nE       AssertionError: 'Goodbye, Mars!' != 'Hello, World!'\nE       - Goodbye, Mars!\nE       + Hello, World!\nE        : \nE       \nE       This test expects a return of the string 'Hello, World!' \nE       Did you use print('Hello, World!') by mistake?\n \nhello_world_test.py:30: AssertionError\n=========================== short test summary info ============================\nFAILED hello_world_test.py::HelloWorldTest::test_say_hi - AssertionError: 'Go...\n============================== 1 failed in 0.01s ==============================="

    assert extract_reward_from_pytest_output(message_1) == 0

    message_0 = "============================= here are some random text ==============================="

    assert extract_reward_from_pytest_output(message_0) == 0


def test_filter_non_utf8():
    """Test the filter_non_utf8 function with various inputs."""

    # Test with regular ASCII text
    assert filter_non_utf8("hello world") == "hello world"

    # Test with valid UTF-8 characters
    assert filter_non_utf8("héllo wørld") == "héllo wørld"

    # Test with emoji (valid surrogate pairs)
    assert filter_non_utf8("hello 👋 world 🌍") == "hello 👋 world 🌍"

    # Test with various Unicode characters
    assert filter_non_utf8("こんにちは 你好 مرحبا") == "こんにちは 你好 مرحبا"

    # Test with mixed content
    mixed_text = "Regular text with émoji 🎉 and ünïcode"
    assert filter_non_utf8(mixed_text) == mixed_text

    # Test with empty string
    assert filter_non_utf8("") == ""

    # Test with non-string input (should return as-is)
    with pytest.raises(AttributeError):
        filter_non_utf8(None)
    with pytest.raises(AttributeError):
        filter_non_utf8(123)
    with pytest.raises(AttributeError):
        filter_non_utf8([1, 2, 3])

    # Test with newlines and special characters
    text_with_newlines = "line1\nline2\tTabbed\r\nWindows line ending"
    assert filter_non_utf8(text_with_newlines) == text_with_newlines

    # Test with string containing invalid UTF-8 bytes (simulated)
    # Note: This is tricky to test directly since Python strings are Unicode
    # But we can test the function's behavior with valid input
    text_with_special_chars = "Text with \u0000 null and \uffff characters"
    result = filter_non_utf8(text_with_special_chars)
    # Should preserve valid Unicode characters
    assert isinstance(result, str)

    # Test with very long string
    long_text = "a" * 10000 + "🎉" * 1000
    result = filter_non_utf8(long_text)
    assert len(result) == 11000
    assert result.startswith("a" * 10000)
    assert result.endswith("🎉" * 1000)


def test_filter_non_utf8_edge_cases():
    """Test edge cases for filter_non_utf8 function."""

    # Test with only whitespace
    assert filter_non_utf8("   \t\n  ") == "   \t\n  "

    # Test with only special Unicode characters
    assert (
        filter_non_utf8("\u200b\u200c\u200d") == "\u200b\u200c\u200d"
    )  # Zero-width characters

    # Test with combining characters
    combining_text = "e\u0301"  # e with acute accent as combining character
    result = filter_non_utf8(combining_text)
    assert result == combining_text

    # Test with different types that should raise.
    test_cases = [42, 3.14, True, False, [], {}, set()]
    for input_val in test_cases:
        with pytest.raises(AttributeError):
            filter_non_utf8(input_val)


def test_filter_non_utf8_preserves_json_serializable():
    """Test that the function preserves JSON-serializable content."""
    import json

    test_strings = [
        "simple text",
        "text with émoji 🎉",
        '{"key": "value with ünicöde"}',
        "line1\nline2\ttab",
    ]

    for text in test_strings:
        filtered = filter_non_utf8(text)
        # Should be JSON serializable
        json_str = json.dumps({"text": filtered})
        # Should be able to parse back
        parsed = json.loads(json_str)
        assert parsed["text"] == filtered


def test_cleanup_pytest_output():
    message = "============================= test session starts ==============================\n===============================\n==============================\n=============================\n"
    cleaned_message = cleanup_pytest_output(message)
    expected = "============================= test session starts ==============================\n====\n====\n====\n"
    assert cleaned_message == expected

    message = "----------------------------- test session starts ------------------------------\n-------------------------------\n------------------------------\n-----------------------------\n"
    cleaned_message = cleanup_pytest_output(message)
    expected = "----------------------------- test session starts ------------------------------\n----\n----\n----\n"
    assert cleaned_message == expected

    message = "============================= test session starts ==============================\nplatform linux -- Python 3.12.3, pytest-8.3.3, pluggy-1.5.0 -- /datadrive/eric_work_space/venvs2024/be/bin/python\ncachedir: .pytest_cache\nrootdir: /tmp/RepoEnv-2lpnkhwv\nplugins: anyio-4.3.0\ncollecting ... collected 21 items\n\nphone_number_test.py::PhoneNumberTest::test_area_code FAILED\nphone_number_test.py::PhoneNumberTest::test_cleans_numbers_with_dots FAILED\nphone_number_test.py::PhoneNumberTest::test_cleans_numbers_with_multiple_spaces FAILED\nphone_number_test.py::PhoneNumberTest::test_cleans_the_number FAILED\nphone_number_test.py::PhoneNumberTest::test_invalid_if_area_code_starts_with_0 FAILED\nphone_number_test.py::PhoneNumberTest::test_invalid_if_area_code_starts_with_0_on_valid_11_digit_number FAILED\nphone_number_test.py::PhoneNumberTest::test_invalid_if_area_code_starts_with_1 FAILED\nphone_number_test.py::PhoneNumberTest::test_invalid_if_area_code_starts_with_1_on_valid_11_digit_number FAILED\nphone_number_test.py::PhoneNumberTest::test_invalid_if_exchange_code_starts_with_0 FAILED\nphone_number_test.py::PhoneNumberTest::test_invalid_if_exchange_code_starts_with_0_on_valid_11_digit_number FAILED\nphone_number_test.py::PhoneNumberTest::test_invalid_if_exchange_code_starts_with_1 FAILED\n"
    cleaned_message = cleanup_pytest_output(message)
    expected = "============================= test session starts ==============================\ncollecting ... collected 21 items\n\nphone_number_test.py::PhoneNumberTest::test_area_code FAILED\nphone_number_test.py::PhoneNumberTest::test_cleans_numbers_with_dots FAILED\nphone_number_test.py::PhoneNumberTest::test_cleans_numbers_with_multiple_spaces FAILED\nphone_number_test.py::PhoneNumberTest::test_cleans_the_number FAILED\nphone_number_test.py::PhoneNumberTest::test_invalid_if_area_code_starts_with_0 FAILED\nphone_number_test.py::PhoneNumberTest::test_invalid_if_area_code_starts_with_0_on_valid_11_digit_number FAILED\nphone_number_test.py::PhoneNumberTest::test_invalid_if_area_code_starts_with_1 FAILED\nphone_number_test.py::PhoneNumberTest::test_invalid_if_area_code_starts_with_1_on_valid_11_digit_number FAILED\nphone_number_test.py::PhoneNumberTest::test_invalid_if_exchange_code_starts_with_0 FAILED\nphone_number_test.py::PhoneNumberTest::test_invalid_if_exchange_code_starts_with_0_on_valid_11_digit_number FAILED\nphone_number_test.py::PhoneNumberTest::test_invalid_if_exchange_code_starts_with_1 FAILED\n"
    assert cleaned_message == expected

    message = "Ran 15 tests in 0.09s\nSomething else\n"
    cleaned_message = cleanup_pytest_output(message)
    expected = "\nSomething else\n"
    assert cleaned_message == expected

    message = "Ran 1 test in 12.25s\nSomething else\n"
    cleaned_message = cleanup_pytest_output(message)
    expected = "\nSomething else\n"
    assert cleaned_message == expected


def test_filter_problems():
    dataset = [f"problem{i+1}" for i in range(5)]

    # Test retrieving all problem IDs
    problem_ids = filter_problems(dataset, problems="all")
    assert isinstance(problem_ids, list), "Expected a list of problem IDs"
    assert len(problem_ids) == len(dataset), "Expected all problem IDs"
    assert all(
        isinstance(pid, str) for pid in problem_ids
    ), "All problem IDs should be strings"

    # Test retrieving a single valid problem ID
    task_name = dataset[4]  # e.g., "problem4"
    problem_ids = filter_problems(dataset, problems=task_name)
    assert isinstance(problem_ids, list), "Expected a list of problem IDs"
    assert len(problem_ids) == 1, "Expected exactly one problem ID"
    assert problem_ids[0] == task_name, f"Expected problem ID to be {task_name}"

    # Test retrieving an invalid problem ID
    invalid_task_name = "non_existent_task"
    with pytest.raises(
        ValueError, match=f"Invalid split or problem id: '{invalid_task_name}'"
    ):
        filter_problems(dataset, problems=invalid_task_name)

    # Test with empty dataset
    empty_dataset = []
    assert filter_problems(empty_dataset, problems="all") == []

    # Test with duplicate items in dataset
    duplicate_dataset = ["problem1", "problem2", "problem1", "problem3"]
    result = filter_problems(duplicate_dataset, problems="all")
    assert result == duplicate_dataset  # Should preserve duplicates and order

    # Test with excluded_ids
    result = filter_problems(
        dataset, problems="all", excluded_ids=["problem2", "problem4"]
    )
    assert result == ["problem1", "problem3", "problem5"]

    # Test with custom_splits
    custom_splits = {
        "easy": ["problem1", "problem2"],
        "mixed": ["problem1", "problem3", "problem5"],
    }
    result = filter_problems(dataset, problems="easy", custom_splits=custom_splits)
    assert result == ["problem1", "problem2"]

    result = filter_problems(dataset, problems="mixed", custom_splits=custom_splits)
    assert result == ["problem1", "problem3", "problem5"]

    # Test with excluded_ids and custom_splits combined
    result = filter_problems(
        dataset, problems="all", excluded_ids=["problem2"], custom_splits=custom_splits
    )
    assert result == ["problem1", "problem3", "problem4", "problem5"]

    # Test error cases
    # Invalid problem ID in list
    with pytest.raises(ValueError, match="Invalid problem id: 'nonexistent'"):
        filter_problems(dataset, problems=["problem1", "nonexistent"])

    # Invalid problem ID as string
    with pytest.raises(ValueError, match="Invalid split or problem id: 'nonexistent'"):
        filter_problems(dataset, problems="nonexistent")

    # Duplicate problem IDs in list
    with pytest.raises(ValueError, match="Duplicate problem IDs found in the list"):
        filter_problems(dataset, problems=["problem1", "problem2", "problem1"])

    # Test edge case: excluded_ids with custom split that includes excluded items
    custom_splits_with_excluded = {"test_split": ["problem1", "problem2", "problem3"]}
    result = filter_problems(
        dataset,
        problems="test_split",
        custom_splits=custom_splits_with_excluded,
        excluded_ids=["problem2"],  # excluded_ids only affects "all"
    )
    assert result == [
        "problem1",
        "problem2",
        "problem3",
    ]  # excluded_ids doesn't affect custom splits
