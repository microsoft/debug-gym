from pathlib import Path

import pytest

from debug_gym.gym.utils import (
    _walk,
    clean_code,
    cleanup_pytest_output,
    create_ignore_file,
    extract_max_score_from_pytest_output,
    extract_reward_from_pytest_output,
    is_subdirectory,
    make_file_matcher,
    show_line_number,
    str2bool,
    unescape,
)


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
    breakpoints_state = {
        "path/to/code.py|||2": "b 2",
        "path/to/code.py|||3": "b 3, bar > 4",
    }
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


def test_make_file_matcher(tmp_path):
    working_dir = Path(tmp_path)
    ignore_file = working_dir / ".debugignore"

    for with_negation in [False, True]:
        debugignore_contents = "\n".join(
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
                ".debugignore",
                "log/",
                "data/",
            ]
        )
        if with_negation is True:
            debugignore_contents += "\n!data/unignore/*"
        with open(ignore_file, "w") as f:
            f.write(debugignore_contents)
        is_ignored = make_file_matcher(ignore_file, patterns=["source/*.frog"])

        assert is_ignored(working_dir / "foo.py") is False
        assert is_ignored(working_dir / "source/source.py") is False
        assert is_ignored(working_dir / "source/__init__.py") is False
        assert is_ignored(working_dir / "source/main.frog") is True
        assert is_ignored(working_dir / "utils/main.frog") is False
        assert is_ignored(working_dir / ".DS_Store") is True
        assert is_ignored(working_dir / "foo.pyc") is True
        assert is_ignored(working_dir / "foo_test.py") is True
        assert is_ignored(working_dir / "testy.py") is True
        assert is_ignored(working_dir / "data/foo.py") is True
        assert is_ignored(working_dir / "docs/source_code.py") is False
        assert is_ignored(working_dir / ".docs/source_code.py") is True
        assert is_ignored(working_dir / "this_is_code.md") is True
        assert is_ignored(working_dir / ".debugignore") is True
        assert is_ignored(working_dir / "log/foo.py") is True
        assert is_ignored(working_dir / "source/fotesto.py") is True
        assert is_ignored(working_dir / ".meta/important.cc") is True
        assert is_ignored(working_dir / "data/specific.py") is True
        if with_negation is True:
            assert is_ignored(working_dir / "data/unignore/foo.py") is False
        else:
            assert is_ignored(working_dir / "data/unignore/foo.py") is True


def test_create_ignore_file(tmp_path):
    # Test without including .gitignore
    test_dir = tmp_path / "test_dir"
    debugignore_path = test_dir / ".debugignore"
    test_dir.mkdir()
    create_ignore_file(
        debugignore_path, patterns=["*.pyc", "*.log"], include_gitignore=False
    )
    assert debugignore_path.exists()
    with open(debugignore_path) as f:
        contents = f.read().splitlines()
    assert contents == ["*.pyc", "*.log", ".debugignore"]

    # Test with including .gitignore
    gitignore_path = test_dir / ".gitignore"
    with open(gitignore_path, "w") as f:
        f.write("*.tmp\n*.bak\n")
    create_ignore_file(
        debugignore_path, patterns=["*.pyc", "*.log"], include_gitignore=True
    )
    with open(debugignore_path) as f:
        contents = f.read().splitlines()
    assert contents == ["*.pyc", "*.log", "*.tmp", "*.bak", ".debugignore"]

    # Test with empty patterns and without including .gitignore
    create_ignore_file(debugignore_path, patterns=[], include_gitignore=False)
    with open(debugignore_path) as f:
        contents = f.read().splitlines()
    assert contents == [".debugignore"]

    # Test with empty patterns and including .gitignore
    create_ignore_file(debugignore_path, patterns=[], include_gitignore=True)
    with open(debugignore_path) as f:
        contents = f.read().splitlines()
    assert contents == ["*.tmp", "*.bak", ".debugignore"]


def test_str2bool():
    assert str2bool("True") is True
    assert str2bool("true") is True
    assert str2bool("t") is True
    assert str2bool("1") is True
    assert str2bool("Yes") is True
    assert str2bool("Y") is True
    assert str2bool("False") is False
    assert str2bool("false") is False
    assert str2bool("f") is False
    assert str2bool("0") is False
    assert str2bool("No") is False
    assert str2bool("N") is False
    assert str2bool(True) is True
    assert str2bool(False) is False
    with pytest.raises(Exception, match="Boolean value expected."):
        str2bool("Maybe")
    with pytest.raises(Exception, match="Boolean value expected."):
        str2bool("yeah")
    with pytest.raises(Exception, match="Boolean value expected."):
        str2bool("nah")


def test_is_subdirectory():

    assert is_subdirectory("/path/to/file", "/path/to") is True
    assert is_subdirectory("/path/to/file", "/path/to/") is True
    assert is_subdirectory("/path/to/file", "/path/to/file") is True
    assert is_subdirectory("/path/to/file", "/path/too") is False
    assert is_subdirectory("/path/to/file", "/path/too/file") is False
    assert is_subdirectory("/some/random/file", "/path/to") is False
    assert is_subdirectory("some/random/file", "/path/to") is True
    assert is_subdirectory("file", "/path/to") is True
    assert is_subdirectory("/path/file", "/path/to") is False


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


def test_walk():
    path = "data/mini_nightmare"

    # depth 0
    path_list = []
    for p in _walk(path, 0):
        path_list.append(p)
    assert path_list == []

    # depth 1
    path_list = []
    for p in _walk(path, 1):
        path_list.append(p)
    expected = [
        Path("data/mini_nightmare/config"),
        Path("data/mini_nightmare/counter"),
        Path("data/mini_nightmare/grader"),
        Path("data/mini_nightmare/mini_nightmare.md"),
        Path("data/mini_nightmare/pandas_dataframe"),
        Path("data/mini_nightmare/patcher"),
        Path("data/mini_nightmare/purr"),
        Path("data/mini_nightmare/scientific_calculator"),
        Path("data/mini_nightmare/shopping_cart"),
        Path("data/mini_nightmare/sum_tree"),
        Path("data/mini_nightmare/tomorrow_date"),
    ]
    # sort the list
    path_list.sort()
    expected.sort()
    assert path_list == expected

    # depth 2
    path_list = []
    for p in _walk(path, 2):
        path_list.append(p)
    expected = [
        Path("data/mini_nightmare/config"),
        Path("data/mini_nightmare/config/config_code.py"),
        Path("data/mini_nightmare/config/test.py"),
        Path("data/mini_nightmare/config/.debugignore"),
        Path("data/mini_nightmare/config/.debugreadonly"),
        Path("data/mini_nightmare/counter"),
        Path("data/mini_nightmare/counter/counter_code.py"),
        Path("data/mini_nightmare/counter/test.py"),
        Path("data/mini_nightmare/counter/.debugignore"),
        Path("data/mini_nightmare/counter/.debugreadonly"),
        Path("data/mini_nightmare/grader"),
        Path("data/mini_nightmare/grader/grader_code.py"),
        Path("data/mini_nightmare/grader/test.py"),
        Path("data/mini_nightmare/grader/.debugignore"),
        Path("data/mini_nightmare/grader/.debugreadonly"),
        Path("data/mini_nightmare/mini_nightmare.md"),
        Path("data/mini_nightmare/pandas_dataframe"),
        Path("data/mini_nightmare/pandas_dataframe/pandas_dataframe_code.py"),
        Path("data/mini_nightmare/pandas_dataframe/test.py"),
        Path("data/mini_nightmare/pandas_dataframe/.debugignore"),
        Path("data/mini_nightmare/pandas_dataframe/.debugreadonly"),
        Path("data/mini_nightmare/patcher"),
        Path("data/mini_nightmare/patcher/patcher_code.py"),
        Path("data/mini_nightmare/patcher/test.py"),
        Path("data/mini_nightmare/patcher/source_code.txt"),
        Path("data/mini_nightmare/patcher/.debugignore"),
        Path("data/mini_nightmare/patcher/.debugreadonly"),
        Path("data/mini_nightmare/purr"),
        Path("data/mini_nightmare/purr/purr_code.py"),
        Path("data/mini_nightmare/purr/test.py"),
        Path("data/mini_nightmare/purr/.debugignore"),
        Path("data/mini_nightmare/purr/.debugreadonly"),
        Path("data/mini_nightmare/scientific_calculator"),
        Path("data/mini_nightmare/scientific_calculator/scientific_calculator_code.py"),
        Path("data/mini_nightmare/scientific_calculator/test.py"),
        Path("data/mini_nightmare/scientific_calculator/scientific_calculator_tool.py"),
        Path("data/mini_nightmare/scientific_calculator/.debugignore"),
        Path("data/mini_nightmare/scientific_calculator/.debugreadonly"),
        Path("data/mini_nightmare/shopping_cart"),
        Path("data/mini_nightmare/shopping_cart/shopping_cart_code.py"),
        Path("data/mini_nightmare/shopping_cart/test.py"),
        Path("data/mini_nightmare/shopping_cart/.debugignore"),
        Path("data/mini_nightmare/shopping_cart/.debugreadonly"),
        Path("data/mini_nightmare/sum_tree"),
        Path("data/mini_nightmare/sum_tree/sum_tree_code.py"),
        Path("data/mini_nightmare/sum_tree/test.py"),
        Path("data/mini_nightmare/sum_tree/.debugignore"),
        Path("data/mini_nightmare/sum_tree/.debugreadonly"),
        Path("data/mini_nightmare/tomorrow_date"),
        Path("data/mini_nightmare/tomorrow_date/tomorrow_date_code.py"),
        Path("data/mini_nightmare/tomorrow_date/test.py"),
        Path("data/mini_nightmare/tomorrow_date/.debugignore"),
        Path("data/mini_nightmare/tomorrow_date/.debugreadonly"),
    ]
    # sort the list
    path_list.sort()
    expected.sort()
    assert path_list == expected

    # depth is None (max)
    path_list = []
    for p in _walk(path, None):
        path_list.append(p)
    expected = [
        Path("data/mini_nightmare/config"),
        Path("data/mini_nightmare/config/config_code.py"),
        Path("data/mini_nightmare/config/test.py"),
        Path("data/mini_nightmare/config/.debugignore"),
        Path("data/mini_nightmare/config/.debugreadonly"),
        Path("data/mini_nightmare/counter"),
        Path("data/mini_nightmare/counter/counter_code.py"),
        Path("data/mini_nightmare/counter/test.py"),
        Path("data/mini_nightmare/counter/.debugignore"),
        Path("data/mini_nightmare/counter/.debugreadonly"),
        Path("data/mini_nightmare/grader"),
        Path("data/mini_nightmare/grader/grader_code.py"),
        Path("data/mini_nightmare/grader/test.py"),
        Path("data/mini_nightmare/grader/.debugignore"),
        Path("data/mini_nightmare/grader/.debugreadonly"),
        Path("data/mini_nightmare/mini_nightmare.md"),
        Path("data/mini_nightmare/pandas_dataframe"),
        Path("data/mini_nightmare/pandas_dataframe/pandas_dataframe_code.py"),
        Path("data/mini_nightmare/pandas_dataframe/test.py"),
        Path("data/mini_nightmare/pandas_dataframe/.debugignore"),
        Path("data/mini_nightmare/pandas_dataframe/.debugreadonly"),
        Path("data/mini_nightmare/patcher"),
        Path("data/mini_nightmare/patcher/patcher_code.py"),
        Path("data/mini_nightmare/patcher/test.py"),
        Path("data/mini_nightmare/patcher/source_code.txt"),
        Path("data/mini_nightmare/patcher/.debugignore"),
        Path("data/mini_nightmare/patcher/.debugreadonly"),
        Path("data/mini_nightmare/purr"),
        Path("data/mini_nightmare/purr/purr_code.py"),
        Path("data/mini_nightmare/purr/test.py"),
        Path("data/mini_nightmare/purr/.debugignore"),
        Path("data/mini_nightmare/purr/.debugreadonly"),
        Path("data/mini_nightmare/scientific_calculator"),
        Path("data/mini_nightmare/scientific_calculator/scientific_calculator_code.py"),
        Path("data/mini_nightmare/scientific_calculator/test.py"),
        Path("data/mini_nightmare/scientific_calculator/scientific_calculator_tool.py"),
        Path("data/mini_nightmare/scientific_calculator/.debugignore"),
        Path("data/mini_nightmare/scientific_calculator/.debugreadonly"),
        Path("data/mini_nightmare/shopping_cart"),
        Path("data/mini_nightmare/shopping_cart/shopping_cart_code.py"),
        Path("data/mini_nightmare/shopping_cart/test.py"),
        Path("data/mini_nightmare/shopping_cart/.debugignore"),
        Path("data/mini_nightmare/shopping_cart/.debugreadonly"),
        Path("data/mini_nightmare/sum_tree"),
        Path("data/mini_nightmare/sum_tree/sum_tree_code.py"),
        Path("data/mini_nightmare/sum_tree/test.py"),
        Path("data/mini_nightmare/sum_tree/.debugignore"),
        Path("data/mini_nightmare/sum_tree/.debugreadonly"),
        Path("data/mini_nightmare/tomorrow_date"),
        Path("data/mini_nightmare/tomorrow_date/tomorrow_date_code.py"),
        Path("data/mini_nightmare/tomorrow_date/test.py"),
        Path("data/mini_nightmare/tomorrow_date/.debugignore"),
        Path("data/mini_nightmare/tomorrow_date/.debugreadonly"),
    ]
    # sort the list
    path_list.sort()
    expected.sort()
    assert path_list == expected


def test_unescape_surrogate_pairs():
    # Test with regular string
    regular_string = "This is a regular string with escapes \\n\\t"
    assert unescape(regular_string) == "This is a regular string with escapes \n\t"

    # Test with surrogate pairs that would cause UTF-8 encoding issues
    surrogate_string = "Test with surrogate \\ud800\\udc00 pair"
    result = unescape(surrogate_string)

    # Verify we can encode the result to UTF-8 without errors
    try:
        result.encode("utf-8")
    except UnicodeEncodeError:
        assert False, "Unescaped string still has invalid surrogate pairs"

    # The result should replace the surrogate with a replacement character
    assert "Test with surrogate" in result


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
