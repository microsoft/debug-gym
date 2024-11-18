from unittest.mock import patch

import pytest

from froggy.utils import (
    HistoryTracker,
    TimeoutException,
    _walk,
    clean_code,
    extract_max_score_from_pytest_output,
    extract_reward_from_pytest_output,
    is_subdirectory,
    load_config,
    make_is_readonly,
    show_line_number,
    str2bool,
    time_limit,
    trim_prompt_messages,
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

    messages = [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "User message 1"},
        {"role": "assistant", "content": "Assistant message 1"},
        {"role": "user", "content": "User message 2"},
        {"role": "assistant", "content": "Assistant message 2"},
        {"role": "user", "content": "User message 3"},
        {"role": "assistant", "content": "Assistant message 3"},
        {"role": "user", "content": "User message 4"},
    ]
    expected = [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "User message 3"},
        {"role": "assistant", "content": "Assistant message 3"},
        {"role": "user", "content": "User message 4"},
    ]
    assert trim_prompt_messages(messages, 65, token_counter) == expected


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


def test_make_is_readonly():
    import atexit
    import tempfile
    from pathlib import Path

    # do the test in a tmp folder
    tempdir = tempfile.TemporaryDirectory(prefix="TestFroggyignore-")
    working_dir = Path(tempdir.name)
    ignore_file = working_dir / ".froggyignore"
    atexit.register(tempdir.cleanup)  # Make sure to cleanup that folder once done.

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
    assert ht.prompt_response_pairs == [
        []
    ]  # at 0-th step, there is no prompt-response pair

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
    ht.save_prompt_response_pairs(
        [("prompt_3_1", "response_3_1"), ("prompt_3_2", "response_3_2")]
    )
    ht.save_prompt_response_pairs([("prompt_4_1", "response_4_1")])
    ht.save_prompt_response_pairs(
        [("prompt_5_1", "response_5_1"), ("prompt_5_2", "response_5_2")]
    )

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

    # output token_usage if it exists
    assert ht.json(3) == {
        "step_id": 3,
        "action": "action4",
        "obs": "obs4",
        "token_usage": 12345,
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


def test_load_config():
    import atexit
    import tempfile
    from pathlib import Path

    import yaml

    # do the test in a tmp folder
    tempdir = tempfile.TemporaryDirectory(prefix="TestLoadConfig-")
    working_dir = Path(tempdir.name)
    config_file = working_dir / "config.yaml"
    atexit.register(tempdir.cleanup)  # Make sure to cleanup that folder once done.

    config_contents = {}
    config_contents["zero_shot"] = {
        "random_seed": 42,
        "max_steps": 100,
        "llm_name": "gpt2",
        "llm_temperature": [0.5],
    }
    config_contents["cot"] = {
        "random_seed": 43,
        "max_steps": 50,
        "cot_style": "standard",
        "llm_name": "gpt20",
        "llm_temperature": [0.3, 0.5],
    }

    # write the config file into yaml
    with open(config_file, "w") as f:
        yaml.dump(config_contents, f)

    # now test
    with patch(
        "sys.argv",
        [
            "config_file",
            str(config_file),
            "--agent",
            "zero_shot",
            "-p",
            "zero_shot.random_seed=123",
            "cot.llm_temperature=[0.8, 0.8]",
            "-v",
            "--debug",
        ],
    ):

        _config, _args = load_config()
    assert _args.agent == "zero_shot"
    assert "zero_shot" in _config.keys()
    assert "cot" in _config.keys()
    assert _config["zero_shot"]["random_seed"] == 123
    assert _config["zero_shot"]["max_steps"] == 100
    assert _config["zero_shot"]["llm_name"] == "gpt2"
    assert _config["zero_shot"]["llm_temperature"] == [0.5]
    assert _config["cot"]["random_seed"] == 43
    assert _config["cot"]["max_steps"] == 50
    assert _config["cot"]["cot_style"] == "standard"
    assert _config["cot"]["llm_name"] == "gpt20"
    assert _config["cot"]["llm_temperature"] == [0.8, 0.8]
    assert _args.debug is True
    assert _args.verbose is True


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
    message_15 = '============================= test session starts ==============================\ncollecting ... collected 15 items\n\ntwelve_days_test.py::TwelveDaysTest::test_eighth_day_eight_maids_a_milking FAILED\ntwelve_days_test.py::TwelveDaysTest::test_eleventh_day_eleven_pipers_piping FAILED\ntwelve_days_test.py::TwelveDaysTest::test_fifth_day_five_gold_rings FAILED\ntwelve_days_test.py::TwelveDaysTest::test_first_day_a_partridge_in_a_pear_tree PASSED\ntwelve_days_test.py::TwelveDaysTest::test_fourth_day_four_calling_birds FAILED\ntwelve_days_test.py::TwelveDaysTest::test_ninth_day_nine_ladies_dancing FAILED\ntwelve_days_test.py::TwelveDaysTest::test_recites_first_three_verses_of_the_song PASSED\ntwelve_days_test.py::TwelveDaysTest::test_recites_the_whole_song PASSED\ntwelve_days_test.py::TwelveDaysTest::test_recites_three_verses_from_the_middle_of_the_song PASSED\ntwelve_days_test.py::TwelveDaysTest::test_second_day_two_turtle_doves FAILED\ntwelve_days_test.py::TwelveDaysTest::test_seventh_day_seven_swans_a_swimming FAILED\ntwelve_days_test.py::TwelveDaysTest::test_sixth_day_six_geese_a_laying FAILED\ntwelve_days_test.py::TwelveDaysTest::test_tenth_day_ten_lords_a_leaping FAILED\ntwelve_days_test.py::TwelveDaysTest::test_third_day_three_french_hens FAILED\ntwelve_days_test.py::TwelveDaysTest::test_twelfth_day_twelve_drummers_drumming FAILED\n\n=================================== FAILURES ===================================\n_____________ TwelveDaysTest.test_eighth_day_eight_maids_a_milking _____________\n\nself = <twelve_days_test.TwelveDaysTest testMethod=test_eighth_day_eight_maids_a_milking>\n\n    def test_eighth_day_eight_maids_a_milking(self):\n        expected = [\n            "On the eighth day of Christmas my true love gave to me: "\n            "eight Maids-a-Milking, "\n            "seven Swans-a-Swimming, "\n            "six Geese-a-Laying, "\n            "five Gold Rings, "\n            "four Calling Birds, "\n            "three French Hens, "\n            "two Turtle Doves, "\n            "and a Partridge in a Pear Tree."\n        ]\n>       self.assertEqual(recite(8, 8), expected)\nE       AssertionError: Lists differ: [\'On [48 chars] me: a Partridge in a Pear Tree, two Turtle Do[128 chars]ng.\'] != [\'On [48 chars] me: eight Maids-a-Milking, seven Swans-a-Swim[128 chars]ee.\']\nE       \nE       First differing element 0:\nE       \'On t[47 chars] me: a Partridge in a Pear Tree, two Turtle Do[127 chars]ing.\'\nE       \'On t[47 chars] me: eight Maids-a-Milking, seven Swans-a-Swim[127 chars]ree.\'\nE       \nE       Diff is 650 characters long. Set self.maxDiff to None to see it.\n\ntwelve_days_test.py:99: AssertionError\n____________ TwelveDaysTest.test_eleventh_day_eleven_pipers_piping _____________\n\nself = <twelve_days_test.TwelveDaysTest testMethod=test_eleventh_day_eleven_pipers_piping>\n\n    def test_eleventh_day_eleven_pipers_piping(self):\n        expected = [\n            "On the eleventh day of Christmas my true love gave to me: "\n            "eleven Pipers Piping, "\n            "ten Lords-a-Leaping, "\n            "nine Ladies Dancing, "\n            "eight Maids-a-Milking, "\n            "seven Swans-a-Swimming, "\n            "six Geese-a-Laying, "\n            "five Gold Rings, "\n            "four Calling Birds, "\n            "three French Hens, "\n            "two Turtle Doves, "\n            "and a Partridge in a Pear Tree."\n        ]\n>       self.assertEqual(recite(11, 11), expected)\nE       AssertionError: Lists differ: [\'On [50 chars] me: a Partridge in a Pear Tree, two Turtle Do[192 chars]ng.\'] != [\'On [50 chars] me: eleven Pipers Piping, ten Lords-a-Leaping[192 chars]ee.\']\nE       \nE       First differing element 0:\nE       \'On t[49 chars] me: a Partridge in a Pear Tree, two Turtle Do[191 chars]ing.\'\nE       \'On t[49 chars] me: eleven Pipers Piping, ten Lords-a-Leaping[191 chars]ree.\'\nE       \nE       Diff is 801 characters long. Set self.maxDiff to None to see it.\n\ntwelve_days_test.py:147: AssertionError\n________________ TwelveDaysTest.test_fifth_day_five_gold_rings _________________\n\nself = <twelve_days_test.TwelveDaysTest testMethod=test_fifth_day_five_gold_rings>\n\n    def test_fifth_day_five_gold_rings(self):\n        expected = [\n            "On the fifth day of Christmas my true love gave to me: "\n            "five Gold Rings, "\n            "four Calling Birds, "\n            "three French Hens, "\n            "two Turtle Doves, "\n            "and a Partridge in a Pear Tree."\n        ]\n>       self.assertEqual(recite(5, 5), expected)\nE       AssertionError: Lists differ: [\'On [47 chars] me: a Partridge in a Pear Tree, two Turtle Do[61 chars]gs.\'] != [\'On [47 chars] me: five Gold Rings, four Calling Birds, thre[61 chars]ee.\']\nE       \nE       First differing element 0:\nE       \'On t[46 chars] me: a Partridge in a Pear Tree, two Turtle Do[60 chars]ngs.\'\nE       \'On t[46 chars] me: five Gold Rings, four Calling Birds, thre[60 chars]ree.\'\nE       \nE       - [\'On the fifth day of Christmas my true love gave to me: a Partridge in a Pear \'\nE       ?                                                          ^^^^^^^  ^^^  ^^^^^^\nE       \nE       + [\'On the fifth day of Christmas my true love gave to me: five Gold Rings, four \'\nE       ?                                                          ^ ++++++ ^^  ^^^^^^^\nE       \nE       -  \'Tree, two Turtle Doves, three French Hens, and four Calling Birds, five Gold \'\nE       -  \'Rings.\']\nE       +  \'Calling Birds, three French Hens, two Turtle Doves, and a Partridge in a \'\nE       +  \'Pear Tree.\']\n\ntwelve_days_test.py:60: AssertionError\n______________ TwelveDaysTest.test_fourth_day_four_calling_birds _______________\n\nself = <twelve_days_test.TwelveDaysTest testMethod=test_fourth_day_four_calling_birds>\n\n    def test_fourth_day_four_calling_birds(self):\n        expected = [\n            "On the fourth day of Christmas my true love gave to me: "\n            "four Calling Birds, "\n            "three French Hens, "\n            "two Turtle Doves, "\n            "and a Partridge in a Pear Tree."\n        ]\n>       self.assertEqual(recite(4, 4), expected)\nE       AssertionError: Lists differ: [\'On [48 chars] me: a Partridge in a Pear Tree, two Turtle Do[44 chars]ds.\'] != [\'On [48 chars] me: four Calling Birds, three French Hens, tw[44 chars]ee.\']\nE       \nE       First differing element 0:\nE       \'On t[47 chars] me: a Partridge in a Pear Tree, two Turtle Do[43 chars]rds.\'\nE       \'On t[47 chars] me: four Calling Birds, three French Hens, tw[43 chars]ree.\'\nE       \nE       - [\'On the fourth day of Christmas my true love gave to me: a Partridge in a \'\nE       ?                                                            ^^^^^^^^^^^  ^^\nE       \nE       + [\'On the fourth day of Christmas my true love gave to me: four Calling Birds, \'\nE       ?                                                           ++++++ ^^  ^^^^^^^^\nE       \nE       -  \'Pear Tree, two Turtle Doves, and three French Hens, four Calling Birds.\']\nE       +  \'three French Hens, two Turtle Doves, and a Partridge in a Pear Tree.\']\n\ntwelve_days_test.py:49: AssertionError\n______________ TwelveDaysTest.test_ninth_day_nine_ladies_dancing _______________\n\nself = <twelve_days_test.TwelveDaysTest testMethod=test_ninth_day_nine_ladies_dancing>\n\n    def test_ninth_day_nine_ladies_dancing(self):\n        expected = [\n            "On the ninth day of Christmas my true love gave to me: "\n            "nine Ladies Dancing, "\n            "eight Maids-a-Milking, "\n            "seven Swans-a-Swimming, "\n            "six Geese-a-Laying, "\n            "five Gold Rings, "\n            "four Calling Birds, "\n            "three French Hens, "\n            "two Turtle Doves, "\n            "and a Partridge in a Pear Tree."\n        ]\n>       self.assertEqual(recite(9, 9), expected)\nE       AssertionError: Lists differ: [\'On [47 chars] me: a Partridge in a Pear Tree, two Turtle Do[149 chars]ng.\'] != [\'On [47 chars] me: nine Ladies Dancing, eight Maids-a-Milkin[149 chars]ee.\']\nE       \nE       First differing element 0:\nE       \'On t[46 chars] me: a Partridge in a Pear Tree, two Turtle Do[148 chars]ing.\'\nE       \'On t[46 chars] me: nine Ladies Dancing, eight Maids-a-Milkin[148 chars]ree.\'\nE       \nE       Diff is 709 characters long. Set self.maxDiff to None to see it.\n\ntwelve_days_test.py:114: AssertionError\n_______________ TwelveDaysTest.test_second_day_two_turtle_doves ________________\n\nself = <twelve_days_test.TwelveDaysTest testMethod=test_second_day_two_turtle_doves>\n\n    def test_second_day_two_turtle_doves(self):\n        expected = [\n            "On the second day of Christmas my true love gave to me: "\n            "two Turtle Doves, "\n            "and a Partridge in a Pear Tree."\n        ]\n>       self.assertEqual(recite(2, 2), expected)\nE       AssertionError: Lists differ: [\'On [41 chars]gave to me: and a Partridge in a Pear Tree, two Turtle Doves.\'] != [\'On [41 chars]gave to me: two Turtle Doves, and a Partridge in a Pear Tree.\']\nE       \nE       First differing element 0:\nE       \'On t[39 chars] gave to me: and a Partridge in a Pear Tree, two Turtle Doves.\'\nE       \'On t[39 chars] gave to me: two Turtle Doves, and a Partridge in a Pear Tree.\'\nE       \nE       - [\'On the second day of Christmas my true love gave to me: and a Partridge in a \'\nE       ?                                                           ^^^^^^^^  ^^^^  ^^^^\nE       \nE       + [\'On the second day of Christmas my true love gave to me: two Turtle Doves, \'\nE       ?                                                           ^^^^^^  ^  ^^^^^^\nE       \nE       -  \'Pear Tree, two Turtle Doves.\']\nE       +  \'and a Partridge in a Pear Tree.\']\n\ntwelve_days_test.py:30: AssertionError\n____________ TwelveDaysTest.test_seventh_day_seven_swans_a_swimming ____________\n\nself = <twelve_days_test.TwelveDaysTest testMethod=test_seventh_day_seven_swans_a_swimming>\n\n    def test_seventh_day_seven_swans_a_swimming(self):\n        expected = [\n            "On the seventh day of Christmas my true love gave to me: "\n            "seven Swans-a-Swimming, "\n            "six Geese-a-Laying, "\n            "five Gold Rings, "\n            "four Calling Birds, "\n            "three French Hens, "\n            "two Turtle Doves, "\n            "and a Partridge in a Pear Tree."\n        ]\n>       self.assertEqual(recite(7, 7), expected)\nE       AssertionError: Lists differ: [\'On [49 chars] me: a Partridge in a Pear Tree, two Turtle Do[105 chars]ng.\'] != [\'On [49 chars] me: seven Swans-a-Swimming, six Geese-a-Layin[105 chars]ee.\']\nE       \nE       First differing element 0:\nE       \'On t[48 chars] me: a Partridge in a Pear Tree, two Turtle Do[104 chars]ing.\'\nE       \'On t[48 chars] me: seven Swans-a-Swimming, six Geese-a-Layin[104 chars]ree.\'\nE       \nE       - [\'On the seventh day of Christmas my true love gave to me: a Partridge in a \'\nE       ?                                                            ^^^^^^^^^^ ^^  --\nE       \nE       + [\'On the seventh day of Christmas my true love gave to me: seven \'\nE       ?                                                            ^ ^^\nE       \nE       -  \'Pear Tree, two Turtle Doves, three French Hens, four Calling Birds, five \'\nE       -  \'Gold Rings, and six Geese-a-Laying, seven Swans-a-Swimming.\']\nE       +  \'Swans-a-Swimming, six Geese-a-Laying, five Gold Rings, four Calling Birds, \'\nE       +  \'three French Hens, two Turtle Doves, and a Partridge in a Pear Tree.\']\n\ntwelve_days_test.py:85: AssertionError\n_______________ TwelveDaysTest.test_sixth_day_six_geese_a_laying _______________\n\nself = <twelve_days_test.TwelveDaysTest testMethod=test_sixth_day_six_geese_a_laying>\n\n    def test_sixth_day_six_geese_a_laying(self):\n        expected = [\n            "On the sixth day of Christmas my true love gave to me: "\n            "six Geese-a-Laying, "\n            "five Gold Rings, "\n            "four Calling Birds, "\n            "three French Hens, "\n            "two Turtle Doves, "\n            "and a Partridge in a Pear Tree."\n        ]\n>       self.assertEqual(recite(6, 6), expected)\nE       AssertionError: Lists differ: [\'On [47 chars] me: a Partridge in a Pear Tree, two Turtle Do[81 chars]ng.\'] != [\'On [47 chars] me: six Geese-a-Laying, five Gold Rings, four[81 chars]ee.\']\nE       \nE       First differing element 0:\nE       \'On t[46 chars] me: a Partridge in a Pear Tree, two Turtle Do[80 chars]ing.\'\nE       \'On t[46 chars] me: six Geese-a-Laying, five Gold Rings, four[80 chars]ree.\'\nE       \nE       - [\'On the sixth day of Christmas my true love gave to me: a Partridge in a Pear \'\nE       ?                                                           ^^ ^^^^^^^^  ^^^^^^^\nE       \nE       + [\'On the sixth day of Christmas my true love gave to me: six Geese-a-Laying, \'\nE       ?                                                          ++++++++++ ^^ ^  ^^\nE       \nE       -  \'Tree, two Turtle Doves, three French Hens, four Calling Birds, and five Gold \'\nE       -  \'Rings, six Geese-a-Laying.\']\nE       +  \'five Gold Rings, four Calling Birds, three French Hens, two Turtle Doves, \'\nE       +  \'and a Partridge in a Pear Tree.\']\n\ntwelve_days_test.py:72: AssertionError\n______________ TwelveDaysTest.test_tenth_day_ten_lords_a_leaping _______________\n\nself = <twelve_days_test.TwelveDaysTest testMethod=test_tenth_day_ten_lords_a_leaping>\n\n    def test_tenth_day_ten_lords_a_leaping(self):\n        expected = [\n            "On the tenth day of Christmas my true love gave to me: "\n            "ten Lords-a-Leaping, "\n            "nine Ladies Dancing, "\n            "eight Maids-a-Milking, "\n            "seven Swans-a-Swimming, "\n            "six Geese-a-Laying, "\n            "five Gold Rings, "\n            "four Calling Birds, "\n            "three French Hens, "\n            "two Turtle Doves, "\n            "and a Partridge in a Pear Tree."\n        ]\n>       self.assertEqual(recite(10, 10), expected)\nE       AssertionError: Lists differ: [\'On [47 chars] me: a Partridge in a Pear Tree, two Turtle Do[170 chars]ng.\'] != [\'On [47 chars] me: ten Lords-a-Leaping, nine Ladies Dancing,[170 chars]ee.\']\nE       \nE       First differing element 0:\nE       \'On t[46 chars] me: a Partridge in a Pear Tree, two Turtle Do[169 chars]ing.\'\nE       \'On t[46 chars] me: ten Lords-a-Leaping, nine Ladies Dancing,[169 chars]ree.\'\nE       \nE       Diff is 751 characters long. Set self.maxDiff to None to see it.\n\ntwelve_days_test.py:130: AssertionError\n_______________ TwelveDaysTest.test_third_day_three_french_hens ________________\n\nself = <twelve_days_test.TwelveDaysTest testMethod=test_third_day_three_french_hens>\n\n    def test_third_day_three_french_hens(self):\n        expected = [\n            "On the third day of Christmas my true love gave to me: "\n            "three French Hens, "\n            "two Turtle Doves, "\n            "and a Partridge in a Pear Tree."\n        ]\n>       self.assertEqual(recite(3, 3), expected)\nE       AssertionError: Lists differ: [\'On [47 chars] me: a Partridge in a Pear Tree, and two Turtl[24 chars]ns.\'] != [\'On [47 chars] me: three French Hens, two Turtle Doves, and [24 chars]ee.\']\nE       \nE       First differing element 0:\nE       \'On t[46 chars] me: a Partridge in a Pear Tree, and two Turtl[23 chars]ens.\'\nE       \'On t[46 chars] me: three French Hens, two Turtle Doves, and [23 chars]ree.\'\nE       \nE       - [\'On the third day of Christmas my true love gave to me: a Partridge in a Pear \'\nE       ?                                                          ^^^^ ^ --- -- ^^^^ ^^\nE       \nE       + [\'On the third day of Christmas my true love gave to me: three French Hens, \'\nE       ?                                                          ^^ ^^^^   ^^^^ ^^^\nE       \nE       -  \'Tree, and two Turtle Doves, three French Hens.\']\nE       +  \'two Turtle Doves, and a Partridge in a Pear Tree.\']\n\ntwelve_days_test.py:39: AssertionError\n___________ TwelveDaysTest.test_twelfth_day_twelve_drummers_drumming ___________\n\nself = <twelve_days_test.TwelveDaysTest testMethod=test_twelfth_day_twelve_drummers_drumming>\n\n    def test_twelfth_day_twelve_drummers_drumming(self):\n        expected = [\n            "On the twelfth day of Christmas my true love gave to me: "\n            "twelve Drummers Drumming, "\n            "eleven Pipers Piping, "\n            "ten Lords-a-Leaping, "\n            "nine Ladies Dancing, "\n            "eight Maids-a-Milking, "\n            "seven Swans-a-Swimming, "\n            "six Geese-a-Laying, "\n            "five Gold Rings, "\n            "four Calling Birds, "\n            "three French Hens, "\n            "two Turtle Doves, "\n            "and a Partridge in a Pear Tree."\n        ]\n>       self.assertEqual(recite(12, 12), expected)\nE       AssertionError: Lists differ: [\'On [49 chars] me: a Partridge in a Pear Tree, two Turtle Do[218 chars]ng.\'] != [\'On [49 chars] me: twelve Drummers Drumming, eleven Pipers P[218 chars]ee.\']\nE       \nE       First differing element 0:\nE       \'On t[48 chars] me: a Partridge in a Pear Tree, two Turtle Do[217 chars]ing.\'\nE       \'On t[48 chars] me: twelve Drummers Drumming, eleven Pipers P[217 chars]ree.\'\nE       \nE       Diff is 857 characters long. Set self.maxDiff to None to see it.\n\ntwelve_days_test.py:165: AssertionError\n=========================== short test summary info ============================\nFAILED twelve_days_test.py::TwelveDaysTest::test_eighth_day_eight_maids_a_milking\nFAILED twelve_days_test.py::TwelveDaysTest::test_eleventh_day_eleven_pipers_piping\nFAILED twelve_days_test.py::TwelveDaysTest::test_fifth_day_five_gold_rings - ...\nFAILED twelve_days_test.py::TwelveDaysTest::test_fourth_day_four_calling_birds\nFAILED twelve_days_test.py::TwelveDaysTest::test_ninth_day_nine_ladies_dancing\nFAILED twelve_days_test.py::TwelveDaysTest::test_second_day_two_turtle_doves\nFAILED twelve_days_test.py::TwelveDaysTest::test_seventh_day_seven_swans_a_swimming\nFAILED twelve_days_test.py::TwelveDaysTest::test_sixth_day_six_geese_a_laying\nFAILED twelve_days_test.py::TwelveDaysTest::test_tenth_day_ten_lords_a_leaping\nFAILED twelve_days_test.py::TwelveDaysTest::test_third_day_three_french_hens\nFAILED twelve_days_test.py::TwelveDaysTest::test_twelfth_day_twelve_drummers_drumming\n========================= 11 failed, 4 passed in 0.05s =========================\n'

    assert extract_max_score_from_pytest_output(message_15) == 15

    message_1 = "============================= test session starts ==============================\ncollecting ... collected 1 item\n \nhello_world_test.py::HelloWorldTest::test_say_hi FAILED\n \n=================================== FAILURES ===================================\n__________________________ HelloWorldTest.test_say_hi __________________________\n \nself = <hello_world_test.HelloWorldTest testMethod=test_say_hi>\n \n    def test_say_hi(self):\n        msg = \"\n\nThis test expects a return of the string 'Hello, World!' \nDid you use print('Hello, World!') by mistake?\"\n>       self.assertEqual(hello(), \"Hello, World!\", msg=msg)\nE       AssertionError: 'Goodbye, Mars!' != 'Hello, World!'\nE       - Goodbye, Mars!\nE       + Hello, World!\nE        : \nE       \nE       This test expects a return of the string 'Hello, World!' \nE       Did you use print('Hello, World!') by mistake?\n \nhello_world_test.py:30: AssertionError\n=========================== short test summary info ============================\nFAILED hello_world_test.py::HelloWorldTest::test_say_hi - AssertionError: 'Go...\n============================== 1 failed in 0.01s ==============================="

    assert extract_max_score_from_pytest_output(message_1) == 1

    message_0 = "============================= here are some random text ==============================="
    with pytest.raises(ValueError, match="No test cases found in the pytest output."):
        extract_max_score_from_pytest_output(message_0)


def test_extract_reward_from_pytest_output():
    message_15 = '============================= test session starts ==============================\ncollecting ... collected 15 items\n\ntwelve_days_test.py::TwelveDaysTest::test_eighth_day_eight_maids_a_milking FAILED\ntwelve_days_test.py::TwelveDaysTest::test_eleventh_day_eleven_pipers_piping FAILED\ntwelve_days_test.py::TwelveDaysTest::test_fifth_day_five_gold_rings FAILED\ntwelve_days_test.py::TwelveDaysTest::test_first_day_a_partridge_in_a_pear_tree PASSED\ntwelve_days_test.py::TwelveDaysTest::test_fourth_day_four_calling_birds FAILED\ntwelve_days_test.py::TwelveDaysTest::test_ninth_day_nine_ladies_dancing FAILED\ntwelve_days_test.py::TwelveDaysTest::test_recites_first_three_verses_of_the_song PASSED\ntwelve_days_test.py::TwelveDaysTest::test_recites_the_whole_song PASSED\ntwelve_days_test.py::TwelveDaysTest::test_recites_three_verses_from_the_middle_of_the_song PASSED\ntwelve_days_test.py::TwelveDaysTest::test_second_day_two_turtle_doves FAILED\ntwelve_days_test.py::TwelveDaysTest::test_seventh_day_seven_swans_a_swimming FAILED\ntwelve_days_test.py::TwelveDaysTest::test_sixth_day_six_geese_a_laying FAILED\ntwelve_days_test.py::TwelveDaysTest::test_tenth_day_ten_lords_a_leaping FAILED\ntwelve_days_test.py::TwelveDaysTest::test_third_day_three_french_hens FAILED\ntwelve_days_test.py::TwelveDaysTest::test_twelfth_day_twelve_drummers_drumming FAILED\n\n=================================== FAILURES ===================================\n_____________ TwelveDaysTest.test_eighth_day_eight_maids_a_milking _____________\n\nself = <twelve_days_test.TwelveDaysTest testMethod=test_eighth_day_eight_maids_a_milking>\n\n    def test_eighth_day_eight_maids_a_milking(self):\n        expected = [\n            "On the eighth day of Christmas my true love gave to me: "\n            "eight Maids-a-Milking, "\n            "seven Swans-a-Swimming, "\n            "six Geese-a-Laying, "\n            "five Gold Rings, "\n            "four Calling Birds, "\n            "three French Hens, "\n            "two Turtle Doves, "\n            "and a Partridge in a Pear Tree."\n        ]\n>       self.assertEqual(recite(8, 8), expected)\nE       AssertionError: Lists differ: [\'On [48 chars] me: a Partridge in a Pear Tree, two Turtle Do[128 chars]ng.\'] != [\'On [48 chars] me: eight Maids-a-Milking, seven Swans-a-Swim[128 chars]ee.\']\nE       \nE       First differing element 0:\nE       \'On t[47 chars] me: a Partridge in a Pear Tree, two Turtle Do[127 chars]ing.\'\nE       \'On t[47 chars] me: eight Maids-a-Milking, seven Swans-a-Swim[127 chars]ree.\'\nE       \nE       Diff is 650 characters long. Set self.maxDiff to None to see it.\n\ntwelve_days_test.py:99: AssertionError\n____________ TwelveDaysTest.test_eleventh_day_eleven_pipers_piping _____________\n\nself = <twelve_days_test.TwelveDaysTest testMethod=test_eleventh_day_eleven_pipers_piping>\n\n    def test_eleventh_day_eleven_pipers_piping(self):\n        expected = [\n            "On the eleventh day of Christmas my true love gave to me: "\n            "eleven Pipers Piping, "\n            "ten Lords-a-Leaping, "\n            "nine Ladies Dancing, "\n            "eight Maids-a-Milking, "\n            "seven Swans-a-Swimming, "\n            "six Geese-a-Laying, "\n            "five Gold Rings, "\n            "four Calling Birds, "\n            "three French Hens, "\n            "two Turtle Doves, "\n            "and a Partridge in a Pear Tree."\n        ]\n>       self.assertEqual(recite(11, 11), expected)\nE       AssertionError: Lists differ: [\'On [50 chars] me: a Partridge in a Pear Tree, two Turtle Do[192 chars]ng.\'] != [\'On [50 chars] me: eleven Pipers Piping, ten Lords-a-Leaping[192 chars]ee.\']\nE       \nE       First differing element 0:\nE       \'On t[49 chars] me: a Partridge in a Pear Tree, two Turtle Do[191 chars]ing.\'\nE       \'On t[49 chars] me: eleven Pipers Piping, ten Lords-a-Leaping[191 chars]ree.\'\nE       \nE       Diff is 801 characters long. Set self.maxDiff to None to see it.\n\ntwelve_days_test.py:147: AssertionError\n________________ TwelveDaysTest.test_fifth_day_five_gold_rings _________________\n\nself = <twelve_days_test.TwelveDaysTest testMethod=test_fifth_day_five_gold_rings>\n\n    def test_fifth_day_five_gold_rings(self):\n        expected = [\n            "On the fifth day of Christmas my true love gave to me: "\n            "five Gold Rings, "\n            "four Calling Birds, "\n            "three French Hens, "\n            "two Turtle Doves, "\n            "and a Partridge in a Pear Tree."\n        ]\n>       self.assertEqual(recite(5, 5), expected)\nE       AssertionError: Lists differ: [\'On [47 chars] me: a Partridge in a Pear Tree, two Turtle Do[61 chars]gs.\'] != [\'On [47 chars] me: five Gold Rings, four Calling Birds, thre[61 chars]ee.\']\nE       \nE       First differing element 0:\nE       \'On t[46 chars] me: a Partridge in a Pear Tree, two Turtle Do[60 chars]ngs.\'\nE       \'On t[46 chars] me: five Gold Rings, four Calling Birds, thre[60 chars]ree.\'\nE       \nE       - [\'On the fifth day of Christmas my true love gave to me: a Partridge in a Pear \'\nE       ?                                                          ^^^^^^^  ^^^  ^^^^^^\nE       \nE       + [\'On the fifth day of Christmas my true love gave to me: five Gold Rings, four \'\nE       ?                                                          ^ ++++++ ^^  ^^^^^^^\nE       \nE       -  \'Tree, two Turtle Doves, three French Hens, and four Calling Birds, five Gold \'\nE       -  \'Rings.\']\nE       +  \'Calling Birds, three French Hens, two Turtle Doves, and a Partridge in a \'\nE       +  \'Pear Tree.\']\n\ntwelve_days_test.py:60: AssertionError\n______________ TwelveDaysTest.test_fourth_day_four_calling_birds _______________\n\nself = <twelve_days_test.TwelveDaysTest testMethod=test_fourth_day_four_calling_birds>\n\n    def test_fourth_day_four_calling_birds(self):\n        expected = [\n            "On the fourth day of Christmas my true love gave to me: "\n            "four Calling Birds, "\n            "three French Hens, "\n            "two Turtle Doves, "\n            "and a Partridge in a Pear Tree."\n        ]\n>       self.assertEqual(recite(4, 4), expected)\nE       AssertionError: Lists differ: [\'On [48 chars] me: a Partridge in a Pear Tree, two Turtle Do[44 chars]ds.\'] != [\'On [48 chars] me: four Calling Birds, three French Hens, tw[44 chars]ee.\']\nE       \nE       First differing element 0:\nE       \'On t[47 chars] me: a Partridge in a Pear Tree, two Turtle Do[43 chars]rds.\'\nE       \'On t[47 chars] me: four Calling Birds, three French Hens, tw[43 chars]ree.\'\nE       \nE       - [\'On the fourth day of Christmas my true love gave to me: a Partridge in a \'\nE       ?                                                            ^^^^^^^^^^^  ^^\nE       \nE       + [\'On the fourth day of Christmas my true love gave to me: four Calling Birds, \'\nE       ?                                                           ++++++ ^^  ^^^^^^^^\nE       \nE       -  \'Pear Tree, two Turtle Doves, and three French Hens, four Calling Birds.\']\nE       +  \'three French Hens, two Turtle Doves, and a Partridge in a Pear Tree.\']\n\ntwelve_days_test.py:49: AssertionError\n______________ TwelveDaysTest.test_ninth_day_nine_ladies_dancing _______________\n\nself = <twelve_days_test.TwelveDaysTest testMethod=test_ninth_day_nine_ladies_dancing>\n\n    def test_ninth_day_nine_ladies_dancing(self):\n        expected = [\n            "On the ninth day of Christmas my true love gave to me: "\n            "nine Ladies Dancing, "\n            "eight Maids-a-Milking, "\n            "seven Swans-a-Swimming, "\n            "six Geese-a-Laying, "\n            "five Gold Rings, "\n            "four Calling Birds, "\n            "three French Hens, "\n            "two Turtle Doves, "\n            "and a Partridge in a Pear Tree."\n        ]\n>       self.assertEqual(recite(9, 9), expected)\nE       AssertionError: Lists differ: [\'On [47 chars] me: a Partridge in a Pear Tree, two Turtle Do[149 chars]ng.\'] != [\'On [47 chars] me: nine Ladies Dancing, eight Maids-a-Milkin[149 chars]ee.\']\nE       \nE       First differing element 0:\nE       \'On t[46 chars] me: a Partridge in a Pear Tree, two Turtle Do[148 chars]ing.\'\nE       \'On t[46 chars] me: nine Ladies Dancing, eight Maids-a-Milkin[148 chars]ree.\'\nE       \nE       Diff is 709 characters long. Set self.maxDiff to None to see it.\n\ntwelve_days_test.py:114: AssertionError\n_______________ TwelveDaysTest.test_second_day_two_turtle_doves ________________\n\nself = <twelve_days_test.TwelveDaysTest testMethod=test_second_day_two_turtle_doves>\n\n    def test_second_day_two_turtle_doves(self):\n        expected = [\n            "On the second day of Christmas my true love gave to me: "\n            "two Turtle Doves, "\n            "and a Partridge in a Pear Tree."\n        ]\n>       self.assertEqual(recite(2, 2), expected)\nE       AssertionError: Lists differ: [\'On [41 chars]gave to me: and a Partridge in a Pear Tree, two Turtle Doves.\'] != [\'On [41 chars]gave to me: two Turtle Doves, and a Partridge in a Pear Tree.\']\nE       \nE       First differing element 0:\nE       \'On t[39 chars] gave to me: and a Partridge in a Pear Tree, two Turtle Doves.\'\nE       \'On t[39 chars] gave to me: two Turtle Doves, and a Partridge in a Pear Tree.\'\nE       \nE       - [\'On the second day of Christmas my true love gave to me: and a Partridge in a \'\nE       ?                                                           ^^^^^^^^  ^^^^  ^^^^\nE       \nE       + [\'On the second day of Christmas my true love gave to me: two Turtle Doves, \'\nE       ?                                                           ^^^^^^  ^  ^^^^^^\nE       \nE       -  \'Pear Tree, two Turtle Doves.\']\nE       +  \'and a Partridge in a Pear Tree.\']\n\ntwelve_days_test.py:30: AssertionError\n____________ TwelveDaysTest.test_seventh_day_seven_swans_a_swimming ____________\n\nself = <twelve_days_test.TwelveDaysTest testMethod=test_seventh_day_seven_swans_a_swimming>\n\n    def test_seventh_day_seven_swans_a_swimming(self):\n        expected = [\n            "On the seventh day of Christmas my true love gave to me: "\n            "seven Swans-a-Swimming, "\n            "six Geese-a-Laying, "\n            "five Gold Rings, "\n            "four Calling Birds, "\n            "three French Hens, "\n            "two Turtle Doves, "\n            "and a Partridge in a Pear Tree."\n        ]\n>       self.assertEqual(recite(7, 7), expected)\nE       AssertionError: Lists differ: [\'On [49 chars] me: a Partridge in a Pear Tree, two Turtle Do[105 chars]ng.\'] != [\'On [49 chars] me: seven Swans-a-Swimming, six Geese-a-Layin[105 chars]ee.\']\nE       \nE       First differing element 0:\nE       \'On t[48 chars] me: a Partridge in a Pear Tree, two Turtle Do[104 chars]ing.\'\nE       \'On t[48 chars] me: seven Swans-a-Swimming, six Geese-a-Layin[104 chars]ree.\'\nE       \nE       - [\'On the seventh day of Christmas my true love gave to me: a Partridge in a \'\nE       ?                                                            ^^^^^^^^^^ ^^  --\nE       \nE       + [\'On the seventh day of Christmas my true love gave to me: seven \'\nE       ?                                                            ^ ^^\nE       \nE       -  \'Pear Tree, two Turtle Doves, three French Hens, four Calling Birds, five \'\nE       -  \'Gold Rings, and six Geese-a-Laying, seven Swans-a-Swimming.\']\nE       +  \'Swans-a-Swimming, six Geese-a-Laying, five Gold Rings, four Calling Birds, \'\nE       +  \'three French Hens, two Turtle Doves, and a Partridge in a Pear Tree.\']\n\ntwelve_days_test.py:85: AssertionError\n_______________ TwelveDaysTest.test_sixth_day_six_geese_a_laying _______________\n\nself = <twelve_days_test.TwelveDaysTest testMethod=test_sixth_day_six_geese_a_laying>\n\n    def test_sixth_day_six_geese_a_laying(self):\n        expected = [\n            "On the sixth day of Christmas my true love gave to me: "\n            "six Geese-a-Laying, "\n            "five Gold Rings, "\n            "four Calling Birds, "\n            "three French Hens, "\n            "two Turtle Doves, "\n            "and a Partridge in a Pear Tree."\n        ]\n>       self.assertEqual(recite(6, 6), expected)\nE       AssertionError: Lists differ: [\'On [47 chars] me: a Partridge in a Pear Tree, two Turtle Do[81 chars]ng.\'] != [\'On [47 chars] me: six Geese-a-Laying, five Gold Rings, four[81 chars]ee.\']\nE       \nE       First differing element 0:\nE       \'On t[46 chars] me: a Partridge in a Pear Tree, two Turtle Do[80 chars]ing.\'\nE       \'On t[46 chars] me: six Geese-a-Laying, five Gold Rings, four[80 chars]ree.\'\nE       \nE       - [\'On the sixth day of Christmas my true love gave to me: a Partridge in a Pear \'\nE       ?                                                           ^^ ^^^^^^^^  ^^^^^^^\nE       \nE       + [\'On the sixth day of Christmas my true love gave to me: six Geese-a-Laying, \'\nE       ?                                                          ++++++++++ ^^ ^  ^^\nE       \nE       -  \'Tree, two Turtle Doves, three French Hens, four Calling Birds, and five Gold \'\nE       -  \'Rings, six Geese-a-Laying.\']\nE       +  \'five Gold Rings, four Calling Birds, three French Hens, two Turtle Doves, \'\nE       +  \'and a Partridge in a Pear Tree.\']\n\ntwelve_days_test.py:72: AssertionError\n______________ TwelveDaysTest.test_tenth_day_ten_lords_a_leaping _______________\n\nself = <twelve_days_test.TwelveDaysTest testMethod=test_tenth_day_ten_lords_a_leaping>\n\n    def test_tenth_day_ten_lords_a_leaping(self):\n        expected = [\n            "On the tenth day of Christmas my true love gave to me: "\n            "ten Lords-a-Leaping, "\n            "nine Ladies Dancing, "\n            "eight Maids-a-Milking, "\n            "seven Swans-a-Swimming, "\n            "six Geese-a-Laying, "\n            "five Gold Rings, "\n            "four Calling Birds, "\n            "three French Hens, "\n            "two Turtle Doves, "\n            "and a Partridge in a Pear Tree."\n        ]\n>       self.assertEqual(recite(10, 10), expected)\nE       AssertionError: Lists differ: [\'On [47 chars] me: a Partridge in a Pear Tree, two Turtle Do[170 chars]ng.\'] != [\'On [47 chars] me: ten Lords-a-Leaping, nine Ladies Dancing,[170 chars]ee.\']\nE       \nE       First differing element 0:\nE       \'On t[46 chars] me: a Partridge in a Pear Tree, two Turtle Do[169 chars]ing.\'\nE       \'On t[46 chars] me: ten Lords-a-Leaping, nine Ladies Dancing,[169 chars]ree.\'\nE       \nE       Diff is 751 characters long. Set self.maxDiff to None to see it.\n\ntwelve_days_test.py:130: AssertionError\n_______________ TwelveDaysTest.test_third_day_three_french_hens ________________\n\nself = <twelve_days_test.TwelveDaysTest testMethod=test_third_day_three_french_hens>\n\n    def test_third_day_three_french_hens(self):\n        expected = [\n            "On the third day of Christmas my true love gave to me: "\n            "three French Hens, "\n            "two Turtle Doves, "\n            "and a Partridge in a Pear Tree."\n        ]\n>       self.assertEqual(recite(3, 3), expected)\nE       AssertionError: Lists differ: [\'On [47 chars] me: a Partridge in a Pear Tree, and two Turtl[24 chars]ns.\'] != [\'On [47 chars] me: three French Hens, two Turtle Doves, and [24 chars]ee.\']\nE       \nE       First differing element 0:\nE       \'On t[46 chars] me: a Partridge in a Pear Tree, and two Turtl[23 chars]ens.\'\nE       \'On t[46 chars] me: three French Hens, two Turtle Doves, and [23 chars]ree.\'\nE       \nE       - [\'On the third day of Christmas my true love gave to me: a Partridge in a Pear \'\nE       ?                                                          ^^^^ ^ --- -- ^^^^ ^^\nE       \nE       + [\'On the third day of Christmas my true love gave to me: three French Hens, \'\nE       ?                                                          ^^ ^^^^   ^^^^ ^^^\nE       \nE       -  \'Tree, and two Turtle Doves, three French Hens.\']\nE       +  \'two Turtle Doves, and a Partridge in a Pear Tree.\']\n\ntwelve_days_test.py:39: AssertionError\n___________ TwelveDaysTest.test_twelfth_day_twelve_drummers_drumming ___________\n\nself = <twelve_days_test.TwelveDaysTest testMethod=test_twelfth_day_twelve_drummers_drumming>\n\n    def test_twelfth_day_twelve_drummers_drumming(self):\n        expected = [\n            "On the twelfth day of Christmas my true love gave to me: "\n            "twelve Drummers Drumming, "\n            "eleven Pipers Piping, "\n            "ten Lords-a-Leaping, "\n            "nine Ladies Dancing, "\n            "eight Maids-a-Milking, "\n            "seven Swans-a-Swimming, "\n            "six Geese-a-Laying, "\n            "five Gold Rings, "\n            "four Calling Birds, "\n            "three French Hens, "\n            "two Turtle Doves, "\n            "and a Partridge in a Pear Tree."\n        ]\n>       self.assertEqual(recite(12, 12), expected)\nE       AssertionError: Lists differ: [\'On [49 chars] me: a Partridge in a Pear Tree, two Turtle Do[218 chars]ng.\'] != [\'On [49 chars] me: twelve Drummers Drumming, eleven Pipers P[218 chars]ee.\']\nE       \nE       First differing element 0:\nE       \'On t[48 chars] me: a Partridge in a Pear Tree, two Turtle Do[217 chars]ing.\'\nE       \'On t[48 chars] me: twelve Drummers Drumming, eleven Pipers P[217 chars]ree.\'\nE       \nE       Diff is 857 characters long. Set self.maxDiff to None to see it.\n\ntwelve_days_test.py:165: AssertionError\n=========================== short test summary info ============================\nFAILED twelve_days_test.py::TwelveDaysTest::test_eighth_day_eight_maids_a_milking\nFAILED twelve_days_test.py::TwelveDaysTest::test_eleventh_day_eleven_pipers_piping\nFAILED twelve_days_test.py::TwelveDaysTest::test_fifth_day_five_gold_rings - ...\nFAILED twelve_days_test.py::TwelveDaysTest::test_fourth_day_four_calling_birds\nFAILED twelve_days_test.py::TwelveDaysTest::test_ninth_day_nine_ladies_dancing\nFAILED twelve_days_test.py::TwelveDaysTest::test_second_day_two_turtle_doves\nFAILED twelve_days_test.py::TwelveDaysTest::test_seventh_day_seven_swans_a_swimming\nFAILED twelve_days_test.py::TwelveDaysTest::test_sixth_day_six_geese_a_laying\nFAILED twelve_days_test.py::TwelveDaysTest::test_tenth_day_ten_lords_a_leaping\nFAILED twelve_days_test.py::TwelveDaysTest::test_third_day_three_french_hens\nFAILED twelve_days_test.py::TwelveDaysTest::test_twelfth_day_twelve_drummers_drumming\n========================= 11 failed, 4 passed in 0.05s =========================\n'

    assert extract_reward_from_pytest_output(message_15) == 4

    message_1 = "============================= test session starts ==============================\ncollecting ... collected 1 item\n \nhello_world_test.py::HelloWorldTest::test_say_hi FAILED\n \n=================================== FAILURES ===================================\n__________________________ HelloWorldTest.test_say_hi __________________________\n \nself = <hello_world_test.HelloWorldTest testMethod=test_say_hi>\n \n    def test_say_hi(self):\n        msg = \"\n\nThis test expects a return of the string 'Hello, World!' \nDid you use print('Hello, World!') by mistake?\"\n>       self.assertEqual(hello(), \"Hello, World!\", msg=msg)\nE       AssertionError: 'Goodbye, Mars!' != 'Hello, World!'\nE       - Goodbye, Mars!\nE       + Hello, World!\nE        : \nE       \nE       This test expects a return of the string 'Hello, World!' \nE       Did you use print('Hello, World!') by mistake?\n \nhello_world_test.py:30: AssertionError\n=========================== short test summary info ============================\nFAILED hello_world_test.py::HelloWorldTest::test_say_hi - AssertionError: 'Go...\n============================== 1 failed in 0.01s ==============================="

    assert extract_reward_from_pytest_output(message_1) == 0

    message_0 = "============================= here are some random text ==============================="

    assert extract_reward_from_pytest_output(message_0) == 0


def test_time_limit():
    import time

    with time_limit(3):
        time.sleep(2)
    assert True

    with pytest.raises(TimeoutException, match="Timed out!"):
        with time_limit(1):
            time.sleep(2)

    with time_limit(None):
        time.sleep(0.2)
    assert True


def test_walk():
    path = "data/terminal_simulator"

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
        "data/terminal_simulator/.froggyignore",
        "data/terminal_simulator/buggy",
        "data/terminal_simulator/code",
        "data/terminal_simulator/README.md",
        "data/terminal_simulator/test_part_1.py",
        "data/terminal_simulator/test_part_2.py",
        "data/terminal_simulator/test.py",
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
        "data/terminal_simulator/.froggyignore",
        "data/terminal_simulator/buggy",
        "data/terminal_simulator/buggy/buggy_code_info_20241031-205241.json",
        "data/terminal_simulator/code",
        "data/terminal_simulator/code/__init__.py",
        "data/terminal_simulator/code/base_simulator.py",
        "data/terminal_simulator/code/run_terminal_simulator.py",
        "data/terminal_simulator/code/some_random_code.py",
        "data/terminal_simulator/code/terminal_simulator.py",
        "data/terminal_simulator/README.md",
        "data/terminal_simulator/test_part_1.py",
        "data/terminal_simulator/test_part_2.py",
        "data/terminal_simulator/test.py",
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
        "data/terminal_simulator/.froggyignore",
        "data/terminal_simulator/buggy",
        "data/terminal_simulator/buggy/buggy_code_info_20241031-205241.json",
        "data/terminal_simulator/code",
        "data/terminal_simulator/code/__init__.py",
        "data/terminal_simulator/code/base_simulator.py",
        "data/terminal_simulator/code/run_terminal_simulator.py",
        "data/terminal_simulator/code/some_random_code.py",
        "data/terminal_simulator/code/terminal_simulator.py",
        "data/terminal_simulator/README.md",
        "data/terminal_simulator/test_part_1.py",
        "data/terminal_simulator/test_part_2.py",
        "data/terminal_simulator/test.py",
    ]
    # sort the list
    path_list.sort()
    expected.sort()
    assert path_list == expected
