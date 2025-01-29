import argparse
import codecs
import logging
import os
import re
import signal
from contextlib import contextmanager
from os.path import join as pjoin
from pathlib import Path
from typing import Optional

import yaml


def clean_code(code):
    assert isinstance(code, str)
    code_line = unescape(code).split("\n")
    # Remove trailing white spaces with rstrip.
    return "\n".join(line.rstrip() for line in code_line)


def unescape(s):
    return codecs.decode(s, "unicode_escape")


def show_line_number(code_string, code_path=None, breakpoints_state=None):
    # Show line number for each line
    # code_path is the path of the code file in view
    # breakpoints_state is a dict, the keys are "|||".join([file_path, str(line_number)]) and values are breakpoint_command
    # line numbers are 1-indexed
    # line numbers and code lines are separated by a tab

    assert code_string is not None, "code_string should not be None"
    assert isinstance(
        code_string, str
    ), f"code_string should be a string, but got {type(code_string)}"
    code_line = code_string.split("\n")

    output = []
    line_number_digit = len(str(len(code_line) + 1))  # e.g., 999 lines -> 4 digits
    # 1-4 digits: 4
    # 5-8 digits: 8...
    line_number_digit = (line_number_digit - 1) // 4 * 4 + 4
    for i, line in enumerate(code_line):
        has_breakpoint = False
        if code_path is not None and len(breakpoints_state) > 0:
            _key = "|||".join([code_path, str(i + 1)])
            if _key in breakpoints_state.keys():
                has_breakpoint = True
        _tmp = ""
        if has_breakpoint:
            _tmp += "B"
        _tmp = "{:<2}{:>{}} {}".format(_tmp, i + 1, line_number_digit, line)
        output.append(_tmp)
    return "\n".join(output)


def make_is_readonly(full_path, base_dir=None, patterns: list[str] = None):
    if patterns is None:
        patterns = []
    # Ref: gitignore_parser.parse_gitignore
    from gitignore_parser import _normalize_path, handle_negation, rule_from_pattern

    base_dir = _normalize_path(base_dir or os.path.dirname(full_path))

    lines = []
    if os.path.isfile(full_path):
        with open(full_path) as ignore_file:
            lines = ignore_file.readlines()

    lines += patterns

    rules = []
    for i, line in enumerate(lines):
        line = line.rstrip("\n")
        rule = rule_from_pattern(line.rstrip("\n"), base_dir, (full_path, i))
        if rule:
            rules.append(rule)

    if not any(r.negation for r in rules):
        return lambda file_path: any(r.match(file_path) for r in rules)
    else:
        # We have negation rules. We can't use a simple "any" to evaluate them.
        # Later rules override earlier rules.
        return lambda file_path: handle_negation(file_path, rules)


def _walk(path, depth: Optional[int]):
    """recursively list files and directories up to a certain depth"""
    depth = 1e5 if depth is None else depth
    if depth == 0:
        return

    with os.scandir(path) as p:
        for entry in p:
            yield entry.path
            if entry.is_dir() and depth > 0:
                yield from _walk(entry.path, depth - 1)


# Helper class to control boolean flags from the command line with argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        import argparse

        raise argparse.ArgumentTypeError("Boolean value expected.")


def is_subdirectory(path, directory):
    directory = str(directory)
    if not path.startswith(directory):
        path = pjoin(directory, path)
    return not os.path.relpath(path, directory).startswith("..")


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds: Optional[int]):
    if seconds is None:
        yield
        return

    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def cleanup_pytest_output(output):
    # Remove timing, root dir, and platform to avoid randomizing LLM's response.
    res = re.sub(
        r"^Ran \d+ tests? in \d+\.\d+s$",
        "",
        output,
        flags=re.MULTILINE,
    )
    res = re.sub(r"^====*$", "====", res, flags=re.MULTILINE)
    res = re.sub(r"^----*$", "----", res, flags=re.MULTILINE)
    res = re.sub(r"^platform .*\n", "", res, flags=re.MULTILINE)
    res = re.sub(r"^rootdir: .*\n", "", res, flags=re.MULTILINE)
    res = re.sub(r"^plugins: .*\n", "", res, flags=re.MULTILINE)
    res = re.sub(r"^cachedir: .*\n", "", res, flags=re.MULTILINE)

    return res


def extract_max_score_from_pytest_output(output):
    # ... collected 25 items
    # ... collected 1 item
    match = re.search(r"collected (\d+) items?", output)
    if match:
        return max(int(match.group(1)), 1.0)
    else:
        raise ValueError("No test cases found in the pytest output.", output)


def extract_reward_from_pytest_output(output):
    # We extract the number of tests passed from the pytest output.
    # The number of tests passed is the reward.
    # e.g. ========================= 25 failed in 0.06s =========================
    # e.g. ========================= 23 failed, 2 passed in 0.06s =========================
    match = re.search(r"(\d+) passed", output)
    if match:
        return int(match.group(1))

    return 0


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file")
    parser.add_argument("--agent", help="zero_shot, cot, tadpole", default="zero_shot")
    parser.add_argument(
        "--debug", action="store_true", help="Before sending action to the environment."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-v",
        "--verbose",
        dest="logging_level",
        action="store_const",
        const=logging.DEBUG,
        help="Verbose mode",
    )
    group.add_argument(
        "--logging-level",
        dest="logging_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level",
    )
    parser.add_argument(
        "--force-all",
        action="store_true",
        help="Force running all problems even if they are already done.",
    )
    parser.add_argument(
        "--force-failed",
        action="store_true",
        help="Force running only problems that have failed.",
    )
    parser.add_argument(
        "--keep-completed-tasks",
        action="store_true",
        help="Keep displaying completed tasks in the workers panel.",
    )
    parser.add_argument(
        "-vv", "--very-verbose", action="store_true", help="Set logging level to DEBUG"
    )
    parser.add_argument(
        "--force-all",
        action="store_true",
        help="Force running all problems even if they are already done.",
    )
    parser.add_argument(
        "--force-failed",
        action="store_true",
        help="Force running only problems that have failed.",
    )
    parser.add_argument(
        "--keep-completed-tasks",
        action="store_true",
        help="Keep displaying completed tasks in the workers panel.",
    )
    parser.add_argument(
        "-p",
        "--params",
        nargs="+",
        metavar="my.setting=value",
        default=[],
        help="override params of the config file," " e.g. -p 'cot.random_seed=123'",
    )
    args = parser.parse_args()
    assert os.path.exists(args.config_file), "Invalid config file"
    with open(args.config_file) as reader:
        config = yaml.safe_load(reader)

    # Parse overriden params.
    for param in args.params:
        fqn_key, value = param.split("=")
        entry_to_change = config
        keys = fqn_key.split(".")
        for k in keys[:-1]:
            entry_to_change = entry_to_change[k]
        entry_to_change[keys[-1]] = yaml.safe_load(value)

    return config, args
