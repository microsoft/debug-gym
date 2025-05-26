import codecs
import os
import re
from os.path import join as pjoin
from pathlib import Path
from typing import Callable


def clean_code(code):
    assert isinstance(code, str)
    code_line = unescape(code).split("\n")
    # Remove trailing white spaces with rstrip.
    return "\n".join(line.rstrip() for line in code_line)


def unescape(s):
    try:
        # First, try the normal unescape
        result = codecs.decode(s, "unicode_escape")
        # Test if it can be encoded to UTF-8 (which will happen during JSON encoding)
        result.encode("utf-8")
        return result
    except UnicodeEncodeError:
        # If it contains surrogate pairs that can't be encoded to UTF-8,
        # replace them with the Unicode replacement character (U+FFFD)
        result = codecs.decode(s, "unicode_escape")
        return result.encode("utf-8", errors="replace").decode("utf-8")


def get_code_length(code_string):
    # Get the number of lines in the code string
    assert isinstance(
        code_string, str
    ), f"code_string should be a string, but got {type(code_string)}"
    code_line = code_string.split("\n")
    return len(code_line)


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


def make_file_matcher(pattern_file, base_dir=None, patterns: list[str] = None):
    """
    Creates a file matcher function based on ignore patterns from a file and additional patterns.

    Args:
        pattern_file (str): Path to the file containing gitignore-like patterns.
        base_dir (str, optional): Base directory to resolve relative paths. Defaults to the directory of the pattern_file.
        patterns (list[str], optional): Additional patterns to include. Defaults to an empty list.

    Returns:
        function: A function that takes a file path as input and returns True if the file matches any of the patterns, False otherwise.
    """
    if patterns is None:
        patterns = []
    # Ref: gitignore_parser.parse_gitignore
    from gitignore_parser import _normalize_path, handle_negation, rule_from_pattern

    base_dir = _normalize_path(base_dir or os.path.dirname(pattern_file))

    lines = []
    if os.path.isfile(pattern_file):
        with open(pattern_file) as ignore_file:
            lines = ignore_file.readlines()

    lines += patterns

    rules = []
    for i, line in enumerate(lines):
        line = line.rstrip("\n")
        rule = rule_from_pattern(line.rstrip("\n"), base_dir, (pattern_file, i))
        if rule:
            rules.append(rule)

    if not any(r.negation for r in rules):
        return lambda file_path: any(r.match(file_path) for r in rules)
    else:
        # We have negation rules. We can't use a simple "any" to evaluate them.
        # Later rules override earlier rules.
        return lambda file_path: handle_negation(file_path, rules)


def create_ignore_file(
    filepath: str | Path, patterns: list[str] = [], include_gitignore: bool = True
):
    """
    Creates a file at the specified `filepath` containing gitignore-like patterns.

    Files and directories matching the patterns in that file will be treated differently.
    E.g., Files in a `.debugignore` file will be ignored by the environment.
          Files in a `.debugreadonly` file will be marked as readonly.

    Args:
        filepath (str): The file path where to create the ignore file.
        patterns (list[str]): A list of patterns to include in the ignore file.
        include_gitignore (bool): If True, includes the contents of an existing .gitignore file
                                  in the ignore file. Default is True.
    """
    path = Path(filepath)
    gitignore_file = path.parent / ".gitignore"
    if include_gitignore and gitignore_file.exists():
        with open(gitignore_file) as f:
            patterns = patterns + f.read().splitlines()

    with open(path, "w") as f:
        f.write("\n".join(patterns + [path.name]))


def _walk(path, depth: int | None = None, skip: Callable | None = None):
    """recursively list files and directories up to a certain depth"""
    depth = 1e5 if depth is None else depth
    if depth == 0:
        return

    with os.scandir(path) as p:
        for entry in sorted(p, key=lambda x: x.name):
            if skip and skip(entry.path):
                continue

            yield Path(entry)
            if entry.is_dir() and depth > 0:
                yield from _walk(entry.path, depth=depth - 1, skip=skip)


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
        raise ValueError("Cannot extract max score from pytest output.", output)


def extract_reward_from_pytest_output(output):
    # We extract the number of tests passed from the pytest output.
    # The number of tests passed is the reward.
    # e.g. ========================= 25 failed in 0.06s =========================
    # e.g. ========================= 23 failed, 2 passed in 0.06s =========================
    match = re.search(r"(\d+) passed", output)
    if match:
        return int(match.group(1))

    return 0
