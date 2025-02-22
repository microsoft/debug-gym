import re


def strip_ansi(s):
    return re.sub(r"\x1B[@-_][0-?]*[ -/]*[@-~]", "", s)
