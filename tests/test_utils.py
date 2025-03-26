from debug_gym.utils import strip_ansi


def test_strip_ansi():
    message = "\x1b[31mThis is a test message.\x1b[0m"
    assert strip_ansi(message) == "This is a test message."
    message = "\x1b[32mThis is another test message.\x1b[0m"
    assert strip_ansi(message) == "This is another test message."
