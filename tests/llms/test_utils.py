from debug_gym.llms.utils import print_messages


def test_print_messages(logger_mock):
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "system", "content": "System message"},
    ]
    print_messages(messages, logger_mock)
    assert logger_mock._log_history == [
        "[cyan]Hello[/cyan]",
        "[cyan]Hi[/cyan]",
        "[yellow]System message[/yellow]",
    ]
