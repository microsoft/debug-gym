from debug_gym.llms.utils import print_messages


def test_print_messages(logger_mock):
    messages = [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": "Hi",
            "tool_calls": ["Tool call 1", "Tool call 2"],
        },
        {"role": "tool", "content": "Tool message"},
        {
            "role": "user",
            "content": [{"type": "tool_result", "content": "Tool result"}],
        },
        {"role": "assistant", "tool_calls": [{"key": 3, "key2": "value"}]},
        {"role": "system", "content": 12345},
    ]
    print_messages(messages, logger_mock)
    assert logger_mock._log_history == [
        "[yellow]System message[/yellow]",
        "[magenta]Hello[/magenta]",
        "[cyan]Hi[/cyan]",
        "[cyan]Tool call: Tool call 1[/cyan]",
        "[cyan]Tool call: Tool call 2[/cyan]",
        "[green]Tool message[/green]",
        "[magenta]Tool result[/magenta]",
        "[cyan]Tool call: {'key': 3, 'key2': 'value'}[/cyan]",
        "[yellow]12345[/yellow]",
    ]


def test_print_messages_unknown_role(logger_mock):
    bad_message = [{"role": "unknown", "content": "bad"}]
    try:
        print_messages(bad_message, logger_mock)
    except ValueError as e:
        assert "Unknown role" in str(e)
    else:
        assert False, "ValueError not raised for unknown role"
