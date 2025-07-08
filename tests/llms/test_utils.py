from debug_gym.llms.utils import print_messages


def test_print_messages(logger_mock):
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "system", "content": "System message"},
        {"role": "tool", "content": "Tool message"},
        {
            "role": "user",
            "content": [{"type": "tool_result", "content": "Tool result"}],
        },
        {"role": "assistant", "tool_calls": ["Tool call 1", "Tool call 2"]},
        {"role": "assistant", "tool_calls": "Single tool call"},
        {"role": "system", "content": 12345},
    ]
    print_messages(messages, logger_mock)
    assert logger_mock._log_history == [
        "[cyan]Hello[/cyan]",
        "[cyan]Hi[/cyan]",
        "[yellow]System message[/yellow]",
        "[magenta]Tool message[/magenta]",
        "[magenta]Tool result[/magenta]",
        "[cyan]Tool call 1[/cyan]",
        "[cyan]Tool call 2[/cyan]",
        "[cyan]Single tool call[/cyan]",
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
