from rich.markup import escape

from debug_gym.logger import DebugGymLogger


def _log_with_color(logger: DebugGymLogger, message: str, color: str):
    """Log a message with a specific color, escape it
    for Rich, and mark it as already escaped for DebugGymLogger."""
    logger.info(
        f"[{color}]{escape(message)}[/]{color}",
        extra={"already_escaped": True},
    )


def print_messages(messages: list[dict], logger: DebugGymLogger):
    """Print messages coloring each role differently.
    Colors:
        green: selected tool or assistant messages
        magenta: result of tool calls
        cyan: user messages
        yellow: system message
    """

    for m in messages:
        role = m["role"]
        if role == "tool":
            _log_with_color(logger, m["content"], "magenta")
        elif role == "user":
            if isinstance(m["content"], list):
                for item in m["content"]:
                    if item["type"] == "tool_result":
                        _log_with_color(logger, str(item["content"]), "magenta")
                    else:
                        _log_with_color(logger, str(item), "cyan")
            else:
                _log_with_color(logger, str(m["content"]), "cyan")
        elif role == "assistant":
            content = m.get("content", m.get("tool_calls", m))
            if isinstance(content, list):
                for item in content:
                    _log_with_color(logger, str(item), "cyan")
            else:
                _log_with_color(logger, str(content), "cyan")
        elif role == "assistant":
            content = m.get("content", m.get("tool_calls", m))
            _log_with_color(logger, str(content), "green")
        elif role == "system":
            _log_with_color(logger, str(m["content"]), "yellow")
        else:
            raise ValueError(f"Unknown role: {m['content']}")
