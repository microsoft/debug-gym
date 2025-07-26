from debug_gym.logger import DebugGymLogger, log_with_color


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
            log_with_color(logger, m["content"], "magenta")
        elif role == "user":
            if isinstance(m["content"], list):
                for item in m["content"]:
                    if item["type"] == "tool_result":
                        log_with_color(logger, str(item["content"]), "magenta")
                    else:
                        log_with_color(logger, str(item), "magenta")
            else:
                log_with_color(logger, str(m["content"]), "magenta")
        elif role == "assistant":
            content = m.get("content")
            if content:
                if isinstance(content, list):
                    for item in content:
                        log_with_color(logger, str(item), "cyan")
                else:
                    log_with_color(logger, str(content), "cyan")
            if "tool_calls" in m:
                for tool_call in m["tool_calls"]:
                    log_with_color(logger, f"Tool call: {tool_call}", "cyan")
        elif role == "system":
            log_with_color(logger, str(m["content"]), "yellow")
        else:
            raise ValueError(f"Unknown role: {m['content']}")
