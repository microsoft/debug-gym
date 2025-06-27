from debug_gym.logger import DebugGymLogger


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
            logger.info(f"[magenta]{m['content']}\n[/magenta]")
        elif role == "user":
            if isinstance(m["content"], list):
                for item in m["content"]:
                    if item["type"] == "tool_result":
                        logger.info(f"[magenta]{item['content']}\n[/magenta]")
                    else:
                        logger.info(f"[cyan]{item}\n[/cyan]")
            else:
                logger.info(f"[cyan]{m['content']}\n[/cyan]")
        elif role == "assistant":
            content = m.get("content", m.get("tool_calls", m))
            logger.info(f"[green]{content}\n[/green]")
        elif role == "system":
            logger.info(f"[yellow]{m['content']}\n[/yellow]")
        else:
            raise ValueError(f"Unknown role: {m['content']}")
