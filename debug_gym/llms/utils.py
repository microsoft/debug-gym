from rich.markup import escape

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
            logger.info(f"[magenta]{escape(m['content'])}\n[/magenta]")
        elif role == "user":
            if isinstance(m["content"], list):
                for item in m["content"]:
                    if item["type"] == "tool_result":
                        logger.info(
                            f"[magenta]{escape(str(item['content']))}\n[/magenta]"
                        )
                    else:
                        logger.info(f"[cyan]{escape(str(item))}\n[/cyan]")
            else:
                logger.info(f"[cyan]{escape(str(m['content']))}\n[/cyan]")
        elif role == "assistant":
            content = m.get("content", m.get("tool_calls", m))
            logger.info(f"[green]{escape(str(content))}\n[/green]")
        elif role == "system":
            logger.info(f"[yellow]{escape(str(m['content']))}\n[/yellow]")
        else:
            raise ValueError(f"Unknown role: {escape(str(m['content']))}")
