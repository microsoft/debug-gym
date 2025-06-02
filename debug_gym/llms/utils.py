
from termcolor import colored

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
            logger.info(colored(f"{m['content']}\n", "magenta"))
        elif role == "user":
            if isinstance(m["content"], list):
                for item in m["content"]:
                    if item["type"] == "tool_result":
                        logger.info(colored(f"{item["content"]}\n", "magenta"))
                    else:
                        logger.info(colored(f"{item}\n", "cyan"))
            else:
                logger.info(colored(f"{m['content']}\n", "cyan"))
        elif role == "assistant":
            content = m.get("content", m.get("tool_calls", m))
            logger.info(colored(f"{content}\n", "green"))
        elif role == "system":
            logger.info(colored(f"{m['content']}\n", "yellow"))
        else:
            raise ValueError(f"Unknown role: {m['content']}")
