import argparse
import logging
import os

import yaml
from termcolor import colored

from debug_gym.logger import DebugGymLogger


def print_messages(messages: list[dict], logger: DebugGymLogger):
    """Print messages coloring each role differently.
    Messages is a list of dictionaries with role and content keys."""
    for m in messages:
        if m["role"] == "user":
            logger.info(colored(f"{m['content']}\n", "cyan"))
        elif m["role"] == "assistant":
            logger.info(colored(f"{m['content']}\n", "green"))
        elif m["role"] == "system":
            logger.info(colored(f"{m['content']}\n", "yellow"))
        else:
            raise ValueError(f"Unknown role: {m['content']}")


def merge_messages(messages: list[dict]) -> list[dict]:
    """Merge consecutive messages with same role into one message."""
    messages_out = []
    to_merge = []

    def merge():
        content = "\n\n".join(m["content"] for m in to_merge if m["content"])
        if content:
            messages_out.append({"role": current_role, "content": content})

    current_role = None
    for message in messages:
        if current_role == message["role"]:
            to_merge.append(message)
        else:
            merge()  # merge the previous messages
            current_role = message["role"]
            to_merge = [message]
    merge()  # merge the last messages
    return messages_out


def trim(text: str, max_length: int, count_tokens: callable, where: str = "middle"):
    # Get an approximate number of characters per token ratio in the text.
    nb_tokens = count_tokens(text=text)
    if nb_tokens == 0:
        return text

    chars_per_token = len(text) / nb_tokens
    # Adjust the max_length based on the chars_per_token ratio.
    max_length = int(max_length * chars_per_token)

    if len(text) <= max_length:
        return text

    ellipsis = "…"
    if max_length <= len(ellipsis):
        return ellipsis[:max_length]

    match where:
        case "end":
            return text[: max_length - len(ellipsis)] + ellipsis
        case "start":
            return ellipsis + text[-(max_length - len(ellipsis)) :]
        case "middle":
            half_length = (max_length - len(ellipsis)) // 2
            return text[:half_length] + ellipsis + text[-half_length:]
        case _:
            raise ValueError(f"Invalid value for `where`: {where!r}.")

    return text


def trim_prompt_messages(
    messages: list[dict], context_length: int, count_tokens: callable
):
    # Trim message content to context length
    # messages: list of dict, each dict has keys "content" and "role"
    # context_length: int, maximum number of tokens
    # count_tokens: function, count the number of tokens in a string
    # messages should not be empty
    assert len(messages) > 0, "messages should not be empty"
    # all messages should be dictionaries with keys "content" and "role"
    assert all(
        isinstance(item, dict) and "content" in item and "role" in item
        for item in messages
    ), 'all messages should be dictionaries with keys "content" and "role"'
    # the last message should be from the user
    assert messages[-1]["role"] == "user", "the last message should be from the user"
    # if two consecutive messages are from the same role, they should be merged
    assert all(
        messages[i]["role"] != messages[i + 1]["role"] for i in range(len(messages) - 1)
    ), "if two consecutive messages are from the same role, they should be merged first"
    # context_length should be non-negative
    assert context_length >= 0, "context_length should be non-negative"

    message_lengths = [count_tokens(item["content"]) for item in messages]
    total_length = sum(message_lengths)
    if total_length <= context_length:
        return messages

    # keep the first (system) message and last (user) message if possible
    new_messages, new_length = [], 0
    if messages[0]["role"] == "system":
        new_messages.append(messages[0])
        new_length += message_lengths[0]

    assert (
        new_length <= context_length
    ), f"The system message exceeds: {new_length} > {context_length}!"

    new_messages.append(dict(messages[-1]))
    new_length += message_lengths[-1]
    if new_length > context_length:
        token_space_remaining = context_length - (new_length - message_lengths[-1])
        # just keep the system message and trim the last message
        new_messages[-1]["content"] = trim(
            new_messages[-1]["content"],
            token_space_remaining,
            count_tokens=count_tokens,
            where="middle",
        )
    else:
        # adding back the messages in between (from latest to earliest)
        start = 1 if messages[0]["role"] == "system" else 0
        for i in range(len(messages) - 2, start, -1):
            if new_length + message_lengths[i] > context_length:
                break
            if start == 0:
                new_messages = [messages[i]] + new_messages
            else:
                new_messages = new_messages[:1] + [messages[i]] + new_messages[1:]
            new_length += message_lengths[i]

    return new_messages


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file")
    parser.add_argument(
        "--agent",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Break before sending action to the environment.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-v",
        "--verbose",
        dest="logging_level",
        action="store_const",
        const=logging.INFO,
        help="Verbose mode",
        default=logging.WARNING,
    )
    group.add_argument(
        "-vv",
        "--very-verbose",
        dest="logging_level",
        action="store_const",
        const=logging.DEBUG,
        help="Verbose mode",
        default=logging.WARNING,
    )
    group.add_argument(
        "--logging-level",
        dest="logging_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level",
    )
    parser.add_argument(
        "--force-all",
        action="store_true",
        help="Force running all problems even if they are already done.",
    )
    parser.add_argument(
        "--force-failed",
        action="store_true",
        help="Force running only problems that have failed.",
    )
    parser.add_argument(
        "--keep-completed-tasks",
        action="store_true",
        help="Keep displaying completed tasks in the workers panel.",
    )
    parser.add_argument(
        "-p",
        "--params",
        nargs="+",
        metavar="my.setting=value",
        default=[],
        help="override params of the config file,"
        " e.g. -p 'rewrite_only.random_seed=123'",
    )
    args = parser.parse_args()
    assert os.path.exists(args.config_file), "Invalid config file"
    with open(args.config_file) as reader:
        config = yaml.safe_load(reader)

    # Parse overriden params.
    for param in args.params:
        fqn_key, value = param.split("=")
        entry_to_change = config
        keys = fqn_key.split(".")
        for k in keys[:-1]:
            entry_to_change = entry_to_change[k]
        entry_to_change[keys[-1]] = yaml.safe_load(value)

    available_agents = [item for item in list(config.keys()) if item != "base"]

    if not args.agent:
        # pick first agent
        args.agent = available_agents[0]
    elif args.agent not in available_agents:
        raise ValueError(
            f"Invalid agent: {args.agent}. Available agents: {available_agents}"
        )

    if "base" in config:
        # base config is specified (shared across agents)
        return_config = config["base"]
        agent_specific_config = config[args.agent]
        for key in agent_specific_config:
            # override base config with agent specific config
            return_config[key] = agent_specific_config[key]
    else:
        # base config is not specified
        return_config = config[args.agent]

    # assume agent type is the key if not specified by the user
    if not return_config.get("agent_type"):
        return_config["agent_type"] = args.agent

    return return_config, args
