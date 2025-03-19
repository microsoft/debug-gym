import argparse
import copy
import json
import logging
import os
from dataclasses import asdict

import yaml

from debug_gym.agents.llm_api import LLMResponse
from debug_gym.gym.envs.env import EnvInfo
from debug_gym.gym.utils import unescape


class HistoryTracker:
    def __init__(self, history_steps: int) -> None:
        self.history_steps = history_steps
        self.reset()

    def reset(self) -> None:
        self.memory: list[EnvInfo] = []
        self.prompt_response_pairs: list[LLMResponse | None] = []

    def step(
        self,
        new_info: EnvInfo,
        llm_responses: list[LLMResponse] | LLMResponse | None = None,
    ) -> None:
        """llm_responses can be None since the initial state does not have prompt and response"""
        self.memory.append(copy.deepcopy(new_info))

        llm_responses = llm_responses or []
        if not isinstance(llm_responses, list):
            llm_responses = [llm_responses]
        self.prompt_response_pairs.append(copy.deepcopy(llm_responses))

    def get(self):
        # return the history_steps latest steps
        return self.memory[-self.history_steps :]

    def get_all(self):
        return self.memory

    def json(self, game_step=None, include_prompt_response_pairs=False):
        if len(self.memory) == 0:
            return {}
        if game_step is None:
            # retrieve the most recent step
            game_step = len(self.memory) - 1
        if game_step == 0:
            # initial state
            json_out = {
                "step_id": game_step,
                "action": None,  # env reset
                "obs": self.memory[0].step_observation.observation,
            }
            if include_prompt_response_pairs:
                json_out["prompt_response_pairs"] = None
        else:
            json_out = {
                "step_id": game_step,
                "action": self.memory[game_step].action,
                "obs": self.memory[game_step].step_observation.observation,
            }
            # prompt_response_pairs could be empty for the initial state
            prp = self.prompt_response_pairs[game_step]

            if prp and include_prompt_response_pairs:
                json_out["prompt_response_pairs"] = [
                    # doesn't include None values
                    asdict(
                        p,
                        dict_factory=lambda x: {k: v for (k, v) in x if v is not None},
                    )
                    for p in prp
                ]

        return json_out

    def score(self):
        return sum([memory.score for memory in self.memory])

    def __len__(self):
        return len(self.memory)

    def clone(self):
        return copy.deepcopy(self)

    def filter_out(self, actions: list[str]):
        history = HistoryTracker(self.history_steps)
        for info, llm_response in zip(self.memory, self.prompt_response_pairs):
            if info.action not in actions:
                history.step(info, llm_response)

        return history


def trim(text: str, max_length: int, token_counter: callable, where: str = "middle"):

    # Get an approximate number of characters per token ratio in the text.
    nb_tokens = token_counter(text=text)
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
    messages: list[dict], context_length: int, token_counter: callable
):
    # Trim message content to context length
    # messages: list of dict, each dict has keys "content" and "role"
    # context_length: int, maximum number of tokens
    # token_counter: function, count the number of tokens in a string
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

    message_lengths = [token_counter(text=item["content"]) for item in messages]
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
    ), f"The system message execeeds: {new_length} > {context_length}!"

    new_messages.append(dict(messages[-1]))
    new_length += message_lengths[-1]
    if new_length > context_length:
        token_space_remaining = context_length - (new_length - message_lengths[-1])
        # just keep the system message and trim the last message
        new_messages[-1]["content"] = trim(
            new_messages[-1]["content"],
            token_space_remaining,
            token_counter=token_counter,
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


def build_history_conversation(
    history: HistoryTracker, reset_prompt_history_after_rewrite: bool = False
):
    _history = history.get()
    # Find the latest rewrite step
    if len(_history) == 0 or reset_prompt_history_after_rewrite is False:
        latest_rewrite_step = 0
    else:
        for i in range(len(_history)):
            if _history[i].rewrite_counter == _history[-1].rewrite_counter:
                latest_rewrite_step = i
                break
    _messages = []
    for history_info in _history[latest_rewrite_step:]:
        if history_info.action is not None:
            _messages.append({"role": "assistant", "content": f"{history_info.action}"})
        _messages.append(
            {"role": "user", "content": f"{history_info.step_observation.observation}"}
        )
    return _messages


def build_history_non_conversation(
    history: HistoryTracker, reset_prompt_history_after_rewrite: bool = False
):
    _history = history.get()
    # Find the latest rewrite step
    if len(_history) == 0 or reset_prompt_history_after_rewrite is False:
        latest_rewrite_step = 0
    else:
        for i in range(len(_history)):
            if _history[i].rewrite_counter == _history[-1].rewrite_counter:
                latest_rewrite_step = i
                break
    _history_prompt = []
    _history = _history[latest_rewrite_step:]
    for _i, history_info in enumerate(_history):
        _m = {
            "step": _i,
            "command": (None if history_info.action is None else history_info.action),
            "stdout": history_info.step_observation.observation,
        }
        _history_prompt.append(_m)
    return _history_prompt


def build_history_prompt(
    history: HistoryTracker,
    use_conversational_prompt: bool = True,
    reset_prompt_history_after_rewrite: bool = False,
):
    messages = []
    if use_conversational_prompt is True:
        conversation_history = build_history_conversation(
            history, reset_prompt_history_after_rewrite
        )
        if len(conversation_history) == 0:
            messages.append(
                {
                    "role": "user",
                    "content": "No history of command and terminal outputs.",
                }
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": f"History of command and terminal outputs (the last {(len(conversation_history) + 1) // 2} steps):",
                }
            )
            messages.extend(conversation_history)
    else:
        history_prompt = build_history_non_conversation(
            history, reset_prompt_history_after_rewrite
        )
        if len(history_prompt) == 0:
            prompt = ["No history of command and terminal outputs."]
        else:
            prompt = [
                f"History of command and terminal outputs (the last {len(history_prompt)} steps):"
            ]
            prompt += ["\n" + unescape(json.dumps(history_prompt, indent=4)) + "\n"]
        messages.append({"role": "user", "content": "\n".join(prompt)})
    return messages


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
