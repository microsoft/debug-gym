import copy
import json
from dataclasses import asdict

from example_agent.llm_api import LLMResponse
from froggy.envs.env import EnvInfo
from froggy.utils import unescape


class HistoryTracker:
    def __init__(self, history_steps: int) -> None:
        self.history_steps = history_steps
        self.reset()

    def reset(self) -> None:
        self.memory: list[EnvInfo] = []
        self.prompt_response_pairs: list[LLMResponse | None] = []

    def step(self, new_info: EnvInfo, llm_response: LLMResponse | None = None) -> None:
        """llm_response can be None since the initial state does not have prompt and response"""
        self.memory.append(copy.deepcopy(new_info))
        self.prompt_response_pairs.append([copy.deepcopy(llm_response)])

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

    new_messages.append(messages[-1])
    new_length += message_lengths[-1]
    if new_length > context_length:
        # just keep the last message, remove the system message
        new_messages = [messages[-1]]
        new_length = message_lengths[-1]
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
