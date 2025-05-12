import copy
from dataclasses import asdict

from debug_gym.agents.llm_api import LLMResponse
from debug_gym.gym.envs.env import EnvInfo


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
        self.memory.append(new_info)  # was deepcopy needed?

        llm_responses = llm_responses or []
        if not isinstance(llm_responses, list):
            llm_responses = [llm_responses]
        self.prompt_response_pairs.append(copy.deepcopy(llm_responses))

    def get(self):
        # return the history_steps latest steps
        return (
            self.memory[-self.history_steps :],
            self.prompt_response_pairs[-self.history_steps :],
        )

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


def build_history_conversation(
    history: HistoryTracker, reset_prompt_history_after_rewrite: bool = False
):
    _history, _prompt_response_pairs = history.get()
    # Find the latest rewrite step
    if len(_history) == 0 or reset_prompt_history_after_rewrite is False:
        latest_rewrite_step = 0
    else:
        for i in range(len(_history)):
            if _history[i].rewrite_counter == _history[-1].rewrite_counter:
                latest_rewrite_step = i
                break
    _messages = []
    for history_info, response in zip(
        _history[latest_rewrite_step:], _prompt_response_pairs[latest_rewrite_step:]
    ):
        if hasattr(response[0].response, "role"):  # GPT
            _messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        response[0].response.tool_calls[0]
                    ],  # ChatCompletionMessageToolCall(id='call_jFkY53qCEQLKDXQqdDd7AZyL', function=Function(arguments='{"command":"b 13"}', name='pdb'), type='function')
                }
            )
            _messages.append(
                {
                    "role": "tool",
                    "tool_call_id": history_info.action.id,  # 'call_jFkY53qCEQLKDXQqdDd7AZyL'
                    "name": history_info.action.name,  # 'pdb'
                    "content": f"{history_info.step_observation.observation}",  # 'Breakpoint 1 at /tmp/RepoEnv-9uqllb7j/hangman.py:13\nlist .\n1  ->\t"""The pytest entry point."""\r\n  2  \t\r\n  3  \tfrom __future__ import annotations\r\n  4  \t\r\n  5  \timport pytest\r\n  6  \t\r\n  7  \t\r\n  8  \tif __name__ == "__main__":\r\n  9  \t    raise SystemExit(pytest.console_main())\r\n[EOF]'
                }
            )
        else:  # Claude
            _messages.append(
                {
                    "role": "assistant",  # "assistant"
                    "content": [
                        {
                            "type": "tool_use",
                            "id": history_info.action.id,  # 'toolu_01SdR84CsnTKRpdH4zwFjvGj'
                            "name": history_info.action.name,  # 'view'
                            "input": history_info.action.arguments,  # {'path': 'hangman_test.py'}
                        }
                    ],
                }
            )
            _messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": history_info.action.id,  # 'toolu_01SdR84CsnTKRpdH4zwFjvGj'
                            "content": f"{history_info.step_observation.observation}",  # 'Viewing `hangman_test.py`. The file is read-only, it is not editable.'
                        }
                    ],
                }
            )
    return _messages


def build_history_prompt(
    history: HistoryTracker,
    reset_prompt_history_after_rewrite: bool = False,
):
    messages = []
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
    return messages
