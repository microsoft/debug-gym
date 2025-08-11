import copy
from dataclasses import asdict

from debug_gym.gym.envs.env import EnvInfo
from debug_gym.llms.base import LLM, LLMResponse


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
        return (
            self.memory[-self.history_steps :],
            self.prompt_response_pairs[-self.history_steps :],
        )

    def get_all(self):
        return self.memory

    def json(self, game_step=None):
        if len(self.memory) == 0:
            return {}
        if game_step is None:
            # retrieve the most recent step
            game_step = len(self.memory) - 1
        if game_step == 0:
            # initial state
            json_out = {
                "step_id": game_step,
                "reasoning": None,
                "content": None,
                "action": None,  # env reset
                "obs": self.memory[0].step_observation.observation,
                "rewrite_consumed": 0,
                "prompt_response_pairs": None,
            }
        else:
            json_out = {
                "step_id": game_step,
                "content": self.memory[game_step].action_content,
                "reasoning": self.memory[game_step].action_reasoning,
                "action": asdict(self.memory[game_step].action_tool_call),
                "obs": self.memory[game_step].step_observation.observation,
                "rewrite_consumed": self.memory[game_step].rewrite_counter,
            }
            # prompt_response_pairs could be empty for the initial state
            if self.prompt_response_pairs[game_step]:
                json_out["prompt_response_pairs"] = [
                    # doesn't include None values
                    asdict(
                        p,
                        dict_factory=lambda x: {k: v for (k, v) in x if v is not None},
                    )
                    for p in self.prompt_response_pairs[game_step]
                ]

        return json_out

    def score(self):
        return sum([memory.score for memory in self.memory])

    def __len__(self):
        return len(self.memory)

    def clone(self):
        return copy.deepcopy(self)


def build_history_prompt(
    history: HistoryTracker, llm: LLM, reset_prompt_history_after_rewrite: bool = False
):
    _history, _prompt_response_pairs = history.get()
    latest_rewrite_step = 0
    # Find the latest rewrite step when reset_prompt_history_after_rewrite
    if reset_prompt_history_after_rewrite:
        for i in range(len(_history)):
            if _history[i].rewrite_counter == _history[-1].rewrite_counter:
                latest_rewrite_step = i
                break
    _messages = []
    for history_info, response in zip(
        _history[latest_rewrite_step:], _prompt_response_pairs[latest_rewrite_step:]
    ):
        _messages.extend(llm.format_tool_call_history(history_info, response))
    return _messages
