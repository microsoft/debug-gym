import copy
from dataclasses import asdict

from debug_gym.gym.envs.env import EnvInfo
from debug_gym.llms.base import LLM, LLMResponse


class HistoryTracker:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.system_prompt: dict | None = None
        self.problem_prompt: dict | None = None
        self.env_initial_observation: EnvInfo | None = None
        self.env_observations: list[EnvInfo] = []
        self.llm_responses: list[LLMResponse | None] = []

    def init(
        self, system_prompt: str, problem_prompt: str, env_initial_observation: EnvInfo
    ) -> None:
        self.system_prompt = system_prompt
        self.problem_prompt = problem_prompt
        self.env_initial_observation = copy.deepcopy(env_initial_observation)
        self.llm_responses = []
        self.env_observations = []

    def step(
        self,
        llm_response: LLMResponse,
        env_observation: EnvInfo,
    ) -> None:
        """llm_responses can be None since the initial state does not have prompt and response"""
        self.llm_responses.append(copy.deepcopy(llm_response))
        self.env_observations.append(copy.deepcopy(env_observation))

    def get(self):
        # return the history_steps latest steps
        return (
            self.env_observations,
            self.llm_responses,
        )

    def json(self, game_step=None):
        if len(self.env_observations) == 0 and self.env_init is None:
            return {}

        if game_step >= len(self.env_observations):
            raise ValueError(
                f"Invalid game_step: {game_step}. Max step: {len(self.env_observations)-1}"
            )

        if game_step is None:
            # retrieve the most recent step
            game_step = len(self.env_observations) - 1

        if game_step == 0:
            # initial state
            json_out = {
                "step_id": game_step,
                "reasoning": None,
                "content": None,
                "action": None,  # env reset
                "obs": self.env_initial_observation.step_observation.observation,
                "rewrite_consumed": 0,
                "prompt_response_pairs": None,
                "system_prompt": self.system_prompt,
                "problem_prompt": self.problem_prompt,
            }
        else:
            json_out = {
                "step_id": game_step,
                "content": self.env_observations[game_step].action_content,
                "reasoning": self.env_observations[game_step].action_reasoning,
                "action": asdict(self.env_observations[game_step].action_tool_call),
                "obs": self.env_observations[game_step].step_observation.observation,
                "rewrite_consumed": self.env_observations[game_step].rewrite_counter,
            }
            # prompt_response_pairs could be empty for the initial state
            if self.llm_responses[game_step]:
                json_out["prompt_response_pairs"] = [
                    # doesn't include None values
                    asdict(
                        self.llm_responses[game_step],
                        dict_factory=lambda x: {k: v for (k, v) in x if v is not None},
                    )
                ]
        return json_out

    def score(self):
        return sum([memory.score for memory in self.env_observations])

    def __len__(self):
        return len(self.env_observations)

    def clone(self):
        return copy.deepcopy(self)
