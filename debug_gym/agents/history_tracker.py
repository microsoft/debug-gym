import copy
from dataclasses import asdict

from debug_gym.gym.envs.env import EnvInfo
from debug_gym.llms.base import LLMResponse


class HistoryTracker:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.system_message: dict | None = None
        self.problem_message: dict | None = None
        self.env_initial_observation: EnvInfo | None = None
        self.env_observations: list[EnvInfo] = []
        self.llm_responses: list[LLMResponse | None] = []

    def init(
        self,
        system_message: dict,
        problem_message: dict,
        env_initial_observation: EnvInfo,
    ) -> None:
        self.system_message = system_message
        self.problem_message = problem_message
        self.env_initial_observation = copy.deepcopy(env_initial_observation)
        self.llm_responses = []
        self.env_observations = []

    def step(
        self,
        env_observation: EnvInfo,
        llm_response: LLMResponse,
    ) -> None:
        """llm_responses can be None since the initial state does not have prompt and response"""
        self.env_observations.append(copy.deepcopy(env_observation))
        self.llm_responses.append(copy.deepcopy(llm_response))

    def get(self):
        """Returns the full history of environment observations and LLM responses."""
        return (
            self.env_observations,
            self.llm_responses,
        )

    def json(self, game_step: int | None = None):
        if len(self.env_observations) == 0 and self.env_initial_observation is None:
            return {}

        # Retrieve the most recent step by default.
        game_step = (
            game_step if game_step is not None else len(self.env_observations) - 1
        )
        if game_step < 0 or game_step >= len(self.env_observations):
            raise ValueError(
                f"Invalid game_step: {game_step}; should be between [0, {len(self.env_observations)-1}]."
            )

        if game_step == 0:
            # initial state
            json_out = {
                "step_id": game_step,
                "reasoning": None,
                "content": None,
                "action": None,  # env reset
                "obs": self.env_initial_observation.step_observation.observation,
                "edit_consumed": 0,
                "prompt_response_pairs": None,
                "system_message": self.system_message,
                "problem_message": self.problem_message,
            }
        else:
            json_out = {
                "step_id": game_step,
                "content": self.env_observations[game_step].action_content,
                "reasoning": self.env_observations[game_step].action_reasoning,
                "action": asdict(self.env_observations[game_step].action_tool_call),
                "obs": self.env_observations[game_step].step_observation.observation,
                "edit_consumed": self.env_observations[game_step].edit_counter,
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
        return sum([obs.score for obs in self.env_observations])

    def __len__(self):
        return len(self.env_observations)

    def clone(self):
        return copy.deepcopy(self)
