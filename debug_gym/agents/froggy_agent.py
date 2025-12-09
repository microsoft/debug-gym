from dataclasses import dataclass

from debug_gym.agents.base_agent import (
    AGENT_REGISTRY,
    AgentArgs,
    BaseAgent,
    register_agent,
)
from debug_gym.gym.envs.env import EnvInfo


@dataclass
class FroggyAgentArgs(AgentArgs):
    max_rewrite_steps: int = -1
    show_current_breakpoints: bool = False


@register_agent
class FroggyAgent(BaseAgent):
    name: str = "froggy_agent"
    args_class = FroggyAgentArgs
    system_prompt: str = "{{ agent._default_system_prompt(info) }}"

    def should_stop(self, step: int, info: EnvInfo):
        should_stop, reason = super().should_stop(step, info)
        if (
            self.args.max_rewrite_steps >= 0
            and info.rewrite_counter > self.args.max_rewrite_steps
        ):
            should_stop = True
            reason = "max_rewrite_steps reached"
        return should_stop, reason

    def shortcut_features(self):
        features = []
        if self.env.has_tool("pdb"):
            if self.args.show_current_breakpoints:
                features.append(
                    "The environment will show the current breakpoints in the system prompt."
                )
            if self.env.get_tool("pdb").persistent_breakpoints:
                features.append(
                    "The environment will automatically restore existing breakpoints "
                    "when a new PDB session is started (e.g., after a rewrite)."
                )
            if self.env.get_tool("pdb").auto_list:
                features.append(
                    "After every valid PDB tool calling, the environment will "
                    "automatically call the PDB tool again with a `list .` command, "
                    "which will show the code around the current frame."
                )
        return features

    def _default_system_prompt(self, info) -> str:
        """Return the default system prompt as pretty JSON.
        Trimmed to fit within the token limit."""

        system_prompt_dict = {
            "Instructions": info.instructions,
        }

        if self.args.show_current_breakpoints:
            system_prompt_dict["Current breakpoints"] = info.current_breakpoints

        if (
            info
            and info.eval_observation
            and getattr(info.eval_observation, "observation", "")
        ):
            system_prompt_dict["Evaluation output of current code"] = self.trim_message(
                info.eval_observation.observation,
                max_length_percentage=0.8,
                where="middle",
            )

        shortcut_features = self.shortcut_features()
        if shortcut_features:
            system_prompt_dict["Shortcut features"] = shortcut_features

        return self.to_pretty_json(system_prompt_dict)


# Backward compatibility for configs still referencing "froggy".
# TODO: remove when all configs and consumers migrate to "froggy_agent".
AGENT_REGISTRY.setdefault("froggy", FroggyAgent)
