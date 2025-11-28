from dataclasses import dataclass
from typing import Any, Dict

from debug_gym.agents.base_agent import AgentArgs, BaseAgent, register_agent
from debug_gym.agents.history_tracker import build_history_prompt


@dataclass
class FroggyAgentArgs(AgentArgs):
    show_directory_tree: int = 0
    show_current_breakpoints: bool = False
    reset_prompt_history_after_rewrite: bool = False
    n_rewrites_before_pdb: int = 0


@register_agent
class FroggyAgent(BaseAgent):
    name: str = "froggy"

    def __init__(
        self,
        agent_args: FroggyAgentArgs | Dict[str, Any],
        *args,
        **kwargs,
    ):

        agent_args = (
            FroggyAgentArgs.from_dict(agent_args)
            if isinstance(agent_args, dict)
            else agent_args
        )
        super().__init__(agent_args, *args, **kwargs)

    def build_history_prompt(self):
        messages = build_history_prompt(
            self.history,
            self.llm,
            self.args.reset_prompt_history_after_rewrite,
        )
        return messages

    def _auto_eval_on_rewrite(self):
        """Check if auto eval on rewrite is enabled."""
        try:
            return self.env.get_tool("eval").auto_eval_on_rewrite
        except KeyError:
            return False  # no eval tool

    def shortcut_features(self):
        features = []
        if self._auto_eval_on_rewrite():
            features.append(
                "After successful rewrites, the environment will automatically "
                "call the Eval tool to evaluate the rewritten code. Therefore, "
                "you do not need to call the Eval tool yourself. The evaluation "
                "output will be updated automatically in the system prompt."
            )
        if self.args.show_directory_tree:
            features.append(
                "The environment will show the directory tree of the repository in the system prompt."
            )
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
            "Overall task": self.system_prompt,
            "Instructions": info.instructions,
        }

        if self.args.show_directory_tree > 0:
            system_prompt_dict["Repo directory tree"] = self.trim_message(
                self.env.workspace.display_files(self.args.show_directory_tree),
                max_length_percentage=0.1,
                where="end",
            )

        if self.args.show_current_breakpoints:
            system_prompt_dict["Current breakpoints"] = info.current_breakpoints

        if self._auto_eval_on_rewrite():
            system_prompt_dict["Evaluation output of current code"] = self.trim_message(
                info.eval_observation.observation,
                max_length_percentage=0.8,
                where="middle",
            )

        shortcut_features = self.shortcut_features()
        if shortcut_features:
            system_prompt_dict["Shortcut features"] = shortcut_features

        return self.to_pretty_json(system_prompt_dict)
