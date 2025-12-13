from typing import Any, Dict

from debug_gym.agents.base_agent import BaseAgent, register_agent
from debug_gym.gym.envs.env import EnvInfo, RepoEnv
from debug_gym.gym.tools.tool import ToolCall
from debug_gym.llms.base import LLM, LLMResponse


@register_agent
class AgentSolution(BaseAgent):
    """Agent that applies the gold patch and submits - used for testing environments."""

    name: str = "solution_agent"

    def __init__(
        self,
        llm: LLM | None = None,
        **kwargs,
    ):
        super().__init__(llm=llm, **kwargs)

    def _env_implements_apply_gold_patch(self):
        """Fail early if the environment does not implement apply_gold_patch."""
        return hasattr(self.env, "apply_gold_patch")

    def _run_pdb_sanity_checks(self, info: EnvInfo):
        """Run PDB sanity checks if PDB tool is available."""
        if not self.env.has_tool("pdb"):
            return

        # Make a simple pdb call to make sure it is working.
        action = ToolCall(name="pdb", id="pdb", arguments={"command": "help help"})
        pdb_help_info = self.env.step(action, None, None)
        assert "h(elp)" in pdb_help_info.step_observation.observation, (
            "PDB command did not return expected help message.\n"
            f"{pdb_help_info.step_observation.observation}"
        )

        # Send a pdb continue command, and check the output matches the one from env.reset.
        action = ToolCall(name="pdb", id="pdb", arguments={"command": "continue"})
        pdb_continue_info = self.env.step(action, None, None)

        pdb_observation = pdb_continue_info.step_observation.observation
        expected_messages = [
            "Reached the end of the program. Restarting the debugging session.",
            "Uncaught exception. Entering post mortem debugging",
        ]
        reset_observation = info.step_observation.observation
        if reset_observation.splitlines():
            expected_messages.append(reset_observation.splitlines()[-1])

        assert any(
            msg in pdb_observation for msg in expected_messages
        ), f"PDB command did not return expected continue message.\n{pdb_observation}"

    def step(self, info: EnvInfo) -> EnvInfo:
        tool_call = ToolCall(name="submit", id="submit", arguments={})
        return LLMResponse([], tool=tool_call)

    def execute_action(self, llm_response, **kwargs):
        self.env.apply_gold_patch()
        info = self.env.step(llm_response.tool, None, None)
        return info

    def init(self, info: EnvInfo) -> None:
        self._run_pdb_sanity_checks(info)
