from debug_gym.agents.base_agent import BaseAgent, register_agent
from debug_gym.gym.envs.env import EnvInfo, RepoEnv
from debug_gym.gym.tools.tool import ToolCall
from debug_gym.llms.base import LLM


@register_agent
class AgentSolution(BaseAgent):
    name: str = "solution_agent"

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

        assert any(msg in pdb_observation for msg in expected_messages), (
            f"PDB command did not return expected continue message.\n{pdb_observation}"
        )

    def init(
        self, env: RepoEnv, llm: LLM | None = None, reset_env: bool = True
    ) -> EnvInfo:
        """Initialize the solution agent.

        Args:
            env: The environment to interact with.
            llm: Not used by SolutionAgent (can be None).
            reset_env: Whether to reset the environment (default True).

        Returns:
            The initial EnvInfo after setup.
        """
        self.env = env
        self.llm = llm  # Not used, but stored for compatibility

        if not self._env_implements_apply_gold_patch():
            raise NotImplementedError(
                f"The environment {type(self.env)} is not compatible with SolutionAgent."
                " Check the README.md to see which environments are compatible."
            )

        if reset_env:
            info = self.env.reset()
        else:
            info = self.env.info

        self.logger.info(f"Score: {info.score}/{info.max_score or '-'}")

        # Run PDB sanity checks
        self._run_pdb_sanity_checks(info)

        return info

    def step(self, info: EnvInfo, debug: bool = False) -> EnvInfo:
        """Apply the gold patch and submit.

        Args:
            info: Current environment info.
            debug: Whether to drop into debugger before submit.

        Returns:
            New EnvInfo after applying gold patch and submitting.
        """
        self.env.apply_gold_patch()

        if debug:
            breakpoint()

        action = ToolCall(name="submit", id="submit", arguments={})
        info = self.env.step(action, None, None)

        self.logger.info(f"Score: {info.score}/{info.max_score or '-'}")
        assert info.resolved, (
            "The task is not done after applying the gold patch.\n"
            f"{info.step_observation.observation}"
        )

        return info
