from debug_gym.agents.base_agent import BaseAgent, register_agent
from debug_gym.gym.tools.tool import ToolCall


@register_agent
class AgentSolution(BaseAgent):
    name: str = "solution_agent"

    def _report_progress(self, task_name, info, status):
        self.logger.report_progress(
            problem_id=task_name,
            step=1,
            total_steps=1,
            score=getattr(info, "score", 0),
            max_score=getattr(info, "max_score", 0),
            status=status,
        )

    def _env_implements_apply_gold_patch(self):
        """Fail early if the environment does not implement apply_gold_patch."""
        return hasattr(self.env, "apply_gold_patch")

    def run(self, env, llm=None, debug=False):
        self.env = env
        info = None
        try:
            if not self._env_implements_apply_gold_patch():
                raise NotImplementedError(
                    f"The environment {type(self.env)} is not compatible with SolutionAgent."
                    " Check the README.md to see which environments are compatible."
                )

            info = self.env.reset()

            if info.resolved is True:
                self._report_progress(env.task_name, info, "resolved")
                return True

            self.logger.info(f"Score: {info.score}/{info.max_score or '-'}")

            if env.has_tool("pdb"):
                # Make a simple pdb call to make sure it is working.
                action = ToolCall(
                    name="pdb", id="pdb", arguments={"command": "help help"}
                )
                pdb_help_info = self.env.step(action, None, None)
                assert "h(elp)" in pdb_help_info.step_observation.observation, (
                    "PDB command did not return expected help message.\n"
                    f"{pdb_help_info.step_observation.observation}"
                )

                # Send a pdb continue command, and check the output matches the one from env.reset.
                action = ToolCall(
                    name="pdb", id="pdb", arguments={"command": "continue"}
                )
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
            self._report_progress(env.task_name, info, "resolved")
        except Exception:
            self._report_progress(env.task_name, info, "error")
            raise
        return info.resolved
