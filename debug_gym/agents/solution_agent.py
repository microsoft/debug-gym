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
            score=info.score,
            max_score=info.max_score,
            status=status,
        )

    def run(self, task_name=None, debug=False):
        self.history.reset()
        info = self.env.reset(options={"task_name": task_name})
        self.history.step(info)

        if info.done is True:
            self._report_progress(task_name, info, "done")
            return True

        self.logger.info(
            f"Score: {info.score}/{info.max_score} ({info.score/info.max_score:.1%})"
        )
        try:
            # Make a simple pdb call to make sure it is working.
            action = ToolCall(name="pdb", id="pdb", arguments={"command": "help help"})
            pdb_help_info = self.env.step(action, "")
            assert "h(elp)" in pdb_help_info.step_observation.observation, (
                "PDB command did not return expected help message.\n"
                f"{pdb_help_info.step_observation.observation}"
            )

            # Send a pdb continue command, and check the output matches the one from env.reset.
            action = ToolCall(name="pdb", id="pdb", arguments={"command": "continue"})
            pdb_continue_info = self.env.step(action, "")

            assert (
                "Reached the end of the program. Restarting the debugging session."
                in pdb_continue_info.step_observation.observation
            ) or (
                info.step_observation.observation.splitlines()[-1]
                in pdb_continue_info.step_observation.observation
            ), (
                "PDB command did not return expected continue message.\n"
                f"{pdb_continue_info.step_observation.observation}"
            )

            self.env.apply_gold_patch()

            if debug:
                breakpoint()

            action = ToolCall(name="eval", id="eval", arguments={})
            info = self.env.step(action, "")

            self.history.step(info)

            self.logger.info(
                f"Score: {info.score}/{info.max_score} ({info.score/info.max_score:.1%})"
            )
            assert info.done, (
                "The task is not done after applying the gold patch.\n"
                f"{info.step_observation.observation}"
            )
            self._report_progress(task_name, info, "done")
            return info.done
        except NotImplementedError:
            self._report_progress(task_name, info, "failed")
            self.logger.error(
                f"The environment {type(self.env)} is not compatible with SolutionAgent"
                "Check the README.md to see which environments are compatible."
            )
            raise
        except AssertionError:
            self._report_progress(task_name, info, "failed")
            raise
