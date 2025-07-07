from debug_gym.agents.base_agent import BaseAgent, register_agent
from debug_gym.gym.tools.tool import ToolCall


@register_agent
class AgentSolution(BaseAgent):
    name: str = "solution_agent"

    def run(self, task_name=None, debug=False):
        self.history.reset()

        info = self.env.reset(options={"task_name": task_name})
        self.history.step(info)

        if info.done is True:
            return True

        self.logger.info(
            f"Score: {info.score}/{info.max_score} ({info.score/info.max_score:.1%})"
        )

        # Make a simple pdb call to make sure it is working.
        action = ToolCall(name="pdb", id="pdb", arguments={"command": "help help"})
        pdb_help_info = self.env.step(action, "")
        assert (
            "h(elp)" in pdb_help_info.step_observation.observation
        ), f"PDB command did not return expected help message.\n{pdb_help_info.step_observation.observation}"

        # Send a pdb continue command, and check the output matches the one from env.reset.
        action = ToolCall(name="pdb", id="pdb", arguments={"command": "continue"})
        pdb_continue_info = self.env.step(action, "")

        assert (
            "Reached the end of the program. Restarting the debugging session."
            in pdb_continue_info.step_observation.observation
        ) or (
            info.step_observation.observation.splitlines()[-1]
            in pdb_continue_info.step_observation.observation
        ), f"PDB command did not return expected continue message.\n{pdb_continue_info.step_observation.observation}"

        try:
            self.env.apply_gold_patch()
        except NotImplementedError as e:
            self.logger.error(
                f"The environment {type(self.env)} is not compatible with SolutionAgent"
                "Check the README.md to see which environments are compatible."
            )
            raise

        if debug:
            breakpoint()

        action = ToolCall(name="eval", id="eval", arguments={})
        info = self.env.step(action, "")

        self.history.step(info)

        self.logger.info(
            f"Score: {info.score}/{info.max_score} ({info.score/info.max_score:.1%})"
        )
        assert (
            info.done
        ), f"The task is not done after applying the gold patch.\n{info.step_observation.observation}"

        return info.done
