import subprocess

from debug_gym.agents.base_agent import BaseAgent, register_agent
from debug_gym.gym.tools.tool import ToolCall


@register_agent
class AgentSolution(BaseAgent):
    name: str = "solution"

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
        pdb_help_info = self.env.step(action)
        assert (
            "h(elp)" in pdb_help_info.step_observation.observation
        ), "PDB command did not return expected help message."

        # Send a pdb continue command, and check the output matches the one from env.reset.
        action = ToolCall(name="pdb", id="pdb", arguments={"command": "continue"})
        pdb_continue_info = self.env.step(action)

        assert (
            "Reached the end of the program. Restarting the debugging session."
            in pdb_continue_info.step_observation.observation
        ) or (
            info.step_observation.observation.splitlines()[-1]
            in pdb_continue_info.step_observation.observation
        ), "PDB command did not return expected continue message."

        try:
            self.logger.info(f"Applying gold patch to {self.env.working_dir}.")
            command = f"git -C {self.env.working_dir} apply {getattr(self.env, "git_apply_args", "")} -"
            cmd_out = subprocess.run(
                command.split(),
                input=self.env.gold_patch,
                text=True,
                check=True,
                capture_output=True,
            )
            self.logger.info("Patch applied successfully.")
            self.logger.debug(cmd_out)
        except subprocess.CalledProcessError as e:
            self.logger.debug(e)
            self.logger.debug(f"stderr: {e.stderr}")
            self.logger.debug(f"stdout: {e.stdout}")
            raise

        if debug:
            breakpoint()

        action = ToolCall(name="eval", id="eval", arguments={})
        info = self.env.step(action)

        self.history.step(info)

        self.logger.info(
            f"Score: {info.score}/{info.max_score} ({info.score/info.max_score:.1%})"
        )
        assert info.done, "The task should be done after applying the gold patch."

        return info.done
