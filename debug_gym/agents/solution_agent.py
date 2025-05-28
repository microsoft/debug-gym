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
