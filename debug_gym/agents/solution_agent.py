import subprocess

from debug_gym.agents.base_agent import BaseAgent, register_agent
from debug_gym.gym.tools.tool import ToolCall


@register_agent
class AgentSolution(BaseAgent):
    name: str = "solution"

    # def __init__(
    #     self,
    #     config: dict,
    #     env: RepoEnv,
    #     logger: FroggyLogger | None = None,
    # ):
    #     self.config = config
    #     self.env = env
    #     self.logger = logger or FroggyLogger("froggy")
    #     self.llm = instantiate_llm(self.config, logger=self.logger)
    #     self._uuid = self.config.get("uuid", str(uuid.uuid4()))
    #     self._output_path = pjoin(self.config["output_path"], self._uuid)

    # os.makedirs(self._output_path, exist_ok=True)

    def run(self, task_name=None, debug=False):
        self.history.reset()
        info = self.env.reset(options={"task_name": task_name})
        self.history.step(info)

        if info.done is True:
            return True

        highscore = info.score

        for step in self.logger.tqdm(range(self.config["max_steps"])):
            highscore = max(highscore, info.score)
            self.logger.info(
                f"Score: {info.score}/{info.max_score} ({info.score/info.max_score:.1%}) [Best: {highscore}]".format(
                    info.score
                )
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

            assert (
                info.done is True
            ), "The task should be done after applying the gold patch."

            if info.done or info.rewrite_counter >= self.config["max_rewrite_steps"]:
                self.logger.info(
                    f"Score: {info.score}/{info.max_score} ({info.score/info.max_score:.1%})".format(
                        info.score
                    )
                )
                break

        return info.done
