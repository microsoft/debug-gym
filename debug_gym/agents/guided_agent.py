import logging

from debug_gym.agents.base_agent import register_agent
from debug_gym.agents.rewrite_agent import RewriteAgent
from debug_gym.llms.base import LLM
from debug_gym.logger import DebugGymLogger


@register_agent
class GuidedRewriteAgent(RewriteAgent):
    name: str = "guided_agent"

    def try_rewrite(self, task_name):
        # make a copy of the env for the llm
        cloned_env = self.env.clone()

        # Only keep the rewrite tool in the cloned env
        for tool in cloned_env.tools:
            if tool.name != "rewrite":
                cloned_env.remove_tool(tool.name)

        # Reset the cloned environment and replay the history.
        info = cloned_env.reset(options={"task_name": task_name})
        # replay the history up to the current step
        for step in self.history.get_all():
            assert not step.done
            info = cloned_env.step(step.action)

        prompt = self.build_prompt(info)
        response = self.llm(prompt, info.tools)
        info = cloned_env.step(response.response)

        return info.done

    def run(self, task_name=None, debug=False):
        self.llm.logger = DebugGymLogger(name="LLM", level=logging.ERROR)
        self.human = LLM.instantiate(llm_name="human", logger=self.logger)

        self.history.reset()
        info = self.env.reset(options={"task_name": task_name})
        # initial state does not have prompt and response
        self.history.step(info, None)

        if info.done is True:
            # msg = "Environment started with entrypoint passing without errors."
            return True

        highscore = info.score

        for step in self.logger.tqdm(range(self.config["max_steps"])):
            highscore = max(highscore, info.score)
            self.logger.info(
                f"Score: {info.score}/{info.max_score} ({info.score/info.max_score:.1%}) [Best: {highscore}]"
            )

            llm_done = self.try_rewrite(task_name)
            if llm_done:
                self.logger.info(
                    f"*** The rewrite-only agent with {self.llm.model_name} managed to solve the task with the current context. ***"
                )
                break

            # If the LLM did not manage to solve the task, we continue with the guided approach.
            prompt = self.build_prompt(info)
            human_response = self.human(prompt, info.tools)

            if debug:
                breakpoint()

            # step the environment with the human response
            info = self.env.step(human_response.response)
            # log the human response
            self.history.step(info, human_response)

            if info.done:
                self.logger.info(
                    "You managed to provide the patch that solves the task before the LLM. Congrats!"
                )
                break

        return info.done
