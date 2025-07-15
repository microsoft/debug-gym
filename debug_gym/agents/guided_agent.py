import logging

from debug_gym.agents.base_agent import register_agent
from debug_gym.agents.rewrite_agent import RewriteAgent
from debug_gym.gym.entities import Event
from debug_gym.gym.tools.tool import ToolCall
from debug_gym.gym.tools.toolbox import Toolbox
from debug_gym.llms.base import LLM
from debug_gym.logger import DebugGymLogger


@register_agent
class GuidedRewriteAgent(RewriteAgent):
    name: str = "guided_agent"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create a dedicated env for the guided rewrite agent.
        self.llm.logger = DebugGymLogger(
            name="LLM",
            level=logging.DEBUG,
            log_dir=self.logger.log_file.parent,
            icon="🤖",
        )

        # Create a human interface for the guided agent.
        self.logger.level = logging.DEBUG
        self.logger.icon = "👤"
        self.human = LLM.instantiate(llm_name="human", logger=self.logger)

    def try_rewrite_and_rollback(self, last_info):
        prompt = self.build_prompt(last_info)

        # Git commit the current state before trying to rewrite.
        self.env.terminal.run("git add . && git commit -m 'Before rewrite attempt'")

        # Remove all tools except the rewrite tool.
        tools = [tool for tool in last_info.tools if tool.name == "rewrite"]
        response = self.llm(prompt, tools)
        self.llm.logger.info(f"LLM response: {response.response}")
        self.llm.logger.info(f"LLM tool: {response.tool}")

        # Temporarily disable the REWRITE_SUCCESS event.
        self.env.event_hooks.mute(Event.REWRITE_SUCCESS)
        info_after_rewrite = self.env.step(response.tool)
        info = self.env.step(ToolCall(id="eval", name="eval", arguments={}))
        self.env.event_hooks.unmute(Event.REWRITE_SUCCESS)

        self.llm.logger.info(f"LLM observation: {info.eval_observation.observation}.")

        # Rollback any changes made by the LLM.
        self.env.terminal.run("git reset --hard HEAD")

        return info.done

    def run(self, task_name=None, debug=False):
        step = 0
        max_steps = self.config["max_steps"]
        try:
            self.history.reset()
            info = self.env.reset(options={"task_name": task_name})
            # initial state does not have prompt and response
            self.history.step(info, None)

            # First make sure git is setup correctly.
            self.env.terminal.run(
                "git init && git config user.name 'debug-gym' && git config user.email '<>'"
            )

            if info.done is True:
                self.logger.report_progress(
                    problem_id=task_name,
                    step=1,
                    total_steps=1,
                    score=info.score,
                    max_score=info.max_score,
                    status="resolved",
                )
                return True

            highscore = info.score

            for step in range(max_steps):
                self.logger.info(f"\n{'='*20} STEP {step+1} {'='*20}\n")
                highscore = max(highscore, info.score)
                self.logger.info(
                    f"Step: {step} | Score: {info.score}/{info.max_score} ({info.score/info.max_score:.1%}) [Best: {highscore}]"
                )

                llm_done = self.try_rewrite_and_rollback(info)
                if llm_done:
                    msg = f"[green]*** The rewrite-only agent with {self.llm.model_name} managed to solve the task with the current context. ***[/green]"
                    self.llm.logger.error(msg)
                    break
                else:
                    msg = f"[red]*** The rewrite-only agent with {self.llm.model_name} failed to solve the task with the current context. ***[/red]"
                    self.llm.logger.error(msg)

                # If the LLM did not manage to solve the task, we continue with the guided approach.
                prompt = self.build_prompt(info)
                human_response = self.human(prompt, info.tools)
                if not llm_done:
                    msg = f"[red]*** The rewrite-only agent with {self.llm.model_name} failed to solve the task with the current context. ***[/red]"
                    self.llm.logger.error(msg)

                if debug:
                    breakpoint()

                # step the environment with the human response
                info = self.env.step(human_response.tool)
                # log the human response
                self.history.step(info, human_response)

                if info.done:
                    self.logger.info(
                        "You managed to provide the patch that solves the task before the LLM. Congrats!"
                    )
                    # early stop, set current step and total steps to be the same
                    self.logger.report_progress(
                        problem_id=task_name,
                        step=step + 1,
                        total_steps=step + 1,
                        score=info.score,
                        max_score=info.max_score,
                        status="resolved" if info.done else "unresolved",
                    )
                    break
                # keep progress bar running until max_steps is reached
                self.logger.report_progress(
                    problem_id=task_name,
                    step=step + 1,
                    total_steps=max_steps + 1,
                    score=info.score,
                    max_score=info.max_score,
                    status="running",
                )
            # max_steps was reached, task was either resolved or unresolved
            self.logger.report_progress(
                problem_id=task_name,
                step=step + 1,
                total_steps=step + 1,
                score=info.score,
                max_score=info.max_score,
                status="resolved" if info.done else "unresolved",
            )

            return info.done
        except Exception:
            # report any error that happens during the run
            self.logger.report_progress(
                problem_id=task_name,
                step=step + 1,
                total_steps=step + 1,
                score=info.score if info else 0,
                max_score=info.max_score if info else 1,
                status="error",
            )
            raise
