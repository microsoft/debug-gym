import logging

from debug_gym.agents.base_agent import register_agent
from debug_gym.agents.history_tracker import build_history_prompt
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

        # Initialize the different LLM rewriters.
        self.llms = [
            LLM.instantiate(
                llm_name=llm_name,
                logger=DebugGymLogger(
                    name=llm_name,
                    level=logging.DEBUG,
                    log_dir=self.logger.log_file.parent,
                    icon="🤖",
                ),
            )
            for llm_name in self.config["llms"]
        ]

        # Create logger for the main guide, e.g. (a human).
        self.llm.logger = DebugGymLogger(
            name=self.config["llm_name"],
            level=logging.DEBUG,
            log_dir=self.logger.log_file.parent,
            icon="👤",
        )

    def build_prompt(self, info, llm):
        messages = []
        messages.extend(self.build_system_prompt(info))
        messages.extend(self.build_history_prompt(llm))
        messages.extend(self.build_question_prompt())
        return messages

    def build_history_prompt(self, llm):
        messages = build_history_prompt(
            self.history,
            llm,
            self.config["reset_prompt_history_after_rewrite"],
        )
        return messages

    def try_rewrite_and_rollback(self, llm, last_info):
        prompt = self.build_prompt(last_info, llm)

        # Git commit the current state before trying to rewrite.
        self.env.terminal.run("git add . && git commit -m 'Before rewrite attempt'")

        # Remove all tools except the rewrite tool.
        tools = [tool for tool in last_info.tools if tool.name == "rewrite"]
        response = llm(prompt, tools)
        llm.logger.info(f"LLM response: {response.response}")
        llm.logger.info(f"LLM tool: {response.tool}")

        # Temporarily disable the REWRITE_SUCCESS event.
        self.env.event_hooks.mute(Event.REWRITE_SUCCESS)
        info_after_rewrite = self.env.step(response.tool)
        llm_info = self.env.step(ToolCall(id="eval", name="eval", arguments={}))
        self.env.event_hooks.unmute(Event.REWRITE_SUCCESS)

        llm.logger.info(f"LLM observation: {llm_info.eval_observation.observation}.")

        if not llm_info.done:
            # Rollback any changes made by the LLM if it hasn't solved the task yet.
            self.env.terminal.run("git reset --hard HEAD")

        return llm_info

    def run(self, task_name=None, debug=False):
        step = 0
        max_steps = self.config["max_steps"]
        info = None
        llm_done = False
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

                solved = None
                for llm in self.llms:
                    llm_info = self.try_rewrite_and_rollback(llm, info)
                    if llm_info.done:
                        solved = llm_info
                        msg = f"[green]*** The rewrite-only agent with {llm.model_name} managed to solve the task with the current context. ***[/green]"
                        llm.logger.error(msg)
                    else:
                        msg = f"[red]*** The rewrite-only agent with {llm.model_name} failed to solve the task with the current context. ***[/red]"
                        llm.logger.error(msg)

                if solved is not None:
                    llm_info = solved
                    break

                # If the LLM did not manage to solve the task, we continue with the guided approach.
                prompt = self.build_prompt(info, self.llm)
                guide_response = self.llm(prompt, info.tools)

                if debug:
                    breakpoint()

                # step the environment with the guide response
                info = self.env.step(guide_response.tool)
                # log the guide response
                self.history.step(info, guide_response)

                if info.done:
                    self.logger.info(
                        "You managed to provide the patch that solves the task before the LLM. Congrats!"
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
            # self.logger.report_progress(
            #     problem_id=task_name,
            #     step=step + 1,
            #     total_steps=step + 1,
            #     score=info.score,
            #     max_score=info.max_score,
            #     status="resolved" if info.done or llm_info.done else "unresolved",
            # )

            return info.done or llm_info.done
        except Exception as e:
            # report any error that happens during the run
            if info:
                self.logger.report_progress(
                    problem_id=task_name,
                    step=step + 1,
                    total_steps=step + 1,
                    score=info.score if info else 0,
                    max_score=info.max_score if info else 1,
                    status="error",
                )
            raise
