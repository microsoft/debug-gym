import json
import os
import subprocess
import uuid
from os.path import join as pjoin

import numpy as np

from froggy.agents.base_agent import BaseAgent, register_agent
from froggy.agents.llm_api import instantiate_llm
from froggy.agents.utils import HistoryTracker, build_history_prompt
from froggy.logger import FroggyLogger
from froggy.pond.envs.env import RepoEnv
from froggy.pond.utils import unescape


@register_agent
class PdbAgent(BaseAgent):
    name = "pdb_agent"
    system_prompt = "Your goal is to debug a Python program to make sure it can pass a set of test functions. You have access to the pdb debugger tools, you can use them to investigate the code, set breakpoints, and print necessary values to identify the bugs. Once you have gained enough information, propose a rewriting patch to fix the bugs. Avoid rewriting the entire code, focus on the bugs only."
    action_prompt = "Based on the instruction, the current code, the last execution output, and the history information, continue your debugging process using pdb commands or to propose a patch using rewrite command. Output a single command, nothing else. Do not repeat your previous commands unless they can provide more information."


@register_agent
class PdbAfterRewrites(PdbAgent):
    name: str = "pdb_after_rewrites"

    def run(self, task_name=None, debug=False):
        # remove the pdb tool from the environment
        assert "pdb" in self.env.tools, "pdb not found in env tools"
        pdb_tool = self.env.tools.pop("pdb")

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

            prompt = self.build_prompt(info)

            llm_response = self.llm(
                prompt, info, temperature=self.config["llm_temperature"][0]
            )

            if debug:
                breakpoint()

            info = self.env.step(llm_response.response)

            # re-introduce pdb tool at the right time
            if (
                info.rewrite_counter >= self.config["n_rewrites_before_pdb"]
                and pdb_tool.name not in self.env.tools
            ):
                self.env.add_tool(pdb_tool)
                self.env.tools["pdb"].start_pdb()
                # update info tools related fields after adding pdb so it's included when building the next prompt
                info.instructions = self.env.instructions
                info.tools = self.env.tool_instructions

            self.history.step(info, llm_response)

            if info.done or info.rewrite_counter >= self.config["max_rewrite_steps"]:
                self.logger.info(
                    f"Score: {info.score}/{info.max_score} ({info.score/info.max_score:.1%})"
                )
                break

        return info.done

@register_agent
class PdbHumanInTheLoop(PdbAgent):
    name: str = "pdb_hitl"

    def run(self, task_name=None, debug=False):
        # instantiate the human in the loop
        hitl_config = self.config.copy()
        hitl_config["llm_name"] = "human"
        self.hitl = instantiate_llm(hitl_config, logger=self.logger)

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

            prompt = self.build_prompt(info)

            llm_response = self.llm(
                prompt, info, temperature=self.config["llm_temperature"][0]
            )

            if debug:
                breakpoint()

            # make a copy of the env for the human in the loop
            self.hitl_env = self.env.clone()
            hitl_info = self.hitl_env.reset(options={"task_name": task_name})
            # replay the history up to the current step
            for step in self.history.get_all():
                if step.done:
                    break
                hitl_info = self.hitl_env.step(step.action)

            info = self.env.step(llm_response.response)

            self.history.step(info, llm_response)

            if info.done or info.rewrite_counter >= self.config["max_rewrite_steps"]:
                self.logger.info(
                    f"Score (llm): {info.score}/{info.max_score} ({info.score/info.max_score:.1%})"
                )
                break

            # call the human in the loop
            hitl_response = self.hitl(
                prompt, hitl_info, temperature=self.config["llm_temperature"][0]
            )
            hitl_info = self.hitl_env.step(hitl_response.response)

            if hitl_info.done or hitl_info.rewrite_counter >= self.config["max_rewrite_steps"]:
                self.logger.info(
                    f"Score (human): {hitl_info.score}/{hitl_info.max_score} ({hitl_info.score/hitl_info.max_score:.1%})"
                )
                break


        return info.done