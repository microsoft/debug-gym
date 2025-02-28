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
    action_prompt = "Based on the instruction, the current code, the last execution output, and the history information, continue your debugging process using pdb commands or to propose a patch using rewrite command. Output a single command, nothing else. Do not repeat your previous commands unless they can provide more information. You must be concise and avoid overthinking."


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
            llm_response.response = self.parse_r1_response(llm_response.response)

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
