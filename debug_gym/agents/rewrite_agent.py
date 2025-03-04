import json
import os
import subprocess
import uuid
from os.path import join as pjoin

import numpy as np

from debug_gym.agents.base_agent import BaseAgent, register_agent
from debug_gym.agents.llm_api import instantiate_llm
from debug_gym.agents.utils import HistoryTracker, build_history_prompt
from debug_gym.logger import DebugGymLogger
from debug_gym.gym.envs.env import RepoEnv
from debug_gym.gym.utils import unescape


@register_agent
class RewriteOnly(BaseAgent):
    name: str = "rewrite_only"
    system_prompt: str = (
        "Your goal is to debug a Python program to make sure it can pass a set of test functions. You need to propose a rewriting patch to fix the bugs. Avoid rewriting the entire code, focus on the bugs only."
    )
    action_prompt: str = (
        "Based on the instruction, the current code, the last execution output, and the history information, continue your debugging process to propose a patch using rewrite command. Output a single command, nothing else. Do not repeat your previous commands unless they can provide more information. You must be concise and avoid overthinking."
    )
