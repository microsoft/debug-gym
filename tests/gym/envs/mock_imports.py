import os
import re
import shutil
import subprocess
from ast import literal_eval
from pathlib import Path

import datasets
import docker
from tqdm import tqdm

# Import from swesmith
import swesmith
from swesmith.constants import MAP_REPO_TO_SPECS
from swesmith.utils import clone_repo

from debug_gym.gym.entities import EvalOutput
from debug_gym.gym.envs.env import RepoEnv
from debug_gym.gym.terminal import DockerTerminal, Terminal
from debug_gym.gym.utils import create_ignore_file


# Define constants that might be needed
class TestStatus:
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    XFAIL = "xfailed"


# Non-test extensions similar to SWE-bench
NON_TEST_EXTS = [".md", ".rst", ".txt", ".yml", ".yaml", ".json", ".ini", ".toml", ".cfg"]


class SWEBenchEnvMock(RepoEnv):
    """Mock version of SWEBenchEnv for testing purposes"""
    CACHE = Path.joinpath(Path.home(), ".cache", "debug_gym", "swe-bench")

    def __init__(
        self,
        dataset_id: str = "princeton-nlp/SWE-bench_Verified",
        split: str = "test",
        instance_ids: list[str] | None = None,
        terminal: Terminal | None = None,
        **kwargs,
    ):
        terminal = terminal or DockerTerminal(logger=kwargs.get("logger"))
        super().__init__(terminal=terminal, **kwargs)
        self.dataset_id = dataset_id
        self.split = split
        self.instance_ids = instance_ids

    @property
    def instructions(self):
        return {}


# Replace SWEBenchEnv with our mock in the imports to avoid issues
import sys
import debug_gym.gym.envs
sys.modules['debug_gym.gym.envs.swe_bench'] = type('ModuleMock', (), {'SWEBenchEnv': SWEBenchEnvMock})

# Import our SWESmithEnv
from debug_gym.gym.envs.swe_smith import SWESmithEnv