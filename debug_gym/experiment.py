import datetime
import json
import os
import subprocess
from pathlib import Path

from debug_gym import version as dg_version
from debug_gym.gym.envs import select_env
from debug_gym.gym.terminals import select_terminal
from debug_gym.gym.tools.toolbox import Toolbox
from debug_gym.logger import DebugGymLogger


def create_env(config: dict, task_data: dict, logger: DebugGymLogger):
    terminal = select_terminal(config.get("terminal"), logger, uuid=config["uuid"])
    env_class = select_env(task_data["env_type"])
    env = env_class(
        task_data=task_data,
        terminal=terminal,
        logger=logger,
        **config.get("env", {}),
    )

    add_tools(env, config, logger)
    return env


def add_tools(env, config: dict, logger: DebugGymLogger):
    """Add tools to the environment"""
    for tool in config["tools"]:
        tool_config = {}
        if isinstance(tool, dict):
            assert len(tool) == 1, "Tool dict must have exactly one key"
            tool, tool_config = list(tool.items())[0]
        if isinstance(config["tools"], dict) and isinstance(
            config["tools"][tool], dict
        ):
            tool_config.update(config["tools"][tool])

        tool_instantiated = Toolbox.get_tool(tool, **tool_config)
        env.add_tool(tool_instantiated)
        logger.debug(f"Adding tool to toolbox: {tool_instantiated.__class__.__name__}")


def dump_experiment_info(config: dict, args: dict):
    """Dump experiment information to a JSONL file.
    Each line is one experiment run with its metadata."""

    try:  # Get git commit hash
        git_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)
            )
            .decode()
            .strip()
        )
    except Exception:
        git_hash = ""

    try:  # Get git diff
        git_diff = subprocess.check_output(
            ["git", "diff"], cwd=os.path.dirname(__file__)
        ).decode()
    except Exception:
        git_diff = ""

    version_info = {
        "debug_gym_version": dg_version.__version__,
        "datetime": datetime.datetime.now().isoformat(),
        "git_hash": git_hash,
        "git_diff": git_diff,
        "config": config,
        "args": vars(args),
        "python_version": os.sys.version,
    }

    file = Path(config["output_path"]) / config["uuid"] / "experiment_info.jsonl"
    with open(file, "a") as f:
        f.write(f"{json.dumps(version_info)}\n")
