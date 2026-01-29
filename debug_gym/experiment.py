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

    env_class = select_env(task_data.get("env_type"))
    env = env_class(
        task_data=task_data,
        terminal=terminal,
        logger=logger,
        **config.get("env", {}),
    )

    # Register MCP servers before adding tools (must happen per-process)
    register_mcp_servers(config, logger)

    add_tools(env, config, logger)
    return env


def register_mcp_servers(config: dict, logger: DebugGymLogger):
    """Register MCP servers from config. Must be called per-process."""
    mcp_servers = config.get("mcp_servers", [])
    if not mcp_servers:
        return

    from debug_gym.gym.tools.mcp_proxy import register_mcp_server_sse

    for server_config in mcp_servers:
        server_id = server_config.get("server_id")
        url = server_config.get("url")
        if not server_id or not url:
            logger.warning(f"Skipping invalid MCP server config: {server_config}")
            continue

        try:
            tools = register_mcp_server_sse(
                server_id=server_id,
                url=url,
                headers=server_config.get("headers"),
                tool_prefix=server_config.get("tool_prefix", ""),
            )
            logger.debug(f"Registered MCP server '{server_id}' with {len(tools)} tools")
        except Exception as e:
            logger.error(f"Failed to register MCP server '{server_id}': {e}")


def add_tools(env, config: dict, logger: DebugGymLogger):
    """Add tools to the environment"""
    for tool in config.get("tools", []):
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

    file = Path(config["output_path"]) / "experiment_info.jsonl"
    with open(file, "a") as f:
        f.write(f"{json.dumps(version_info)}\n")
