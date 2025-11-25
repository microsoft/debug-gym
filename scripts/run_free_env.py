"""Standalone runner for FreeEnv + FreeAgent with human-visible logging."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from debug_gym.agents.free_agent import FreeAgent
from debug_gym.gym.envs.free_env import FreeEnv
from debug_gym.gym.terminals import select_terminal
from debug_gym.gym.terminals.terminal import Terminal
from debug_gym.gym.tools.toolbox import Toolbox
from debug_gym.llms.base import LLM
from debug_gym.llms.human import Human
from debug_gym.logger import DebugGymLogger


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser that exposes the runner configuration flag."""
    parser = argparse.ArgumentParser(description="Run FreeAgent against FreeEnv.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("scripts/config_free_env.yaml"),
        help="Path to the YAML configuration file.",
    )
    return parser


def load_app_config(path: Path) -> dict:
    """Load the YAML configuration used to seed the environment and agent."""
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_llm(config: dict, logger: DebugGymLogger):
    """Instantiate the LLM (or human driver) based on configuration defaults."""
    llm_cfg = config.get("llm") or {}
    llm_name = llm_cfg.get("name") or config.get("llm_name") or "human"

    if llm_name.lower() == "human":
        return Human(model_name="human", logger=logger)

    return LLM.instantiate(
        llm_name=llm_name,
        llm_config_file_path=llm_cfg.get("config_file")
        or config.get("llm_config_file_path"),
        logger=logger,
    )


def resolve_terminal(
    env_config: Mapping[str, Any],
    logger: DebugGymLogger,
) -> Terminal | None:
    """Resolve the requested terminal backend, normalizing legacy config shapes."""
    terminal_setting = env_config.get("terminal")

    if isinstance(terminal_setting, Terminal):
        return terminal_setting

    if terminal_setting is None:
        terminal_config: dict[str, Any] = {"type": "docker"}
    elif isinstance(terminal_setting, str):
        terminal_config = {"type": terminal_setting}
    elif isinstance(terminal_setting, Mapping):
        terminal_config = dict(terminal_setting)
    else:
        raise TypeError(
            "terminal configuration must be a mapping, string, Terminal, or None",
        )

    terminal_config.setdefault("type", "docker")
    terminal_config["type"] = str(terminal_config["type"]).lower()
    terminal_config.setdefault("base_image", env_config["image"])
    terminal_config.setdefault(
        "working_dir", env_config.get("workspace_dir", "/testbed")
    )

    setup_commands = env_config.get("setup_commands")
    if setup_commands:
        terminal_config.setdefault("setup_commands", list(setup_commands))

    overrides = dict(env_config.get("terminal_kwargs") or {})
    terminal_config.update(overrides)

    return select_terminal(terminal_config, logger=logger)


def add_tools(env: FreeEnv, tools_config: list[Any], logger: DebugGymLogger) -> None:
    """Instantiate tools defined in config, honoring optional per-tool kwargs."""

    for tool_entry in tools_config:
        tool_kwargs: dict[str, Any] = {}
        if isinstance(tool_entry, Mapping):
            if len(tool_entry) != 1:
                raise ValueError("Tool mapping entries must contain a single tool name")
            tool_entry = dict(tool_entry)
            tool_name, tool_kwargs = next(iter(tool_entry.items()))
        else:
            tool_name = str(tool_entry)

        if tool_name == "submit" and "eval_on_submit" not in tool_kwargs:
            tool_kwargs = {**tool_kwargs, "eval_on_submit": False}

        env.add_tool(Toolbox.get_tool(tool_name, **tool_kwargs))
        logger.debug("Added tool %s with options %s", tool_name, tool_kwargs)


def main() -> int:
    """Entrypoint for running FreeAgent against FreeEnv from the command line."""
    args = build_parser().parse_args()
    config = load_app_config(args.config)

    logger = DebugGymLogger("free-agent-run")

    env_cfg = config["environment"]
    terminal = resolve_terminal(env_cfg, logger)
    # Copy only the knobs understood by FreeEnv, leaving unrelated config behind.
    env_kwargs = dict(
        image=env_cfg["image"],
        terminal=terminal,
        mount_path=env_cfg.get("mount_path"),
        setup_commands=env_cfg.get("setup_commands"),
        instructions=env_cfg.get("instructions"),
        init_git=env_cfg.get("init_git", True),
        workspace_dir=env_cfg.get("workspace_dir", "/testbed"),
        logger=logger,
        dir_tree_depth=env_cfg.get("dir_tree_depth", 2),
    )

    # Instantiate the environment once the terminal and core parameters are ready.
    env = FreeEnv(**env_kwargs)

    tools_config = config.get("tools")
    if not tools_config:
        raise ValueError(
            "Configuration must specify a non-empty 'tools' list for FreeEnv sessions."
        )

    add_tools(env, tools_config, logger)

    llm = build_llm(config, logger)
    agent_config = config.get("agent", {})
    agent = FreeAgent(config=agent_config, env=env, llm=llm, logger=logger)

    task_name = config.get("task_name", "free-session")

    try:
        resolved = agent.run(task_name=task_name)
        agent.save_trajectory(task_name=task_name)
        agent.save_patch(task_name=task_name)
        logger.info(f"Run complete. Resolved={resolved}")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
