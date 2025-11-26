"""Standalone runner for FreeEnv + FreeAgent with human-visible logging."""

from __future__ import annotations

import argparse
from pathlib import Path

from debug_gym.agents.free_agent import FreeAgent
from debug_gym.gym.envs.free_env import FreeEnv
from debug_gym.gym.tools.toolbox import Toolbox
from debug_gym.llms.base import LLM
from debug_gym.llms.human import Human
from debug_gym.logger import DebugGymLogger

DEFAULT_TOOLS = ["listdir", "view", "grep", "rewrite", "bash"]


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


def main() -> int:
    """Entrypoint for running FreeAgent against FreeEnv from the command line."""
    args = build_parser().parse_args()
    config = load_app_config(args.config)

    logger = DebugGymLogger("free-agent-run")

    env_cfg = config["environment"]
    # Copy only the knobs understood by FreeEnv, leaving unrelated config behind.
    env_kwargs = dict(
        image=env_cfg["image"],
        terminal=env_cfg.get("terminal"),
        mount_path=env_cfg.get("mount_path"),
        setup_commands=env_cfg.get("setup_commands"),
        workspace_dir=env_cfg.get("workspace_dir", "/testbed"),
        logger=logger,
        dir_tree_depth=env_cfg.get("dir_tree_depth", 2),
        terminal_kwargs=env_cfg.get("terminal_kwargs", {}),
    )

    env = FreeEnv(**env_kwargs)

    for tool_name in config.get("tools", DEFAULT_TOOLS):
        env.add_tool(Toolbox.get_tool(tool_name))

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
