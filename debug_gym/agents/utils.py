import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from debug_gym.logger import DebugGymLogger


@dataclass
class BaseConfig:
    """Base configuration dataclass for debug-gym.

    This class defines the structure and defaults for configuration options.
    It can be used with or without a YAML config file - all values can be
    specified via command line arguments.
    """

    # Environment configs
    output_path: str = "exps/default"
    benchmark: str | None = None
    problems: str | list[str] = "all"
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    terminal: dict[str, Any] = field(default_factory=lambda: {"type": "local"})
    tools: list[str | dict] = field(
        default_factory=lambda: ["pdb", "view", "rewrite", "eval"]
    )

    # LLM configs
    llm_name: str = "gpt-4o"
    llm_config_file_path: str | None = None

    # Agent configs
    agent_type: str | None = None
    random_seed: int = 42
    max_steps: int = 50
    max_rewrite_steps: int = 10
    memory_size: int = 20
    save_patch: bool = True
    reset_prompt_history_after_rewrite: bool = False
    system_prompt_template_file: str | None = None

    # Shortcut features
    show_current_breakpoints: bool = False
    show_directory_tree: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseConfig":
        """Create config from dictionary, ignoring unknown fields."""
        field_names = {f.name for f in fields(cls)}
        known = {k: v for k, v in data.items() if k in field_names}
        return cls(**known)

    def update(self, data: dict[str, Any]) -> None:
        """Update config with values from dictionary."""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file",
        nargs="?",
        default=None,
        help="Path to config file (optional if using command line arguments)",
    )
    parser.add_argument(
        "--agent",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Break before sending action to the environment.",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=None,
        help=(
            "Number of workers to use, default is 1 (no parallelism). "
            "Can be set via DEBUG_GYM_WORKERS environment variable."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available agents and problems.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-v",
        "--verbose",
        dest="logging_level",
        action="store_const",
        const=logging.INFO,
        help="Verbose mode",
        default=logging.WARNING,
    )
    group.add_argument(
        "-vv",
        "--very-verbose",
        dest="logging_level",
        action="store_const",
        const=logging.DEBUG,
        help="Verbose mode",
        default=logging.WARNING,
    )
    group.add_argument(
        "--logging-level",
        dest="logging_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level",
    )
    parser.add_argument(
        "--force-all",
        action="store_true",
        help="Force running all problems even if they are already done.",
    )
    parser.add_argument(
        "--force-failed",
        action="store_true",
        help="Force running only problems that have failed.",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=0,
        help="Timeout in seconds for each problem. Default: 0 seconds (no timeout).",
    )
    parser.add_argument(
        "--keep-completed-tasks",
        action="store_true",
        help="Keep displaying completed tasks in the workers panel.",
    )
    parser.add_argument(
        "--no-live-display",
        action="store_true",
        help="Disable rich live progress display.",
    )
    parser.add_argument(
        "--max-display",
        type=int,
        default=20,
        help="Maximum number of tasks to display in the progress bar.",
    )
    parser.add_argument(
        "-p",
        "--params",
        nargs="+",
        action="extend",
        metavar="my.setting=value",
        default=[],
        help="override params of the config file,"
        " e.g. -p 'rewrite_only.random_seed=123'",
    )

    # Config arguments that can be used without a config file
    config_group = parser.add_argument_group(
        "config options",
        "These options can be used instead of or to override a config file",
    )
    config_group.add_argument(
        "--output-path",
        type=str,
        help="Output path for experiment results",
    )
    config_group.add_argument(
        "--benchmark",
        type=str,
        help="Benchmark to run (e.g., 'aider', 'mini_nightmare', 'swebench-debug')",
    )
    config_group.add_argument(
        "--problems",
        type=str,
        help="Problems to run ('all' or comma-separated list)",
    )
    config_group.add_argument(
        "--llm-name",
        type=str,
        help="Name of the LLM to use (e.g., 'gpt-4o', 'claude-3-sonnet')",
    )
    config_group.add_argument(
        "--llm-config-file-path",
        type=str,
        help="Path to LLM config file",
    )
    config_group.add_argument(
        "--random-seed",
        type=int,
        help="Random seed for reproducibility",
    )
    config_group.add_argument(
        "--max-steps",
        type=int,
        help="Maximum number of steps per problem",
    )
    config_group.add_argument(
        "--max-rewrite-steps",
        type=int,
        help="Maximum number of rewrite steps",
    )
    config_group.add_argument(
        "--memory-size",
        type=int,
        help="Memory size for agent",
    )
    config_group.add_argument(
        "--save-patch",
        action="store_true",
        default=None,
        help="Save patch after solving",
    )
    config_group.add_argument(
        "--no-save-patch",
        action="store_true",
        help="Do not save patch after solving",
    )
    config_group.add_argument(
        "--terminal-type",
        type=str,
        choices=["local", "docker", "kubernetes"],
        help="Terminal type to use",
    )
    config_group.add_argument(
        "--tools",
        type=str,
        help="Comma-separated list of tools (e.g., 'pdb,view,rewrite,eval')",
    )
    config_group.add_argument(
        "--env-kwargs",
        type=str,
        help="Environment keyword arguments as JSON string",
    )

    args = parser.parse_args()

    if args.config_file is not None:
        # Load config from YAML file
        if not os.path.exists(args.config_file):
            raise FileNotFoundError(f"Config file not found: {args.config_file}")
        with open(args.config_file) as reader:
            config = yaml.safe_load(reader)

        # Parse overriden params.
        for param in args.params:
            fqn_key, value = param.split("=")
            entry_to_change = config
            keys = fqn_key.split(".")
            for k in keys[:-1]:
                entry_to_change = entry_to_change[k]
            entry_to_change[keys[-1]] = yaml.safe_load(value)

        available_agents = [item for item in list(config.keys()) if item != "base"]

        if not args.agent:
            # pick first agent
            args.agent = available_agents[0]
        elif args.agent not in available_agents:
            raise ValueError(
                f"Invalid agent: {args.agent}. Available agents: {available_agents}"
            )

        if "base" in config:
            # base config is specified (shared across agents)
            return_config = config["base"]
            agent_specific_config = config[args.agent]
            for key in agent_specific_config:
                # override base config with agent specific config
                return_config[key] = agent_specific_config[key]
        else:
            # base config is not specified
            return_config = config[args.agent]

        # assume agent type is the key if not specified by the user
        if not return_config.get("agent_type"):
            return_config["agent_type"] = args.agent
    else:
        # No config file provided - use BaseConfig defaults and CLI args
        base_config = BaseConfig()
        return_config = base_config.to_dict()

        # Set agent_type from --agent or default to "rewrite_agent"
        if args.agent:
            return_config["agent_type"] = args.agent
        else:
            args.agent = "rewrite_agent"
            return_config["agent_type"] = "rewrite_agent"

    # Apply command-line overrides to config
    return_config = _apply_cli_overrides(return_config, args)

    return return_config, args


def _apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict:
    """Apply command-line argument overrides to config dictionary."""
    if args.output_path is not None:
        config["output_path"] = args.output_path

    if args.benchmark is not None:
        config["benchmark"] = args.benchmark

    if args.problems is not None:
        # Parse comma-separated problems or keep as string if "all"
        if args.problems.lower() == "all":
            config["problems"] = "all"
        else:
            config["problems"] = [p.strip() for p in args.problems.split(",")]

    if args.llm_name is not None:
        config["llm_name"] = args.llm_name

    if args.llm_config_file_path is not None:
        config["llm_config_file_path"] = args.llm_config_file_path

    if args.random_seed is not None:
        config["random_seed"] = args.random_seed

    if args.max_steps is not None:
        config["max_steps"] = args.max_steps

    if args.max_rewrite_steps is not None:
        config["max_rewrite_steps"] = args.max_rewrite_steps

    if args.memory_size is not None:
        config["memory_size"] = args.memory_size

    if args.no_save_patch:
        config["save_patch"] = False
    elif args.save_patch is not None:
        config["save_patch"] = args.save_patch

    if args.terminal_type is not None:
        if "terminal" not in config:
            config["terminal"] = {}
        config["terminal"]["type"] = args.terminal_type

    if args.tools is not None:
        config["tools"] = [t.strip() for t in args.tools.split(",")]

    if args.env_kwargs is not None:
        try:
            env_kwargs = json.loads(args.env_kwargs)
            if "env_kwargs" not in config:
                config["env_kwargs"] = {}
            config["env_kwargs"].update(env_kwargs)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON for --env-kwargs: {e}")

    return config


def save_patch(env, problem_path: Path, logger: DebugGymLogger):
    """Persist the current environment patch to disk."""
    problem_path.mkdir(parents=True, exist_ok=True)
    patch_path = problem_path / "debug_gym.patch"
    with open(patch_path, "w") as f:
        f.write(env.patch)

    logger.debug(f"Patch saved in {patch_path}")


def save_trajectory(agent, problem: str, problem_path: Path, logger: DebugGymLogger):
    """Persist the agent trajectory to disk."""
    problem_path.mkdir(parents=True, exist_ok=True)
    trajectory = agent.build_trajectory(task_name=problem)
    json_file = problem_path / "trajectory.json"
    with open(json_file, "w") as f:
        json.dump(trajectory, f, indent=4)

    logger.debug(f"Trajectory saved in {json_file}")
