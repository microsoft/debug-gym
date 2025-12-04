import argparse
import json
import logging
import os
from pathlib import Path

import yaml

from debug_gym.logger import DebugGymLogger


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file")
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
    args = parser.parse_args()
    assert os.path.exists(args.config_file), "Invalid config file"
    with open(args.config_file) as reader:
        config = yaml.safe_load(reader)

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
        # Override base config with agent specific config
        for key, value in config[args.agent].items():
            return_config[key] = value
    else:
        # base config is not specified
        return_config = config[args.agent]

    # Parse overriden params.
    for param in args.params:
        fqn_key, value = param.split("=")
        entry_to_change = return_config
        keys = fqn_key.split(".")
        for k in keys[:-1]:
            if k not in entry_to_change:
                entry_to_change[k] = {}

            entry_to_change = entry_to_change[k]
        entry_to_change[keys[-1]] = yaml.safe_load(value)

    # assume agent type is the key if not specified by the user
    if not return_config.get("agent_type"):
        return_config["agent_type"] = args.agent

    return return_config, args


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
