import argparse
import json
import logging
import os
from pathlib import Path

import yaml

from debug_gym.gym.tools.tool import ToolCall
from debug_gym.llms.base import LLMResponse
from debug_gym.logger import DebugGymLogger


def load_config(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to config file")
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
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for terminal or timeout failures"
        " (e.g., spot instance eviction, container deletion). "
        "Steps are replayed, including the step that failed, "
        "after which new steps are generated. Default: 3.",
    )
    parser.add_argument(
        "-p",
        "--params",
        nargs="+",
        action="extend",
        metavar="my.setting=value",
        default=[],
        help="override params of the config file,"
        " e.g. -p 'edit_only.random_seed=123'",
    )
    args = parser.parse_args(args)
    config = {}
    if args.config is not None:
        assert os.path.exists(args.config), "Invalid config file"
        with open(args.config) as reader:
            config = yaml.safe_load(reader)

    # Parse overriden params.
    for param in args.params:
        fqn_key, value = param, ""
        if "=" in param:
            fqn_key, value = param.split("=")

        entry_to_change = config
        keys = fqn_key.split(".")
        for k in keys[:-1]:
            if k not in entry_to_change:
                entry_to_change[k] = {}

            entry_to_change = entry_to_change[k]
        entry_to_change[keys[-1]] = yaml.safe_load(value)

    return config, args


def save_patch(env, problem_path: Path, logger: DebugGymLogger):
    """Persist the current environment patch to disk."""
    problem_path.mkdir(parents=True, exist_ok=True)
    patch_path = problem_path / "debug_gym.patch"
    with open(patch_path, "w") as f:
        f.write(env.patch)

    logger.debug(f"Patch saved in {patch_path}")


def save_trajectory(agent, problem_path: Path, logger: DebugGymLogger):
    """Persist the agent trajectory to disk."""
    problem_path.mkdir(parents=True, exist_ok=True)
    trajectory = agent.build_trajectory()
    json_file = problem_path / "trajectory.json"
    with open(json_file, "w") as f:
        json.dump(trajectory, f, indent=4)

    logger.debug(f"Trajectory saved in {json_file}")


def load_trajectory(
    problem_path: Path, logger: DebugGymLogger
) -> list[LLMResponse] | None:
    """Load a previous trajectory and reconstruct LLMResponse objects for replay.

    Follows the same approach as replay.py for accurate reconstruction of LLMResponse
    objects, including token counts and original prompt data from prompt_response_pairs.

    Returns a list of LLMResponse objects that can be passed to agent.execute_action(),
    or None if no trajectory exists.
    """
    json_file = problem_path / "trajectory.json"
    if not json_file.exists():
        return None

    try:
        with open(json_file, "r") as f:
            trajectory = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load trajectory from {json_file}: {e}")
        return None

    log = trajectory.get("log", [])
    if not log:
        return None

    llm_responses = []
    for step in log:
        # Skip step 0 (initial state with no action)
        if step.get("step_id") == 0 or step.get("action") is None:
            continue

        # Reconstruct ToolCall from saved action
        action_data = step.get("action", {})
        if not action_data:
            continue

        tool_call = ToolCall(
            id=action_data.get("id", ""),
            name=action_data.get("name", ""),
            arguments=action_data.get("arguments", {}),
        )

        # Extract data from prompt_response_pairs if available (like replay.py does)
        prompt_response_pairs = step.get("prompt_response_pairs", [])
        if prompt_response_pairs and len(prompt_response_pairs) > 0:
            prompt_response = prompt_response_pairs[0]
            token_usage = prompt_response.get("token_usage", {})
            llm_response = LLMResponse(
                prompt=prompt_response.get("prompt", []),
                response=prompt_response.get("response"),
                reasoning_response=prompt_response.get("reasoning_response"),
                tool=tool_call,
                prompt_token_count=token_usage.get("prompt", 0),
                response_token_count=token_usage.get("response", 0),
            )
        else:
            # Fallback to step-level data if prompt_response_pairs not available
            llm_response = LLMResponse(
                prompt=[],
                response=step.get("content"),
                reasoning_response=step.get("reasoning"),
                tool=tool_call,
            )
        llm_responses.append(llm_response)

    if llm_responses:
        logger.info(
            f"Loaded {len(llm_responses)} steps from previous trajectory for replay"
        )

    return llm_responses if llm_responses else None
