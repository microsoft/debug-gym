"""Interactive FreeEnv demo that runs a container image with a human operator."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

from debug_gym.gym.envs.free_env import FreeEnv
from debug_gym.gym.terminals import select_terminal
from debug_gym.gym.terminals.terminal import Terminal
from debug_gym.gym.tools.toolbox import Toolbox
from debug_gym.llms.human import Human
from debug_gym.logger import DebugGymLogger

DEFAULT_IMAGE = "swesmith.x86_64.amueller__word_cloud.ec24191c"
DEFAULT_TOOLS = [
    "listdir",
    "view",
    "grep",
    "rewrite",
    "bash",
    {"submit": {"eval_on_submit": False}},
]


def format_observations(env_info) -> list[dict]:
    messages = [
        {
            "role": "system",
            "content": env_info.instructions or "Interact with the repository.",
        }
    ]

    instructions_text = (env_info.instructions or "").strip()
    for index, observation in enumerate(env_info.all_observations):
        text = observation.observation.strip()
        if index == 0 and text == instructions_text:
            continue
        prefix = f"[{observation.source}] " if observation.source else ""
        messages.append({"role": "user", "content": f"{prefix}{text}"})
    return messages


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch a FreeEnv session with human-in-the-loop control.",
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help="Docker image name to load inside the environment.",
    )
    parser.add_argument(
        "--terminal",
        default="docker",
        choices=["docker", "kubernetes"],
        help="Terminal backend to use.",
    )
    parser.add_argument(
        "--registry",
        default=None,
        help="Optional registry prefix (e.g. ghcr.io/swe-bench).",
    )
    parser.add_argument(
        "--workspace-dir",
        default="/testbed",
        help="Working directory inside the container or pod.",
    )
    parser.add_argument(
        "--mount-path",
        type=Path,
        default=None,
        help="Optional host path whose contents should be copied into the environment.",
    )
    parser.add_argument(
        "--setup-command",
        action="append",
        default=[],
        help="Additional setup commands to run when the terminal starts (repeatable).",
    )
    parser.add_argument(
        "--tool",
        dest="tools",
        action="append",
        default=None,
        help="Tool name to add to the toolbox (can be specified multiple times).",
    )
    parser.add_argument(
        "--init-git",
        action="store_true",
        help="Initialize a git repository inside the environment (disabled by default).",
    )
    parser.add_argument(
        "--instructions",
        default=None,
        help="Custom instruction text displayed at reset.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=10,
        help="Maximum number of retries for invalid human tool calls.",
    )
    parser.add_argument(
        "--dir-tree-depth",
        type=int,
        default=2,
        help="Depth of the directory tree shown in observations.",
    )
    return parser


def _add_tools(env: FreeEnv, tool_specs: Iterable[Any], logger: DebugGymLogger) -> None:
    """Attach toolbox entries, defaulting submit to eval_on_submit=False for humans."""

    for spec in tool_specs:
        tool_kwargs: dict[str, Any] = {}
        if isinstance(spec, dict):
            if len(spec) != 1:
                raise ValueError("Tool dictionary must contain exactly one entry")
            spec = dict(spec)
            tool_name, tool_kwargs = next(iter(spec.items()))
        else:
            tool_name = str(spec)

        if tool_name == "submit" and "eval_on_submit" not in tool_kwargs:
            tool_kwargs = {**tool_kwargs, "eval_on_submit": False}

        env.add_tool(Toolbox.get_tool(tool_name, **tool_kwargs))
        logger.debug("Loaded tool %s with options %s", tool_name, tool_kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logger = DebugGymLogger("free-env-demo")

    tool_specs: list[Any]
    if args.tools:
        # User-specified tools override defaults but still respect submit behaviour.
        tool_specs = list(args.tools)
    else:
        tool_specs = list(DEFAULT_TOOLS)

    terminal_config: dict[str, Any] = {
        "type": args.terminal,
        "base_image": args.image,
        "working_dir": args.workspace_dir,
    }
    if args.setup_command:
        terminal_config["setup_commands"] = list(args.setup_command)
    if args.registry:
        terminal_config["registry"] = args.registry

    terminal: Terminal | None = select_terminal(terminal_config, logger=logger)

    env = FreeEnv(
        image=args.image,
        terminal=terminal,
        mount_path=args.mount_path,
        setup_commands=args.setup_command,
        instructions=args.instructions,
        init_git=args.init_git,
        workspace_dir=args.workspace_dir,
        logger=logger,
        dir_tree_depth=args.dir_tree_depth,
    )

    _add_tools(env, tool_specs, logger)
    logger.info("Loaded tools: %s", env.tool_names)

    info = env.reset()
    human = Human(logger=logger, max_retries=args.max_retries)

    try:
        while True:
            messages = format_observations(info)
            response = human(messages, env.tools)
            logger.info(
                "Running %s with arguments %s",
                response.tool.name,
                response.tool.arguments,
            )
            info = env.step(
                response.tool,
                action_content=response.response,
            )
    except KeyboardInterrupt:
        logger.info("Session interrupted by user.")
    except ValueError as exc:
        logger.error("Session terminated: %s", exc)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
