"""Interactive FreeEnv demo that runs a container image with a human operator."""

from __future__ import annotations

import argparse
from pathlib import Path

from debug_gym.gym.envs.free_env import FreeEnv
from debug_gym.gym.tools.toolbox import Toolbox
from debug_gym.llms.human import Human
from debug_gym.logger import DebugGymLogger

DEFAULT_IMAGE = "swesmith.x86_64.amueller__word_cloud.ec24191c"
DEFAULT_TOOLS = ["listdir", "view", "rewrite", "bash"]


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
        choices=["docker", "kubernetes", "k8s"],
        help="Terminal backend to use.",
    )
    parser.add_argument(
        "--registry",
        default=None,
        help="Optional registry prefix (e.g. ghcr.io/swe-bench).",
    )
    parser.add_argument(
        "--workspace-dir",
        default="/workspace",
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logger = DebugGymLogger("free-env-demo")

    tool_names = args.tools if args.tools else DEFAULT_TOOLS

    terminal_kwargs: dict[str, str] = {}
    if args.registry:
        terminal_kwargs["registry"] = args.registry

    env = FreeEnv(
        image=args.image,
        terminal=args.terminal,
        mount_path=args.mount_path,
        setup_commands=args.setup_command,
        instructions=args.instructions,
        init_git=args.init_git,
        workspace_dir=args.workspace_dir,
        logger=logger,
        terminal_kwargs=terminal_kwargs,
        dir_tree_depth=args.dir_tree_depth,
    )

    for tool_name in tool_names:
        env.add_tool(Toolbox.get_tool(tool_name))
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
