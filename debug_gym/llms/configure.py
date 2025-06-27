import argparse
import os
from pathlib import Path

from rich.console import Console

from debug_gym.llms.constants import DEFAULT_LLM_CONFIG, LLM_CONFIG_TEMPLATE


def init_llm_config(dest_dir: str = None):
    """Copy the llm config template to the specified
    directory or the user's home directory."""

    parser = argparse.ArgumentParser(description="Create an LLM config template.")
    parser.add_argument(
        "destination",
        nargs="?",
        type=str,
        help=f"Destination directory (positional). Defaults to {DEFAULT_LLM_CONFIG.parent}",
    )
    parser.add_argument("-d", "--dest", type=str, help="Destination directory")
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Override the file if it already exists",
    )

    console = Console()
    args = parser.parse_args()
    force = args.force

    if args.destination is not None:
        dest_dir = Path(args.destination)
    elif args.dest is not None:
        dest_dir = Path(args.dest)
    else:
        dest_dir = Path.home() / ".config" / "debug_gym"

    os.makedirs(dest_dir, exist_ok=True)

    destination = dest_dir / "llm.yaml"
    if not destination.exists():
        with open(destination, "w") as f:
            f.write(LLM_CONFIG_TEMPLATE)
        console.print(f"LLM config template created at `{destination}`.", style="green")
    elif force:
        with open(destination, "w") as f:
            f.write(LLM_CONFIG_TEMPLATE)
        console.print(
            f"LLM config template overridden at `{destination}`.", style="yellow"
        )
    else:
        console.print(
            f"LLM config template already exists at `{destination}`.", style="red"
        )

    console.print(
        f"Please edit `{destination}` to configure your LLM settings.",
        style="bold green",
    )


if __name__ == "__main__":
    init_llm_config()
