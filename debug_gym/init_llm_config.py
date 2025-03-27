import argparse
import os
from pathlib import Path

from debug_gym.agents.llm_api import LLM_CONFIG_TEMPLATE


def init_llm_config(dest_dir: str = None):
    """Copy the llm config template to the specified
    directory or the user's home directory."""

    parser = argparse.ArgumentParser(
        description="Create an LLM config template in the specified directory or `~/.config/debug_gym`."
    )
    parser.add_argument(
        "destination", nargs="?", type=str, help="Destination directory (positional)"
    )
    parser.add_argument("-d", "--dest", type=str, help="Destination directory")
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Override the file if it already exists",
    )

    args = parser.parse_args()
    force = args.force

    if args.destination is not None:
        dest_dir = Path(args.destination)
    elif args.dest is not None:
        dest_dir = Path(args.dest)
    else:
        dest_dir = Path.joinpath(Path.home(), ".config", "debug_gym")

    os.makedirs(dest_dir, exist_ok=True)

    destination = dest_dir / "llm.yaml"
    if not os.path.exists(destination):
        with open(destination, "w") as f:
            f.write(LLM_CONFIG_TEMPLATE)
        print(f"LLM config template created at `{destination}`.")
    elif force:
        with open(destination, "w") as f:
            f.write(LLM_CONFIG_TEMPLATE)
        print(f"LLM config template overridden at `{destination}`.")
    else:
        print(f"LLM config template already exists at `{destination}`.")

    print("Please edit the file to configure your LLM settings.")


if __name__ == "__main__":
    init_llm_config()
