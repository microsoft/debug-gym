import argparse
import importlib.resources
import os
import shutil
from pathlib import Path


def copy_llm_config_template(dest_dir: str = None):
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
    template_name = "llm.template.yaml"

    if args.destination is not None:
        dest_dir = Path(args.destination)
    elif args.dest is not None:
        dest_dir = Path(args.dest)
    else:
        dest_dir = Path.joinpath(Path.home(), ".config", "debug_gym")

    os.makedirs(dest_dir, exist_ok=True)

    try:
        # Look for the template file within the package
        source = Path(importlib.resources.files("debug_gym")) / template_name
    except (ImportError, ModuleNotFoundError):
        # Fallback to relative path for development mode
        source = Path(__file__).parent.absolute() / template_name

    destination = dest_dir / template_name
    if not os.path.exists(destination):
        shutil.copy2(source, destination)
        print(f"LLM config template created at `{destination}`.")
    elif force:
        shutil.copy2(source, destination)
        print(f"LLM config template overridden at `{destination}`.")
    else:
        print(f"LLM config template already exists at `{destination}`.")

    print("Please edit the file to configure your LLM settings.")
