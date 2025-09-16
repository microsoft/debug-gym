import json
from glob import glob
from pathlib import Path

import pandas as pd
from termcolor import colored


def main(args):
    # Collect all *.jsonl files in the output directory
    for jsonl_name in ["froggy.jsonl", "debug_gym.jsonl", "trajectory.json"]:
        log_files = glob(f"{args.path}/**/{jsonl_name}", recursive=True)
        # Use pandas to read the logs
        results = []
        for log_file in sorted(log_files):
            try:
                with open(log_file, "r") as f:
                    data = json.load(f)

                # Extract the relative path from the log file path
                log_path = Path(log_file)
                relative_path = log_path.relative_to(args.path).parent

                result = {
                    "success": data["success"],
                    "uuid": data["uuid"],
                    "agent_type": data["agent_type"],
                    "problem": data["problem"],
                    "path": str(relative_path).split("/")[
                        0
                    ],  # Get the first part of the path
                }
                results.append(result)

                if args.verbose:
                    # Print agent_type, uuid, and problem colored by success, and path to the log.
                    color = "green" if result["success"] else "red"
                    if args.show_failed_only and result["success"]:
                        continue

                    print(
                        colored(
                            f"{result['agent_type']} {result['uuid']} {result['problem']}",
                            color,
                        ),
                        f"\t({log_file})",
                    )

            except Exception as e:
                print(colored(f"Error reading {log_file}. ({e!r})", "red"))

    df = pd.DataFrame(results)

    # Group by agent type, path, and uuid
    grouped = df.groupby(["agent_type", "path", "uuid"])

    # Print success rate for each agent
    for (agent_type, path, uuid), group in grouped:
        total = len(group)
        nb_successes = group["success"].sum()
        success_rate = nb_successes / total
        print(
            colored(
                f"{agent_type} ({path}) {uuid}: {success_rate:.2%} ({nb_successes} out of {total})"
            )
        )


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="Folder where to find the logs.")
    parser.add_argument(
        "--agents",
        nargs="+",
        help="Agent UUID(s) for which to collect the logs. Default: all agent found in `path`.",
    )
    parser.add_argument(
        "--show-failed-only",
        action="store_true",
        help="Only print out failed experiments",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
