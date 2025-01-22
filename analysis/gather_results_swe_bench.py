import json
import os
from glob import glob
from os.path import join as pjoin
from pathlib import Path
import pandas as pd

from termcolor import colored

from froggy.envs import SWEBenchEnv


def main(args):
    # Collect all *.jsonl files in the output directory
    log_files = glob(str(args.path / "**" / "froggy.jsonl"), recursive=True)

    # Use pandas to read the logs
    results = []
    for log_file in log_files:
        try:
            with open(log_file, "r") as f:
                data = json.load(f)

            result = {
                "success": data["success"],
                "uuid": data["uuid"],
                "agent_type": data.get("agent_type", "solution"),
                "problem": data["problem"],
            }
            results.append(result)

            if args.verbose:
                # Print agent_type, uuid, and problem colored by success, and path to the log.
                color = "green" if result["success"] else "red"
                if args.show_failed_only and result["success"]:
                    continue

                print(colored(f"{result['agent_type']} {result['uuid']} {result['problem']}", color), f"\t({log_file})")


        except Exception as e:
            print(colored(f"Error reading {log_file}. ({e!r})", "red"))

    # TODO: check for duplicated experiments.

    # If needed, get the list of all problems.
    problem_list = None
    if not args.ignore_missing:
        # create environment
        env = SWEBenchEnv(
            dir_tree_depth=1,
            run_on_rewrite=True,
            auto_view_change=True,
        )
        problem_list = env.dataset.keys()  # all tasks

    df = pd.DataFrame(results)

    # Group by agent type and uuid
    grouped = df.groupby(["agent_type", "uuid"])

    # Print success rate for each agent
    for agent_type, group in grouped:
        total = len(group) if args.ignore_missing else len(problem_list)
        nb_successes = group["success"].sum()
        success_rate = nb_successes / total
        print(colored(f"{agent_type}: {success_rate:.2%} ({nb_successes} out of {total})"))


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="Folder where to find the logs.")
    parser.add_argument("--agents", nargs="+", help="Agent UUID(s) for which to collect the logs. Default: all agent found in `path`.")
    parser.add_argument("--ignore-missing", action="store_true", help="Ignore missing experiments")
    parser.add_argument("--show-failed-only", action="store_true", help="Only print out failed experiments")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
