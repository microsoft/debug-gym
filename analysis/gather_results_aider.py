import json
import os
from os.path import join as pjoin

from termcolor import colored

from froggy.envs import AiderBenchmarkEnv


def main():

    # create environment
    env = AiderBenchmarkEnv(
        dir_tree_depth=1,
        run_on_rewrite=True,
        auto_view_change=True,
    )
    """
        "zero_shot": "input uuid here",  # uuid correspond to a folder in the output directory
        "cot": "input uuid here",
        "tadpole": "input uuid here",
        "zero_shot_nopdb": "input uuid here",
        "cot_nopdb": "input uuid here",
    """
    problem_list = env.dataset.keys()  # all tasks
    agent_uuids = {
        "zero_shot": "aa3baf71-dcdc-4d5c-9b41-8e689d6ef263",  # uuid correspond to a folder in the output directory
        "cot": "14ecc5fc-09d1-4761-ba40-1f1638be50c5",
        "tadpole": "36d5cef9-26ac-49b6-844e-aa20a450586a",
        "zero_shot_nopdb": "7a0f0fee-14c8-4f60-a444-6b4c6c2318c0",
        "cot_nopdb": "540ee2c7-ca01-47de-9501-a75e5a5c443a",
    }
    for agent_name in agent_uuids:
        log_path = pjoin("output_aider", agent_uuids[agent_name])
        _results = {}
        for problem_name in problem_list:
            log_file = pjoin(log_path, problem_name, "froggy.jsonl")
            if not os.path.exists(log_file):
                continue
            try:
                with open(log_file, "r") as f:
                    _log = json.load(f)
                assert "success" in _log
                _results[problem_name] = 1.0 if _log["success"] else 0.0
            except:
                _results[problem_name] = 0.0
                print(colored(f"Error reading {log_file}", "red"))

        # print result statistics
        success_rate = sum(_results.values()) / len(_results) if len(_results) > 0 else 0.0
        print(
            colored(
                f"{agent_name}: {success_rate:.2f} ({sum(_results.values())} out of {len(_results)})",
                "green",
            )
        )


if __name__ == "__main__":
    main()
