import json
import os
from os.path import join as pjoin

from termcolor import colored

from froggy.envs import TerminalSimulatorEnv


def main():

    # create environment
    env = TerminalSimulatorEnv(
        bug_free_code_path="data/terminal_simulator",
        buggy_code_path="data/terminal_simulator/buggy/buggy_code_info_20241031-205241.json",
        dir_tree_depth=2,
        run_on_rewrite=True,
        auto_view_change=True,
    )

    problem_list = env.dataset.keys()  # all tasks
    agent_uuids = {
        "zero_shot": "input uuid here",  # uuid correspond to a folder in the output directory
        "zero_shot_nopdb": "input uuid here",
    }

    for agent_name in agent_uuids:
        log_path = pjoin("output_terminal_simulator", agent_uuids[agent_name])
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
