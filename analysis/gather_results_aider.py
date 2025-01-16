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
    problem_list = env.dataset.keys()  # all tasks
    jsonl_folder_name = './analysis/parse_folders.jsonl'
    # Check if the jsonl file exists in the subfolder
    if os.path.isfile(jsonl_folder_name):
        with open(jsonl_folder_name, 'r') as file:
            jsonl_dic = json.load(file)
            agent_uuids = {k: v for k, v in jsonl_dic.items() if k not in ["baseline", "agent", "output_name_path"]}
    for agent_name in agent_uuids:
        for i in range(len(agent_uuids[agent_name])):
            log_path = pjoin("output_aider", agent_uuids[agent_name][i])
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
                    f"{agent_name} {agent_uuids[agent_name][i]}: {success_rate:.2f} ({sum(_results.values())} out of {len(_results)})",
                    "green",
                )
            )


if __name__ == "__main__":
    main()
