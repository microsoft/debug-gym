import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

plt.rcParams.update(
    {
        "font.size": 22,  # Base font size
        "axes.labelsize": 22,  # Axis labels
        "axes.titlesize": 22,  # Plot title
        "xtick.labelsize": 22,  # X-axis tick labels
        "ytick.labelsize": 22,  # Y-axis tick labels
        "legend.fontsize": 22,  # Legend text
    }
)

agent_name_map = {
    "rewrite": "rewrite",
    "pdb": "debug",
    "seq": "debug(5)",
}


def analyze_froggy_results(model_name):
    """
    Analyzes froggy.jsonl files for a given model to extract success rates and rewrite counts.

    Args:
        model_name (str): Path to the model directory (e.g. 'exps/aider/rewrite_4o_0')

    Returns:
        pd.DataFrame: DataFrame containing results by task
    """
    model_dir = os.path.join(model_name)
    results = []

    for jsonl_name in ["froggy.jsonl", "debug_gym.jsonl"]:
        for jsonl_file in glob.glob(f"{model_dir}/**/{jsonl_name}", recursive=True):
            # Get task name from directory path
            task = os.path.dirname(jsonl_file).split("/")[-1]

            with open(jsonl_file) as f:
                data = json.load(f)
                # Extract success status
                success = data.get("success", False)

                results.append(
                    {
                        "task": task,
                        "success": success,
                    }
                )

    df = pd.DataFrame(results)
    return df


def analyze_froggy_results_with_seeds(base_model_name, seeds=[0, 1, 2]):
    """
    Analyzes and averages results across different seeds for a base model name

    Args:
        base_model_name (str): Base path without seed (e.g. '../exps/aider/rewrite_o3-mini')
        seeds (list): List of seeds to average over

    Returns:
        pd.DataFrame: DataFrame containing averaged results by task
    """
    all_dfs = []

    for seed in seeds:
        model_path = f"{base_model_name}_{seed}"
        try:
            df = analyze_froggy_results(model_path)
        except:
            continue
        df["seed"] = seed
        all_dfs.append(df)

    # Combine all DataFrames
    combined_df = pd.concat(all_dfs)
    return combined_df


def plot_overlap_winning_games_between_agents(df_dict, figsize=(12, 7)):
    """
    There are three agents: rewrite, pdb, and seq. Each agent has 3 runs (seeds 0, 1, 2).
    Each agent has a different number of games won. The goal is to show how many games that are won by all agents by at least once, at least twice, or all three times.
    Creates a grouped bar plot showing it.
    Args:
        df_dict (dict): Dictionary mapping model names to their DataFrames with averaged results
        figsize (tuple): Figure size (width, height)
    """
    all_data = []
    all_llms = set()
    for model_name, df in df_dict.items():
        # extract the index of games that are won by this agent at least once, twice, or all three times
        # get the index of games won by this agent (across different seeds)
        won_games = (
            df[df["success"] == True].groupby("task").size().reset_index(name="count")
        )
        # won_at_least_once
        task_name_won_at_least_once = won_games[won_games["count"] >= 1].task.unique()
        # won_at_least_twice
        task_name_won_at_least_twice = won_games[won_games["count"] >= 2].task.unique()
        # won_all
        task_name_won_all = won_games[won_games["count"] == 3].task.unique()
        # add to all_data
        if model_name.split("_")[-1] not in all_llms:
            all_llms.add(model_name.split("_")[-1])
        all_data.append(
            {
                "agent": model_name.split("_")[0],
                "llm": model_name.split("_")[-1],
                "indices_won_at_least_once": task_name_won_at_least_once,
                "indices_won_at_least_twice": task_name_won_at_least_twice,
                "indices_won_all": task_name_won_all,
            }
        )

    new_data = []
    for _llm in ["llama33-70b", "4o", "4o-mini", "o1", "o3-mini", "claude37"]:
        # get the indices of all games won by this
        _indices_1, _indices_2, _indices_3 = [], [], []
        for _data in all_data:
            if _data["llm"] == _llm:
                _indices_1.append(_data["indices_won_at_least_once"])
                _indices_2.append(_data["indices_won_at_least_twice"])
                _indices_3.append(_data["indices_won_all"])
        # get the intersection of all indices
        _indices_1 = set.intersection(*map(set, _indices_1))
        _indices_2 = set.intersection(*map(set, _indices_2))
        _indices_3 = set.intersection(*map(set, _indices_3))
        # get the length of each indices
        _indices_1 = len(_indices_1)
        _indices_2 = len(_indices_2)
        _indices_3 = len(_indices_3)
        # add to new_data
        new_data.append(
            {
                "llm": _llm,
                ">= once": _indices_1,
                ">= twice": _indices_2,
                "three times": _indices_3,
            }
        )
    # create a dataframe from new_data
    df = pd.DataFrame(new_data)
    # melt the dataframe to long format
    df = pd.melt(df, id_vars=["llm"], value_vars=[">= once", ">= twice", "three times"])
    # rename the columns
    df.columns = ["llm", "won", "count"]
    # create a grouped bar plot
    palette = sns.color_palette("Set2")
    # set color
    sns.set_palette(palette)
    plt.figure(figsize=figsize)
    sns.barplot(data=df, x="llm", y="count", hue="won")
    plt.xlabel("LLM backbone")
    plt.ylabel("#Games succeeded by all agents (out of 300)")
    plt.yticks(
        np.arange(0, 141, 20),
        [
            "0",
            "20",
            "40",
            "60",
            "80",
            "100",
            "120",
            "140",
        ],
    )  # Set y-ticks for the first subplot
    plt.legend()
    # add grid
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_overlap_between_agents(df_dict, figsize=(12, 7)):
    """
    There are three agents: rewrite, pdb, and seq. Each agent has 3 runs (seeds 0, 1, 2).
    Because both the pdb and seq agents are developed based on the rewrite agent.
    We want to understand when the rewrite agent is able to solve a task (win at least once out of 3 runs), what is the probability that the pdb and seq agents are also able to solve the same task (win at least once out of 3 runs).
    Creates a grouped bar plot showing it.
    Args:
        df_dict (dict): Dictionary mapping model names to their DataFrames with averaged results
        figsize (tuple): Figure size (width, height)
    """
    all_data = []
    all_llms = set()
    for model_name, df in df_dict.items():
        # extract the index of games that are won by this agent at least once, twice, or all three times
        # get the index of games won by this agent (across different seeds)
        won_games = (
            df[df["success"] == True].groupby("task").size().reset_index(name="count")
        )
        # won_at_least_once
        task_name_won_at_least_once = won_games[won_games["count"] >= 1].task.unique()
        # add to all_data
        if model_name.split("_")[-1] not in all_llms:
            all_llms.add(model_name.split("_")[-1])
        all_data.append(
            {
                "agent": model_name.split("_")[0],
                "llm": model_name.split("_")[-1],
                "indices_won_at_least_once": task_name_won_at_least_once,
            }
        )

    new_data = []
    for _llm in ["llama33-70b", "4o", "4o-mini", "o1", "o3-mini", "claude37"]:
        # for _llm in ["o3-mini", "claude37"]:
        _indices_rewrite, _indices_pdb, _indices_seq = None, None, None
        for _data in all_data:
            if _data["llm"] != _llm:
                continue
            if _data["agent"] == "rewrite":
                _indices_rewrite = _data["indices_won_at_least_once"]
            elif _data["agent"] == "pdb":
                _indices_pdb = _data["indices_won_at_least_once"]
            elif _data["agent"] == "seq":
                _indices_seq = _data["indices_won_at_least_once"]
        # get the intersection between agents
        _indices_rewrite_pdb = set.intersection(
            set(_indices_rewrite), set(_indices_pdb)
        )
        _indices_rewrite_seq = set.intersection(
            set(_indices_rewrite), set(_indices_seq)
        )
        new_data.append(
            {
                "llm": _llm,
                "debug/rewrite": len(_indices_rewrite_pdb) / len(_indices_rewrite),
                "debug(5)/rewrite": len(_indices_rewrite_seq) / len(_indices_rewrite),
            }
        )
    # create a dataframe from new_data
    df = pd.DataFrame(new_data)
    # melt the dataframe to long format

    # melt the dataframe to long format
    df = pd.melt(df, id_vars=["llm"], value_vars=["debug/rewrite", "debug(5)/rewrite"])
    # rename the columns
    df.columns = ["llm", "won", "count"]
    # create a grouped bar plot
    palette = sns.color_palette("Set2")
    # set color
    sns.set_palette(palette)
    plt.figure(figsize=figsize)
    sns.barplot(data=df, x="llm", y="count", hue="won")
    plt.xlabel("LLM backbone")
    plt.ylabel("Proportion")
    plt.yticks(
        np.arange(0, 1.1, 0.2),
        [
            "0",
            "20%",
            "40%",
            "60%",
            "80%",
            "100%",
        ],
    )  # Set y-ticks for the first subplot
    plt.legend()
    # add grid
    plt.grid(True, alpha=0.3)
    plt.show()
    # import pdb; pdb.set_trace()


# Example usage:
model_paths = [
    "../exps/swe-bench/rewrite_llama33-70b",
    "../exps/swe-bench/rewrite_4o",
    "../exps/swe-bench/rewrite_4o-mini",
    "../exps/swe-bench/rewrite_o1",
    "../exps/swe-bench/rewrite_o3-mini",
    "../exps/swe-bench/rewrite_claude37",
    "../exps/swe-bench/pdb_llama33-70b",
    "../exps/swe-bench/pdb_4o",
    "../exps/swe-bench/pdb_4o-mini",
    "../exps/swe-bench/pdb_o1",
    "../exps/swe-bench/pdb_o3-mini",
    "../exps/swe-bench/pdb_claude37",
    "../exps/swe-bench/seq_llama33-70b",
    "../exps/swe-bench/seq_4o",
    "../exps/swe-bench/seq_4o-mini",
    "../exps/swe-bench/seq_o1",
    "../exps/swe-bench/seq_o3-mini",
    "../exps/swe-bench/seq_claude37",
]

# Analyze all models with seed averaging
results_dict = {}
for _path in tqdm(model_paths):
    _name = _path.split("/")[-1]
    results_dict[_name] = analyze_froggy_results_with_seeds(
        _path + "/" + _name, seeds=[0, 1, 2]
    )
# Plot comparison
plot_overlap_between_agents(results_dict)
