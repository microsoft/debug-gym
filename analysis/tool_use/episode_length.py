import argparse
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
ONLY_SUCCESS = True


def analyze_froggy_results(model_name):
    """
    Analyzes froggy.jsonl files for a given model to extract success rates and episode lengths.

    Args:
        model_name (str): Path to the model directory (e.g. 'exps/swe-bench/rewrite_4o_0')

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

                # Count episode length
                episode_length = 0
                for step in data.get("log", []):
                    episode_length += 1
                    if episode_length == 50:
                        break

                results.append(
                    {
                        "task": task,
                        "success": success,
                        "episode_length": episode_length,
                    }
                )

    df = pd.DataFrame(results)
    return df


def analyze_froggy_results_with_seeds(base_model_name, seeds=[0, 1, 2]):
    """
    Analyzes and averages results across different seeds for a base model name

    Args:
        base_model_name (str): Base path without seed (e.g. '../exps/swe-bench/rewrite_o3-mini')
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


def plot_episode_length(df_dict, model_paths, figsize=(12, 7)):
    """
    Creates a grouped bar chart showing episode lengths for multiple models, grouped by agent types (rewrite, debug), each bar is averaged over seeds with error bars.
    Args:
        df_dict (dict): Dictionary mapping model names to their DataFrames with averaged results
        model_paths (list): List of model paths for custom x-tick labels
        figsize (tuple): Figure size (width, height)
    """
    plt.figure(figsize=figsize)

    all_data = []
    # Create plot for each model
    for model_name, df in df_dict.items():
        # ignore the data points where the agent failed
        if ONLY_SUCCESS:
            df = df[df["success"]]
        for agent in ["rewrite", "debug"]:
            if agent not in model_name:
                continue
            episode_length_mean = df["episode_length"].mean()
            episode_length_std = df["episode_length"].std()
            all_data.append(
                [
                    model_name,
                    model_name[len(agent) + 1 :],
                    agent,
                    float(round(episode_length_mean, 2)),
                    float(round(episode_length_std, 2)),
                ]
            )
    print(all_data)
    # convert to DataFrame
    all_data = pd.DataFrame(
        all_data, columns=["name", "model", "agent", "episode length", "std"]
    )
    # bar chart
    sns.barplot(
        data=all_data, x="name", y="episode length", hue="agent", palette="Set2"
    )
    # add error bars
    plt.errorbar(
        x=all_data["name"],
        y=all_data["episode length"],
        yerr=all_data["std"],
        fmt="none",
        capsize=5,
        color="black",
    )

    plt.ylim(0, 40)
    plt.ylabel("Steps")
    plt.xlabel("Backbone LLM")
    plt.xticks(rotation=90)
    # custom x ticks
    plt.xticks(
        np.arange(len(all_data)),
        [item.split("_")[-1] for item in model_paths],
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Analyze and plot episode lengths for debug-gym experiments"
    )
    parser.add_argument(
        "--exp-path",
        required=True,
        help="Path to experiments directory, e.g., '../../exps/'",
    )
    parser.add_argument(
        "--exp-uuid-list",
        nargs="+",  # Accept one or more arguments
        required=True,
        help="A list of experiment UUID/name to analyze, e.g., exp1 exp2 exp3",
    )

    args = parser.parse_args()
    model_paths = [os.path.join(args.exp_path, item) for item in args.exp_uuid_list]

    # Analyze all models with seed averaging
    results_dict = {}
    for _path in tqdm(model_paths):
        _name = _path.split("/")[-1]
        results_dict[_name] = analyze_froggy_results_with_seeds(_path, seeds=[0])

    # Plot comparison
    plot_episode_length(results_dict, model_paths)


if __name__ == "__main__":
    main()
