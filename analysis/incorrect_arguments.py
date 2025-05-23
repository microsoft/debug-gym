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
    Analyzes froggy.jsonl files for a given model to extract success rates and rewrite counts.

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

                # Count rewrite commands
                total_incorrect_arguments = 0
                episode_length = 0
                for step in data.get("log", []):
                    episode_length += 1
                    if episode_length == 50:
                        break
                    if "got an unexpected keyword argument" in step["obs"]:
                        total_incorrect_arguments += 1

                results.append(
                    {
                        "task": task,
                        "success": success,
                        "incorrect_arguments": total_incorrect_arguments
                        / episode_length
                        * 100.0,
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


def plot_incorrect_arguments(df_dict, figsize=(12, 7)):
    """
    Creates a grouped bar chart showing episode lengths for multiple models, grouped by agent types (rewrite, pdb, seq), each bar is averaged over seeds (0, 1, 2, with error bars)
    Args:
        df_dict (dict): Dictionary mapping model names to their DataFrames with averaged results
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
            incorrect_arguments_mean = df["incorrect_arguments"].mean()
            incorrect_arguments_std = df["incorrect_arguments"].std()
            all_data.append(
                [
                    model_name,
                    model_name[len(agent) + 1 :],
                    agent,
                    float(round(incorrect_arguments_mean, 2)),
                    float(round(incorrect_arguments_std, 2)),
                ]
            )
    print(all_data)
    # convert to DataFrame
    all_data = pd.DataFrame(
        all_data, columns=["name", "model", "agent", "incorrect arguments", "std"]
    )
    # bar chart
    sns.barplot(
        data=all_data, x="name", y="incorrect arguments", hue="agent", palette="Set2"
    )
    # add error bars
    plt.errorbar(
        x=all_data["name"],
        y=all_data["incorrect arguments"],
        yerr=all_data["std"],
        fmt="none",
        capsize=5,
        color="black",
    )

    plt.ylim(0, 60)
    plt.ylabel("Incorrect Arguments (%)")
    plt.xlabel("Backbone LLM")
    plt.xticks(rotation=90)
    # custom x ticks
    plt.xticks(
        np.arange(len(all_data)),
        [
            "4o",
            "4o1",
            "o3",
            "o3-high",
            "o3-mini",
            "o4-mini",
            "4o",
            "4o1",
            "o3",
            "o3-high",
            "o3-mini",
            "o4-mini",
        ],
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Example usage:
model_paths = [
    "../exps/may22/rewrite_4o",
    "../exps/may22/rewrite_o1",
    "../exps/may22/rewrite_o3",
    "../exps/may22/rewrite_o3-high",
    "../exps/may22/rewrite_o3-mini",
    "../exps/may22/rewrite_o4-mini",
    "../exps/may22/debug_4o",
    "../exps/may22/debug_o1",
    "../exps/may22/debug_o3",
    "../exps/may22/debug_o3-high",
    "../exps/may22/debug_o3-mini",
    "../exps/may22/debug_o4-mini",
]

# Analyze all models with seed averaging
results_dict = {}
for _path in tqdm(model_paths):
    _name = _path.split("/")[-1]
    results_dict[_name] = analyze_froggy_results_with_seeds(_path, seeds=[0])


# Plot comparison
plot_incorrect_arguments(results_dict)
