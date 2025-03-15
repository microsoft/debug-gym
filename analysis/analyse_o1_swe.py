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
        "font.size": 20,  # Base font size
        "axes.labelsize": 20,  # Axis labels
        "axes.titlesize": 20,  # Plot title
        "xtick.labelsize": 20,  # X-axis tick labels
        "ytick.labelsize": 20,  # Y-axis tick labels
        "legend.fontsize": 20,  # Legend text
    }
)


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

    for jsonl_file in glob.glob(f"{model_dir}/**/froggy.jsonl", recursive=True):
        # Get task name from directory path
        task = os.path.dirname(jsonl_file).split("/")[-1]

        with open(jsonl_file) as f:
            data = json.load(f)

            # Extract success status
            success = data.get("success", False)

            # Count rewrite commands
            episode_length = 0

            rewrite_success = []
            rewrite_length = []
            rewrite_repeat = 0
            seen_rewrite = set()

            for step in data.get("log", []):
                episode_length += 1
                if episode_length > 50:
                    break
                if step.get("action") is None:
                    continue
                if step["action"].strip().startswith("```rewrite"):
                    action_length = len(step["action"].split())
                    rewrite_length.append(action_length)
                    success = "Rewriting done." in step["obs"]
                    rewrite_success.append(int(success))
                    if step["action"].strip() not in seen_rewrite:
                        seen_rewrite.add(step["action"].strip())
                    else:
                        rewrite_repeat += 1

            # if len(seen_rewrite) == len(rewrite_length):
            #     rewrite_repeat = -1
            # else:
            #     rewrite_repeat = rewrite_repeat / float(len(rewrite_length)) if len(rewrite_length) > 0 else 0
            if rewrite_repeat > 0:
                rewrite_repeat = 1

            results.append(
                {
                    "task": task,
                    "success": success,
                    "episode_length": episode_length,
                    "rewrite_success": rewrite_success,
                    "rewrite_length": rewrite_length,
                    "rewrite_repeat": rewrite_repeat,
                }
            )

    df = pd.DataFrame(results)
    # import pdb; pdb.set_trace()
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


def plot_rewrite_stats(df_dict, figsize=(12, 7)):
    """
    Compute the average rewrite success rate and average rewrite length for each model
    Args:
        df_dict (dict): Dictionary mapping model names to their DataFrames with averaged results
        figsize (tuple): Figure size (width, height)
    """
    plt.figure(figsize=figsize)

    all_data = []
    for model_name, df in df_dict.items():
        # Compute the average rewrite success rate and average rewrite length
        avg_rewrite_success = df["rewrite_success"].apply(lambda x: np.mean(x)).mean()
        avg_rewrite_length = df["rewrite_length"].apply(lambda x: np.mean(x)).mean()
        # ignore if rewrite_repeat is -1
        avg_rewrite_repeat = (
            df[df["rewrite_repeat"] != -1]["rewrite_repeat"].mean()
            if len(df[df["rewrite_repeat"] != -1]) > 0
            else 0
        )

        all_data.append(
            [model_name, avg_rewrite_success, avg_rewrite_length, avg_rewrite_repeat]
        )

    # all_data = np.array(all_data)
    print(all_data)
    import pdb

    pdb.set_trace()
    # convert to DataFrame
    all_data = pd.DataFrame(
        all_data,
        columns=["name", "avg_rewrite_success", "avg_rewrite_length"],
    )
    # import pdb; pdb.set_trace()
    # nice palette
    palette = sns.color_palette("Set2")
    # set color
    sns.set_palette(palette)
    # stacked bar plot showing the distribution of PDB command categories for each model
    all_data.set_index("name")[
        ["view", "listdir", "pdb", "rewrite", "eval", "other"]
    ].plot(kind="bar", stacked=True, figsize=figsize)
    plt.title("Distribution of tool calls")
    plt.xlabel("Backbone LLM")
    plt.ylabel("Percentage")
    plt.xticks(rotation=90)
    # custom x ticks
    plt.xticks(
        np.arange(len(all_data)),
        ["rewrite", "debug", "second chance"],
    )

    # plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Example usage:
model_paths = [
    "../exps/swe-bench/rewrite_4o",
    "../exps/swe-bench/pdb_4o",
    "../exps/swe-bench/seq_4o",
    "../exps/swe-bench/rewrite_o1",
    "../exps/swe-bench/pdb_o1",
    "../exps/swe-bench/seq_o1",
    "../exps/swe-bench/rewrite_o3-mini",
    "../exps/swe-bench/pdb_o3-mini",
    "../exps/swe-bench/seq_o3-mini",
]

# Analyze all models with seed averaging
results_dict = {}
for _path in tqdm(model_paths):
    _name = _path.split("/")[-1]
    results_dict[_name] = analyze_froggy_results_with_seeds(
        _path + "/" + _name, seeds=[0, 1, 2]
    )

# Plot comparison
plot_rewrite_stats(results_dict)
