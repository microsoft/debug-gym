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

    for jsonl_file in glob.glob(f"{model_dir}/**/froggy.jsonl", recursive=True):
        # Get task name from directory path
        task = os.path.dirname(jsonl_file).split("/")[-1]

        with open(jsonl_file) as f:
            data = json.load(f)

            # Extract success status
            success = data.get("success", False)

            # Count rewrite commands
            total_prompt_tokens = 0
            total_response_tokens = 0
            rewrite_count = 0
            episode_length = 0
            for step in data.get("log", []):
                episode_length += 1
                if step.get("action") and "```rewrite" in step["action"]:
                    rewrite_count += 1

                # Extract token usage from prompt_response_pairs
                if step.get("prompt_response_pairs"):
                    for pair in step["prompt_response_pairs"]:
                        if isinstance(pair.get("token_usage"), dict):
                            total_prompt_tokens += pair["token_usage"].get("prompt", 0)
                            total_response_tokens += pair["token_usage"].get(
                                "response", 0
                            )

            results.append(
                {
                    "task": task,
                    "success": success,
                    "rewrite_count": rewrite_count,
                    "prompt_tokens": total_prompt_tokens,
                    "response_tokens": total_response_tokens,
                    "episode_length": episode_length,
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

    # Group by task and calculate means
    averaged_df = (
        combined_df.groupby("task")
        .agg({"success": "mean", "rewrite_count": "mean"})
        .reset_index()
    )

    return combined_df


agent_name_map = {
    "rewrite": "rewrite",
    "pdb": "debug",
    "seq": "debug(5)",
}


def plot_episode_length(df_dict, figsize=(12, 7)):
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
        df = df[df["success"]]
        for agent in ["rewrite", "pdb", "seq"]:
            if agent not in model_name:
                continue
            episode_length_mean = df["episode_length"].mean()
            episode_length_std = df["episode_length"].std()
            all_data.append(
                [
                    model_name,
                    model_name[len(agent) + 1 :],
                    agent_name_map[agent],
                    float(round(episode_length_mean, 2)),
                    float(round(episode_length_std, 2)),
                ]
            )
    # all_data = np.array(all_data)
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

    plt.ylim(0, 30)
    plt.ylabel("Steps")
    plt.xlabel("Backbone LLM")
    plt.xticks(rotation=90)
    # custom x ticks
    plt.xticks(
        np.arange(len(all_data)),
        [
            "llama-3b",
            "llama-70b",
            "r1-llama-70b",
            "r1-qwen-32b",
            "4o",
            "4o-mini",
            "o1",
            "o3-mini",
            "llama-3b",
            "llama-70b",
            "r1-llama-70b",
            "r1-qwen-32b",
            "4o",
            "4o-mini",
            "o1",
            "o3-mini",
            "llama-3b",
            "llama-70b",
            "r1-llama-70b",
            "r1-qwen-32b",
            "4o",
            "4o-mini",
            "o1",
            "o3-mini",
        ],
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_episode_response_tokens(df_dict, figsize=(12, 7)):
    """
    Creates a grouped bar chart showing episode lengths for multiple models, grouped by agent types (rewrite, pdb, seq), each bar is averaged over seeds (0, 1, 2, with error bars)
    Args:
        df_dict (dict): Dictionary mapping model names to their DataFrames with averaged results
        figsize (tuple): Figure size (width, height)
    """

    all_data = []
    # Create plot for each model
    for model_name, df in df_dict.items():
        # ignore the data points where the agent failed
        df = df[df["success"]]
        for agent in ["rewrite", "pdb", "seq"]:
            if agent not in model_name:
                continue
            response_tokens_mean = df["response_tokens"].mean()
            response_tokens_std = df["response_tokens"].std()
            all_data.append(
                [
                    model_name,
                    model_name[len(agent) + 1 :],
                    agent_name_map[agent],
                    float(round(response_tokens_mean, 2)),
                    float(round(response_tokens_std, 2)),
                ]
            )
    print(all_data)
    # convert to DataFrame
    all_data = pd.DataFrame(
        all_data, columns=["name", "model", "agent", "response_tokens", "std"]
    )
    # bar chart, with broken y-axis (0-1000) and (3000-7000)
    f, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, sharex=True)
    sns.barplot(
        data=all_data,
        x="name",
        y="response_tokens",
        hue="agent",
        palette="Set2",
        ax=ax1,
    )
    sns.barplot(
        data=all_data,
        x="name",
        y="response_tokens",
        hue="agent",
        palette="Set2",
        ax=ax2,
    )
    # add error bars
    ax1.errorbar(
        x=all_data["name"],
        y=all_data["response_tokens"],
        yerr=all_data["std"],
        fmt="none",
        capsize=5,
        color="black",
    )
    ax2.errorbar(
        x=all_data["name"],
        y=all_data["response_tokens"],
        yerr=all_data["std"],
        fmt="none",
        capsize=5,
        color="black",
    )

    ax1.set_ylim(2000, 17000)
    ax2.set_ylim(0, 1000)

    ax1.set_ylabel("")
    ax2.set_ylabel("")
    f.subplots_adjust(
        left=0.09, right=0.99, bottom=0.31, top=0.97, hspace=0.08, wspace=0.2
    )
    ax2.get_legend().remove()
    plt.xlabel("Backbone LLM")
    plt.ylabel("Response tokens")
    plt.xticks(rotation=90)
    # custom x ticks
    plt.xticks(
        np.arange(len(all_data)),
        [
            "llama-3b",
            "llama-70b",
            "r1-llama-70b",
            "r1-qwen-32b",
            "4o",
            "4o-mini",
            "o1",
            "o3-mini",
            "llama-3b",
            "llama-70b",
            "r1-llama-70b",
            "r1-qwen-32b",
            "4o",
            "4o-mini",
            "o1",
            "o3-mini",
            "llama-3b",
            "llama-70b",
            "r1-llama-70b",
            "r1-qwen-32b",
            "4o",
            "4o-mini",
            "o1",
            "o3-mini",
        ],
    )

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Example usage:
model_paths = [
    "../exps/aider/rewrite_llama32-3b",
    "../exps/aider/rewrite_llama33-70b",
    "../exps/aider/rewrite_r1-distill-llama-70b",
    "../exps/aider/rewrite_r1-distill-qwen-32b",
    "../exps/aider/rewrite_4o",
    "../exps/aider/rewrite_4o-mini",
    "../exps/aider/rewrite_o1",
    "../exps/aider/rewrite_o3-mini",
    "../exps/aider/pdb_llama32-3b",
    "../exps/aider/pdb_llama33-70b",
    "../exps/aider/pdb_r1-distill-llama-70b",
    "../exps/aider/pdb_r1-distill-qwen-32b",
    "../exps/aider/pdb_4o",
    "../exps/aider/pdb_4o-mini",
    "../exps/aider/pdb_o1",
    "../exps/aider/pdb_o3-mini",
    "../exps/aider/seq_llama32-3b",
    "../exps/aider/seq_llama33-70b",
    "../exps/aider/seq_r1-distill-llama-70b",
    "../exps/aider/seq_r1-distill-qwen-32b",
    "../exps/aider/seq_4o",
    "../exps/aider/seq_4o-mini",
    "../exps/aider/seq_o1",
    "../exps/aider/seq_o3-mini",
]

# Analyze all models with seed averaging
results_dict = {}
for _path in tqdm(model_paths):
    _name = _path.split("/")[-1]
    results_dict[_name] = analyze_froggy_results_with_seeds(
        _path + "/" + _name, seeds=[0, 1, 2]
    )
# Plot comparison
plot_episode_length(results_dict)
# plot_episode_response_tokens(results_dict)
