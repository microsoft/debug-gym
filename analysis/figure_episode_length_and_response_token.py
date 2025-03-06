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

    # print("Success rate:", df["success"].mean())
    # print("Average rewrites:", df["rewrite_count"].mean())
    # print("Average prompt tokens:", df["prompt_tokens"].mean())
    # print("Average response tokens:", df["response_tokens"].mean())
    # print("\nResults by task:")
    # print(df)
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

    # print(f"\nAveraged results for {base_model_name}:")
    # print(f"Success rate: {averaged_df['success'].mean():.2%}")
    # print(f"Average rewrites: {averaged_df['rewrite_count'].mean():.2f}")

    return combined_df


agent_name_map = {
    "rewrite": "rewrite",
    "pdb": "debug",
    "seq": "second chance",
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
    # import pdb; pdb.set_trace()
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
    plt.ylabel("Average Episode Length")
    plt.xlabel("Backbone LLM")
    plt.title("Average Episode Lengths (Averaged Across 3 Runs)")
    plt.xticks(rotation=90)
    # custom x ticks
    plt.xticks(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
        ],
        [
            "4o-mini",
            "4o",
            "o1",
            "o3-mini",
            "llama32-3b",
            "llama33-70b",
            "r1-distill-llama-70b",
            "r1-distill-qwen-32b",
            "4o-mini",
            "4o",
            "o1",
            "o3-mini",
            "llama32-3b",
            "llama33-70b",
            "r1-distill-llama-70b",
            "r1-distill-qwen-32b",
            "4o-mini",
            "4o",
            "o1",
            "o3-mini",
            "llama32-3b",
            "llama33-70b",
            "r1-distill-llama-70b",
            "r1-distill-qwen-32b",
        ],
    )
    # # cutsom legend with same three colors as above
    # plt.legend(
    #     ["rewrite", "debug", "second-chance"],
    #     loc="upper left",
    #     bbox_to_anchor=(1, 1),
    # )
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
    # all_data = np.array(all_data)
    print(all_data)
    # convert to DataFrame
    all_data = pd.DataFrame(
        all_data, columns=["name", "model", "agent", "response_tokens", "std"]
    )
    # import pdb; pdb.set_trace()
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

    ax1.set_ylim(2000, 10000)
    ax2.set_ylim(0, 1000)

    ax1.set_ylabel("")
    ax2.set_ylabel("")
    # then, set a new label on the plot (basically just a piece of text) and move it to where it makes sense (requires trial and error)
    f.text(
        0.05,
        0.55,
        "Average Response Tokens (Averaged Across 3 Runs)",
        va="center",
        rotation="vertical",
    )
    f.subplots_adjust(
        left=0.09, right=0.99, bottom=0.31, top=0.97, hspace=0.08, wspace=0.2
    )
    ax2.get_legend().remove()
    plt.xlabel("Backbone LLM")
    # plt.title("Average Response Tokens (Averaged Across 3 Runs)")
    plt.xticks(rotation=90)
    # custom x ticks
    plt.xticks(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
        ],
        [
            "4o-mini",
            "4o",
            "o1",
            "o3-mini",
            "llama32-3b",
            "llama33-70b",
            "r1-distill-llama-70b",
            "r1-distill-qwen-32b",
            "4o-mini",
            "4o",
            "o1",
            "o3-mini",
            "llama32-3b",
            "llama33-70b",
            "r1-distill-llama-70b",
            "r1-distill-qwen-32b",
            "4o-mini",
            "4o",
            "o1",
            "o3-mini",
            "llama32-3b",
            "llama33-70b",
            "r1-distill-llama-70b",
            "r1-distill-qwen-32b",
        ],
    )

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Example usage:
model_names = [
    "../exps/aider/rewrite_4o-mini/rewrite_4o-mini",
    "../exps/aider/rewrite_4o/rewrite_4o",
    "../exps/aider/rewrite_o1/rewrite_o1",
    "../exps/aider/rewrite_o3-mini/rewrite_o3-mini",
    "../exps/aider/rewrite_llama32-3b/rewrite_llama32-3b",
    "../exps/aider/rewrite_llama33-70b/rewrite_llama33-70b",
    "../exps/aider/rewrite_r1-distill-llama-70b/rewrite_r1-distill-llama-70b",
    "../exps/aider/rewrite_r1-distill-qwen-32b/rewrite_r1-distill-qwen-32b",
    "../exps/aider/pdb_4o-mini/pdb_4o-mini",
    "../exps/aider/pdb_4o/pdb_4o",
    "../exps/aider/pdb_o1/pdb_o1",
    "../exps/aider/pdb_o3-mini/pdb_o3-mini",
    "../exps/aider/pdb_llama32-3b/pdb_llama32-3b",
    "../exps/aider/pdb_llama33-70b/pdb_llama33-70b",
    "../exps/aider/pdb_r1-distill-llama-70b/pdb_r1-distill-llama-70b",
    "../exps/aider/pdb_r1-distill-qwen-32b/pdb_r1-distill-qwen-32b",
    "../exps/aider/seq_4o-mini/seq_4o-mini",
    "../exps/aider/seq_4o/seq_4o",
    "../exps/aider/seq_o1/seq_o1",
    "../exps/aider/seq_o3-mini/seq_o3-mini",
    "../exps/aider/seq_llama32-3b/seq_llama32-3b",
    "../exps/aider/seq_llama33-70b/seq_llama33-70b",
    "../exps/aider/seq_r1-distill-llama-70b/seq_r1-distill-llama-70b",
    "../exps/aider/seq_r1-distill-qwen-32b/seq_r1-distill-qwen-32b",
]

# Analyze all models with seed averaging
results_dict = {}
for name in tqdm(model_names):
    results_dict[name.split("/")[-1]] = analyze_froggy_results_with_seeds(name)

# Plot comparison
plot_episode_length(results_dict)
# plot_episode_response_tokens(results_dict)
