import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams.update(
    {
        "font.size": 20,  # Base font size
        "axes.labelsize": 20,  # Axis labels
        "axes.titlesize": 20,  # Plot title
        "xtick.labelsize": 15,  # X-axis tick labels
        "ytick.labelsize": 20,  # Y-axis tick labels
        "legend.fontsize": 20,  # Legend text
    }
)

agent_name_map = {
    "rewrite": "rewrite",
    "pdb": "debug",
    "seq": "second chance",
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


def plot_winning_time_per_game(df_dict, figsize=(12, 7)):
    """
    Creates a grouped bar plot showing how much time (seed) each agent solves the game.
    Args:
        df_dict (dict): Dictionary mapping model names to their DataFrames with averaged results
        figsize (tuple): Figure size (width, height)
    """
    plt.figure(figsize=figsize)

    all_data = []
    # Create plot for each model
    for agent_model, df in df_dict.items():
        # Group by task and calculate means
        grouped_df = (
            df.groupby("task")
            .agg({"success": "mean", "rewrite_count": "mean"})
            .reset_index()
        )
        grouped_df["success"] = grouped_df["success"] * 3
        # import pdb; pdb.set_trace()
        all_data.append([agent_name_map[agent_model.split("_")[0]], grouped_df])
    # ignore tasks where all agent_model failed or succeeded
    keep_index = []
    all_solved = 0
    all_failed = 0
    # import pdb; pdb.set_trace()
    for i in range(len(all_data[0][1])):
        if all([all_data[j][1]["success"][i] == 0 for j in range(len(all_data))]):
            # import pdb; pdb.set_trace()
            all_failed += 1
            continue
        if all([all_data[j][1]["success"][i] == 3 for j in range(len(all_data))]):
            # import pdb; pdb.set_trace()
            all_solved += 1
            continue
        keep_index.append(i)
    print("---------------", len(keep_index), all_solved, all_failed)
    for i in range(len(all_data)):
        all_data[i][1] = all_data[i][1].iloc[keep_index]

    # import pdb; pdb.set_trace()

    # Plot grouped bar plot
    x = np.arange(len(all_data[0][1]["task"]))
    width = 0.2

    # nice palette
    sns.set_palette("Set2")
    for i, (model_name, df) in enumerate(all_data):
        plt.bar(x + i * width, df["success"], width, label=model_name)

    plt.ylabel("Number of success in 3 runs")
    plt.xticks(x + width, all_data[0][1]["task"], rotation=90)
    plt.yticks(np.arange(0, 4, 1))
    plt.legend()
    plt.tight_layout()
    plt.show()


# Example usage:
model_names = [
    "../exps/aider/rewrite_4o/rewrite_4o",
    "../exps/aider/pdb_4o/pdb_4o",
    "../exps/aider/seq_4o/seq_4o",
]

# Analyze all models with seed averaging
results_dict = {
    name.split("/")[-1]: analyze_froggy_results_with_seeds(name) for name in model_names
}
# Plot comparison
plot_winning_time_per_game(results_dict)
