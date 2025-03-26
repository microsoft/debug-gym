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
            for step in data.get("log", []):
                if step.get("action") and step["action"].strip().startswith(
                    "```rewrite"
                ):
                    rewrite_count += 1

                # Extract token usage from prompt_response_pairs
                if step.get("prompt_response_pairs"):
                    for pair in step["prompt_response_pairs"]:
                        if isinstance(pair.get("token_usage"), dict):
                            total_prompt_tokens += pair["token_usage"].get("prompt", 0)
                            total_response_tokens += pair["token_usage"].get(
                                "response", 0
                            )

                if rewrite_count == 10:
                    break

            results.append(
                {
                    "task": task,
                    "success": success,
                    "rewrite_count": rewrite_count,
                    "prompt_tokens": total_prompt_tokens,
                    "response_tokens": total_response_tokens,
                }
            )

    df = pd.DataFrame(results)

    print("Success rate:", df["success"].mean())
    print("Average rewrites:", df["rewrite_count"].mean())
    print("Average prompt tokens:", df["prompt_tokens"].mean())
    print("Average response tokens:", df["response_tokens"].mean())
    print("\nResults by task:")
    print(df)
    return df


agent_name_map = {
    "rewrite_4o": "gpt-4o",
    "rewrite_r1-distill-qwen-32b": "r1-distill-qwen-32b",
}


def plot_multiple_cumulative_success(df_dict, figsize=(12, 7)):
    """
    Creates a comparative plot showing cumulative success rates vs number of rewrites for multiple models

    Args:
        df_dict (dict): Dictionary mapping model names to their DataFrames with averaged results
        figsize (tuple): Figure size (width, height)
    """
    plt.figure(figsize=figsize)

    # Create plot for each model
    for model_name, df in df_dict.items():
        # Sort by number of rewrites
        max_rewrites = max(df["rewrite_count"])

        avg_perf_per_rewrite = []
        std_dev_per_rewrite = []
        for i in range(int(max_rewrites) + 1):
            accumulated_success = []
            for seed in [0, 1, 2]:
                _df = df[df["rewrite_count"] <= i]
                _success = _df[_df["seed"] == seed]["success"].sum() / len(
                    df[df["seed"] == seed]
                )
                # import pdb; pdb.set_trace()
                accumulated_success.append(_success)
            avg_perf_per_rewrite.append(np.mean(accumulated_success))
            std_dev_per_rewrite.append(np.std(accumulated_success))

        final_success_rate = avg_perf_per_rewrite[-1]
        # Plot steps and points with success rate in legend
        plt.step(
            np.arange(max_rewrites + 1),
            avg_perf_per_rewrite,
            where="post",
            label=f"{agent_name_map[model_name]}",
        )
        plt.fill_between(
            np.arange(max_rewrites + 1),
            np.array(avg_perf_per_rewrite) - np.array(std_dev_per_rewrite),
            np.array(avg_perf_per_rewrite) + np.array(std_dev_per_rewrite),
            alpha=0.2,
            step="post",
        )
        # plt.scatter(np.arange(max_rewrites + 1), cumulative_success,
        #            alpha=0.3, marker='o')
        # Add light horizontal line at final average
        plt.axhline(y=final_success_rate, linestyle="--", alpha=0.2)

    plt.xlabel("Average Number of Rewrites")
    plt.ylabel("Cumulative Average Success Rate")
    plt.title("Cumulative Average Success Rates (Averaged Across 3 Runs)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    # plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


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

    print(f"\nAveraged results for {base_model_name}:")
    print(f"Success rate: {averaged_df['success'].mean():.2%}")
    print(f"Average rewrites: {averaged_df['rewrite_count'].mean():.2f}")

    return combined_df


# Example usage:
model_paths = [
    "../exps/aider/rewrite_4o",
    "../exps/aider/rewrite_r1-distill-qwen-32b",
]

# Analyze all models with seed averaging
results_dict = {}
for _path in tqdm(model_paths):
    _name = _path.split("/")[-1]
    results_dict[_name] = analyze_froggy_results_with_seeds(
        _path + "/" + _name, seeds=[0, 1, 2]
    )


# Plot comparison
plot_multiple_cumulative_success(results_dict)
# plot_multiple_cumulative_success_by_resp_tokens(results_dict)
