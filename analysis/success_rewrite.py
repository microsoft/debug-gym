import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def analyze_froggy_results(model_name):
    """
    Analyzes froggy.jsonl files for a given model to extract success rates and rewrite counts.

    Args:
        model_name (str): Path to the model directory (e.g. 'feb24/aider/rewrite_4o_0')

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
        for i in range(int(max_rewrites) + 1):
            avg_perf_per_rewrite.append(
                df[df["rewrite_count"] <= i]["success"].sum() / len(df)
            )

        final_success_rate = avg_perf_per_rewrite[-1]
        # Plot steps and points with success rate in legend
        plt.step(
            np.arange(max_rewrites + 1),
            avg_perf_per_rewrite,
            where="post",
            label=f"{model_name} ({final_success_rate:.1%})",
        )
        # plt.scatter(np.arange(max_rewrites + 1), cumulative_success,
        #            alpha=0.3, marker='o')
        # Add light horizontal line at final average
        plt.axhline(y=final_success_rate, linestyle="--", alpha=0.2)

    plt.xlabel("Average Number of Rewrites")
    plt.ylabel("Cumulative Average Success Rate")
    plt.title("Cumulative Average Success Rates (Averaged Across Seeds)")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_multiple_cumulative_success_by_resp_tokens(
    df_dict, figsize=(12, 7), num_points=100
):
    """
    Creates a comparative plot showing cumulative success rates vs response tokens for multiple models

    Args:
        df_dict (dict): Dictionary mapping model names to their DataFrames with averaged results
        figsize (tuple): Figure size (width, height)
        num_points (int): Number of points to sample for the curve (default: 100)
    """
    plt.figure(figsize=figsize)

    for model_name, df in df_dict.items():
        # Get response tokens array and success array
        tokens = df["response_tokens"].values
        success = df["success"].values

        # Create logarithmically spaced points for better visualization
        max_tokens = tokens.max()
        token_points = np.geomspace(1, max_tokens, num_points)
        token_points = np.unique(np.round(token_points)).astype(int)

        # Vectorized computation of success rates
        avg_perf = np.array(
            [success[tokens <= t].sum() / len(df) for t in token_points]
        )

        final_success_rate = df["success"].mean()

        # Plot with interpolation
        plt.plot(
            token_points, avg_perf, label=f"{model_name} ({final_success_rate:.1%})"
        )
        plt.scatter(token_points, avg_perf, alpha=0.3, marker="o")

        # Add light horizontal line at final average
        plt.axhline(y=final_success_rate, linestyle="--", alpha=0.2)

    plt.xscale("log")  # Use log scale for better visualization
    plt.xlabel("Maximum Response Tokens")
    plt.ylabel("Average Success Rate")
    plt.title("Success Rate vs Response Tokens (Averaged Across Seeds)")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def analyze_froggy_results_with_seeds(base_model_name, seeds=[0, 1, 2]):
    """
    Analyzes and averages results across different seeds for a base model name

    Args:
        base_model_name (str): Base path without seed (e.g. '../feb24/aider/rewrite_o3-mini')
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
model_names = [
    "../feb24/aider/rewrite_o3-mini",
    "../feb24/aider/pdb_o3-mini",
    "../feb24/aider/seq_o3-mini",
]

# Analyze all models with seed averaging
results_dict = {
    name.split("/")[-1]: analyze_froggy_results_with_seeds(name) for name in model_names
}

# Plot comparison
plot_multiple_cumulative_success(results_dict)
plot_multiple_cumulative_success_by_resp_tokens(results_dict)
