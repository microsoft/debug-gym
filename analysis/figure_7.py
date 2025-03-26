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

pdb_command_mapping = {
    "break": "b",
    "continue": "c",
    "clear": "cl",
    "print": "p",
    "next": "n",
    "list": "l",
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
            pdb_command_category = {
                "b": 0,
                "c": 0,
                "cl": 0,
                "p": 0,
                "n": 0,
                "l": 0,
                "other": 0,
            }
            for step in data.get("log", []):
                episode_length += 1
                if step.get("action") and "```rewrite" in step["action"]:
                    rewrite_count += 1

                if step.get("action") and step["action"].strip().startswith("```pdb"):
                    pdb_command = step["action"].split("```pdb", 1)[1].strip()
                    pdb_command = pdb_command.split(" ")[0]
                    pdb_command = pdb_command.strip("`").strip()
                    if (
                        pdb_command not in pdb_command_category
                        and pdb_command not in pdb_command_mapping
                    ):
                        pdb_command_category["other"] += 1
                    elif pdb_command in pdb_command_category:
                        pdb_command_category[pdb_command] += 1
                    else:
                        pdb_command_category[pdb_command_mapping[pdb_command]] += 1

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
                    "pdb_command_category": pdb_command_category,
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


def plot_pdb_command_categories(df_dict, figsize=(12, 7)):
    """
    Creates a grouped hist plot showing the distribution of PDB command categories for each model.
    Args:
        df_dict (dict): Dictionary mapping model names to their DataFrames with averaged results
        figsize (tuple): Figure size (width, height)
    """

    all_data = []
    # Create plot for each model
    for model_name, df in df_dict.items():
        pdb_category_per_model = {
            "b": 0,
            "c": 0,
            "cl": 0,
            "p": 0,
            "n": 0,
            "l": 0,
            "other": 0,
        }
        pdb_call_count = 0
        for _kv in df["pdb_command_category"].items():
            if _kv[1] == {}:
                continue
            # import pdb; pdb.set_trace()
            for k, v in _kv[1].items():
                pdb_call_count += v
                pdb_category_per_model[k] += v
        # percentage
        pdb_category_per_model = {
            k: round(v / pdb_call_count, 2) for k, v in pdb_category_per_model.items()
        }
        all_data.append(
            [
                model_name,
                model_name.split("_")[1],
                pdb_category_per_model["b"],
                pdb_category_per_model["c"],
                pdb_category_per_model["cl"],
                pdb_category_per_model["p"],
                pdb_category_per_model["n"],
                pdb_category_per_model["l"],
                pdb_category_per_model["other"],
            ]
        )
    print(all_data)
    # convert to DataFrame
    all_data = pd.DataFrame(
        all_data, columns=["name", "model", "b", "c", "cl", "p", "n", "l", "other"]
    )
    # nice palette
    palette = sns.color_palette("Set2")
    # set color
    sns.set_palette(palette)
    # stacked bar plot showing the distribution of PDB command categories for each model
    all_data.set_index("name")[["b", "c", "cl", "p", "n", "l", "other"]].plot(
        kind="bar", stacked=True, figsize=figsize
    )
    plt.title("Distribution of PDB command being issued")
    plt.xlabel("Backbone LLM")
    plt.ylabel("Percentage")
    plt.xticks(rotation=45)
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
        ],
    )
    plt.tight_layout()
    plt.show()


# Example usage:
model_paths = [
    "../exps/aider/pdb_llama32-3b",
    "../exps/aider/pdb_llama33-70b",
    "../exps/aider/pdb_r1-distill-llama-70b",
    "../exps/aider/pdb_r1-distill-qwen-32b",
    "../exps/aider/pdb_4o",
    "../exps/aider/pdb_4o-mini",
    "../exps/aider/pdb_o1",
    "../exps/aider/pdb_o3-mini",
]

# Analyze all models with seed averaging
results_dict = {}
for _path in tqdm(model_paths):
    _name = _path.split("/")[-1]
    results_dict[_name] = analyze_froggy_results_with_seeds(
        _path + "/" + _name, seeds=[0, 1, 2]
    )
# Plot comparison
plot_pdb_command_categories(results_dict)
