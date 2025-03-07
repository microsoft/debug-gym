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

            tool_counter = {
                "```view": 0,
                "```listdir": 0,
                "```pdb": 0,
                "```rewrite": 0,
                "```eval": 0,
                "other": 0,
            }

            for step in data.get("log", []):
                episode_length += 1
                if episode_length > 50:
                    break
                if step.get("action") is None:
                    continue
                flag = False
                for tool_key in tool_counter:
                    if step["action"].strip().startswith(tool_key):
                        tool_counter[tool_key] += 1
                        if tool_key == "```pdb":
                            if not step.get("obs").strip().startswith("Tool failure"):
                                print("=" * 20)
                                print(jsonl_file, step.get("step_id"))
                                print(step.get("action"))
                                print(step.get("obs"))
                        flag = True
                        break
                if not flag:
                    # print("=" * 20)
                    # print(step.get("action"))
                    tool_counter["other"] += 1

            results.append(
                {
                    "task": task,
                    "success": success,
                    "episode_length": episode_length,
                    "tool_counter": tool_counter,
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


def plot_tool_use_categories(df_dict, figsize=(12, 7)):
    """
    Creates a grouped hist plot showing the distribution of tool use categories for each model.
    Args:
        df_dict (dict): Dictionary mapping model names to their DataFrames with averaged results
        figsize (tuple): Figure size (width, height)
    """
    plt.figure(figsize=figsize)

    all_data = []
    # Create plot for each model
    for model_name, df in df_dict.items():
        # o1, o3-mini, o1, o3-mini, o1, o3-mini
        tool_category_per_model = {
            "```view": 0,
            "```listdir": 0,
            "```pdb": 0,
            "```rewrite": 0,
            "```eval": 0,
            "other": 0,
        }
        # import pdb; pdb.set_trace()
        tool_call_count = 0
        for _kv in df["tool_counter"].items():
            if _kv[1] == {}:
                continue
            # import pdb; pdb.set_trace()
            for k, v in _kv[1].items():
                tool_call_count += v
                tool_category_per_model[k] += v
        # percentage
        tool_category_per_model = {
            k: round(v / tool_call_count, 2) for k, v in tool_category_per_model.items()
        }
        all_data.append(
            [
                model_name,
                model_name.split("_")[1],
                tool_category_per_model["```view"],
                tool_category_per_model["```listdir"],
                tool_category_per_model["```pdb"],
                tool_category_per_model["```rewrite"],
                tool_category_per_model["```eval"],
                tool_category_per_model["other"],
            ]
        )
    # all_data = np.array(all_data)
    print(all_data)
    # import pdb; pdb.set_trace()
    # convert to DataFrame
    all_data = pd.DataFrame(
        all_data,
        columns=["name", "model", "view", "listdir", "pdb", "rewrite", "eval", "other"],
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
    plt.xticks(rotation=45)
    # custom x ticks
    plt.xticks(
        [0, 1, 2, 3, 4, 5],
        [
            "rewrite_o1",
            "rewrite_o3-mini",
            "debug_o1",
            "debug_o3-mini",
            "sc_o1",
            "sc_o3-mini",
        ],
    )

    # plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Example usage:
model_names = [
    # "../exps/swe-bench/rewrite_o1/rewrite_o1",
    # "../exps/swe-bench/rewrite_o3-mini/rewrite_o3-mini",
    # "../exps/swe-bench/pdb_o1/pdb_o1",
    "../exps/swe-bench/pdb_o3-mini/pdb_o3-mini",
    # "../exps/swe-bench/seq_o1/seq_o1",
    # "../exps/swe-bench/seq_o3-mini/seq_o3-mini",
]

# Analyze all models with seed averaging
results_dict = {
    name.split("/")[-1]: analyze_froggy_results_with_seeds(name) for name in model_names
}
# Plot comparison
# plot_tool_use_categories(results_dict)
