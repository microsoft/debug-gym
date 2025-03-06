import glob
import json
import os

import pandas as pd


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
        if "django" not in task:
            continue

        with open(jsonl_file) as f:
            data = json.load(f)

            # Extract success status
            success = data.get("success", False)

            # Count rewrite commands
            rewrite_actions = []
            rewrite_step_number = []
            episode_length = 0
            for step in data.get("log", []):
                episode_length += 1
                if step.get("action") and "```rewrite" in step["action"]:
                    rewrite_actions.append(step["action"])
                    rewrite_step_number.append(episode_length)

            results.append(
                {
                    "task": task,
                    "success": success,
                    "rewrite_actions": rewrite_actions,
                    "rewrite_step_numbers": rewrite_step_number,
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

    return combined_df


def find_rewrite_test(df_dict, figsize=(12, 7)):
    """
    Scan all rewrite actions, find if any rewrite action was trying to rewrite the test file.
    Args:
        df_dict (dict): Dictionary mapping model names to their DataFrames with averaged results
        figsize (tuple): Figure size (width, height)
    """
    suspicious_actions = []
    # Create plot for each model
    # import pdb; pdb.set_trace()
    for agent_model, df in df_dict.items():
        # Find all rewrite actions
        for _, row in df.iterrows():
            for i, rewrite_action in enumerate(row["rewrite_actions"]):
                # remove content after <c>
                rewrite_action = rewrite_action.split("<c>")[0]
                if "test" in rewrite_action:
                    suspicious_actions.append(
                        {
                            "agent": agent_model,
                            "task": row["task"],
                            "seed": row["seed"],
                            "action": rewrite_action,
                            "step": row["rewrite_step_numbers"][i],
                            "success": row["success"],
                        }
                    )
    # print all suspicious actions
    for action in suspicious_actions:
        if action["success"] is False:
            continue
        print("=" * 20)
        print(
            f"Seed: {action['seed']}, Task: {action['task']}, Step: {action['step']}, Action: {action['action']}, Success: {action['success']}"
        )


# Example usage:
model_names = [
    "../exps/swe-bench/seq_o1/seq_o1",
    # "../exps/aider/pdb_r1-distill-qwen-32b/pdb_r1-distill-qwen-32b",
]

# Analyze all models with seed averaging
results_dict = {
    name.split("/")[-1]: analyze_froggy_results_with_seeds(name) for name in model_names
}
# Plot comparison
find_rewrite_test(results_dict)
