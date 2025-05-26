import codecs
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
ONLY_SUCCESS = False


def clean_code(code):
    assert isinstance(code, str)
    code_line = unescape(code).split("\n")
    # Remove trailing white spaces with rstrip.
    return "\n".join(line.rstrip() for line in code_line)


def unescape(s):
    try:
        # First, try the normal unescape
        result = codecs.decode(s, "unicode_escape")
        # Test if it can be encoded to UTF-8 (which will happen during JSON encoding)
        result.encode("utf-8")
        return result
    except UnicodeEncodeError:
        # If it contains surrogate pairs that can't be encoded to UTF-8,
        # replace them with the Unicode replacement character (U+FFFD)
        result = codecs.decode(s, "unicode_escape")
        return result.encode("utf-8", errors="replace").decode("utf-8")


def parse_line_numbers(line_number_string):
    # only line number is provided
    line_numbers = line_number_string.split(":")
    line_numbers = [item.strip() for item in line_numbers]
    if len(line_numbers) not in [1, 2]:
        return None
    if len(line_numbers) == 1:
        if int(line_numbers[0]) <= 0:
            return None
        # only head is provided (rewrite that line)
        head = int(line_numbers[0]) - 1  # 1-based to 0-based
        tail = head
    else:
        # both head and tail are provided
        if int(line_numbers[0]) <= 0 or int(line_numbers[1]) <= 0:
            return None
        if int(line_numbers[0]) > int(line_numbers[1]):
            return None
        head = int(line_numbers[0]) - 1  # 1-based to 0-based
        tail = int(line_numbers[1]) - 1  # 1-based to 0-based
    return [head, tail]


def is_comment(line):
    line = line.strip()
    if line.startswith("#"):
        return True
    return False


def parse_action(action_string):
    action_string = action_string.split("```rewrite", 1)[1].strip()
    new_code = action_string.split("<c>", 1)[1].split("</c>", 1)[0]
    content = action_string.split("<c>", 1)[0].strip()
    # code/utils.py 4:6
    content_list = content.split()
    if len(content_list) == 0:
        # no file path and line number is provided
        raise ValueError("fail to parse.")
    elif len(content_list) == 1:
        # either file path or line number is provided
        if content_list[0][0].isnumeric():
            # only line number is provided
            head_tail = parse_line_numbers(content_list[0])
    elif len(content_list) == 2:
        # both file path and line number are provided
        head_tail = parse_line_numbers(content_list[1])
    else:
        raise ValueError("fail to parse.")
    _from = head_tail[1] - head_tail[0] + 1

    new_code = clean_code(new_code)  # str
    new_code_lines = new_code.split("\n")
    new_code_length = len(new_code_lines)
    _comment = np.sum([float(is_comment(line)) for line in new_code_lines])
    _to = new_code_length
    return _from, _to, _comment


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
                _from_list = []
                _to_list = []
                _comment_list = []
                episode_length = 0
                for step in data.get("log", []):
                    episode_length += 1
                    if episode_length == 50:
                        break
                    if step.get("action") and "```rewrite" in step["action"]:
                        try:
                            _from, _to, _comment = parse_action(step["action"])
                            _from_list.append(_from)
                            _to_list.append(_to)
                            _comment_list.append(_comment)
                        except:
                            continue
                results.append(
                    {
                        "task": task,
                        "success": success,
                        "_from": _from_list,
                        "_to": _to_list,
                        "comment": _comment_list,
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

    # Group by task and calculate means
    averaged_df = combined_df.groupby("task").agg({"success": "mean"}).reset_index()

    return combined_df


agent_name_map = {
    "rewrite": "rewrite",
    "pdb": "debug",
    "seq": "debug(5)",
}


def plot_rewrite_length_grouped(df_dict, figsize=(12, 7)):
    """
    Creates a grouped bar chart showing how many lines of code the rewrite tool tries to remove and add.
    Args:
        df_dict (dict): Dictionary mapping model names to their DataFrames with averaged results
        figsize (tuple): Figure size (width, height)
    """

    all_data = []
    for agent in ["rewrite", "pdb", "seq"]:
        _tmp_data = []
        # Create plot for each model
        for model_name, df in df_dict.items():
            # ignore the data points where the agent failed
            if ONLY_SUCCESS:
                df = df[df["success"]]
            if agent not in model_name:
                continue
            # df["_from"] is a list of lists, so we need to flatten it
            _from = [item for sublist in df["_from"] for item in sublist]
            _to = [item for sublist in df["_to"] for item in sublist]
            _comment = [item for sublist in df["comment"] for item in sublist]
            # import pdb; pdb.set_trace()
            rewrite_from_mean = np.mean(_from)
            rewrite_to_mean = np.mean(_to)
            rewrite_comment_mean = np.mean(_comment)
            _tmp_data.append(
                [
                    model_name,
                    model_name[len(agent) + 1 :],
                    agent_name_map[agent],
                    float(round(rewrite_from_mean, 2)),
                    float(round(rewrite_to_mean, 2)),
                    float(round(rewrite_comment_mean, 2)),
                ]
            )
        all_data.append(_tmp_data)
    print(all_data)
    # convert to DataFrame
    all_data = [
        pd.DataFrame(
            item, columns=["name", "model", "agent", "deleted", "added", "comment"]
        )
        for item in all_data
    ]
    # melt the DataFrame to long format
    all_data = [
        item.melt(
            id_vars=["name", "model", "agent"],
            value_vars=["deleted", "added", "comment"],
        )
        for item in all_data
    ]
    all_data = [
        item.rename(
            columns={
                "variable": "rewrite type",
                "value": "lines",
            }
        )
        for item in all_data
    ]
    # create three grouped bar plot, one for each agent
    _data_rewrite = all_data[0]
    _data_debug = all_data[1]
    _data_seq = all_data[2]

    palette = sns.color_palette("Set2")
    # set color
    sns.set_palette(palette)
    f, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1, sharey=True)

    sns.barplot(
        data=_data_rewrite,
        x="name",
        y="lines",
        hue="rewrite type",
        palette="Set2",
        hue_order=["deleted", "added", "comment"],
        dodge=True,
        edgecolor="black",
        linewidth=0.5,
        ax=ax1,
    )
    ax1.set_title("rewrite")
    ax1.set_ylabel("# lines")
    ax1.set_ylim(0, 20)
    ax1.set_yticks(np.arange(0, 21, 5))  # Set y-ticks for the first subplot
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("Backbone LLM")
    # there are 6 models, each has two bars (grouped), we only need 6 x ticks.
    ax1.set_xticks(np.arange(6))  # Set x-ticks for the first subplot
    ax1.set_xticklabels(
        [
            "llama33-70b",
            "4o",
            "4o-mini",
            "o1",
            "o3-mini",
            "claude37",
        ]
    )  # Set x-tick labels for the first subplot
    # rotate x tick labels
    ax1.tick_params(axis="x", rotation=90)
    # remove the legend title
    ax1.get_legend().set_title("")

    sns.barplot(
        data=_data_debug,
        x="name",
        y="lines",
        hue="rewrite type",
        palette="Set2",
        hue_order=["deleted", "added", "comment"],
        dodge=True,
        edgecolor="black",
        linewidth=0.5,
        ax=ax2,
    )
    ax2.set_title("debug")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("Backbone LLM")
    # there are 6 models, each has two bars (grouped), we only need 6 x ticks.
    ax2.set_xticks(np.arange(6))  # Set x-ticks for the first subplot
    ax2.set_xticklabels(
        [
            "llama33-70b",
            "4o",
            "4o-mini",
            "o1",
            "o3-mini",
            "claude37",
        ]
    )  # Set x-tick labels for the first subplot
    # rotate x tick labels
    ax2.tick_params(axis="x", rotation=90)
    # remove legend from ax2
    ax2.get_legend().remove()

    sns.barplot(
        data=_data_seq,
        x="name",
        y="lines",
        hue="rewrite type",
        palette="Set2",
        hue_order=["deleted", "added", "comment"],
        dodge=True,
        edgecolor="black",
        linewidth=0.5,
        ax=ax3,
    )
    ax3.set_title("debug(5)")
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel("Backbone LLM")
    # there are 6 models, each has two bars (grouped), we only need 6 x ticks.
    ax3.set_xticks(np.arange(6))  # Set x-ticks for the first subplot
    ax3.set_xticklabels(
        [
            "llama33-70b",
            "4o",
            "4o-mini",
            "o1",
            "o3-mini",
            "claude37",
        ]
    )  # Set x-tick labels for the first subplot
    # rotate x tick labels
    ax3.tick_params(axis="x", rotation=90)
    # remove legend from ax2
    ax3.get_legend().remove()

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Example usage:
model_paths = [
    "../exps/swe-bench/rewrite_llama33-70b",
    "../exps/swe-bench/rewrite_4o",
    "../exps/swe-bench/rewrite_4o-mini",
    "../exps/swe-bench/rewrite_o1",
    "../exps/swe-bench/rewrite_o3-mini",
    "../exps/swe-bench/rewrite_claude37",
    "../exps/swe-bench/pdb_llama33-70b",
    "../exps/swe-bench/pdb_4o",
    "../exps/swe-bench/pdb_4o-mini",
    "../exps/swe-bench/pdb_o1",
    "../exps/swe-bench/pdb_o3-mini",
    "../exps/swe-bench/pdb_claude37",
    "../exps/swe-bench/seq_llama33-70b",
    "../exps/swe-bench/seq_4o",
    "../exps/swe-bench/seq_4o-mini",
    "../exps/swe-bench/seq_o1",
    "../exps/swe-bench/seq_o3-mini",
    "../exps/swe-bench/seq_claude37",
]

# Analyze all models with seed averaging
results_dict = {}
for _path in tqdm(model_paths):
    _name = _path.split("/")[-1]
    results_dict[_name] = analyze_froggy_results_with_seeds(
        _path + "/" + _name, seeds=[0, 1, 2]
    )

# Plot comparison
plot_rewrite_length_grouped(results_dict)
