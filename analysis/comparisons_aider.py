import os
import json

import matplotlib.pyplot as plt

def create_analysis_plots(count_analysis, tool_analysis, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[1, 1])
    
    # Plot count analysis
    bars1 = ax1.bar(count_analysis.keys(), count_analysis.values())
    ax1.set_title('Count Analysis')
    ax1.tick_params(axis='x', rotation=45)
    # Add value labels on top of bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    # Plot tool analysis  
    bars2 = ax2.bar(tool_analysis.keys(), tool_analysis.values())
    ax2.set_title('Tool Analysis')
    ax2.tick_params(axis='x', rotation=45)
    # Add value labels on top of bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    # Adjust layout to prevent overlapping
    plt.tight_layout()
    
    # Save and show the plot
    plt.savefig(output_path)
    plt.close()

def parse_jsonl_files(root_folder, subfolder):
    average_summary = {}
    # Iterate through each subfolder in the root folder
    subfolder_path = os.path.join(root_folder, subfolder)
    # Check if the path is a directory
    if os.path.isdir(subfolder_path):
        task_summary = {}
        for task_name in os.listdir(subfolder_path):
            task_summary[task_name] = {"success": False, "rewrite": 0, "pdb": 0, "view": 0}
            
            jsonl_file_path = os.path.join(subfolder_path, task_name, 'froggy.jsonl')
            # Check if the jsonl file exists in the subfolder
            if os.path.isfile(jsonl_file_path):
                with open(jsonl_file_path, 'r') as file:
                    log_data = json.load(file)
                    logs = log_data['log']
                    for i in range(len(logs)):
                        if logs[i]["action"] is None:
                            continue
                        if "pdb" in logs[i]["action"]:
                            task_summary[task_name]["pdb"] += 1
                        if "view" in logs[i]["action"]:
                            task_summary[task_name]["view"] += 1
                        if "rewrite" in logs[i]["action"]:
                            task_summary[task_name]["rewrite"] += 1
                    if log_data["success"]:
                        task_summary[task_name]["success"] = 1
                    else:
                        task_summary[task_name]["success"] = 0
    return task_summary

def compare_task_summary(base_summary, agent_summary):
    count_analysis = {"reduce rewrite": 0, "increase rewrite": 0, "worse than baseline": 0, "better than baseline": 0, "same as baseline": 0}
    tool_analysis = {"reduce_rewrite_n_pdb_usage": 0, "reduce_rewrite_n_view_usage": 0, "task_reduce": 0, "better_baseline_n_pdb_usage": 0, "better_baseline_n_view_usage": 0, "task_better": 0}
    for task in base_summary:
        #print(task)
        if base_summary[task]["success"] == 1:
            if agent_summary[task]["success"] == 1:
                if base_summary[task]["rewrite"] >= agent_summary[task]["rewrite"]:
                    count_analysis["reduce rewrite"] += 1
                    tool_analysis["reduce_rewrite_n_pdb_usage"] += agent_summary[task]["pdb"]
                    tool_analysis["reduce_rewrite_n_view_usage"] += agent_summary[task]["view"]
                    tool_analysis["task_reduce"] += 1
                else:
                    count_analysis["increase rewrite"] += 1
            else:
                count_analysis["worse than baseline"] += 1
        else:
            if agent_summary[task]["success"] == 1:
                count_analysis["better than baseline"] += 1
                tool_analysis["better_baseline_n_pdb_usage"] += agent_summary[task]["pdb"]
                tool_analysis["better_baseline_n_view_usage"] += agent_summary[task]["view"]
                tool_analysis["task_better"] += 1
            else:
                count_analysis["same as baseline"] += 1 
    tool_analysis["reduce_rewrite_n_pdb_usage"] = tool_analysis["reduce_rewrite_n_pdb_usage"] / tool_analysis["task_reduce"]
    tool_analysis["reduce_rewrite_n_view_usage"] = tool_analysis["reduce_rewrite_n_view_usage"] / tool_analysis["task_reduce"]
    tool_analysis["better_baseline_n_pdb_usage"] = tool_analysis["better_baseline_n_pdb_usage"] / tool_analysis["task_better"]
    tool_analysis["better_baseline_n_view_usage"] = tool_analysis["better_baseline_n_view_usage"] / tool_analysis["task_better"]

    return count_analysis, tool_analysis

def success_rate(summary):
    success = 0
    total = 0
    for task in summary:
        total += 1
        if summary[task]["success"] == 1:
            success += 1
    return success / total

def average_analysis(total_count_analysis, total_tool_analysis, std=False):
    output_count_analysis = {}
    output_tool_analysis = {}
    for agent in total_count_analysis:
        for key in total_count_analysis[agent]:
            if key not in output_count_analysis:
                output_count_analysis[key] = total_count_analysis[agent][key] / len(total_count_analysis)
            else:
                output_count_analysis[key] += total_count_analysis[agent][key] / len(total_count_analysis)
        for key in total_tool_analysis[agent]:
            if key == "task_reduce" or key == "task_better":
                continue
            if key not in output_tool_analysis:
                output_tool_analysis[key] = total_tool_analysis[agent][key] / len(total_tool_analysis)
            else:
                output_tool_analysis[key] += total_tool_analysis[agent][key] / len(total_tool_analysis)
    
    return output_count_analysis, output_tool_analysis

# Define the root folder
root_folder = '../output'
jsonl_folder_name = './parse_folders.jsonl'
# Check if the jsonl file exists in the subfolder
if os.path.isfile(jsonl_folder_name):
    with open(jsonl_folder_name, 'r') as file:
        folder = json.load(file)
        subfolder_name_baseline = folder["subfolder_name_baseline"]
        subfolder_name_agent = folder["subfolder_name_agent"]
        output_fig_path = folder["output_fig_path"]

# Call the function to parse the jsonl files
average_count_analysis = {}
average_tool_analysis = {}
for baseline_name in subfolder_name_baseline:
    total_count_analysis = {}
    total_tool_analysis = {}
    for agent_name in subfolder_name_agent:
        baseline = parse_jsonl_files(root_folder, baseline_name)
        agent = parse_jsonl_files(root_folder, agent_name)

        count_analysis, tool_analysis = compare_task_summary(baseline, agent)
        total_count_analysis[agent_name] = count_analysis
        total_tool_analysis[agent_name] = tool_analysis
    
    average_count_analysis[baseline_name], average_tool_analysis[baseline_name] = average_analysis(total_count_analysis, total_tool_analysis)
    print(average_count_analysis[baseline_name])
    print(average_tool_analysis[baseline_name])

final_count, final_tool = average_analysis(average_count_analysis, average_tool_analysis, std=False)
# Create the plots
create_analysis_plots(final_count, final_tool, output_fig_path)
