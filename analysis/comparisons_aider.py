import os
import json
import matplotlib.pyplot as plt
# Define base command dictionary template
base_commands = {
        'h': 0, 'w': 0, 'd': 0, 'u': 0, 'b': 0, 'cl': 0, 's': 0, 'n': 0,
        'unt': 0, 'c': 0, 'l': 0, 'll': 0, 'a': 0, 'p': 0, 'pp': 0,
        'q': 0, 'whatis': 0, 'wrong': 0
    }
def setup_bar_plot(ax, data, title, ylabel, color_scheme=None, integer_values=False):
    bars = ax.bar(data.keys(), data.values())
    
    if color_scheme:
        for i, color in color_scheme.items():
            if isinstance(i, int):
                bars[i].set_color(color)
            else:
                for bar in bars:
                    bar.set_color(color)
    
    # Configure axis
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', rotation=45)
    
    # Set y-limit and add value labels
    max_height = max(data.values())
    ax.set_ylim(0, max_height * 1.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        # Determine the format string and value based on integer_values flag
        format_str = '{:d}' if integer_values else '{:.2f}'
        value = int(height) if integer_values else height
        
        ax.text(bar.get_x() + bar.get_width()/2., height,
                format_str.format(value),
                ha='center', va='bottom')
    
    return bars

def create_analysis_plots(list_dict, list_of_titles, list_of_yaxis, list_of_colors, output_path):
    # Create figure with four subplots
    fig, ax = plt.subplots(len(list_dict), 1, figsize=(10, 16), height_ratios=[1]*len(list_dict))
    
    # Create all plots
    for i in range(len(list_dict)):
        setup_bar_plot(ax[i], list_dict[i], list_of_titles[i], list_of_yaxis[i], list_of_colors[i], integer_values=False)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def parse_jsonl_files(root_folder, subfolder):
    average_summary = {}
    command_dict = {
        'help': 'h',
        'where': 'w',
        'down': 'd',
        'up': 'u',
        'break': 'b',
        'clear': 'cl',
        'step': 's',
        'next': 'n',
        'until': 'unt',
        'continue': 'c',
        'cont' : 'c',
        'list': 'l',
        'longlist': 'll',
        'args': 'a',
        'whatis': 'whatis'
    }
    # Iterate through each subfolder in the root folder
    subfolder_path = os.path.join(root_folder, subfolder)
    # Check if the path is a directory
    if os.path.isdir(subfolder_path):
        task_summary = {}
        for task_name in os.listdir(subfolder_path):
            task_summary[task_name] = {"success": False, "rewrite": 0, "pdb": 0, "view": 0, "pdb_type": base_commands.copy(), 
                "pdb_stack": []}
            
            jsonl_file_path = os.path.join(subfolder_path, task_name, 'froggy.jsonl')
            # Check if the jsonl file exists in the subfolder
            if os.path.isfile(jsonl_file_path):
                with open(jsonl_file_path, 'r') as file:
                    log_data = json.load(file)
                    logs = log_data['log']
                    for i in range(len(logs)):
                        if logs[i]["action"] is None:
                            continue
                        if "```pdb" in logs[i]["action"]:
                            task_summary[task_name]["pdb"] += 1
                            
                            try:
                                pdb_type_parse = logs[i]["action"].split()[1].split("```")[0]
                                try:
                                    task_summary[task_name]["pdb_type"][pdb_type_parse] += 1
                                except:
                                    try:
                                        task_summary[task_name]["pdb_type"][command_dict[pdb_type_parse]] += 1
                                    except:
                                        task_summary[task_name]["pdb_type"]['wrong'] += 1
                            except:
                                print(logs[i]["action"])
                            task_summary[task_name]["pdb_stack"].append(pdb_type_parse)
                        if "view" in logs[i]["action"]:
                            task_summary[task_name]["view"] += 1
                        if "rewrite" in logs[i]["action"]:
                            task_summary[task_name]["rewrite"] += 1
                    if log_data["success"]:
                        task_summary[task_name]["success"] = 1
                    else:
                        task_summary[task_name]["success"] = 0

    return task_summary

def update_pdb_analysis(pdb_analysis, category, task_data):
    for cmd_type, count in task_data["pdb_type"].items():
        pdb_analysis[f"pdb_action_sum_{category}"][cmd_type] += count

def update_tool_metrics(tool_analysis, prefix, task_data):
    tool_analysis[f"{prefix} # pdb"] += task_data["pdb"]
    tool_analysis[f"{prefix} # view"] += task_data["view"]
    if 'reduce' in prefix:
        tool_analysis[f"task_reduce"] += 1
    elif 'baseline-fail froggy-success':
        tool_analysis[f"task_baseline-fail froggy-success"] += 1

def normalize_tool_metrics(tool_analysis, metric_prefix, task_count_key):
    for metric in ['# pdb', '# view']:
        key = f"{metric_prefix} {metric}"
        if not tool_analysis[task_count_key]==0:
            tool_analysis[key] /= tool_analysis[task_count_key]
    return tool_analysis

def normalize_pdb_metrics(pdb_analysis, category, denominator):
    pdb_analysis[f"pdb_action_sum_{category}"] = {
        cmd: float(count / denominator) if denominator != 0 else 0
        for cmd, count in pdb_analysis[f"pdb_action_sum_{category}"].items()
    }
    return pdb_analysis
    
def task_write(txt_file, task):
    if os.path.isfile(txt_file):
        with open(txt_file, 'r') as f:
            if task not in f.read():
                with open(txt_file, 'a') as f_append:
                    f_append.write(f"{task}\n")  
    else:
        with open(txt_file, 'a') as f_append: 
            f_append.write(f"{task}\n")  
        
def compare_task_summary(base_summary, agent_summary, baseline_name, agent_name, output_name_path):
    count_analysis = {"reduce # rewrite": 0, "same # rewrite":0, "increase # rewrite": 0, "success both": 0, "baseline-success froggy-fail": 0, "baseline-fail froggy-success": 0, "failed both": 0}
    tool_analysis = {"reduce rewrite # pdb": 0, "reduce rewrite # view": 0, "task_reduce": 0, "baseline-fail froggy-success # pdb": 0, "baseline-fail froggy-success # view": 0, "task_baseline-fail froggy-success": 0}
    
    # Create pdb_analysis with different categories using the template
    pdb_analysis = {
        f"pdb_action_sum_{category}": base_commands.copy()
        for category in ['rewrite', 'success', 'trial', 'fail', 'difficult']
    }
    
    for task in base_summary:
        base_success = base_summary[task]["success"] == 1
        agent_success = agent_summary[task]["success"] == 1
        
        if base_success and agent_success:
            count_analysis["success both"] += 1
            if base_summary[task]["rewrite"] > agent_summary[task]["rewrite"]:
                count_analysis["reduce # rewrite"] += 1
                update_tool_metrics(tool_analysis, "reduce rewrite", agent_summary[task])
                update_pdb_analysis(pdb_analysis, "rewrite", agent_summary[task])
            else:
                update_pdb_analysis(pdb_analysis, "trial", agent_summary[task])
                count_analysis["same # rewrite" if base_summary[task]["rewrite"] == agent_summary[task]["rewrite"] 
                             else "increase # rewrite"] += 1
        
        elif base_success and (not agent_success):
            count_analysis["baseline-success froggy-fail"] += 1
            update_pdb_analysis(pdb_analysis, "fail", agent_summary[task])
            task_write('./analysis/results/only_baseline_success_task_'+output_name_path+'.txt', agent_name+'|'+task)        
        elif agent_success and (not base_success):
            count_analysis["baseline-fail froggy-success"] += 1
            update_tool_metrics(tool_analysis, "baseline-fail froggy-success", agent_summary[task])
            update_pdb_analysis(pdb_analysis, "success", agent_summary[task])
            task_write('./analysis/results/only_froggy_success_task_'+output_name_path+'.txt', agent_name+'|'+task)  
        else:
            count_analysis["failed both"] += 1
            update_pdb_analysis(pdb_analysis, "difficult", agent_summary[task])
            task_write('./analysis/results/failed_both_task_'+output_name_path+'.txt', agent_name+'|'+task)  
            
    tool_analysis = normalize_tool_metrics(tool_analysis, "reduce rewrite", "task_reduce")
    tool_analysis = normalize_tool_metrics(tool_analysis, "baseline-fail froggy-success", "task_baseline-fail froggy-success")
    
    pdb_analysis = normalize_pdb_metrics(pdb_analysis, "success", count_analysis["baseline-fail froggy-success"])
    pdb_analysis = normalize_pdb_metrics(pdb_analysis, "rewrite", count_analysis["reduce # rewrite"])
    pdb_analysis = normalize_pdb_metrics(pdb_analysis, "trial", count_analysis["increase # rewrite"] + count_analysis["same # rewrite"])
    pdb_analysis = normalize_pdb_metrics(pdb_analysis, "fail", count_analysis["baseline-success froggy-fail"])
    pdb_analysis = normalize_pdb_metrics(pdb_analysis, "difficult", count_analysis["failed both"])
    return count_analysis, tool_analysis, pdb_analysis

def success_rate(summary):
    success = 0
    total = 0
    for task in summary:
        total += 1
        if summary[task]["success"] == 1:
            success += 1
    return success / total

def average_pdb_analysis(total_pdb_analysis):
    output_pdb_analysis = {
        "pdb_action_sum_rewrite": {},
        "pdb_action_sum_success": {},
        "pdb_action_sum_trial": {},
        "pdb_action_sum_fail": {},
        "pdb_action_sum_difficult": {}
    }
    # For each agent's data
    for agent in total_pdb_analysis:
        for key in total_pdb_analysis[agent].keys():
            for cmd_type in total_pdb_analysis[agent][key]:
                if cmd_type not in output_pdb_analysis[key]:
                    output_pdb_analysis[key][cmd_type] = total_pdb_analysis[agent][key][cmd_type] / len(total_pdb_analysis)
                else:
                    output_pdb_analysis[key][cmd_type] += total_pdb_analysis[agent][key][cmd_type] / len(total_pdb_analysis)
            
    return output_pdb_analysis

def average_analysis(total_count_analysis, total_tool_analysis=None, std=False):
    output_count_analysis = {}
    output_tool_analysis = {}
    for agent in total_count_analysis:
        for key in total_count_analysis[agent]:
            if key not in output_count_analysis:
                output_count_analysis[key] = total_count_analysis[agent][key] / len(total_count_analysis)
            else:
                output_count_analysis[key] += total_count_analysis[agent][key] / len(total_count_analysis)
        if not total_tool_analysis is None:
            for key in total_tool_analysis[agent]:
                if key == "task_reduce" or key == "task_baseline-fail froggy-success":
                    continue
                if key not in output_tool_analysis:
                    output_tool_analysis[key] = total_tool_analysis[agent][key] / len(total_tool_analysis)
                else:
                    output_tool_analysis[key] += total_tool_analysis[agent][key] / len(total_tool_analysis)
        
    return output_count_analysis, output_tool_analysis


def main():
    # Define the root folder
    root_folder = './output_aider'
    jsonl_folder_name = './analysis/parse_folders.jsonl'
    # Check if the jsonl file exists in the subfolder
    if os.path.isfile(jsonl_folder_name):
        with open(jsonl_folder_name, 'r') as file:
            folder = json.load(file)
            subfolder_name_baseline = folder["subfolder_name_baseline"]
            subfolder_name_agent = folder["subfolder_name_agent"]
            output_name_path = folder["output_name_path"]

    # Call the function to parse the jsonl files
    average_count_analysis = {}
    average_tool_analysis = {}
    average_total_pdb_analysis = {}
    for baseline_name in subfolder_name_baseline:
        total_count_analysis = {}
        total_tool_analysis = {}
        total_pdb_analysis = {}

        for agent_name in subfolder_name_agent:
            baseline = parse_jsonl_files(root_folder, baseline_name)
            agent = parse_jsonl_files(root_folder, agent_name)

            count_analysis, tool_analysis, pdb_analysis = compare_task_summary(baseline, agent, baseline_name, agent_name, output_name_path)
            total_count_analysis[agent_name] = count_analysis
            total_tool_analysis[agent_name] = tool_analysis
            total_pdb_analysis[agent_name] = pdb_analysis
            
        # Get the averaged analysis
        averaged_analysis = average_pdb_analysis(total_pdb_analysis)
        average_total_pdb_analysis[baseline_name] = {
            "pdb_action_sum_rewrite": {},
            "pdb_action_sum_success": {},
            "pdb_action_sum_trial": {},
            "pdb_action_sum_fail": {},
            "pdb_action_sum_difficult": {},
        }
        
        # Correctly assign the respective parts using a loop
        for key in average_total_pdb_analysis[baseline_name]:
            average_total_pdb_analysis[baseline_name][key] = averaged_analysis[key]
        
        average_count_analysis[baseline_name], average_tool_analysis[baseline_name] = average_analysis(total_count_analysis, total_tool_analysis)

    final_count, final_tool = average_analysis(average_count_analysis, average_tool_analysis, std=False)
    final_pdb_analysis = average_pdb_analysis(average_total_pdb_analysis)

    # Define color schemes
    count_colors = {0: 'lightcoral', 1: 'coral', 2: 'coral', 
                    3: 'pink', 4:'lightblue', 5: 'lightgreen', 6:'blue'}
    tool_colors = {0: 'lightcoral', 1: 'lightcoral', 2: 'lightgreen', 
                    3: 'lightgreen'}

    create_analysis_plots([final_count, final_tool, final_pdb_analysis["pdb_action_sum_rewrite"], final_pdb_analysis["pdb_action_sum_success"]], 
            ['Success analysis', 'Tool Analysis', 'PDB Commands Usage Analysis (Reduce Cases)','PDB Commands Usage Analysis (Pdb needed Cases)'],
            ['Number of tasks','Average Number of Uses','Average Number of Uses','Average Number of Uses'],
            [count_colors, tool_colors, {'all': 'lightcoral'}, {'all': 'lightgreen'}],
            './analysis/results/'+output_name_path+'.png')
    create_analysis_plots([final_count, final_pdb_analysis["pdb_action_sum_rewrite"], final_pdb_analysis["pdb_action_sum_trial"],  final_pdb_analysis["pdb_action_sum_fail"], final_pdb_analysis["pdb_action_sum_success"], final_pdb_analysis["pdb_action_sum_difficult"]],
            ['Success analysis', 'PDB Commands Usage Analysis (Rewrite Cases)', 'PDB Commands Usage Analysis (Trial Cases)', 'PDB Commands Usage Analysis (Fail Cases)', 'PDB Commands Usage Analysis (Success Cases)', 'PDB Commands Usage Analysis (Difficult Cases)'],
            ['Number of tasks','Average Number of Uses','Average Number of Uses','Average Number of Uses','Average Number of Uses','Average Number of Uses'],
            [count_colors, {'all': 'lightcoral'}, {'all': 'coral'},  {'all': 'lightblue'}, {'all': 'lightgreen'}, {'all': 'blue'}],
            './analysis/results/'+output_name_path+'_pdb.png')


if __name__ == "__main__":
    main()