import os
import json
from collections import defaultdict

def write_to_file(file_path, strings):
    with open(file_path, 'w') as file:
        for string in strings:
            file.write(string + '\n')

def add_unsuccessful_half(current_probs, successes, model):
    harder_probs, _ = successes[model]
    ppo_probs = []
    for potential_prob in harder_probs:
        if len(ppo_probs) < len(current_probs) and potential_prob not in current_probs:
            ppo_probs.append(potential_prob)
    return ppo_probs + list(current_probs)

# Need to modify this now that steps are taken into account
def get_overlap_successes(success_problems, print_success_rate=False):
    sets = []
    for model, (successes, success_rate) in success_problems.items():
        if print_success_rate:
            print(model, success_rate, f"| {len(successes)} problems solved.")
        if "7B" not in model:
            sets.append(successes)
    return set.intersection(*sets)

def get_success_problems(outputs_path):
    success_problems = defaultdict(lambda: [set(), 0.0])
    for model in os.listdir(outputs_path):
        model_output_path = os.path.join(outputs_path, model)
        for problem in os.listdir(model_output_path):
            full_path = os.path.join(model_output_path, problem + '/froggy.jsonl')
            with open(full_path, 'r') as f:
                data = json.load(f)
                if data["success"]:
                    success_problems[model][0].add(problem) 
        success_problems[model][1] = len(success_problems[model][0]) / len(os.listdir(model_output_path))
    return success_problems


outputs_path = '/root/outputs/'

# Gets all the successful problems along with the success rate
successes = get_success_problems(outputs_path)

# "Easy" problems for Qwen (all 3 models could solve them)
overlap = get_overlap_successes(successes, print_success_rate=True)

# Add problems that Qwen1.5B could solve but Qwen0.5B couldn't 
# to balance dataset for PPO training
# problems = add_unsuccessful_half(overlap, successes, "Qwen2.5-Coder-1.5B-Instruct")

# Write
# write_to_file("/root/Froggy/easy_problems.txt", problems)