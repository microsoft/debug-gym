import json
import os

from criteria import *
from tqdm import tqdm

exp_folder_path = "jun7"


def satisfy_criteria(json_file_path, trajectory_criteria=None, data_criteria=None):
    """
    Check if the JSON file satisfies the given criteria.

    Args:
        json_file_path: Path to the JSONL file containing trajectory data
        trajectory_criteria: List of criteria functions that take trajectory log
        data_criteria: List of criteria functions that take full data dict

    Returns:
        bool: True if all criteria are satisfied
    """
    if trajectory_criteria is None:
        trajectory_criteria = []
    if data_criteria is None:
        data_criteria = []
    try:
        with open(json_file_path, "r") as f:
            data = json.load(f)

        # Check data-level criteria (e.g., success status)
        if data_criteria:
            for criterion in data_criteria:
                if not criterion(data):
                    return False

        # Check trajectory-level criteria
        if trajectory_criteria:
            trajectory = data.get("log", [])
            for criterion in trajectory_criteria:
                if not criterion(trajectory):
                    return False

        return True
    except Exception as e:
        print(f"Error processing {json_file_path}: {e}")
        return False


def get_json_files(directory):
    """
    Get all JSON files in the specified directory and subdirectories.
    :param directory: Directory to search for JSON files.
    :return: List of JSON file paths.
    """
    json_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json") or file.endswith(".jsonl"):
                json_files.append(os.path.join(root, file))
    return json_files


def filter_trajectories(directory, trajectory_criteria=None, data_criteria=None):
    """
    Filter trajectory files in a directory based on specified criteria.

    Args:
        directory: Directory to search for JSONL files (searches recursively)
        trajectory_criteria: List of criteria functions that take trajectory log
        data_criteria: List of criteria functions that take full data dict

    Returns:
        list: Paths to JSONL files that satisfy all criteria
    """
    # Get all JSONL files in the directory
    jsonl_files = get_json_files(directory)

    # Filter files that satisfy criteria
    matching_files = []

    print(f"Found {len(jsonl_files)} JSONL files to check...")

    for file_path in tqdm(jsonl_files, desc="Filtering trajectories"):
        if satisfy_criteria(file_path, trajectory_criteria, data_criteria):
            matching_files.append(file_path)

    print(f"Found {len(matching_files)} files matching criteria")
    return matching_files


def main():

    trajectory_criteria = [
        follows_proper_debugging_workflow,
        # uses_pdb_print_commands,
        # has_continue_after_setting_breakpoints
        # lambda trajectory: has_consecutive_pdb_calls(trajectory, n=1),
    ]

    data_criteria = [has_successful_outcome]
    # data_criteria = None

    # Directory containing trajectory files
    # exps_dir = "/Users/ericyuan/GitHub_Enterprise/Froggy/exps/may28"
    exps_dir = "/Users/ericyuan/GitHub_Enterprise/Froggy/exps/" + exp_folder_path

    # Filter trajectories
    matching_files = filter_trajectories(exps_dir, trajectory_criteria, data_criteria)

    # Where are the matching files located
    model_name = {}
    for file_path in matching_files:
        if file_path.startswith(exps_dir):
            file_path = file_path[len(exps_dir) + 1 :]
        _mn = file_path.split("/")[0]
        if _mn not in model_name:
            model_name[_mn] = 0
        model_name[_mn] += 1

    print("Criteria used:")
    for criterion in (trajectory_criteria or []) + (data_criteria or []):
        print(f"  {criterion.__name__}")
    print("\nMatching files by model:")

    # print the number of matching files for each model (sorted by model name)
    model_name = dict(sorted(model_name.items()))
    for mn, count in model_name.items():
        print(f"  {mn}: {count} files")

    # extract matching files relative to the experiments directory
    matching_files = [
        os.path.relpath(file_path, exps_dir) for file_path in matching_files
    ]
    # remove the jsonl file names
    matching_files = [os.path.dirname(file_path) for file_path in matching_files]

    # Save data into a JSON file
    output_file = os.path.join(
        exps_dir, f"filtered_trajectories_{exp_folder_path}.json"
    )
    with open(output_file, "w") as f:
        json.dump(
            {
                "criteria": [c.__name__ for c in trajectory_criteria + data_criteria],
                "number_of_matching_files": len(matching_files),
                "matching_files": matching_files,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
