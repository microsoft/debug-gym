import json
import os

from criteria import *
from tqdm import tqdm


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
        # follows_proper_debugging_workflow,
        uses_pdb_print_commands,
        # lambda trajectory: has_consecutive_pdb_calls(trajectory, n=2),
    ]

    data_criteria = [has_successful_outcome]

    # Directory containing trajectory files
    exps_dir = "/Users/ericyuan/GitHub_Enterprise/Froggy/exps/may28"

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
    for criterion in trajectory_criteria + data_criteria:
        print(f"  {criterion.__name__}")
    print("\nMatching files by model:")
    for mn, count in model_name.items():
        print(f"  {mn}: {count} files")


if __name__ == "__main__":
    main()
