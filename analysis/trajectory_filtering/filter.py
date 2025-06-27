import argparse
import json
import os
import zipfile

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


def create_zip_with_matching_files(matching_files, exps_dir, output_zip_path):
    """
    Create a zip file containing all matching JSONL files.

    Args:
        matching_files: List of file paths that match the criteria
        exps_dir: Base experiments directory
        output_zip_path: Path where the zip file should be created
    """
    if not matching_files:
        print("No matching files found. Skipping zip creation.")
        return

    print(f"\nCreating zip file with {len(matching_files)} matching files...")

    try:
        # Create the directory for the zip file if it doesn't exist
        os.makedirs(os.path.dirname(output_zip_path), exist_ok=True)

        with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in tqdm(matching_files, desc="Adding files to zip"):
                # Check if file exists before adding to zip
                if os.path.exists(file_path):
                    # Calculate the relative path within the zip
                    relative_path = os.path.relpath(file_path, exps_dir)

                    # Add the file to the zip with its relative path structure
                    zipf.write(file_path, relative_path)
                else:
                    print(f"Warning: File not found, skipping: {file_path}")

        # Get the file size for user feedback
        zip_size = os.path.getsize(output_zip_path)
        zip_size_mb = zip_size / (1024 * 1024)
        print(f"Zip file created: {output_zip_path} ({zip_size_mb:.2f} MB)")

    except Exception as e:
        print(f"Error creating zip file: {e}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Filter trajectory files based on specified criteria"
    )
    parser.add_argument(
        "--exp-path",
        help="Path to experiments directory",
        required=True,
    )
    parser.add_argument(
        "--exp-uuid",
        help="Experiment UUID/name to analyze",
        required=True,
    )
    parser.add_argument(
        "--output-file",
        help="Custom output file path (default: <exp_path>/filtered_trajectories_<exp_uuid>.json)",
    )
    parser.add_argument(
        "--create-zip",
        action="store_true",
        help="Create a zip file containing all matching JSONL files",
    )
    parser.add_argument(
        "--zip-output",
        help="Custom zip file path (default: <exp_path>/filtered_trajectories_<exp_uuid>.zip)",
    )

    args = parser.parse_args()

    trajectory_criteria = [
        follows_proper_debugging_workflow,
        # uses_pdb_print_commands,
        # has_continue_after_setting_breakpoints
        # lambda trajectory: has_consecutive_pdb_calls(trajectory, n=1),
    ]

    data_criteria = [has_successful_outcome]
    # data_criteria = None

    # Directory containing trajectory files
    exps_dir = os.path.join(args.exp_path, args.exp_uuid)

    # Filter trajectories
    matching_files = filter_trajectories(exps_dir, trajectory_criteria, data_criteria)

    # Store original paths for zip creation if needed
    original_matching_files = matching_files.copy()

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
    output_file = args.output_file or os.path.join(
        exps_dir, f"filtered_trajectories_{args.exp_uuid}.json"
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

    print(f"\nResults saved to: {output_file}")

    # Create zip file if requested
    if args.create_zip:
        zip_output_file = args.zip_output or os.path.join(
            exps_dir, f"filtered_trajectories_{args.exp_uuid}.zip"
        )
        create_zip_with_matching_files(
            original_matching_files, exps_dir, zip_output_file
        )


if __name__ == "__main__":
    main()
