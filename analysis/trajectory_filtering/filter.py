import argparse
import json
import os
import zipfile
from datetime import datetime
from zoneinfo import ZoneInfo

from criteria import *
from tqdm import tqdm


def categorize_trajectory(
    json_file_path, behavior_pattern_criteria=None, outcome_criteria=None
):
    """
    Categorize the JSON file based on which criteria it satisfies.

    Args:
        json_file_path: Path to the JSONL file containing trajectory data
        behavior_pattern_criteria: List of criteria functions that analyze trajectory behavior patterns
        outcome_criteria: List of criteria functions that check trajectory outcomes

    Returns:
        str: Category - "both", "outcome_only", or "failed"
    """
    if behavior_pattern_criteria is None:
        behavior_pattern_criteria = []
    if outcome_criteria is None:
        outcome_criteria = []

    try:
        with open(json_file_path, "r") as f:
            data = json.load(f)

        # Check outcome-level criteria (e.g., success status)
        outcome_criteria_satisfied = True
        if outcome_criteria:
            for criterion in outcome_criteria:
                if not criterion(data):
                    outcome_criteria_satisfied = False
                    break

        # Check behavior pattern criteria
        behavior_pattern_criteria_satisfied = True
        if behavior_pattern_criteria:
            trajectory = data.get("log", [])
            for criterion in behavior_pattern_criteria:
                if not criterion(trajectory):
                    behavior_pattern_criteria_satisfied = False
                    break

        # Categorize based on which criteria are satisfied
        if outcome_criteria_satisfied and behavior_pattern_criteria_satisfied:
            return "both"
        elif outcome_criteria_satisfied:
            return "outcome_only"
        else:
            return "failed"

    except Exception as e:
        print(f"Error processing {json_file_path}: {e}")
        return "failed"


def satisfy_criteria(
    json_file_path, behavior_pattern_criteria=None, outcome_criteria=None
):
    """
    Check if the JSON file satisfies all criteria.

    Args:
        json_file_path: Path to the JSONL file containing trajectory data
        behavior_pattern_criteria: List of criteria functions that analyze trajectory behavior patterns
        outcome_criteria: List of criteria functions that check trajectory outcomes

    Returns:
        bool: True if all criteria are satisfied
    """
    return (
        categorize_trajectory(
            json_file_path, behavior_pattern_criteria, outcome_criteria
        )
        == "both"
    )


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


def categorize_trajectories(
    directory, behavior_pattern_criteria=None, outcome_criteria=None
):
    """
    Categorize trajectory files in a directory based on specified criteria.

    Args:
        directory: Directory to search for JSONL files (searches recursively)
        behavior_pattern_criteria: List of criteria functions that analyze trajectory behavior patterns
        outcome_criteria: List of criteria functions that check trajectory outcomes

    Returns:
        dict: Dictionary with categories as keys and lists of file paths as values
              Categories: "both", "outcome_only", "failed"
    """
    # Get all JSONL files in the directory
    jsonl_files = get_json_files(directory)

    # Initialize categorized files dictionary
    categorized_files = {"both": [], "outcome_only": [], "failed": []}

    print(f"Found {len(jsonl_files)} JSONL files to categorize...")

    for file_path in tqdm(jsonl_files, desc="Categorizing trajectories"):
        category = categorize_trajectory(
            file_path, behavior_pattern_criteria, outcome_criteria
        )
        categorized_files[category].append(file_path)

    print(f"Categorization complete:")
    print(f"  Both criteria satisfied: {len(categorized_files['both'])} files")
    print(f"  Outcome criteria only: {len(categorized_files['outcome_only'])} files")
    print(f"  Failed outcome criteria: {len(categorized_files['failed'])} files")

    return categorized_files


def filter_trajectories(
    directory, behavior_pattern_criteria=None, outcome_criteria=None
):
    """
    Filter trajectory files in a directory based on specified criteria.

    Args:
        directory: Directory to search for JSONL files (searches recursively)
        behavior_pattern_criteria: List of criteria functions that analyze trajectory behavior patterns
        outcome_criteria: List of criteria functions that check trajectory outcomes

    Returns:
        list: Paths to JSONL files that satisfy all criteria
    """
    # Get all JSONL files in the directory
    jsonl_files = get_json_files(directory)

    # Filter files that satisfy criteria
    matching_files = []

    print(f"Found {len(jsonl_files)} JSONL files to check...")

    for file_path in tqdm(jsonl_files, desc="Filtering trajectories"):
        if satisfy_criteria(file_path, behavior_pattern_criteria, outcome_criteria):
            matching_files.append(file_path)

    print(f"Found {len(matching_files)} files matching criteria")
    return matching_files


def create_zip_with_all_files(all_files, exps_dir, output_zip_path):
    """
    Create a zip file containing all JSONL files.

    Args:
        all_files: List of all file paths to include in the zip
        exps_dir: Base experiments directory
        output_zip_path: Path where the zip file should be created
    """
    if not all_files:
        print("No files found. Skipping zip creation.")
        return

    print(f"\nCreating zip file with {len(all_files)} files...")

    try:
        # Create the directory for the zip file if it doesn't exist
        os.makedirs(os.path.dirname(output_zip_path), exist_ok=True)

        with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in tqdm(all_files, desc="Adding files to zip"):
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
        description="Categorize trajectory files based on specified criteria into three groups: "
        "1) trajectories satisfying both outcome and behavior pattern criteria, "
        "2) trajectories satisfying only outcome criteria, "
        "3) trajectories failing outcome criteria"
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
        help="Custom output file path (default: <exp_path>/all_trajectories_<exp_uuid>.json)",
    )
    parser.add_argument(
        "--create-zip",
        action="store_true",
        help="Create a zip file containing all JSONL files (not just matching ones)",
    )
    parser.add_argument(
        "--zip-output",
        help="Custom zip file path (default: <exp_path>/all_trajectories_<exp_uuid>.zip)",
    )

    args = parser.parse_args()

    behavior_pattern_criteria = [
        follows_proper_debugging_workflow,
        # uses_pdb_print_commands,
        # has_continue_after_setting_breakpoints
        # lambda trajectory: has_consecutive_pdb_calls(trajectory, n=1),
    ]

    outcome_criteria = [has_successful_outcome]
    # outcome_criteria = None

    # Directory containing trajectory files
    exps_dir = os.path.join(args.exp_path, args.exp_uuid)

    # Categorize all trajectories
    categorized_files = categorize_trajectories(
        exps_dir, behavior_pattern_criteria, outcome_criteria
    )

    # Get all files for statistics and zip creation
    all_files = []
    for category_files in categorized_files.values():
        all_files.extend(category_files)

    # Where are the files located by model (for statistics)
    model_name = {}
    for file_path in all_files:
        if file_path.startswith(exps_dir):
            file_path = file_path[len(exps_dir) + 1 :]
        _mn = file_path.split("/")[0]
        if _mn not in model_name:
            model_name[_mn] = 0
        model_name[_mn] += 1

    print("Criteria used:")
    for criterion in (behavior_pattern_criteria or []) + (outcome_criteria or []):
        print(f"  {criterion.__name__}")
    print("\nAll files by model:")

    # print the number of files for each model (sorted by model name)
    model_name = dict(sorted(model_name.items()))
    for mn, count in model_name.items():
        print(f"  {mn}: {count} files")

    # Convert file paths to relative paths and remove .jsonl extensions for directory paths
    categorized_relative_files = {}
    for category, file_paths in categorized_files.items():
        relative_paths = [
            os.path.relpath(file_path, exps_dir) for file_path in file_paths
        ]
        # Remove the jsonl file names to get directory paths
        directory_paths = [os.path.dirname(file_path) for file_path in relative_paths]
        categorized_relative_files[category] = directory_paths

    # Save data into a JSON file
    # Replace slashes in exp_uuid for filename
    safe_exp_uuid = args.exp_uuid.replace("/", "_")
    output_file = args.output_file or os.path.join(
        exps_dir, f"filtered_trajectories_{safe_exp_uuid}.json"
    )

    # Get current time in Eastern Time
    eastern = ZoneInfo("America/New_York")
    current_time = datetime.now(eastern)
    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S %Z")

    with open(output_file, "w") as f:
        json.dump(
            {
                "metadata": {"generated_at": timestamp, "timezone": "Eastern Time"},
                "criteria": {
                    "behavior_pattern_criteria": [
                        c.__name__ for c in behavior_pattern_criteria or []
                    ],
                    "outcome_criteria": [c.__name__ for c in outcome_criteria or []],
                },
                "summary": {
                    "total_files": len(all_files),
                    "both_criteria_satisfied": len(categorized_relative_files["both"]),
                    "outcome_criteria_only": len(
                        categorized_relative_files["outcome_only"]
                    ),
                    "failed_outcome_criteria": len(
                        categorized_relative_files["failed"]
                    ),
                },
                "trajectories": {
                    "both_criteria_satisfied": categorized_relative_files["both"],
                    "outcome_criteria_only": categorized_relative_files["outcome_only"],
                    "failed_outcome_criteria": categorized_relative_files["failed"],
                },
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to: {output_file}")

    # Create zip file if requested (include all files)
    if args.create_zip:
        zip_output_file = args.zip_output or os.path.join(
            exps_dir, f"all_trajectories_{safe_exp_uuid}.zip"
        )
        create_zip_with_all_files(all_files, exps_dir, zip_output_file)


if __name__ == "__main__":
    main()
