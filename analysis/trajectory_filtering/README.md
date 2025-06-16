# Trajectory Filtering

This module provides tools to filter trajectory files logged by `debug-gym` agents (in `.jsonl` format) against various criteria. It enables analysis of agent behavior patterns, debugging workflows, and successful/unsuccessful strategies.

### 1. Overview

The trajectory filtering system consists of two main components:

- **`filter.py`**: Main filtering engine that searches for trajectory files and applies criteria
- **`criteria.py`**: Collection of predefined criteria functions for analyzing agent behavior

### 2. Usage Example
Add necessary criteria in `criteria.py`, enable them in `filter.py`, then run:
```bash
# Basic filtering with JSON output
python filter.py --exp-path /path/to/experiments --exp-uuid my_experiment

# Create a zip file containing all matching JSONL files
python filter.py --exp-path /path/to/experiments --exp-uuid my_experiment --create-zip

# Custom output locations for both JSON and zip files
python filter.py --exp-path /path/to/experiments --exp-uuid my_experiment --output-file /custom/path/results.json --create-zip --zip-output /custom/path/trajectories.zip
```

#### 2.1. Command Line Options
- `--exp-path`: Path to experiments directory
- `--exp-uuid`: Experiment UUID/name to analyze
- `--output-file`: Custom output file path (default: `<exp_path>/filtered_trajectories_<exp_uuid>.json`)
- `--create-zip`: Create a zip file containing all matching JSONL files
- `--zip-output`: Custom zip file path (default: `<exp_path>/filtered_trajectories_<exp_uuid>.zip`)

#### 2.2. Output Files
The script generates two types of output:

1. **JSON Summary File**: Contains metadata about the filtering results:
   - List of criteria used
   - Number of matching files
   - Relative paths to matching trajectory directories

2. **Zip Archive** (optional, when `--create-zip` is used): Contains the actual JSONL files that matched the criteria, preserving the original directory structure for easy analysis

### 3. Criteria

A comprehensive collection of criteria functions organized into two categories:

#### 3.1. Data-Level Criteria
These operate on the full trajectory data dictionary:

- **`has_successful_outcome(trajectory_data)`**
  - Checks if `trajectory_data["success"]` is `True`

#### 3.2. Trajectory-Level Criteria
These operate on the trajectory log entries (list of steps):

##### 3.2.1. PDB-Specific Criteria

- **`has_consecutive_pdb_calls(trajectory, n=1)`**
  - Checks for `n` consecutive pdb tool calls

- **`uses_pdb_print_commands(trajectory)`**
  - Checks if agent used pdb print commands (`p`, `print`, `pp`)

- **`uses_specific_pdb_command(trajectory, pdb_command)`**
  - Checks for usage of a specific pdb command (e.g., `"b"`, `"c"`, `"n"`, `"s"`)

- **`has_rewrite_after_pdb(trajectory)`**
  - Checks if any rewrite calls occur after pdb calls

- **`has_rewrite_soon_after_pdb(trajectory, max_steps_between=10)`**
  - Checks if rewrite calls occur within a specified window after pdb calls

- **`has_continue_after_setting_breakpoints(trajectory)`**
  - Checks for logical sequence: set breakpoints → continue execution

- **`follows_proper_debugging_workflow(trajectory)`**
  - Checks for complete debugging sequence: set breakpoints → continue → inspect variables

##### 3.2.2. Pattern Analysis

- **`count_debug_to_code_patterns(trajectory, max_steps_between=10)`**
  - Counts pdb→rewrite patterns (debugging leading to code changes)

- **`has_sufficient_debug_to_code_patterns(trajectory, min_patterns=1, max_steps_between=10)`**
  - Checks for minimum number of debug→code patterns

##### 3.2.3. General Criteria

- **`has_minimum_trajectory_length(trajectory, min_count=5)`**
  - Ensures trajectory has sufficient length

- **`has_sufficient_code_changes(trajectory, min_rewrites=1)`**
  - Checks for minimum number of rewrite actions


