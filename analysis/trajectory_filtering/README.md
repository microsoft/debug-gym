# Trajectory Filtering

This module provides tools to categorize and filter trajectory files logged by `debug-gym` agents (in `.jsonl` format) against various criteria. It enables analysis of agent behavior patterns, debugging workflows, and successful/unsuccessful strategies.

### 1. Overview

The trajectory filtering system consists of two main components:

- **`filter.py`**: Main filtering engine that searches for trajectory files and applies criteria
- **`criteria.py`**: Collection of predefined criteria functions for analyzing agent behavior

### 2. Usage Example
Add necessary criteria in `criteria.py`, enable them in `filter.py`, then run:
```bash
# Basic categorization with JSON output
python filter.py --exp-path /path/to/experiments --exp-uuid my_experiment

# Create a zip file containing ALL JSONL files (not just matching ones)
python filter.py --exp-path /path/to/experiments --exp-uuid my_experiment --create-zip

# Custom output locations for both JSON and zip files
python filter.py --exp-path /path/to/experiments --exp-uuid my_experiment --output-file /custom/path/results.json --create-zip --zip-output /custom/path/trajectories.zip
```

#### 2.1. Command Line Options
- `--exp-path`: Path to experiments directory
- `--exp-uuid`: Experiment UUID/name to analyze
- `--output-file`: Custom output file path (default: `<exp_path>/filtered_trajectories_<exp_uuid>.json`)
- `--create-zip`: Create a zip file containing ALL JSONL files (not just matching ones)
- `--zip-output`: Custom zip file path (default: `<exp_path>/all_trajectories_<exp_uuid>.zip`)

#### 2.2. Output Files
The script generates two types of output:

1. **JSON Summary File**: Contains comprehensive categorization results:
   - Metadata with generation timestamp (Eastern Time) and timezone information
   - List of criteria used (separated by type)
   - Summary statistics for all three categories
   - Complete lists of trajectories in each category:
     - `both_criteria_satisfied`: Trajectories satisfying both outcome and behavior pattern criteria
     - `outcome_criteria_only`: Trajectories satisfying only outcome criteria
     - `failed_outcome_criteria`: Trajectories failing outcome criteria

2. **Zip Archive** (optional, when `--create-zip` is used): Contains ALL JSONL files in the dataset, preserving the original directory structure for comprehensive analysis

### 3. Criteria

A comprehensive collection of criteria functions organized into two categories:

#### 3.1. Outcome Criteria
These operate on the full trajectory data dictionary to check outcomes:

- **`has_successful_outcome(trajectory_data)`**
  - Checks if `trajectory_data["success"]` is `True`

#### 3.2. Behavior Pattern Criteria
These operate on the trajectory log entries to analyze behavioral patterns:

##### 3.2.1. PDB-Specific Behavior Patterns

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

##### 3.2.2. Analysis Patterns

- **`count_debug_to_code_patterns(trajectory, max_steps_between=10)`**
  - Counts pdb→rewrite patterns (debugging leading to code changes)

- **`has_sufficient_debug_to_code_patterns(trajectory, min_patterns=1, max_steps_between=10)`**
  - Checks for minimum number of debug→code patterns

##### 3.2.3. General Behavior Patterns

- **`has_minimum_trajectory_length(trajectory, min_count=5)`**
  - Ensures trajectory has sufficient length

- **`has_sufficient_code_changes(trajectory, min_rewrites=1)`**
  - Checks for minimum number of rewrite actions

### 4. Example Output Structure

The JSON output now includes detailed categorization with automatic timestamp logging:

```json
{
  "metadata": {
    "generated_at": "2025-07-09 12:06:40 EDT",
    "timezone": "Eastern Time"
  },
  "criteria": {
    "behavior_pattern_criteria": ["follows_proper_debugging_workflow"],
    "outcome_criteria": ["has_successful_outcome"]
  },
  "summary": {
    "total_files": 2506,
    "both_criteria_satisfied": 447,
    "outcome_criteria_only": 420,
    "failed_outcome_criteria": 1639
  },
  "trajectories": {
    "both_criteria_satisfied": ["path/to/trajectory1", "path/to/trajectory2"],
    "outcome_criteria_only": ["path/to/trajectory3", "path/to/trajectory4"],
    "failed_outcome_criteria": ["path/to/trajectory5", "path/to/trajectory6"]
  }
}
```