# Trajectory Filtering

This module provides tools to filter trajectory files logged by `debug-gym` agents (in `.jsonl` format) against various criteria. It enables analysis of agent behavior patterns, debugging workflows, and successful/unsuccessful strategies.

## Overview

The trajectory filtering system consists of two main components:

- **`filter.py`**: Main filtering engine that searches for trajectory files and applies criteria
- **`criteria.py`**: Collection of predefined criteria functions for analyzing agent behavior

## Quick Start

```python
from filter import filter_trajectories
from criteria import has_successful_outcome, follows_proper_debugging_workflow

# Define criteria
trajectory_criteria = [follows_proper_debugging_workflow]
data_criteria = [has_successful_outcome]

# Filter trajectories
matching_files = filter_trajectories(
    directory="../../exps/jun7",
    trajectory_criteria=trajectory_criteria,
    data_criteria=data_criteria
)

print(f"Found {len(matching_files)} trajectories matching criteria")
```

#### Usage Example
Add necessary criteria in `criteria.py`, enable them in `filter.py`, then run:
```bash
python filter.py
```

### `criteria.py`

A comprehensive collection of criteria functions organized into two categories:

#### Data-Level Criteria
These operate on the full trajectory data dictionary:

- **`has_successful_outcome(trajectory_data)`**
  - Checks if `trajectory_data["success"]` is `True`

#### Trajectory-Level Criteria
These operate on the trajectory log entries (list of steps):

##### PDB-Specific Criteria

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

##### Pattern Analysis

- **`count_debug_to_code_patterns(trajectory, max_steps_between=10)`**
  - Counts pdb→rewrite patterns (debugging leading to code changes)

- **`has_sufficient_debug_to_code_patterns(trajectory, min_patterns=1, max_steps_between=10)`**
  - Checks for minimum number of debug→code patterns

##### General Criteria

- **`has_minimum_trajectory_length(trajectory, min_count=5)`**
  - Ensures trajectory has sufficient length

- **`has_sufficient_code_changes(trajectory, min_rewrites=1)`**
  - Checks for minimum number of rewrite actions


