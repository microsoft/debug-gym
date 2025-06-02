"""
Criteria functions for filtering trajectories.

This module contains two types of criteria functions:
1. Data-level criteria: operate on the full trajectory data dict
2. Trajectory-level criteria: operate on the trajectory log entries

"""

# =============================================================================
# DATA-LEVEL CRITERIA
# These functions operate on the full trajectory data dict (not just the log)
# =============================================================================


def has_successful_outcome(trajectory_data):
    """
    Check if the trajectory was successful.

    Args:
        trajectory_data: Full trajectory data dict (not just log)

    Returns:
        bool: True if trajectory was successful
    """
    return trajectory_data.get("success", False)


# =============================================================================
# TRAJECTORY-LEVEL CRITERIA
# These functions operate on the trajectory log entries (list of steps)
# =============================================================================

# PDB-specific criteria
# ---------------------


def has_consecutive_pdb_calls(trajectory, n=1):
    """
    Did the agent call the pdb tool (any pdb command) n times in a row?

    Args:
        trajectory: List of trajectory steps (log entries)
        n: Number of consecutive pdb calls to look for

    Returns:
        bool: True if agent called pdb n times in a row
    """
    if n <= 0:
        return True
    consecutive_pdb_count = 0

    for step in trajectory:
        action = step.get("action")
        if action and action.get("name") == "pdb":
            consecutive_pdb_count += 1
        else:
            consecutive_pdb_count = 0
        if consecutive_pdb_count >= n:
            return True
    return False


def uses_pdb_print_commands(trajectory):
    """
    Did the agent call the pdb print command?

    Args:
        trajectory: List of trajectory steps (log entries)

    Returns:
        bool: True if agent used pdb print command (p, print, pp)
    """
    pdb_print_commands = ["p", "print", "pp"]

    for step in trajectory:
        action = step.get("action")
        if action and action.get("name") == "pdb":
            # Get the pdb command from arguments
            arguments = action.get("arguments", {})
            command = arguments.get("command", "")

            # Extract the first word of the command (the pdb command)
            if command:
                pdb_cmd = command.strip().split()[0]
                if pdb_cmd in pdb_print_commands:
                    return True

    return False


def uses_specific_pdb_command(trajectory, pdb_command):
    """
    Check if trajectory contains a specific pdb command.

    Args:
        trajectory: List of trajectory steps (log entries)
        pdb_command: Specific pdb command to look for (e.g., "b", "c", "n", "s")

    Returns:
        bool: True if trajectory contains the pdb command
    """
    for step in trajectory:
        action = step.get("action")
        if action and action.get("name") == "pdb":
            arguments = action.get("arguments", {})
            command = arguments.get("command", "")

            if command:
                # Extract the first word of the command
                cmd = command.strip().split()[0]
                if cmd == pdb_command:
                    return True

    return False


def has_rewrite_after_pdb(trajectory):
    """
    Check if there are any rewrite tool calls after pdb tool calls.
    This indicates that pdb debugging sessions were useful for informing rewriting behavior.

    Args:
        trajectory: List of trajectory steps (log entries)

    Returns:
        bool: True if there's at least one rewrite call after a pdb call
    """
    pdb_seen = False

    for step in trajectory:
        action = step.get("action")
        if action:
            tool_name = action.get("name")

            # Track when we've seen a pdb call
            if tool_name == "pdb":
                pdb_seen = True

            # If we've seen pdb before and now see a rewrite, return True
            elif tool_name == "rewrite" and pdb_seen:
                return True

    return False


def has_rewrite_soon_after_pdb(trajectory, max_steps_between=10):
    """
    Check if there are rewrite tool calls within max_steps_between steps after pdb tool calls.
    This is a more precise check to ensure the pdb and rewrite calls are temporally related.

    Args:
        trajectory: List of trajectory steps (log entries)
        max_steps_between: Maximum number of steps allowed between pdb and rewrite calls

    Returns:
        bool: True if there's at least one rewrite call within max_steps_between steps after a pdb call
    """
    for i, step in enumerate(trajectory):
        action = step.get("action")
        if action and action.get("name") == "pdb":
            # Look ahead for rewrite calls within the specified window
            for j in range(i + 1, min(i + 1 + max_steps_between, len(trajectory))):
                next_step = trajectory[j]
                next_action = next_step.get("action")
                if next_action and next_action.get("name") == "rewrite":
                    return True

    return False


def count_debug_to_code_patterns(trajectory, max_steps_between=10):
    """
    Count how many times a pdb call is followed by a rewrite call within max_steps_between steps.
    This helps quantify how often debugging leads to code changes.

    Args:
        trajectory: List of trajectory steps (log entries)
        max_steps_between: Maximum number of steps allowed between pdb and rewrite calls

    Returns:
        int: Number of pdb->rewrite patterns found
    """
    pattern_count = 0

    for i, step in enumerate(trajectory):
        action = step.get("action")
        if action and action.get("name") == "pdb":
            # Look ahead for rewrite calls within the specified window
            for j in range(i + 1, min(i + 1 + max_steps_between, len(trajectory))):
                next_step = trajectory[j]
                next_action = next_step.get("action")
                if next_action and next_action.get("name") == "rewrite":
                    pattern_count += 1
                    break  # Only count the first rewrite after each pdb call

    return pattern_count


def has_sufficient_debug_to_code_patterns(
    trajectory, min_patterns=1, max_steps_between=10
):
    """
    Check if there are at least min_patterns instances of pdb calls followed by rewrite calls.

    Args:
        trajectory: List of trajectory steps (log entries)
        min_patterns: Minimum number of pdb->rewrite patterns required
        max_steps_between: Maximum number of steps allowed between pdb and rewrite calls

    Returns:
        bool: True if there are enough pdb->rewrite patterns
    """
    return count_debug_to_code_patterns(trajectory, max_steps_between) >= min_patterns


def follows_proper_debugging_workflow(trajectory):
    """
    Check if trajectory follows a logical debugging sequence:
    1. Set breakpoints using "b" or "break"
    2. Continue execution using "c"
    3. Print/inspect variables using "p" or "print"

    These actions don't need to be consecutive but should occur in this order.

    Args:
        trajectory: List of trajectory steps (log entries)

    Returns:
        bool: True if trajectory contains the logical debugging sequence
    """
    breakpoint_commands = ["b", "break"]
    continue_commands = ["c", "continue"]
    print_commands = ["p", "print", "pp"]

    # Track states: 0 = looking for breakpoint, 1 = looking for continue, 2 = looking for print, 3 = found all
    state = 0

    for step in trajectory:
        action = step.get("action")
        if action and action.get("name") == "pdb":
            arguments = action.get("arguments", {})
            command = arguments.get("command", "")

            if command:
                cmd = command.strip().split()[0]

                # State 0: Looking for breakpoint setting
                if state == 0 and cmd in breakpoint_commands:
                    state = 1

                # State 1: Looking for continue (after breakpoint was set)
                elif state == 1 and cmd in continue_commands:
                    state = 2

                # State 2: Looking for print (after continue was used)
                elif state == 2 and cmd in print_commands:
                    state = 3
                    return True  # Found complete sequence

    return False


# General trajectory criteria
# ---------------------------


def has_minimum_trajectory_length(trajectory, min_count=5):
    """
    Check if trajectory has at least min_count steps.

    Args:
        trajectory: List of trajectory steps (log entries)
        min_count: Minimum number of steps required

    Returns:
        bool: True if trajectory has enough steps
    """
    return len(trajectory) >= min_count


def has_sufficient_code_changes(trajectory, min_rewrites=1):
    """
    Check if trajectory has at least min_rewrites rewrite actions.

    Args:
        trajectory: List of trajectory steps (log entries)
        min_rewrites: Minimum number of rewrite actions required

    Returns:
        bool: True if trajectory has enough rewrite actions
    """
    rewrite_actions = 0
    for step in trajectory:
        action = step.get("action")
        if action and action.get("name") == "rewrite":
            rewrite_actions += 1

    return rewrite_actions >= min_rewrites
