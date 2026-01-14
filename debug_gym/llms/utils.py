from debug_gym.logger import DebugGymLogger, log_with_color


def trim(text: str, max_tokens: int, count_tokens: callable, where: str = "middle"):
    """Trim text to fit within max_tokens by working directly at the token level."""
    if max_tokens <= 0:
        return ""

    nb_tokens = count_tokens(text)
    if nb_tokens <= max_tokens:
        return text

    ellipsis = "…"  # assume ellipsis is a single token
    available_tokens = max_tokens - 1  # account for ellipsis

    def find_char_position_for_tokens(
        target_tokens: int, from_start: bool = True
    ) -> int:
        """Binary search to find character position that gives approximately target_tokens."""
        left, right = 0, len(text)
        best_pos = left if from_start else right

        while left <= right:
            mid = (left + right) // 2
            test_text = text[:mid] if from_start else text[mid:]
            test_tokens = count_tokens(test_text)
            if test_tokens <= target_tokens:
                best_pos = mid
                if from_start:
                    left = mid + 1
                else:
                    right = mid - 1
            else:
                if from_start:
                    right = mid - 1
                else:
                    left = mid + 1
        return best_pos

    if where == "end":
        # Keep the beginning, trim the end
        trim_point = find_char_position_for_tokens(available_tokens, from_start=True)
        return text[:trim_point] + ellipsis
    elif where == "start":
        # Keep the end, trim the beginning
        trim_point = find_char_position_for_tokens(available_tokens, from_start=False)
        return ellipsis + text[trim_point:]
    elif where == "middle":
        # Keep both ends, trim the middle
        half_tokens = available_tokens // 2

        # Find how much we can keep from the start
        start_chars = find_char_position_for_tokens(half_tokens, from_start=True)

        # Find how much we can keep from the end with remaining tokens
        remaining_tokens = available_tokens - count_tokens(text[:start_chars])
        end_chars = find_char_position_for_tokens(remaining_tokens, from_start=False)

        return text[:start_chars] + ellipsis + text[end_chars:]
    else:
        raise ValueError(f"Invalid value for `where`: {where!r}.")


def trim_prompt_messages(
    messages: list[dict], context_length: int, count_tokens: callable
):
    """
    Trim messages to fit within context length.

    Priority:
    1. System message (always kept if present)
    2. First user message (task description)
    3. Most recent (assistant, user/tool) pairs

    Orphan messages (without proper pairing) are dropped.

    Args:
        messages: List of message dicts with 'role' and 'content'/'tool_calls' keys
        context_length: Maximum number of tokens allowed
        count_tokens: Function to count tokens in messages

    Returns:
        Trimmed list of messages that fit within context_length
    """
    # Validation
    assert len(messages) > 0, "messages should not be empty"
    assert messages[-1]["role"] in [
        "user",
        "tool",
    ], "the last message should be from the user or the tool"
    assert context_length >= 0, "context_length should be non-negative"

    # Calculate token counts upfront
    tokens = [count_tokens([msg]) for msg in messages]

    # Early exit if already within limit
    if sum(tokens) <= context_length:
        return messages

    # Extract system message (always at index 0 if present)
    has_system = messages[0]["role"] == "system"
    system_msg, system_tokens = (messages[0], tokens[0]) if has_system else (None, 0)

    assert (
        system_tokens <= context_length
    ), f"System message tokens exceed context length: {system_tokens} > {context_length}!"

    # Extract first user message (task description)
    # It's at index 1 if system exists, else index 0
    user_idx = 1 if has_system else 0
    has_user = user_idx < len(messages) and messages[user_idx]["role"] == "user"
    user_msg, user_tokens = (
        (messages[user_idx], tokens[user_idx]) if has_user else (None, 0)
    )

    # Find (assistant, response) pairs where response is user or tool
    pairs = []
    start_idx = (1 if has_system else 0) + (1 if has_user else 0)
    for i in range(start_idx, len(messages) - 1):
        if messages[i]["role"] != "assistant":
            continue
        next_msg = messages[i + 1]
        if next_msg["role"] in ["user", "tool"]:
            pairs.append((messages[i], next_msg, tokens[i] + tokens[i + 1]))

    # Build result
    result = []
    remaining = context_length

    # 1. Always include system message first
    if system_msg:
        result.append(system_msg)
        remaining -= system_tokens

    # 2. Always include user message (task description) if it fits
    if user_msg and user_tokens <= remaining:
        result.append(user_msg)
        remaining -= user_tokens

    # Add most recent pairs that fit
    selected_pairs = []
    for pair in reversed(pairs):
        assistant_msg, response_msg, pair_tokens = pair
        if pair_tokens <= remaining:
            selected_pairs.append((assistant_msg, response_msg))
            remaining -= pair_tokens
        else:
            break

    # Add in chronological order
    for assistant_msg, response_msg in reversed(selected_pairs):
        result.append(assistant_msg)
        result.append(response_msg)

    assert (
        len(result) > 0
    ), f"After trimming, no messages fit within context length: {context_length}!"

    return result


def print_messages(messages: list[dict], logger: DebugGymLogger):
    """Print messages coloring each role differently.
    Colors:
        green: selected tool or assistant messages
        magenta: result of tool calls
        cyan: user messages
        yellow: system message
    """
    for m in messages:
        role = m["role"]
        if role == "tool":
            log_with_color(logger, m["content"], "green")
        elif role == "user":
            if isinstance(m["content"], list):
                for item in m["content"]:
                    if item["type"] == "tool_result":
                        log_with_color(logger, str(item["content"]), "magenta")
                    else:
                        log_with_color(logger, str(item), "magenta")
            else:
                log_with_color(logger, str(m["content"]), "magenta")
        elif role == "assistant":
            content = m.get("content")
            if content:
                if isinstance(content, list):
                    for item in content:
                        log_with_color(logger, str(item), "cyan")
                else:
                    log_with_color(logger, str(content), "cyan")
            tool_calls = m.get("tool_calls")
            if tool_calls:
                for tool_call in tool_calls:
                    log_with_color(logger, f"Tool call: {tool_call}", "cyan")
        elif role == "system":
            log_with_color(logger, str(m["content"]), "yellow")
        else:
            raise ValueError(f"Unknown role: {m['content']}")
