from debug_gym.logger import DebugGymLogger, log_with_color


def get_message_tokens(message, count_tokens):
    """Count tokens in a single message.

    Args:
        message: A single message dict
        count_tokens: Function that takes a list of messages and returns total token count

    Returns:
        Token count for this message
    """
    return count_tokens([message])


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
    Trim message on the assistant-tool pair level to fit context length.

    Strategy:
    1. Keep the system message (assert if system itself is too long)
    2. Keep as many most recent (assistant, tool) pairs as possible
    3. Only when we can keep all (assistant, tool) pairs, keep the user message

    Args:
        messages: List of message dicts with 'role' and 'content'/'tool_calls' keys
        context_length: Maximum number of tokens allowed
        count_tokens: Function to count tokens in a string

    Returns:
        Trimmed list of messages that fit within context_length
    """
    assert len(messages) > 0, "messages should not be empty"
    assert messages[-1]["role"] in [
        "user",
        "tool",
    ], "the last message should be from the user or the tool"
    assert context_length >= 0, "context_length should be non-negative"

    # Calculate token count for all messages
    message_tokens = [get_message_tokens(msg, count_tokens) for msg in messages]
    total_tokens = sum(message_tokens)

    # If we're already within limit, return as-is
    if total_tokens <= context_length:
        return messages

    # Find system message
    system_msg_idx = 0 if messages[0]["role"] == "system" else None
    system_tokens = message_tokens[0] if system_msg_idx is not None else 0

    # Assert system message fits within context
    assert (
        system_tokens <= context_length
    ), f"System message tokens exceed context length: {system_tokens} > {context_length}!"

    # Find user message
    user_msg_idx = None
    for i, msg in enumerate(messages):
        if msg["role"] == "user":
            user_msg_idx = i
            break

    # Find all (assistant, tool) pairs by going backwards
    assistant_tool_pairs = []
    i = len(messages) - 1
    while i >= 0:
        if (
            messages[i]["role"] == "tool"
            and i > 0
            and messages[i - 1]["role"] == "assistant"
        ):
            assistant_tool_pairs.append((i - 1, i))  # (assistant_idx, tool_idx)
            i -= 2
        else:
            i -= 1

    # Start building result with system message
    result = []
    remaining_tokens = context_length

    if system_msg_idx is not None:
        result.append(messages[system_msg_idx])
        remaining_tokens -= system_tokens

    # Add as many recent (assistant, tool) pairs as possible
    included_pairs = []
    for assistant_idx, tool_idx in assistant_tool_pairs:
        pair_tokens = message_tokens[assistant_idx] + message_tokens[tool_idx]
        if pair_tokens <= remaining_tokens:
            included_pairs.append((assistant_idx, tool_idx))
            remaining_tokens -= pair_tokens
        else:
            break

    # Only include user message if we can fit all (assistant, tool) pairs
    include_user = False
    if len(included_pairs) == len(assistant_tool_pairs) and user_msg_idx is not None:
        user_tokens = message_tokens[user_msg_idx]
        if user_tokens <= remaining_tokens:
            include_user = True

    # Build final result
    if include_user:
        result.append(messages[user_msg_idx])

    # Sort by assistant index to maintain chronological order
    included_pairs.sort(key=lambda pair: pair[0])
    for assistant_idx, tool_idx in included_pairs:
        result.append(messages[assistant_idx])
        result.append(messages[tool_idx])

    assert (
        len(result) > 0
    ), f"After trimming, no messages fit within context length: {context_length}!"

    # Verify final token count
    final_tokens = sum(get_message_tokens(msg, count_tokens) for msg in result)
    assert (
        final_tokens <= context_length
    ), f"After trimming, the message length still exceeds: {final_tokens} > {context_length}!"

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
