import pytest

from debug_gym.llms.utils import print_messages, trim, trim_prompt_messages


def test_print_messages(logger_mock):
    messages = [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": "Hi",
            "tool_calls": ["Tool call 1", "Tool call 2"],
        },
        {"role": "tool", "content": "Tool message"},
        {
            "role": "user",
            "content": [{"type": "tool_result", "content": "Tool result"}],
        },
        {"role": "assistant", "tool_calls": [{"key": 3, "key2": "value"}]},
        {"role": "system", "content": 12345},
    ]
    print_messages(messages, logger_mock)
    assert logger_mock._log_history == [
        "[yellow]System message[/yellow]",
        "[magenta]Hello[/magenta]",
        "[cyan]Hi[/cyan]",
        "[cyan]Tool call: Tool call 1[/cyan]",
        "[cyan]Tool call: Tool call 2[/cyan]",
        "[green]Tool message[/green]",
        "[magenta]Tool result[/magenta]",
        "[cyan]Tool call: {'key': 3, 'key2': 'value'}[/cyan]",
        "[yellow]12345[/yellow]",
    ]


def test_print_messages_unknown_role(logger_mock):
    bad_message = [{"role": "unknown", "content": "bad"}]
    try:
        print_messages(bad_message, logger_mock)
    except ValueError as e:
        assert "Unknown role" in str(e)
    else:
        assert False, "ValueError not raised for unknown role"


class TestTrimPromptMessagesValidation:
    """Test input validation for trim_prompt_messages."""

    def count_tokens(self, messages):
        return sum(
            len(msg.get("content", msg.get("tool_calls", ""))) for msg in messages
        )

    def test_empty_messages_raises(self):
        with pytest.raises(AssertionError, match="messages should not be empty"):
            trim_prompt_messages([], 5, self.count_tokens)

    def test_last_message_not_user_or_tool_raises(self):
        messages = [
            {"role": "system", "content": "System"},
            {"role": "assistant", "content": "Assistant"},
        ]
        with pytest.raises(
            AssertionError, match="the last message should be from the user or the tool"
        ):
            trim_prompt_messages(messages, 20, self.count_tokens)

    def test_negative_context_length_raises(self):
        messages = [{"role": "user", "content": "Hi"}]
        with pytest.raises(
            AssertionError, match="context_length should be non-negative"
        ):
            trim_prompt_messages(messages, -1, self.count_tokens)

    def test_system_message_too_long_raises(self):
        messages = [
            {"role": "system", "content": "Very long system message"},
            {"role": "user", "content": "Hi"},
        ]
        with pytest.raises(
            AssertionError, match="System message tokens exceed context length"
        ):
            trim_prompt_messages(messages, 10, self.count_tokens)


class TestTrimPromptMessagesToolBased:
    """Test trim_prompt_messages with tool-based conversations."""

    def count_tokens(self, messages):
        return sum(
            len(msg.get("content", msg.get("tool_calls", ""))) for msg in messages
        )

    def test_all_messages_fit_returns_unchanged(self):
        """When all messages fit, return as-is."""
        messages = [
            {"role": "system", "content": "Sys"},  # 3
            {"role": "user", "content": "Hi"},  # 2
            {"role": "assistant", "content": "Hello"},  # 5
            {"role": "tool", "content": "Result"},  # 6
        ]
        result = trim_prompt_messages(messages, 50, self.count_tokens)
        assert result == messages

    def test_drop_pair_keep_system_user(self):
        """When context is tight, drop pairs but keep system + user."""
        messages = [
            {"role": "system", "content": "Sys"},  # 3
            {"role": "user", "content": "Hi"},  # 2
            {"role": "assistant", "content": "Hello"},  # 5
            {"role": "tool", "content": "Result"},  # 6
        ]
        expected = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "Hi"},
        ]
        # Context = 6: only fits system(3) + user(2) = 5
        result = trim_prompt_messages(messages, 6, self.count_tokens)
        assert result == expected

    def test_multiple_pairs_keep_most_recent(self):
        """With multiple pairs, keep the most recent ones that fit."""
        messages = [
            {"role": "system", "content": "Sys"},  # 3
            {"role": "user", "content": "Hi"},  # 2
            {"role": "assistant", "content": "Hello1"},  # 6
            {"role": "tool", "content": "Result1"},  # 7
            {"role": "assistant", "content": "Hello2"},  # 6
            {"role": "tool", "content": "Result2"},  # 7
        ]
        expected = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello2"},
            {"role": "tool", "content": "Result2"},
        ]
        # Context = 18: sys(3) + user(2) + pair2(13) = 18
        result = trim_prompt_messages(messages, 18, self.count_tokens)
        assert result == expected

    def test_no_system_message(self):
        """Works without system message."""
        messages = [
            {"role": "user", "content": "Hi"},  # 2
            {"role": "assistant", "content": "Hello"},  # 5
            {"role": "tool", "content": "Result"},  # 6
        ]
        result = trim_prompt_messages(messages, 20, self.count_tokens)
        assert result == messages

    def test_orphan_tool_message_dropped(self):
        """Tool message without preceding assistant is dropped during trimming."""
        messages = [
            {"role": "system", "content": "Sys"},  # 3
            {"role": "user", "content": "Hi"},  # 2
            {"role": "tool", "content": "Result"},  # 6 - orphan!
        ]
        expected = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "Hi"},
        ]
        # Context = 10: forces trimming (total is 11), orphan tool is dropped since it's not paired
        result = trim_prompt_messages(messages, 10, self.count_tokens)
        assert result == expected

    def test_assistant_with_tool_calls_attribute(self):
        """Handles assistant messages with tool_calls instead of content."""
        messages = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "test"}}]},
            {"role": "tool", "content": "Result"},
        ]
        result = trim_prompt_messages(messages, 100, self.count_tokens)
        assert len(result) == 4


class TestTrimPromptMessagesSimpleConversation:
    """Test trim_prompt_messages with simple conversations (no tools)."""

    def count_tokens(self, messages):
        return sum(len(msg.get("content", "")) for msg in messages)

    def test_all_messages_fit_returns_unchanged(self):
        """When all messages fit, return as-is."""
        messages = [
            {"role": "system", "content": "Sys"},  # 3
            {"role": "user", "content": "Hi"},  # 2
            {"role": "assistant", "content": "Hello"},  # 5
            {"role": "user", "content": "Bye"},  # 3
        ]
        result = trim_prompt_messages(messages, 50, self.count_tokens)
        assert result == messages

    def test_keep_system_user_and_recent_pair(self):
        """Keep system, first user (task), and most recent (assistant, user) pair."""
        messages = [
            {"role": "system", "content": "Sys"},  # 3
            {"role": "user", "content": "Hi"},  # 2
            {"role": "assistant", "content": "Hello"},  # 5
            {"role": "user", "content": "Bye"},  # 3
        ]
        # Context = 13: system(3) + user(2) + pair(5+3=8) = 13
        expected = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Bye"},
        ]
        result = trim_prompt_messages(messages, 13, self.count_tokens)
        assert result == expected

    def test_drop_pair_if_too_large(self):
        """Drop (assistant, user) pair if it doesn't fit."""
        messages = [
            {"role": "system", "content": "Sys"},  # 3
            {"role": "user", "content": "Hi"},  # 2
            {"role": "assistant", "content": "Hello"},  # 5
            {"role": "user", "content": "Bye"},  # 3
        ]
        # Context = 10: system(3) + user(2) = 5, remaining = 5
        # Pair (5+3=8) doesn't fit
        expected = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "Hi"},
        ]
        result = trim_prompt_messages(messages, 10, self.count_tokens)
        assert result == expected

    def test_no_system_keep_user_and_recent_pair(self):
        """Without system, keep first user (task) and most recent (assistant, user) pair."""
        messages = [
            {"role": "user", "content": "First"},  # 5
            {"role": "assistant", "content": "Response1"},  # 9
            {"role": "user", "content": "Second"},  # 6
            {"role": "assistant", "content": "Response2"},  # 9
            {"role": "user", "content": "Third"},  # 5
        ]
        # Context = 20: First(5) + pair2(9+5=14) = 19
        # Pairs: (Response1, Second)=15, (Response2, Third)=14
        expected = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Response2"},
            {"role": "user", "content": "Third"},
        ]
        result = trim_prompt_messages(messages, 20, self.count_tokens)
        assert result == expected

    def test_single_user_message(self):
        """Single user message works."""
        messages = [{"role": "user", "content": "Hi"}]
        result = trim_prompt_messages(messages, 10, self.count_tokens)
        assert result == messages

    def test_system_and_user_only(self):
        """System + user only, no assistant messages."""
        messages = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "Hi"},
        ]
        result = trim_prompt_messages(messages, 10, self.count_tokens)
        assert result == messages


class TestTrimPromptMessagesAnthropicFormat:
    """Test trim_prompt_messages with Anthropic-style tool results."""

    def count_tokens(self, messages):
        return sum(
            len(str(msg.get("content", msg.get("tool_calls", "")))) for msg in messages
        )

    def test_anthropic_tool_result_recognized(self):
        """Anthropic tool_result format is recognized as tool message."""
        messages = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "123", "name": "test", "input": {}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "123", "content": "Result"}
                ],
            },
        ]
        result = trim_prompt_messages(messages, 500, self.count_tokens)
        assert len(result) == 4

    def test_anthropic_format_keeps_user_and_recent_pair(self):
        """With Anthropic format, keeps user message and most recent pair."""
        messages = [
            {"role": "system", "content": "Sys"},  # 3
            {"role": "user", "content": "Hi"},  # 2
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "1", "name": "old", "input": {}}
                ],
            },  # ~61
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "1", "content": "OldResult"}
                ],
            },  # ~69
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "2", "name": "new", "input": {}}
                ],
            },  # ~61
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "2", "content": "NewResult"}
                ],
            },  # ~69
        ]
        # System(3) + user(2) + pair2(~130) = ~135
        result = trim_prompt_messages(messages, 140, self.count_tokens)
        assert len(result) == 4  # system + user + most recent pair
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "Hi"  # task description, not tool_result
        assert result[2]["role"] == "assistant"
        assert "2" in str(result[2]["content"])  # most recent pair


def test_trim():
    def count_tokens(text):
        return len(text)

    # Test basic cases - no trimming needed
    assert trim("Hello world", 11, count_tokens) == "Hello world"
    assert trim("Hello world", 20, count_tokens) == "Hello world"
    assert trim("Hi", 2, count_tokens) == "Hi"
    assert trim("Hi", 10, count_tokens) == "Hi"
    assert trim("A", 1, count_tokens) == "A"  # Exactly fits, no trimming needed

    # Test edge cases
    assert trim("Hello world", 0, count_tokens) == ""
    assert trim("", 5, count_tokens) == ""
    assert trim("", 0, count_tokens) == ""

    # Test cases requiring trimming to single token (ellipsis only)
    assert trim("Hello world", 1, count_tokens) == "…"
    assert trim("Hi", 1, count_tokens) == "…"
    assert trim("ABC", 1, count_tokens) == "…"

    # Test trimming from the middle (default behavior)
    assert trim("Hello world", 5, count_tokens) == "He…ld"
    assert trim("Hello world", 6, count_tokens) == "He…rld"
    assert trim("Hello world", 7, count_tokens) == "Hel…rld"
    assert trim("123456789", 5, count_tokens) == "12…89"
    assert trim("123456789", 7, count_tokens) == "123…789"

    # Test trimming from the end
    assert trim("Hello world", 5, count_tokens, where="end") == "Hell…"
    assert trim("Hello world", 6, count_tokens, where="end") == "Hello…"
    assert trim("Hello world", 7, count_tokens, where="end") == "Hello …"
    assert trim("123456789", 5, count_tokens, where="end") == "1234…"

    # Test trimming from the start
    assert trim("Hello world", 5, count_tokens, where="start") == "…orld"
    assert trim("Hello world", 6, count_tokens, where="start") == "…world"
    assert trim("Hello world", 7, count_tokens, where="start") == "… world"
    assert trim("123456789", 5, count_tokens, where="start") == "…6789"

    # Test invalid `where` value
    with pytest.raises(ValueError, match="Invalid value for `where`"):
        trim("Hello world", 5, count_tokens, where="invalid")

    # Test with different token counter
    def another_count_tokens(text):
        return len(text) // 2

    # For "1234567890" (10 chars), another_count_tokens returns 5 tokens
    # Original text has 5 tokens, so no trimming needed when max_tokens >= 5
    assert trim("1234567890", 5, another_count_tokens) == "1234567890"
    assert trim("1234567890", 6, another_count_tokens) == "1234567890"

    # When max_tokens < 5, trimming is needed
    # With max_tokens=4, we need 3 tokens for content + 1 for ellipsis
    assert (
        trim("1234567890", 4, another_count_tokens) == "123…67890"
    )  # Result has 4 tokens
    assert (
        trim("1234567890", 3, another_count_tokens) == "123…890"
    )  # Result has 3 tokens
    assert trim("1234567890", 2, another_count_tokens) == "1…890"  # Result has 2 tokens
    assert trim("1234567890", 1, another_count_tokens) == "1…0"  # Result has 1 token

    # Test with different trimming positions using the alternative counter
    assert (
        trim("1234567890", 3, another_count_tokens, where="end") == "12345…"
    )  # Result has 3 tokens
    assert (
        trim("1234567890", 3, another_count_tokens, where="start") == "…67890"
    )  # Result has 3 tokens

    # Test edge case with very short text and alternative counter
    assert trim("AB", 1, another_count_tokens) == "AB"  # "AB" has 1 token, fits exactly
    assert (
        trim("ABCD", 1, another_count_tokens) == "A…D"
    )  # "ABCD" has 2 tokens, needs trimming to 1

    # Test boundary conditions with precise scenarios
    def word_count_tokens(text):
        # Count words as tokens
        return len(text.split())

    text = "Hello world test example"  # 4 words = 4 tokens
    assert trim(text, 4, word_count_tokens) == text  # No trimming needed
    assert (
        trim(text, 3, word_count_tokens, where="middle") == "Hello … example"
    )  # Should fit in 3 tokens
    assert (
        trim(text, 2, word_count_tokens, where="end") == "Hello …"
    )  # Should fit in 2 tokens
    assert (
        trim(text, 2, word_count_tokens, where="start") == "… example"
    )  # Should fit in 2 tokens

    # Test very short max_tokens with word counter
    assert trim("Hello world", 1, word_count_tokens) == "…"  # Only ellipsis fits
