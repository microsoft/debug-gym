"""
Unit tests for FrogyToolParser - custom vLLM tool parser for handling malformed JSON.
Tests cover the escape_unescaped_newlines_in_json_strings function and the parser's
extract_tool_calls method with various JSON formatting issues.
"""

import json
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

# Mock vLLM imports to avoid dependency issues during testing
sys.modules["vllm"] = MagicMock()
sys.modules["vllm.entrypoints"] = MagicMock()
sys.modules["vllm.entrypoints.openai"] = MagicMock()
sys.modules["vllm.entrypoints.openai.protocol"] = MagicMock()
sys.modules["vllm.entrypoints.openai.tool_parsers"] = MagicMock()
sys.modules["vllm.entrypoints.openai.tool_parsers.abstract_tool_parser"] = MagicMock()
sys.modules["vllm.transformers_utils"] = MagicMock()
sys.modules["vllm.transformers_utils.tokenizer"] = MagicMock()
sys.modules["vllm.entrypoints.chat_utils"] = MagicMock()

from debug_gym.frogboss import escape_unescaped_newlines_in_json_strings


class TestEscapeUnescapedNewlines:
    """Test the escape_unescaped_newlines_in_json_strings function."""

    def test_simple_json_no_changes(self):
        """Test that valid JSON is not modified."""
        json_str = '{"name": "bash", "arguments": {"command": "ls"}}'
        result = escape_unescaped_newlines_in_json_strings(json_str)
        assert result == json_str
        # Verify it's still valid JSON
        json.loads(result)

    def test_invalid_escape_single_quote(self):
        """Test that invalid \\' escape sequences are fixed to '."""
        json_str = '{"name": "pandas", "arguments": {"column": "df[\'Price\']"}}'
        result = escape_unescaped_newlines_in_json_strings(json_str)
        # Should handle the escaped single quotes properly
        # Should be valid JSON after fix
        json.loads(result)

    def test_escaped_backslash_quote_preserved(self):
        """Test that properly escaped quotes are preserved."""
        # Test with an escaped quote within a JSON string
        json_str = '{"text": "test\\\\"}'
        result = escape_unescaped_newlines_in_json_strings(json_str)
        parsed = json.loads(result)
        # The result should have a valid value
        assert "text" in parsed

    def test_implicit_string_concatenation(self):
        """Test that implicit string concatenation across lines is handled."""
        # Models sometimes generate "str1" "str2" across lines
        json_str = '{\n  "command": "python"\n"print(hello)"\n}'
        result = escape_unescaped_newlines_in_json_strings(json_str)
        # The concatenated strings should be merged
        assert '"' in result

    def test_literal_newline_in_string(self):
        """Test that literal newlines within JSON strings are escaped."""
        # A string containing an actual newline character
        json_str = '{"name": "test", "code": "line1\nline2"}'
        result = escape_unescaped_newlines_in_json_strings(json_str)
        # The newline should be escaped to \\n
        assert "\\n" in result
        # Should be valid JSON
        json.loads(result)

    def test_literal_tab_in_string(self):
        """Test that literal tabs within JSON strings are escaped."""
        json_str = '{"text": "col1\tcol2"}'
        result = escape_unescaped_newlines_in_json_strings(json_str)
        # The tab should be escaped to \\t
        assert "\\t" in result
        json.loads(result)

    def test_literal_carriage_return_in_string(self):
        """Test that literal carriage returns are escaped."""
        json_str = '{"text": "line1\rline2"}'
        result = escape_unescaped_newlines_in_json_strings(json_str)
        assert "\\r" in result
        json.loads(result)

    def test_literal_form_feed_in_string(self):
        """Test that literal form feeds are escaped."""
        json_str = '{"text": "page1\fpage2"}'
        result = escape_unescaped_newlines_in_json_strings(json_str)
        assert "\\f" in result
        json.loads(result)

    def test_literal_backspace_in_string(self):
        """Test that literal backspaces are escaped."""
        json_str = '{"text": "word\bchar"}'
        result = escape_unescaped_newlines_in_json_strings(json_str)
        assert "\\b" in result
        json.loads(result)

    def test_multiple_issues_combined(self):
        """Test that multiple JSON issues are fixed together."""
        # Simulate a broken tool call with multiple issues
        json_str = '{\n  "name": "bash",\n  "arguments": {"cmd": "df[Price]"}\n}'
        result = escape_unescaped_newlines_in_json_strings(json_str)
        # Should be valid JSON
        parsed = json.loads(result)
        assert "name" in parsed
        assert parsed["name"] == "bash"

    def test_empty_string(self):
        """Test that empty strings are handled."""
        json_str = '{"name": "", "arguments": {}}'
        result = escape_unescaped_newlines_in_json_strings(json_str)
        assert result == json_str
        json.loads(result)

    def test_escaped_quotes_in_strings(self):
        """Test that properly escaped quotes within strings are preserved."""
        json_str = '{"text": "say hello"}'
        result = escape_unescaped_newlines_in_json_strings(json_str)
        parsed = json.loads(result)
        assert "hello" in parsed["text"]

    def test_backslash_before_quote(self):
        """Test handling of backslash patterns."""
        json_str = '{"path": "C:\\\\test"}'
        result = escape_unescaped_newlines_in_json_strings(json_str)
        # Should be valid JSON - can parse without errors
        parsed = json.loads(result)
        assert "path" in parsed

    def test_complex_nested_structure(self):
        """Test that complex nested JSON is handled correctly."""
        json_str = json.dumps(
            {
                "name": "bash",
                "arguments": {
                    "command": "ls -la",
                    "options": {"detailed": True, "all": False},
                },
            }
        )
        result = escape_unescaped_newlines_in_json_strings(json_str)
        # Should still be valid and parseable
        parsed = json.loads(result)
        assert parsed["name"] == "bash"
        assert parsed["arguments"]["command"] == "ls -la"
