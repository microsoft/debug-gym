import os
import tempfile
from pathlib import Path

import pytest

from debug_gym.gym.envs.env import RepoEnv
from debug_gym.gym.tools.grep import GrepTool


@pytest.fixture
def setup_grep_test_repo():
    def _setup_grep_test_repo(base_dir):
        """Setup a test repository with various file types and content for grep testing"""
        working_dir = base_dir / "grep_test_repo"
        working_dir.mkdir()

        # Create subdirectories
        (working_dir / "src").mkdir()
        (working_dir / "tests").mkdir()
        (working_dir / "docs").mkdir()
        (working_dir / "hidden").mkdir()

        # Python files with various content
        with (working_dir / "main.py").open("w") as f:
            f.write(
                """#!/usr/bin/env python3
import os
import sys

def hello_world():
    print("Hello, World!")
    return "success"

class TestClass:
    def __init__(self):
        self.value = 42
        
    def method_with_bug(self):
        # TODO: Fix this bug
        return self.value / 0  # This will cause a division by zero error

if __name__ == "__main__":
    hello_world()
"""
            )

        with (working_dir / "src" / "utils.py").open("w") as f:
            f.write(
                """import re
import json

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def load_config(filename):
    # TODO: Add error handling
    with open(filename, 'r') as f:
        return json.load(f)

class EmailValidator:
    def __init__(self):
        self.pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    def validate(self, email):
        return re.match(self.pattern, email) is not None
"""
            )

        with (working_dir / "tests" / "test_utils.py").open("w") as f:
            f.write(
                """import pytest
from src.utils import validate_email, EmailValidator

def test_validate_email():
    assert validate_email("test@example.com") == True
    assert validate_email("invalid-email") == False

def test_email_validator_class():
    validator = EmailValidator()
    assert validator.validate("hello@world.com") == True
    assert validator.validate("bad-email") == False

# FIXME: This test is broken
def test_broken_function():
    # This test needs to be fixed
    assert False  # This should pass
"""
            )

        # Configuration files
        with (working_dir / "config.json").open("w") as f:
            f.write(
                """{
    "name": "test_project",
    "version": "1.0.0",
    "debug": true,
    "database_url": "sqlite:///test.db"
}"""
            )

        with (working_dir / "requirements.txt").open("w") as f:
            f.write(
                """pytest>=6.0.0
requests>=2.25.0
flask>=2.0.0
sqlalchemy>=1.4.0
"""
            )

        # Documentation
        with (working_dir / "README.md").open("w") as f:
            f.write(
                """# Test Project

This is a test project for grep functionality.

## Features
- Email validation
- Configuration loading
- Unit tests

## TODO
- Add more tests
- Improve error handling
- Fix known bugs

## Installation
```bash
pip install -r requirements.txt
```
"""
            )

        with (working_dir / "docs" / "api.md").open("w") as f:
            f.write(
                """# API Documentation

## EmailValidator Class

The EmailValidator class provides email validation functionality.

### Methods

- `validate(email)`: Returns True if email is valid, False otherwise
- `__init__()`: Initialize the validator with default pattern

## Functions

- `validate_email(email)`: Standalone function for email validation
- `load_config(filename)`: Load configuration from JSON file

## Examples

```python
validator = EmailValidator()
result = validator.validate("user@example.com")
```
"""
            )

        # Binary file (should be ignored)
        with (working_dir / "binary.bin").open("wb") as f:
            f.write(b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09")

        # Log file
        with (working_dir / "app.log").open("w") as f:
            f.write(
                """2024-01-01 10:00:00 INFO Starting application
2024-01-01 10:00:01 DEBUG Loading configuration
2024-01-01 10:00:02 ERROR Failed to connect to database
2024-01-01 10:00:03 WARNING Retrying connection
2024-01-01 10:00:04 INFO Application started successfully
"""
            )

        # Hidden files
        with (working_dir / ".gitignore").open("w") as f:
            f.write(
                """__pycache__/
*.pyc
*.pyo
*.pyd
.env
.venv
venv/
env/
"""
            )

        with (working_dir / "hidden" / "secret.txt").open("w") as f:
            f.write("This is a secret file that should be ignored by default")

        return working_dir

    return _setup_grep_test_repo


@pytest.fixture
def setup_grep_repo_env(setup_grep_test_repo):
    def _setup_grep_repo_env(base_dir, ignore_patterns=None, readonly_patterns=None):
        test_repo = setup_grep_test_repo(base_dir)

        # Create ignore patterns file if specified
        if ignore_patterns:
            with (test_repo / ".debugignore").open("w") as f:
                f.write("\n".join(ignore_patterns))

        # Create readonly patterns file if specified
        if readonly_patterns:
            with (test_repo / ".debugreadonly").open("w") as f:
                f.write("\n".join(readonly_patterns))

        env = RepoEnv(path=str(test_repo))
        grep_tool = GrepTool()
        env.reset()
        return grep_tool, env

    return _setup_grep_repo_env


class TestGrepTool:
    """Test cases for the GrepTool"""

    def test_grep_basic_search(self, tmp_path, setup_grep_repo_env):
        """Test basic string search functionality"""
        grep_tool, env = setup_grep_repo_env(tmp_path)

        # Search for a common word
        result = grep_tool.use(env, pattern="import")
        assert result.source == "grep"
        assert "Found" in result.observation
        assert "main.py" in result.observation
        assert "src/utils.py" in result.observation
        assert "tests/test_utils.py" in result.observation

    def test_grep_regex_search(self, tmp_path, setup_grep_repo_env):
        """Test regex pattern search"""
        grep_tool, env = setup_grep_repo_env(tmp_path)

        # Search for email pattern
        result = grep_tool.use(
            env, pattern=r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        )
        assert result.source == "grep"
        assert (
            "test@example.com" in result.observation
            or "hello@world.com" in result.observation
        )

    def test_grep_case_insensitive(self, tmp_path, setup_grep_repo_env):
        """Test case-insensitive search"""
        grep_tool, env = setup_grep_repo_env(tmp_path)

        # Search case-insensitively
        result = grep_tool.use(env, pattern="TODO", case_sensitive=False)
        assert result.source == "grep"
        assert "TODO" in result.observation or "todo" in result.observation

    def test_grep_specific_file(self, tmp_path, setup_grep_repo_env):
        """Test searching in a specific file"""
        grep_tool, env = setup_grep_repo_env(tmp_path)

        # Search only in main.py
        result = grep_tool.use(env, pattern="hello_world", path="main.py")
        assert result.source == "grep"
        assert "main.py" in result.observation
        assert "src/utils.py" not in result.observation

    def test_grep_specific_directory(self, tmp_path, setup_grep_repo_env):
        """Test searching in a specific directory"""
        grep_tool, env = setup_grep_repo_env(tmp_path)

        # Search only in src directory
        result = grep_tool.use(env, pattern="validate", path="src")
        assert result.source == "grep"
        assert "src/utils.py" in result.observation
        assert "tests/test_utils.py" not in result.observation

    def test_grep_with_ignore_patterns(self, tmp_path, setup_grep_repo_env):
        """Test that grep respects ignore patterns"""
        # Set up environment with ignore patterns
        grep_tool, env = setup_grep_repo_env(
            tmp_path, ignore_patterns=["*.md", "*.log", "hidden/", "*.bin"]
        )

        # Search for content that exists in ignored files
        result = grep_tool.use(env, pattern="Test Project")
        assert result.source == "grep"
        # Should not find matches in README.md since it's ignored
        assert "README.md" not in result.observation

    def test_grep_max_results_limit(self, tmp_path, setup_grep_repo_env):
        """Test max results limitation"""
        grep_tool, env = setup_grep_repo_env(tmp_path)

        # Search with a low max_results limit
        result = grep_tool.use(env, pattern="def", max_results=2)
        assert result.source == "grep"
        # Should mention limit was reached
        assert (
            "search limit reached" in result.observation
            or "Found 2 matches" in result.observation
        )

    def test_grep_no_matches_found(self, tmp_path, setup_grep_repo_env):
        """Test behavior when no matches are found"""
        grep_tool, env = setup_grep_repo_env(tmp_path)

        result = grep_tool.use(env, pattern="nonexistent_pattern_xyz123")
        assert result.source == "grep"
        assert "No matches found" in result.observation

    def test_grep_empty_pattern(self, tmp_path, setup_grep_repo_env):
        """Test behavior with empty pattern"""
        grep_tool, env = setup_grep_repo_env(tmp_path)

        result = grep_tool.use(env, pattern="")
        assert result.source == "grep"
        assert "Pattern cannot be empty" in result.observation

    def test_grep_invalid_regex(self, tmp_path, setup_grep_repo_env):
        """Test behavior with invalid regex pattern"""
        grep_tool, env = setup_grep_repo_env(tmp_path)

        result = grep_tool.use(env, pattern="[invalid(regex")
        assert result.source == "grep"
        assert "Invalid pattern" in result.observation

    def test_grep_nonexistent_path(self, tmp_path, setup_grep_repo_env):
        """Test behavior when searching in non-existent path"""
        grep_tool, env = setup_grep_repo_env(tmp_path)

        result = grep_tool.use(env, pattern="test", path="nonexistent/path")
        assert result.source == "grep"
        assert "Path not found" in result.observation

    def test_grep_line_numbers(self, tmp_path, setup_grep_repo_env):
        """Test that line numbers are included in output"""
        grep_tool, env = setup_grep_repo_env(tmp_path)

        result = grep_tool.use(
            env, pattern="def test_validate_email", line_numbers=True
        )
        assert result.source == "grep"
        # Should contain line numbers in format "   X: content"
        assert ":" in result.observation
        # Check that we have numbered lines
        lines = result.observation.split("\n")
        numbered_lines = [
            line for line in lines if ":" in line and line.strip()[0].isdigit()
        ]
        assert len(numbered_lines) > 0

    def test_grep_without_line_numbers(self, tmp_path, setup_grep_repo_env):
        """Test output without line numbers"""
        grep_tool, env = setup_grep_repo_env(tmp_path)

        result = grep_tool.use(
            env, pattern="def test_validate_email", line_numbers=False
        )
        assert result.source == "grep"
        # Should not contain numbered lines format
        lines = result.observation.split("\n")
        # Skip header lines and file separators
        content_lines = [
            line
            for line in lines
            if line.strip()
            and not line.startswith("===")
            and not line.startswith("Found")
        ]
        numbered_lines = [
            line
            for line in content_lines
            if ":" in line and line.strip().split(":")[0].strip().isdigit()
        ]
        assert len(numbered_lines) == 0

    def test_grep_binary_file_handling(self, tmp_path, setup_grep_repo_env):
        """Test that binary files are handled gracefully"""
        grep_tool, env = setup_grep_repo_env(tmp_path)

        # Search for pattern that might appear in binary data - should not crash
        result = grep_tool.use(env, pattern="\x00")
        assert result.source == "grep"
        # Should either find no matches or handle the binary file gracefully
        assert "Search failed" not in result.observation

    def test_grep_respects_environment_visibility(self, tmp_path, setup_grep_repo_env):
        """Test that grep only searches visible files according to environment rules"""
        # Set up environment with ignore patterns
        grep_tool, env = setup_grep_repo_env(
            tmp_path, ignore_patterns=[".git*", "hidden/", "*.bin"]
        )

        # Search for content that exists in hidden files
        result = grep_tool.use(env, pattern="secret")
        assert result.source == "grep"
        # Should not find the secret file content
        assert "secret.txt" not in result.observation

    def test_grep_file_too_large_line_truncation(self, tmp_path, setup_grep_repo_env):
        """Test that very long lines are truncated properly"""
        grep_tool, env = setup_grep_repo_env(tmp_path)

        # Create a file with a very long line
        long_line_file = env.working_dir / "long_line.txt"
        with long_line_file.open("w") as f:
            f.write("search_term " + "x" * 300 + " end")

        result = grep_tool.use(env, pattern="search_term")
        assert result.source == "grep"
        # Should truncate long lines with "..."
        assert "..." in result.observation

    def test_grep_unicode_handling(self, tmp_path, setup_grep_repo_env):
        """Test handling of Unicode characters"""
        grep_tool, env = setup_grep_repo_env(tmp_path)

        # Create a file with Unicode content
        unicode_file = env.working_dir / "unicode.txt"
        with unicode_file.open("w", encoding="utf-8") as f:
            f.write("Hello 世界 🌍 café naïve résumé\nSearch term with émojis 🚀✨")

        result = grep_tool.use(env, pattern="Search term")
        assert result.source == "grep"
        assert "unicode.txt" in result.observation
        # Should handle Unicode without errors
        assert "Search failed" not in result.observation
