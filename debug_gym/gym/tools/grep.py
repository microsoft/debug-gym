import os
import re
import shlex
from debug_gym.gym.entities import Observation
from debug_gym.gym.tools.tool import EnvironmentTool
from debug_gym.gym.tools.toolbox import Toolbox


@Toolbox.register()
class GrepTool(EnvironmentTool):
    name: str = "grep"

    examples = [
        """grep(pattern="function", path=None) to search for the word "function" in all files in the repository.""",
        """grep(pattern="class.*Test", path="*.py") to search for lines matching the regex pattern "class.*Test" in all files under the 'tests/' directory.""",
        """grep(pattern="import numpy", path="src/main.py") to search for "import numpy" in the specific file 'src/main.py'.""",
        """grep(pattern="TODO") to search for "TODO".""",
        """grep(pattern="bug", max_results=10) to search for "bug" and limit results to 10 matches.""",
    ]
    description = (
        "Search for a pattern in files within the repository. Can search in specific files, directories, or the entire repository. "
        "Supports both literal string matching and regular expressions."
        + "\nExamples (for demonstration purposes only, you need to adjust the tool calling format according to your specific syntax):\n"
        + "\n".join(examples)
    )
    arguments = {
        "pattern": {
            "type": ["string"],
            "description": "The pattern to search for. Can be a literal string or a regular expression (if regex=True).",
        },
        "path": {
            "type": ["string", "null"],
            "description": "Optional glob pattern to search in. If None, searches the entire repository. Path should be relative to the repository root.",
        },
        "max_results": {
            "type": ["number", "null"],
            "description": "Maximum number of matching lines to return. If None, returns 100 matches.",
        },
    }

    def use(
        self,
        environment,
        pattern: str,
        path: str = None,
        regex: bool = True,
        case_sensitive: bool = True,
        line_numbers: bool = True,
        max_results: int = 100,
    ) -> Observation:
        if not pattern:
            return Observation(self.name, "Pattern cannot be empty.")

        # Compile the search pattern
        try:
            flags = 0 if case_sensitive else re.IGNORECASE
            compiled_pattern = re.compile(pattern, flags)
        except re.error as e:
            return Observation(self.name, f"Invalid pattern: {str(e)}")

        # Determine search scope
        if path:
            search_path = environment.resolve_path(path)
            if not os.path.exists(search_path):
                return Observation(self.name, f"Path not found: {path}")
        else:
            search_path = environment.working_dir

        results = []
        files_searched = 0
        max_files = 1000  # Reasonable limit to prevent overwhelming output

        try:
            # If path is a single file
            if os.path.isfile(search_path):
                results.extend(
                    self._search_file(
                        search_path,
                        compiled_pattern,
                        line_numbers,
                        max_results,
                        environment,
                    )
                )
                files_searched = 1
            else:
                # Search directory recursively
                for root, dirs, files in os.walk(search_path):
                    # Skip hidden directories and common build/cache directories
                    dirs[:] = [
                        d
                        for d in dirs
                        if not d.startswith(".")
                        and d
                        not in {
                            "__pycache__",
                            "node_modules",
                            "tmp",
                            ".git",
                            ".venv",
                            "venv",
                            "build",
                            "dist",
                        }
                    ]

                    for file in files:
                        if files_searched >= max_files:
                            break

                        # Skip binary files and common non-text files
                        if self._is_text_file(file):
                            file_path = os.path.join(root, file)
                            try:
                                file_results = self._search_file(
                                    file_path,
                                    compiled_pattern,
                                    line_numbers,
                                    max_results - len(results),
                                    environment,
                                )
                                results.extend(file_results)
                                files_searched += 1

                                if len(results) >= max_results:
                                    break
                            except Exception as e:
                                # Skip files that can't be read
                                continue

                    if files_searched >= max_files or len(results) >= max_results:
                        break

        except Exception as e:
            return Observation(self.name, f"Search failed: {str(e)}")

        # Format output
        if not results:
            search_scope = f"in {path}" if path else "in repository"
            pattern_desc = f"pattern '{pattern}'"
            return Observation(
                self.name, f"No matches found for {pattern_desc} {search_scope}."
            )

        output_lines = []
        if len(results) >= max_results:
            output_lines.append(
                f"Showing last {len(results)} matches (search limit reached):"
            )
        else:
            output_lines.append(f"Found {len(results)} matches:")

        output_lines.append("")

        current_file = None
        for file_path, line_num, line_content in results[-max_results:]:
            # Show relative path from repository root
            rel_path = os.path.relpath(file_path, environment.working_dir)

            if rel_path != current_file:
                if current_file is not None:
                    output_lines.append("")  # Empty line between files
                output_lines.append(f"=== {rel_path} ===")
                current_file = rel_path

            if line_numbers:
                output_lines.append(f"{line_num:4d}: {line_content}")
            else:
                output_lines.append(line_content)

        return Observation(self.name, "\n".join(output_lines))

    def _search_file(
        self, file_path, pattern, show_line_numbers, max_results, environment
    ):
        """Search for pattern in a single file."""
        results = []
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f, 1):
                    if len(results) >= max_results:
                        break

                    if pattern.search(line):
                        # Remove trailing newline and limit line length for display
                        clean_line = line.rstrip("\n\r")
                        if len(clean_line) > 200:
                            clean_line = clean_line[:197] + "..."

                        results.append((file_path, line_num, clean_line))
        except (UnicodeDecodeError, PermissionError):
            # Skip files that can't be read as text
            pass

        return results

    def _is_text_file(self, filename):
        """Check if a file is likely to be a text file based on its extension."""
        text_extensions = {
            ".py",
            ".js",
            ".ts",
            ".html",
            ".css",
            ".md",
            ".txt",
            ".json",
            ".xml",
            ".yaml",
            ".yml",
            ".sh",
            ".bash",
            ".zsh",
            ".fish",
            ".ps1",
            ".bat",
            ".cmd",
            ".sql",
            ".r",
            ".R",
            ".c",
            ".cpp",
            ".h",
            ".hpp",
            ".java",
            ".go",
            ".rs",
            ".php",
            ".rb",
            ".pl",
            ".scala",
            ".swift",
            ".kt",
            ".dart",
            ".lua",
            ".vim",
            ".emacs",
            ".ini",
            ".cfg",
            ".conf",
            ".log",
            ".csv",
            ".tsv",
            ".rst",
            ".tex",
            ".latex",
            ".bib",
            ".dockerfile",
            ".gitignore",
            ".gitattributes",
            ".editorconfig",
            ".prettierrc",
            ".eslintrc",
            ".pylintrc",
            ".toml",
            ".lock",
            ".requirements",
            ".pipfile",
            ".poetry",
            ".gradle",
            ".maven",
            ".makefile",
            ".cmake",
            ".ninja",
            ".gyp",
            ".gn",
            ".bzl",
            ".bazel",
            ".ant",
        }

        # Files without extension that are usually text
        text_files = {
            "README",
            "LICENSE",
            "CHANGELOG",
            "CONTRIBUTING",
            "INSTALL",
            "NEWS",
            "AUTHORS",
            "COPYING",
            "NOTICE",
            "TODO",
            "BUGS",
            "MANIFEST",
            "Makefile",
            "Dockerfile",
            "Vagrantfile",
            "Gemfile",
            "Rakefile",
            "Procfile",
        }

        name = os.path.basename(filename)
        _, ext = os.path.splitext(filename)

        return (
            ext.lower() in text_extensions
            or name in text_files
            or name.upper() in text_files
        )
