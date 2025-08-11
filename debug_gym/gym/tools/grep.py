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
        """Use grep functionality via bash tool as a special case."""
        if not pattern:
            return Observation(self.name, "Pattern cannot be empty.")

        # Build grep command arguments
        grep_args = []

        # Add options
        grep_args.append("-n")  # line numbers
        grep_args.append("-r")  # recursive
        grep_args.append("-E")  # extended regex
        grep_args.append("-H")  # print filename with output

        if not case_sensitive:
            grep_args.append("-i")  # ignore case

        if not regex:
            grep_args.append("-F")  # fixed strings (literal)

        # Add pattern (safely quoted)
        grep_args.append(shlex.quote(pattern))

        # Add path or default to current directory
        if path:
            grep_args.append(shlex.quote(path))
        else:
            grep_args.append(".")

        # Build the complete command
        command = "grep " + " ".join(grep_args)

        # Add exclusions for common non-text directories and limit results
        command += (
            " | grep -v '/.git/' | grep -v '__pycache__' | grep -v '/node_modules/'"
        )

        if max_results:
            command += f" | head -{max_results}"

        try:
            # Assert that the terminal is a Docker terminal
            from debug_gym.gym.terminal import DockerTerminal

            if not isinstance(environment.terminal, DockerTerminal):
                return Observation(
                    self.name,
                    "Error: grep tool requires a Docker terminal. Current terminal type is not supported.",
                )

            # Use the environment's terminal to run the grep command
            success, output = environment.terminal.run(command, timeout=30)

            if success:
                if output.strip():
                    # Process the output to match expected format
                    lines = output.strip().split("\n")

                    if not lines or (len(lines) == 1 and not lines[0]):
                        search_scope = f"in {path}" if path else "in repository"
                        pattern_desc = f"pattern '{pattern}'"
                        return Observation(
                            self.name,
                            f"No matches found for {pattern_desc} {search_scope}.",
                        )

                    if lines[0].startswith("grep: "):
                        # Handle grep error messages
                        return Observation(self.name, f"Grep error: {lines[0][6:]}")

                    # Format output
                    output_lines = []
                    if len(lines) >= max_results:
                        output_lines.append(
                            f"Showing first {len(lines)} matches (search limit reached):"
                        )
                    else:
                        output_lines.append(f"Found {len(lines)} matches:")

                    output_lines.append("")

                    current_file = None
                    for line in lines:
                        if ":" in line:
                            # Parse grep output: filename:line_number:content
                            parts = line.split(":", 2)
                            if len(parts) >= 3:
                                file_path = parts[0]
                                line_num = parts[1]
                                line_content = parts[2]

                                if file_path != current_file:
                                    if current_file is not None:
                                        output_lines.append(
                                            ""
                                        )  # Empty line between files
                                    output_lines.append(f"=== {file_path} ===")
                                    current_file = file_path

                                if len(line_content) >= 300:
                                    line_content = line_content[:300] + "..."

                                if line_numbers:
                                    output_lines.append(
                                        f"{line_num:>4}: {line_content}"
                                    )
                                else:
                                    output_lines.append(line_content)
                            else:
                                # Fallback for unusual grep output
                                output_lines.append(line)
                        else:
                            output_lines.append(line)

                    return Observation(self.name, "\n".join(output_lines))
                else:
                    search_scope = f"in {path}" if path else "in repository"
                    pattern_desc = f"pattern '{pattern}'"
                    return Observation(
                        self.name,
                        f"No matches found for {pattern_desc} {search_scope}.",
                    )
            else:
                return Observation(self.name, f"Grep command failed: {output}")

        except Exception as e:
            return Observation(self.name, f"Error executing grep: {str(e)}")
