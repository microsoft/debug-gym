from debug_gym.gym.entities import Observation
from debug_gym.gym.tools.tool import EnvironmentTool
from debug_gym.gym.tools.toolbox import Toolbox


@Toolbox.register()
class BashTool(EnvironmentTool):
    name: str = "bash"
    examples = [
        """bash(command="ls -la") to list all files and directories with detailed information.""",
        """bash(command="grep -r 'function_name' .") to search for 'function_name' in all files recursively.""",
        """bash(command="find . -name '*.py' | head -10") to find Python files in the current directory.""",
        """bash(command="cat file.txt | head -20") to show the first 20 lines of a file.""",
        """bash(command="sed -n 10,25p path/to/file") to show lines 10 to 25 of a file at relative path.""",
        """bash(command="pip list") to show installed Python packages.""",
    ]
    description = (
        "Run commands in a bash shell. "
        "You have access to common linux and python packages via pip. "
        "State is persistent across command calls within the same session. "
        "\nExamples (for demonstration purposes only, you need to adjust the tool calling format according to your specific syntax):\n"
        + "\n".join(examples)
    )
    arguments = {
        "command": {
            "type": ["string"],
            "description": "The bash command to execute. The command will be run in the current working directory of the environment.",
        },
    }

    def use(self, environment, command: str) -> Observation:
        """Execute a bash command in the environment's terminal and return the result."""
        try:
            # Assert that the terminal is not a local terminal (only in production)
            import os

            from debug_gym.gym.terminals.local import LocalTerminal

            # Require remote terminal unless local is explicitly allowed
            require_remote = (
                os.environ.get("ALLOW_LOCAL_TERMINAL", "false").lower() == "false"
            )
            if require_remote and type(environment.terminal) is LocalTerminal:
                return Observation(
                    self.name,
                    "Error: bash tool requires a non-local terminal. Current terminal type is not supported.",
                )

            # Use the environment's terminal to run the command
            # Set a reasonable timeout (30 seconds) to prevent hanging
            success, output = environment.terminal.run(command, timeout=30)

            if success:
                result = (
                    output
                    if output.strip()
                    else "Command executed successfully (no output)"
                )
            else:
                result = f"Command failed with output:\n{output}"

        except Exception as e:
            result = f"Error executing command: {str(e)}"

        return Observation(self.name, result)
