from debug_gym.gym.entities import Observation
from debug_gym.gym.tools.tool import EnvironmentTool
from debug_gym.gym.tools.toolbox import Toolbox
from debug_gym.gym.utils import show_line_number
from debug_gym.gym.workspace import WorkspaceReadError


@Toolbox.register()
class ViewTool(EnvironmentTool):
    name: str = "view"
    examples = [
        """view(path="main.py") to show the content of a file called 'main.py' in the root. The content will be annotated with line numbers and current breakpoints because include_line_numbers_and_breakpoints is True by default.""",
        """view(path="utils/vector.py", include_line_numbers_and_breakpoints=True) to show the content of a file called 'vector.py' in a subdirectory called 'utils'. The content will be annotated with line numbers and current breakpoints.""",
        """view(path="src/util.py", include_line_numbers_and_breakpoints=False) to show the content of a file called 'util.py' in a subdirectory called 'src'. The line numbers and breakpoints will not be included in the output.""",
        """view(path="funcs/helper.py", start=6, end=24) to show the content of a file called 'helper.py' in a subdirectory called 'funcs', starting from line 6 to line 24. The content will be annotated with line numbers and current breakpoints.""",
        """view(path="src/main.py", start=514) to show the content of a file called 'main.py' in a subdirectory called 'src', starting from line 514 to the end of the file. The content will be annotated with line numbers and current breakpoints.""",
    ]
    description = (
        "Specify a file path to view its content. The file path should be relative to the root directory of the repository."
        + "\nExamples (for demonstration purposes only, you need to adjust the tool calling format according to your specific syntax):\n"
        + "\n".join(examples)
    )
    arguments = {
        "path": {
            "type": ["string"],
            "description": "The path to the file to be viewed. The path should be relative to the root directory of the repository.",
        },
        "start": {
            "type": ["number", "null"],
            "description": "The starting line number (1-based, inclusive) to view. If not provided, starts from the beginning.",
        },
        "end": {
            "type": ["number", "null"],
            "description": "The ending line number (1-based, inclusive) to view. If not provided, shows until the end of the file.",
        },
        "include_line_numbers_and_breakpoints": {
            "type": ["boolean", "null"],
            "description": "Whether to annotate the file content with line numbers and current breakpoints before each line of code. For example, a line can be shown as 'B  426         self.assertEqual(CustomUser._default_manager.count(), 0)', where 'B' indicates a breakpoint before this line of code. '426' is the line number. This argument is optional and defaults to True. If set to False, the file content will be shown without line numbers and breakpoints.",
        },
    }

    def use(
        self,
        environment,
        path: str,
        start: int = None,
        end: int = None,
        include_line_numbers_and_breakpoints: bool = True,
    ) -> Observation:
        new_file = path.strip()
        if not new_file:
            return Observation(
                self.name, "Invalid file path. Please specify a valid file path."
            )

        # RepoEnv.read_file raises WorkspaceReadError if the file
        # does not exist, is not accessible, or is outside the working directory
        try:
            file_content = environment.workspace.read_file(new_file)
        except WorkspaceReadError as e:
            return Observation(self.name, f"View failed. Error message:\n{str(e)}")
        except Exception as e:
            # Catch other exceptions (e.g., terminal errors) and return a user-friendly message
            return Observation(
                self.name,
                f"View failed due to an unexpected error: {type(e).__name__}: {str(e)}",
            )
        file_lines = file_content.splitlines()

        if not file_lines:
            return Observation(self.name, f"The file `{new_file}` is empty.")

        # Convert 1-based line numbers to 0-based indices
        s = (start - 1) if start is not None else 0
        e = min(end, len(file_lines)) if end is not None else len(file_lines)

        # Validate indices
        if s < 0 or s >= len(file_lines):
            return Observation(
                self.name,
                f"Invalid start index: `{start}`. It should be between 1 and {len(file_lines)}.",
            )
        if e < 0 or e > len(file_lines):
            return Observation(
                self.name,
                f"Invalid end index: `{end}`. It should be between 1 and {len(file_lines)}.",
            )
        if s + 1 > e:  # end is inclusive, so we check s + 1
            return Observation(
                self.name,
                f"Invalid range: start index `{start}` is greater than end index `{end}`.",
            )

        selected_lines = file_lines[s:e]
        display_content = "\n".join(selected_lines)

        breakpoints_message = ""
        if include_line_numbers_and_breakpoints:
            display_content = show_line_number(
                display_content,
                code_path=environment.workspace.resolve_path(new_file),
                environment=environment,
                start_index=s + 1,
            )
            if environment.current_breakpoints_state:
                breakpoints_message = (
                    " B indicates breakpoint before a certain line of code."
                )

        line_numbers = (
            f" lines {s + 1}-{e} of {len(file_lines)} total lines."
            if len(file_lines) > 0
            else ""
        )
        read_only = (
            " The file is read-only."
            if not environment.workspace.is_editable(new_file)
            else ""
        )
        obs = (
            f"Viewing `{new_file}`,{line_numbers}{read_only}{breakpoints_message}"
            f"\n\n```\n{display_content}\n```\n\n"
        )
        return Observation(self.name, obs)
