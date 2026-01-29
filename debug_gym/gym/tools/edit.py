import difflib

from debug_gym.gym.entities import Event, Observation
from debug_gym.gym.terminals.terminal import UnrecoverableTerminalError
from debug_gym.gym.tools.tool import EnvironmentTool
from debug_gym.gym.tools.toolbox import Toolbox
from debug_gym.gym.workspace import WorkspaceReadError


@Toolbox.register()
class EditTool(EnvironmentTool):
    name = "edit"
    examples = [
        """edit(path=\"code/utils.py\", start=None, end=None, new_code=\"print('hola')\") will edit the specified file 'code/utils.py' (the entire code) to be print('hola'), because no line number is provided.""",
        """edit(path=\"code/utils.py\", start=10, end=None, new_code=\"    print('bonjour')\") will edit line number 10 of the specified file 'code/utils.py' to be print('bonjour'), with the indents ahead (in this case, 4 spaces).""",
        """edit(path=\"code/utils.py\", start=10, end=20, new_code=\"    print('hello')\\n    print('hi again')\") will replace the chunk of code between line number 10 and 20 in the specified file 'code/utils.py' by the two lines provided, both with indents ahead (in this case, 4 spaces).""",
        """edit(path='code/utils.py', start=4, end=6, new_code=\"        print('buongiorno')\") will replace the chunk of code between line number 4 and 6 in the specified file 'code/utils.py' by the single line provided, with the indent ahead (in this case, 8 spaces).""",
        """edit(path='code/new_utils.py', new_code=\"print('hello new file')\") will create 'code/new_utils.py' with the provided content when the file does not already exist.""",
    ]
    description = (
        "Edit the content of the specified file path, between lines [start, end], with the new code. Line numbers are 1-based. When start is provided and end is None, it's assumed to edit a single line (start). When both start and end are None, it's assumed to edit the whole file, this is not recommended because most of the time the expected change is local. If the file does not exist yet, it will be created with the provided content. The new code should be valid python code include proper indentation (can be determined from context)."
        + "\nExamples (for demonstration purposes only, you need to adjust the tool calling format according to your specific syntax):"
        + "\n".join(examples)
    )
    arguments = {
        "path": {
            "type": ["string"],
            "description": "A file path to be edited.",
        },
        "start": {
            "type": ["number", "null"],
            "description": "The starting line number to be edited. If None, the whole file will be edited.",
        },
        "end": {
            "type": ["number", "null"],
            "description": "The ending line number to be edited. If None, end is the same as start.",
        },
        "new_code": {
            "type": ["string"],
            "description": "The new code to be inserted. The new code should be valid python code include proper indentation (can be determined from context).",
        },
    }

    def _overwrite_file(self, environment, filepath: str, content: str):
        """Persist the new content to disk using the workspace abstraction."""
        environment.workspace.write_file(filepath, content)

    def _edit_file(
        self, environment, file_path, start, end, new_code, file_exists: bool
    ):
        """Apply the requested edit and return the diff plus line count of the new code."""
        try:
            original_content = (
                environment.workspace.read_file(file_path) if file_exists else ""
            )
        except WorkspaceReadError as exc:
            # Surface unexpected read errors back to the caller.
            raise exc
        new_code_lines = new_code.split("\n")
        new_code_length = len(new_code_lines)

        if start is None or not file_exists:
            # No line number is provided or we are creating a new file.
            self._overwrite_file(environment, filepath=file_path, content=new_code)
        else:
            # edit the code given the provided line numbers
            full_code_lines = original_content.split("\n")
            if start >= len(full_code_lines):
                # if start exceeds the number of lines in the file, append the new code to the end of the file
                full_code_lines.extend(new_code_lines)
            else:
                # edit the code
                full_code_lines[start : end + 1] = new_code_lines  # list
            self._overwrite_file(
                environment, filepath=file_path, content="\n".join(full_code_lines)
            )

        # Calculate diff between original and new content
        new_content = environment.workspace.read_file(file_path)
        diff = "".join(
            difflib.unified_diff(
                original_content.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile="original",
                tofile="current",
            )
        )

        return diff, new_code_length

    def fail(self, environment, message: str) -> Observation:
        """Report the failed edit and emit the corresponding event."""
        self.edit_success = False
        message = f"Edit failed. Error message:\n{message}\n"
        self.queue_event(
            environment=environment,
            event=Event.EDIT_FAIL,
            message=message,
        )
        return Observation(self.name, message)

    def use(
        self,
        environment,
        path: str = None,
        start: int = None,
        end: int = None,
        new_code: str = "",
    ) -> Observation:
        """Main entrypoint used by LLM tool calls to edit or create files."""
        self.edit_success = False
        if path is None:
            return self.fail(environment, "File path is None.")
        # Resolve the target path to get absolute path (without checking existence).
        try:
            resolved_path = environment.workspace.resolve_path(path, raises=False)
        except Exception as exc:
            return self.fail(environment, f"Invalid path `{path}`: {exc}")

        workspace_root = environment.workspace.working_dir
        try:
            resolved_path.resolve(strict=False).relative_to(
                workspace_root.resolve(strict=False)
            )
        except ValueError:
            return self.fail(
                environment, f"`{path}` is not within the workspace directory."
            )

        file_exists = environment.workspace.has_file(path)

        # When creating a new file, ignore start/end positions and treat as full file write.
        if file_exists and start is not None:
            end = end or start  # only start is provided (edit that line)
            if start > end:
                return self.fail(
                    environment,
                    "Invalid line number range, start should be less than or equal to end.",
                )
            if start <= 0 or end <= 0:
                return self.fail(
                    environment, "Invalid line number, line numbers are 1-based."
                )
            start, end = start - 1, end - 1  # 1-based to 0-based
        else:
            start = None
            end = None
        try:
            diff, new_code_length = self._edit_file(
                environment, path, start, end, new_code, file_exists=file_exists
            )
        except UnrecoverableTerminalError:
            raise
        except Exception as e:
            return self.fail(environment, str(e))

        self.edit_success = True
        message = f"The file `{path}` has been updated successfully.\n\nDiff:\n\n{diff}"

        self.queue_event(
            environment=environment,
            event=Event.EDIT_SUCCESS,
            message=message,
            file=path,
            # converting head/tail back to 1-based index for breakpoint management
            head=start + 1 if isinstance(start, int) else None,
            tail=end + 1 if isinstance(end, int) else None,
            length=new_code_length,
        )
        return Observation(self.name, message)
