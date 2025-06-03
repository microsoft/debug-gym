import difflib

from debug_gym.gym.entities import Event, Observation
from debug_gym.gym.tools.tool import EnvironmentTool
from debug_gym.gym.tools.toolbox import Toolbox
from debug_gym.gym.utils import clean_code


@Toolbox.register()
class RewriteTool(EnvironmentTool):
    name = "rewrite"
    examples = [
        """rewrite(path="code/utils.py", start=None, end=None, new_code="print('hola')") will rewrite the specified file 'code/utils.py' (the entire code) to be print('hola'), because no line number is provided.""",
        """rewrite(path="code/utils.py", start=10, end=None, new_code="    print('bonjour')") will rewite line number 10 of the specified file 'code/utils.py' to be print('bonjour'), with the indents ahead (in this case, 4 spaces).""",
        """rewrite(path="code/utils.py", start=10, end=20, new_code="    print('hello')\\n    print('hi again')") will replace the chunk of code between line number 10 and 20 in the specified file 'code/utils.py' by the two lines provided, both with indents ahead (in this case, 4 spaces).""",
        """rewrite(path='code/utils.py', start=4, end=6, new_code="        print('buongiorno')") will replace the chunk of code between line number 4 and 6 in the specified file 'code/utils.py' by the single line provided, with the indent ahead (in this case, 8 spaces).""",
    ]
    description = (
        "Rewrite the content of the specified file path, between lines [start, end], with the new code. Line numbers are 1-based. When start and end are None, it's assumed to rewrite the whole file. When start is provided and end is None, it's assumed to rewrite a single line (start). The new code should be valid python code include proper indentation (can be determined from context)."
        + "\nExamples (for demonstration purposes only, you need to adjust the tool calling format according to your specific syntax):"
        + "\n".join(examples)
    )
    arguments = {
        "path": {
            "type": ["string"],
            "description": "A file path to be rewritten.",
        },
        "start": {
            "type": ["number", "null"],
            "description": "The starting line number to be rewritten. If None, the whole file will be rewritten.",
        },
        "end": {
            "type": ["number", "null"],
            "description": "The ending line number to be rewritten. If None, end is the same as start.",
        },
        "new_code": {
            "type": "string",
            "description": "The new code to be inserted. The new code should be valid python code include proper indentation (can be determined from context).",
        },
    }

    def _overwrite_file(self, environment, filepath: str, content: str):
        with open(environment.to_absolute(filepath), "w") as f:
            f.write(content)

    def _rewrite_file(self, environment, file_path, start, end, new_code):
        original_content = environment.read_file(file_path)
        new_code = clean_code(new_code)  # str
        new_code_lines = new_code.split("\n")
        new_code_length = len(new_code_lines)

        if start is None:
            # no line number is provided, rewrite the whole code
            self._overwrite_file(environment, filepath=file_path, content=new_code)
        else:
            # rewrite the code given the provided line numbers
            full_code_lines = environment.read_file(file_path).split("\n")
            if start >= len(full_code_lines):
                # if start exceeds the number of lines in the file, append the new code to the end of the file
                full_code_lines.extend(new_code_lines)
            else:
                # rewrite the code
                full_code_lines[start : end + 1] = new_code_lines  # list
            self._overwrite_file(
                environment, filepath=file_path, content="\n".join(full_code_lines)
            )

        # Calculate diff between original and new content
        new_content = environment.read_file(file_path)
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
        self.rewrite_success = False
        message = f"Rewrite failed. Error message:\n{message}\n"
        self.queue_event(
            environment=environment,
            event=Event.REWRITE_FAIL,
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
        self.rewrite_success = False
        if path is None:
            return self.fail(environment, "File path is None.")
        if not environment.is_editable(path):
            return self.fail(environment, f"File `{path}` is not editable.")
        if start is not None:
            end = end or start  # only start is provided (rewrite that line)
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
        try:
            diff, new_code_length = self._rewrite_file(
                environment, path, start, end, new_code
            )
        except Exception as e:
            return self.fail(environment, str(e))

        self.rewrite_success = True
        message = f"The file `{path}` has been updated successfully.\n\nDiff:\n\n{diff}"

        self.queue_event(
            environment=environment,
            event=Event.REWRITE_SUCCESS,
            message=message,
            file=path,
            # converting head/tail back to 1-based index for breakpoint management
            head=start + 1 if isinstance(start, int) else None,
            tail=end + 1 if isinstance(end, int) else None,
            length=new_code_length,
        )
        return Observation(self.name, message)
