from debug_gym.gym.entities import Event, Observation
from debug_gym.gym.tools.tool import EnvironmentTool
from debug_gym.gym.tools.toolbox import Toolbox
from debug_gym.gym.utils import clean_code


@Toolbox.register()
class RewriteTool(EnvironmentTool):
    name = "rewrite"
    examples = [
        """rewrite(path=None, start=None, end=None, new_code="print('hola')") will rewrite the current file (the entire code) to be print('hola'), because no line number is provided.""",
        """rewrite(path=None, start=10, end=None, new_code="    print('bonjour')") will rewite line number 10 of the current file to be print('bonjour'), with the indents ahead (in this case, 4 spaces).""",
        """rewrite(path=None, start=10, end=20, new_code="    print('hello')\\n    print('hi again')") will replace the chunk of code between line number 10 and 20 in the current file by the two lines provided, both with indents ahead (in this case, 4 spaces).""",
        """rewrite(path='code/utils.py', start=4, end=6, new_code="        print('buongiorno')") will replace the chunk of code between line number 4 and 6 in the file code/utils.py by the single line provided, with the indent ahead (in this case, 8 spaces).""",
    ]
    tool_description = {
        "type": "function",
        "function": {
            "name": "rewrite",
            "description": "Rewrite the content of the specified file path, between lines [start, end], with the new code. Line numbers are 1-based. When file path is None, it's assumed to rewrite the current file. When start and end are None, it's assumed to rewrite the whole file. When start is provided and end is None, it's assumed to rewrite a single line (start). The new code should be valid python code include proper indentation (can be determined from context)."
            + "\nExamples (for demonstration purposes only, you need to adjust the tool calling format according to your specific syntax):\n"
            + "\n".join(examples),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": ["string", "null"],
                        "description": "A file path to be rewritten. If None, the current file will be used.",
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
                },
                "required": ["path", "start", "end", "new_code"],
                "additionalProperties": False,
            },
        },
    }

    def _rewrite_file(self, file_path, start, end, new_code):
        assert file_path is not None, "No file is currently open."
        if file_path.startswith(str(self.environment.working_dir)):
            file_path = file_path[len(str(self.environment.working_dir)) + 1 :]
        assert (
            file_path in self.environment.all_files
        ), f"File {file_path} does not exist or is not in the current repository."
        assert (
            file_path in self.environment.editable_files
        ), f"File {file_path} is not editable."

        new_code = clean_code(new_code)  # str
        new_code_lines = new_code.split("\n")
        new_code_length = len(
            new_code_lines
        )  # number of lines in the newly generated code
        if start is None:
            # no line number is provided, rewrite the whole code
            self.environment.overwrite_file(filepath=file_path, content=new_code)
            if file_path == self.environment.current_file:
                self.environment.load_current_file(file_path)
        else:
            # rewrite the code given the provided line numbers
            full_code_lines = self.environment.load_file(file_path).split("\n")
            if start >= len(full_code_lines):
                # if start exceeds the number of lines in the file, append the new code to the end of the file
                full_code_lines.extend(new_code_lines)
            else:
                # rewrite the code
                full_code_lines[start : end + 1] = new_code_lines  # list
            self.environment.overwrite_file(
                filepath=file_path, content="\n".join(full_code_lines)
            )
            if file_path == self.environment.current_file:
                self.environment.load_current_file(file_path)
        return new_code_length

    def fail(self, message: str) -> Observation:
        self.rewrite_success = False
        message = "\n".join([message, "Rewrite failed."])
        self.queue_event(Event.REWRITE_FAIL, message=message)
        return Observation(self.name, message)

    def use(
        self, path: str = None, start: int = None, end: int = None, new_code: str = ""
    ) -> Observation:
        self.rewrite_success = False
        if path is None:
            # by default, rewrite the current file
            path = self.environment.current_file
        if start is not None:
            if end is None:
                # only start is provided (rewrite that line)
                end = start
            if start > end:
                return self.fail(
                    "Invalid line number range, start should be less than or equal to end."
                )
            if start <= 0 or end <= 0:
                return self.fail("Invalid line number, line numbers are 1-based.")
            start, end = start - 1, end - 1  # 1-based to 0-based
        try:
            new_code_length = self._rewrite_file(path, start, end, new_code)
        except Exception as e:
            return self.fail(f"Error while rewriting the file: {str(e)}")

        self.rewrite_success = True
        message = "Rewriting done."
        self.queue_event(
            Event.REWRITE_SUCCESS,
            message=message,
            file=path,
            # converting head/tail back to 1-based index for breakpoint management
            head=start + 1 if isinstance(start, int) else None,
            tail=end + 1 if isinstance(end, int) else None,
            length=new_code_length,
        )
        return Observation(self.name, message)
