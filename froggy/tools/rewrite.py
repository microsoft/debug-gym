from froggy.tools.tool import EnvironmentTool
from froggy.tools.toolbox import Toolbox
from froggy.utils import clean_code


@Toolbox.register()
class RewriteTool(EnvironmentTool):
    name = "rewrite"
    instructions = {
        "template": "rewrite(path: str, start: int, end: int, new_code: str)",
        "description": "Rewrite the code in the specified path, replace the content between lines [start, end] with the new code. Line numbers are 1-based. When file path is not provided, it's assumed to rewrite the current file. When start and end are not provided, it's assumed to rewrite the whole code. When only start is provided, it's assumed to rewrite that single line. The new code should be valid python code include proper indentation (can be determined from context), the special tokens <c> and </c> are used to wrap the new code. ",
        "examples": [
            """rewrite(new_code="print('hola')") will rewrite the current file (the entire code) to be print('hola'), because no line number is provided.""",
            """rewrite(start=10, new_code="    print('bonjour')") will rewite line number 10 of the current file to be print('bonjour'), with the indents ahead (in this case, 4 spaces).""",
            """rewrite(start=10, end=20, new_code="    print('hello')\\n    print('hi again')") will replace the chunk of code between line number 10 and 20 in the current file by the two lines provided, both with indents ahead (in this case, 4 spaces).""",
            """rewrite(path='code/utils.py', start=4, end=6, new_code="        print('buongiorno')") will replace the chunk of code between line number 4 and 6 in the file code/utils.py by the single line provided, with the indent ahead (in this case, 8 spaces).""",
            """rewrite('code/greetings.py', 2, 7, new_code="    print('nihao')") will replace the chunk of code between line number 2 and 7 in the file code/greetings.py by the single line provided, with the indent ahead (in this case, 4 spaces).""",
        ],
    }

    def __init__(self):
        super().__init__()
        self.rewrite_success = False

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

    def use(
        self, path: str = None, start: int = None, end: int = None, new_code: str = ""
    ):
        if path is None:
            # by default, rewrite the current file
            path = self.environment.current_file
        if start is not None:
            if end is None:
                # only start is provided (rewrite that line)
                end = start
            if start > end:
                return "Invalid line number range, start should be less than or equal to end.\nRewrite failed."
            if start <= 0 or end <= 0:
                return "Invalid line number, line numbers are 1-based.\nRewrite failed."
            start, end = start - 1, end - 1  # 1-based to 0-based

        try:
            new_code_length = self._rewrite_file(path, start, end, new_code)
        except Exception as e:
            self.rewrite_success = False
            return f"Error while rewriting the file: {str(e)}\nRewrite failed."

        if (
            hasattr(self.environment, "tools")
            and isinstance(self.environment.tools, dict)
            and "pdb" in self.environment.tools
        ):
            self.environment.tools["pdb"].breakpoint_modify(
                path,
                start + 1 if isinstance(start, int) else None,
                end + 1 if isinstance(end, int) else None,
                new_code_length,
            )  # converting head/tail back to 1-based index for breakpoint management
        self.rewrite_success = True
        return "Rewriting done."
