from debug_gym.gym.entities import Event, Observation
from debug_gym.gym.tools.tool import EnvironmentTool
from debug_gym.gym.tools.toolbox import Toolbox
from debug_gym.gym.utils import clean_code


@Toolbox.register()
class RewriteTool(EnvironmentTool):
    name = "rewrite"
    instructions = {
        "template": "```rewrite file/path.py start:end <c>new_code</c>```",
        "description": "Rewrite the code in file/path.py between lines [start, end] with the new code. Line numbers are 1-based. When file path is not provided, it's assumed to rewrite the current file. When start and end are not provided, it's assumed to rewrite the whole code. When only start is provided, it's assumed to rewrite that single line. The new code should be valid python code include proper indentation (can be determined from context), the special tokens <c> and </c> are used to wrap the new code. ",
        "examples": [
            "```rewrite <c>print('hola')</c>``` will rewrite the current file (the entire code) to be print('hola'), because no line number is provided.",
            "```rewrite 10 <c>    print('bonjour')</c>``` will rewite line number 10 of the current file to be print('bonjour'), with the indents ahead (in this case, 4 spaces).",
            "```rewrite 10:20 <c>    print('hello')\\n    print('hi again')</c>``` will replace the chunk of code between line number 10 and 20 in the current file by the two lines provided, both with indents ahead (in this case, 4 spaces).",
            "```rewrite code/utils.py 4:6 <c>        print('buongiorno')</c>``` will replace the chunk of code between line number 4 and 6 in the file code/utils.py by the single line provided, with the indent ahead (in this case, 8 spaces).",
        ],
    }

    def _rewrite_file(self, file_path, head, tail, new_code):
        if file_path is None:
            # by default, rewrite the current file
            file_path = self.environment.current_file

        if file_path is None:
            return "No file is currently open.", False, None
        if file_path.startswith(str(self.environment.working_dir)):
            file_path = file_path[len(str(self.environment.working_dir)) + 1 :]
        if file_path not in self.environment.all_files:
            return (
                f"File {file_path} does not exist or is not in the current repository.",
                False,
                None,
            )
        if file_path not in self.environment.editable_files:
            return f"File {file_path} is not editable.", False, None

        success = True
        new_code = clean_code(new_code)  # str
        new_code_lines = new_code.split("\n")
        new_code_length = len(
            new_code_lines
        )  # number of lines in the newly generated code
        if head is None and tail is None:
            # no line number is provided, rewrite the whole code
            try:
                self.environment.overwrite_file(filepath=file_path, content=new_code)
                if file_path == self.environment.current_file:
                    self.environment.load_current_file(file_path)
            except:
                success = False
        else:
            # rewrite the code given the provided line numbers
            if tail is None:
                # only head is provided (rewrite that line)
                tail = head
            try:
                full_code_lines = self.environment.load_file(file_path).split("\n")
                if head >= len(full_code_lines):
                    # if head exceeds the number of lines in the file, append the new code to the end of the file
                    full_code_lines.extend(new_code_lines)
                else:
                    # rewrite the code
                    full_code_lines[head : tail + 1] = new_code_lines  # list
                self.environment.overwrite_file(
                    filepath=file_path, content="\n".join(full_code_lines)
                )
                if file_path == self.environment.current_file:
                    self.environment.load_current_file(file_path)
            except:
                success = False
        return "", success, new_code_length

    def parse_line_numbers(self, line_number_string):

        # only line number is provided
        line_numbers = line_number_string.split(":")
        line_numbers = [item.strip() for item in line_numbers]
        if len(line_numbers) not in [1, 2]:
            return "Invalid line number format.", None, None
        if len(line_numbers) == 1:
            if int(line_numbers[0]) <= 0:
                return "Invalid line number, line numbers are 1-based.", None, None
            # only head is provided (rewrite that line)
            head = int(line_numbers[0]) - 1  # 1-based to 0-based
            tail = head
        else:
            # len(line_numbers) == 2:
            # both head and tail are provided
            if int(line_numbers[0]) <= 0 or int(line_numbers[1]) <= 0:
                return "Invalid line number, line numbers are 1-based.", None, None
            if int(line_numbers[0]) > int(line_numbers[1]):
                return (
                    "Invalid line number range, head should be less than or equal to tail.",
                    None,
                    None,
                )
            head = int(line_numbers[0]) - 1  # 1-based to 0-based
            tail = int(line_numbers[1]) - 1  # 1-based to 0-based
        return "", head, tail

    def use(self, tool_args) -> Observation:
        content = tool_args
        # parse content to get file_path, head, tail, and new_code
        # code/utils.py 4:6 <c>        print('buongiorno')</c>
        self.environment.rewrite_counter += 1
        file_path, head, tail = None, None, None
        message = ""
        try:
            new_code = content.split("<c>", 1)[1].split("</c>", 1)[0]
            content = content.split("<c>", 1)[0].strip()
            # code/utils.py 4:6
            content_list = content.split()
            if len(content_list) == 0:
                # no file path and line number is provided
                pass
            elif len(content_list) == 1:
                # either file path or line number is provided
                if content_list[0][0].isnumeric():
                    # only line number is provided
                    message, head, tail = self.parse_line_numbers(content_list[0])
                else:
                    # only file path is provided
                    file_path = content_list[0]
            elif len(content_list) == 2:
                # both file path and line number are provided
                file_path = content_list[0]
                message, head, tail = self.parse_line_numbers(content_list[1])
            else:
                message = "SyntaxError: invalid syntax."
        except:
            message = "SyntaxError: invalid syntax."
        if "" != message:
            self.rewrite_success = False
            message = "\n".join([message, "Rewrite failed."])
            self.queue_event(Event.REWRITE_FAIL, message=message)
            return Observation(self.name, message)

        message, success, new_code_length = self._rewrite_file(
            file_path, head, tail, new_code
        )
        if success is True:
            self.rewrite_success = True
            message = "Rewriting done."
            self.queue_event(
                Event.REWRITE_SUCCESS,
                message=message,
                file=file_path,
                # converting head/tail back to 1-based index for breakpoint management
                head=head + 1 if isinstance(head, int) else None,
                tail=tail + 1 if isinstance(tail, int) else None,
                length=new_code_length,
            )
            return Observation(self.name, message)

        self.rewrite_success = False
        message = "\n".join([message, "Rewrite failed."])
        self.queue_event(Event.REWRITE_FAIL, message=message)
        return Observation(self.name, message)
