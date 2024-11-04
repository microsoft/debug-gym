from froggy.tools import EnvironmentTool
from froggy.utils import clean_code


class CodePatcher(EnvironmentTool):
    action = "```rewrite"

    def __init__(self):
        super().__init__()
        self.rewrite_success = False

    @staticmethod
    def get(patch_type):
        if patch_type == "udiff":
            return UDiffPatcher()
        elif patch_type == "whole":
            return WholePatcher()
        elif patch_type == "substitution":
            return SubstitutionPatcher()
        else:
            raise ValueError("Invalid patch type!")


class UDiffPatcher(CodePatcher):
    name = "udiff_patcher"
    description = "Creates patches of code given UDiff format."
    instructions = {
        "template": "```rewrite <unified_diff>```",
        "description": "After debugging, or whenever a sufficient amount of information has been gathered, use the following command to rewrite the code.\n For example if the code is:\n def greet():\n    print('Hello, world!')\n and the user wants to rewrite the full code, the command would be:\n```rewrite\n@@ -1,2 +1,2 @@\n-def greet():\n-    print('Hello, world!')\n+def greet(name):\n+    print(f'Hello, {name}!')\n```\n this will result in the following code:\ndef greet(name):\n    print(f'Hello, {name}!')\n alternatively, you can rewrite only a line by providing line numbers, e.g.:\n```rewrite\n@@ -2 +2 @@\n-    print('Hello, world!')\n+    print(f'Hello everyone!')\n```\n this will result in the following code:\ndef greet(name):\n    print(f'Hello everyone!')\n",
    }

    def is_triggered(self, action):
        return action.startswith(self.action)

    def use(self, patch):
        code = self.environment.current_file_content
        patch_lines = patch.split("```diff")[1].split("```")[0]
        patched_code = self._apply_unified_diff(
            code, patch_lines.splitlines(keepends=True)
        )

        if patched_code:
            new_code = clean_code("".join(patched_code))
            self.environment.overwrite_file(
                filepath=self.environment.current_file, content=new_code
            )
            self.environment.load_current_file(self.environment.current_file)
            # TODO: to support breakpoint management
            self.environment.current_breakpoints_state = {}
            self.rewrite_success = True
            return "Rewrite successful."

        self.rewrite_success = False
        return "Rewrite failed."

    def _apply_unified_diff(self, code_lines, patch_lines):
        import re

        patched_lines = code_lines.copy()
        line_offset = 0

        hunk_header_pattern = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
        idx = 0

        while idx < len(patch_lines):
            line = patch_lines[idx]
            if line.startswith("@@"):
                match = hunk_header_pattern.match(line)
                if not match:
                    return None  # Invalid hunk header

                # Extract hunk header information
                src_start = int(match.group(1)) - 1  # Convert to 0-based index
                src_len = int(match.group(2) or "1")
                tgt_start = int(match.group(3)) - 1
                tgt_len = int(match.group(4) or "1")

                idx += 1
                hunk_lines = []
                while idx < len(patch_lines) and not patch_lines[idx].startswith("@@"):
                    hunk_lines.append(patch_lines[idx])
                    idx += 1

                # Apply the hunk
                removed = []
                added = []
                for hunk_line in hunk_lines:
                    if hunk_line.startswith("-"):
                        removed.append(hunk_line[1:])
                    elif hunk_line.startswith("+"):
                        added.append(hunk_line[1:])
                    elif hunk_line.startswith(" "):
                        continue
                    elif hunk_line.startswith("\\"):
                        continue  # Handle \ No newline at end of file
                    else:
                        return None  # Invalid hunk line

                # Replace the lines in the original code
                patched_lines[
                    src_start + line_offset : src_start + line_offset + len(removed)
                ] = added
                line_offset += len(added) - len(removed)
            else:
                idx += 1

        return patched_lines


class WholePatcher(CodePatcher):
    name = "whole_patcher"
    description = "Rewrites the full code."
    instructions = {
        "template": "```rewrite <codef>```",
        "description": "After debugging, or whenever a sufficient amount of information has been gathered, use the following command to rewrite the code.\n For example if the code is:\n def greet():\n    print('Hello, world!')\n and the user wants to rewrite it, the command would be:\n```rewrite\ndef greet(name):\n    print(f'Hello, {name}!')\n```\n this will result in the following code:\ndef greet(name):\n    print(f'Hello, {name}!')\n",
    }

    def is_triggered(self, action):
        return action.startswith(self.action)

    def use(self, patch):
        content = patch.split(self.action)[1].split("```")[0]
        if content:
            new_code = clean_code(content)
            self.environment.overwrite_file(
                filepath=self.environment.current_file, content=new_code
            )
            self.environment.load_current_file(self.environment.current_file)
            # TODO: to support breakpoint management
            self.environment.current_breakpoints_state = {}
            self.rewrite_success = True
            return "Rewrite successful."

        self.rewrite_success = False
        return "Rewrite failed."


class SubstitutionPatcher(CodePatcher):
    name = "substitution_patcher"
    description = "Creates patches of code given start and end lines."
    instructions = {
        "template": "```rewrite file/path.py head:tail <c>new_code</c>```",
        "description": "Rewrite the code in file/path.py between lines [head, tail] with the new code. Line numbers are 1-based. When file path is not provided, it's assumed to rewrite the current file. When head and tail are not provided, it's assumed to rewrite the whole code. When only head is provided, it's assumed to rewrite that single line. The new code should be valid python code include proper indentation (can be determined from context), the special tokens <c> and </c> are used to wrap the new code. ",
        "examples": [
            "```rewrite <c>print('hola')</c>``` will rewrite the current file (the entire code) to be print('hola'), because no line number is provided.",
            "```rewrite 10 <c>    print('bonjour')</c>``` will rewite line number 10 of the current file to be print('bonjour'), with the indents ahead (in this case, 4 spaces).",
            "```rewrite 10:20 <c>    print('hello')\\n    print('hi again')</c>``` will replace the chunk of code between line number 10 and 20 in the current file by the two lines provided, both with indents ahead (in this case, 4 spaces).",
            "```rewrite code/utils.py 4:6 <c>        print('buongiorno')</c>``` will replace the chunk of code between line number 4 and 6 in the file code/utils.py by the single line provided, with the indent ahead (in this case, 8 spaces).",
        ],
    }

    def is_triggered(self, action):
        return action.startswith(self.action)

    def _rewrite_file(self, file_path, head, tail, new_code):
        if file_path is None:
            # by default, rewrite the current file
            file_path = self.environment.current_file

        if file_path is None:
            return "No file is currently open.", False, None
        if file_path.startswith(str(self.environment.working_dir)):
            file_path = file_path[len(str(self.environment.working_dir)) + 1 :]
        if file_path not in self.environment.all_files:
            return f"File {file_path} does not exist or is not in the current repository.", False, None
        if file_path not in self.environment.editable_files:
            return f"File {file_path} is not editable.", False, None

        success = True
        new_code = clean_code(new_code)  # str
        new_code_lines = new_code.split("\n")
        new_code_length = len(new_code_lines)  # number of lines in the newly generated code
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
                # rewrite the code
                full_code_lines = self.environment.load_file(file_path).split("\n")
                full_code_lines[head : tail + 1] = new_code_lines  # list
                self.environment.overwrite_file(
                    filepath=file_path, 
                    content="\n".join(full_code_lines)
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
        assert len(line_numbers) in [1, 2], "Invalid line number format."
        if len(line_numbers) == 1:
            # only head is provided (rewrite that line)
            head = int(line_numbers[0]) - 1  # 1-based to 0-based
            tail = head
        else:
            # len(line_numbers) == 2:
            # both head and tail are provided
            head = int(line_numbers[0]) - 1  # 1-based to 0-based
            tail = int(line_numbers[1]) - 1  # 1-based to 0-based
        return head, tail

    def use(self, patch):
        content = patch.split(self.action)[1].split("```")[0].strip()
        # parse content to get file_path, head, tail, and new_code
        # code/utils.py 4:6 <c>        print('buongiorno')</c>
        file_path, head, tail = None, None, None
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
                    head, tail = self.parse_line_numbers(content_list[0])
                else:
                    # only file path is provided
                    file_path = content_list[0]
            elif len(content_list) == 2:
                # both file path and line number are provided
                file_path = content_list[0]
                head, tail = self.parse_line_numbers(content_list[1])
            else:
                raise ValueError("Invalid content format.")
        except:
            return "Rewrite failed."

        message, success, new_code_length = self._rewrite_file(file_path, head, tail, new_code)
        if success is True:
            if (
                hasattr(self.environment, "tools")
                and isinstance(self.environment.tools, dict)
                and "pdb" in self.environment.tools
            ):
                self.environment.tools["pdb"].breakpoint_modify(
                    file_path,
                    head + 1 if isinstance(head, int) else None,
                    tail + 1 if isinstance(tail, int) else None,
                    new_code_length,
                )  # converting head/tail back to 1-based index for breakpoint management
            self.rewrite_success = True
            return "Rewriting done."
        return "\n".join([message, "Rewrite failed."])
