

class Patcher():
    """
    Rewrite the code between lines [start, end] with the new code. Line numbers are 1-based. When start and end are not provided, it's assumed to rewrite the whole code. When only start is provided, it's assumed to rewrite that single line. The new code should be valid python code include proper indentation (can be determined from context), the special tokens <c> and </c> are used to wrap the new code.
    A few examples:
        <c>print('hola')</c> will rewrite the file (the entire code) to be print('hola'), because no line number is provided."
        10 <c>    print('bonjour')</c> will rewite line number 10 of the file to be print('bonjour'), with the indents ahead (in this case, 4 spaces)."
        10:20 <c>    print('hello')\\n    print('hi again')</c> will replace the chunk of code between line number 10 and 20 in the current file by the two lines provided, both with indents ahead (in this case, 4 spaces)."
    """

    def __init__(self):
        pass

    def parse_line_numbers(self, line_number_string):

        # e.g., 4:6 or 4
        line_numbers = line_number_string.split(":")
        line_numbers = [item.strip() for item in line_numbers]
        if len(line_numbers) not in [1, 2]:
            return "Invalid line number format.", None, None
        if len(line_numbers) == 1:
            # only head is provided (rewrite that line)
            head = int(line_numbers[0]) - 1  # 1-based to 0-based
            tail = head
        else:
            # len(line_numbers) == 2:
            # both head and tail are provided
            head = int(line_numbers[0]) - 1  # 1-based to 0-based
            tail = int(line_numbers[1]) - 1  # 1-based to 0-based
        return "", head, tail
    
    def _rewrite_file(self, code, head, tail, new_code):

        new_code_lines = new_code.split("\n")
        if head is None and tail is None:
            # no line number is provided, rewrite the whole code
            output = "\n".join(new_code_lines)
        else:
            # rewrite the code given the provided line numbers
            if tail is None:
                # only head is provided (rewrite that line)
                tail = head
            try:
                full_code_lines = code.split("\n")
                if head >= len(full_code_lines):
                    # if head exceeds the number of lines in the file, append the new code to the end of the file
                    full_code_lines.extend(new_code_lines)
                else:
                    # rewrite the code
                    full_code_lines[head : tail + 1] = new_code_lines  # list
                output = "\n".join(full_code_lines)
            except:
                output = "Rewrite failed."
        return output

    def use(self, code, patch):
        content = patch
        # parse content to get head, tail, and new_code
        # 4:6 <c>        print('buongiorno')</c>
        head, tail = None, None
        message = ""
        try:
            new_code = content.split("<c>", 1)[1].split("</c>", 1)[0]
            content = content.split("<c>", 1)[0].strip()
            # 4:6
            if content == "":
                # no line number is provided
                pass
            elif content[0].isnumeric():
                # line number is provided
                message, head, tail = self.parse_line_numbers(content)
            else:
                message = "SyntaxError: invalid syntax."
        except:
            message = "SyntaxError: invalid syntax."
        if "" != message:
            return "\n".join([message, "Rewrite failed."])

        output = self._rewrite_file(code, head, tail, new_code)
        return output
