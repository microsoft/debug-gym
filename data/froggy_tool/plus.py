from tool import EnvironmentTool


class PlusTool(EnvironmentTool):
    name: str = "plus"
    action_str = "```plus"
    instructions = {
        "template": "```plus <number1> <number2>```",
        "description": "Adds two numbers together.",
        "examples": [
            "```plus 1 2``` will return 3.",
            "```plus 3 4``` will return 7.",
            "```plus -3 3``` will return 0.",
        ],
    }

    def use(self, input_text):
        # parse the input text
        try:
            numbers = input_text.split(self.action)[1].split("```")[0].strip()
        except:
            return "SyntaxError: invalid syntax."
        number_1, number_2 = numbers.split()
        # calculate the result
        assert number_1.isdigit() and number_2.isdigit(), "ValueError: invalid value."
        result = int(number_1) + int(number_2)
        return result
        