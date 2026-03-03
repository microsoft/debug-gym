from debug_gym.gym.entities import Observation
from debug_gym.gym.tools.tool import EnvironmentTool
from debug_gym.gym.tools.toolbox import Toolbox


@Toolbox.register()
class SubmitTool(EnvironmentTool):
    name = "submit"
    examples = [
        """Use submit with `message`: "Fixed the off-by-one error in the loop condition" to submit the solution with a summary of the changes made.""",
        """Use submit with `message`: null to submit without a message.""",
    ]
    description = (
        "Should be called when the task is complete."
        + "\nExamples (for demonstration purposes only, you need to adjust the tool calling format according to your specific syntax):\n"
        + "\n".join(examples)
    )
    arguments = {
        "message": {
            "type": ["string", "null"],
            "description": "An optional message to conclude the task, summarizing what was done.",
        }
    }

    def __init__(self, eval_on_submit=True):
        super().__init__()
        self.eval_on_submit = eval_on_submit

    def use(self, environment, message: str = None, **kwargs) -> Observation:
        output = ""
        if message:
            output = f"Agent message: {message}\n\n"

        if self.eval_on_submit:
            output += environment.eval().output
        else:
            output += "The agent terminated the session."

        environment.terminated = True
        return Observation(self.name, output)
