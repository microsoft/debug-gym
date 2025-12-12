from debug_gym.gym.entities import Observation
from debug_gym.gym.tools.tool import EnvironmentTool
from debug_gym.gym.tools.toolbox import Toolbox


@Toolbox.register()
class SubmitTool(EnvironmentTool):
    name = "submit"
    description = "Submit your changes once the task is complete."
    arguments = {
        "message": {
            "type": "string",
            "description": "An optional message to conclude the task, summarizing what was done.",
            "required": False,
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
