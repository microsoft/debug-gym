from debug_gym.gym.entities import Observation
from debug_gym.gym.tools.tool import EnvironmentTool
from debug_gym.gym.tools.toolbox import Toolbox


@Toolbox.register()
class SubmitTool(EnvironmentTool):
    name = "submit"
    description = "Submit your changes once the task is complete."
    arguments = {}

    def __init__(self, eval_on_submit=True):
        super().__init__()
        self.eval_on_submit = eval_on_submit

    def use(self, environment, **kwargs) -> Observation:
        output = "The agent terminated the session."
        if self.eval_on_submit:
            output = environment.eval().output

        environment.terminated = True
        return Observation(self.name, output)
