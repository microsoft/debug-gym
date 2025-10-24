from debug_gym.gym.entities import Observation
from debug_gym.gym.tools.tool import EnvironmentTool
from debug_gym.gym.tools.toolbox import Toolbox


@Toolbox.register()
class SubmitTool(EnvironmentTool):
    name = "submit"
    description = "Submit your changes once the task is complete."
    arguments = {}

    def use(self, environment, **kwargs) -> Observation:
        eval_output = environment.eval()
        environment.terminated = True
        return Observation(self.name, eval_output.output)
