from debug_gym.gym.entities import Observation
from debug_gym.gym.tools.tool import EnvironmentTool
from debug_gym.gym.tools.toolbox import Toolbox


@Toolbox.register()
class SubmitTool(EnvironmentTool):
    name = "submit"
    description = (
        "Finalize the task: applies the hidden official benchmark test patch,"
        "runs the benchmark evaluation, and returns the final results. "
        "After this call the environment is marked as submitted and further development will stop."
    )
    arguments = {}

    def use(self, environment, **kwargs) -> Observation:
        eval_output = environment.eval()
        return Observation(self.name, eval_output.output)
