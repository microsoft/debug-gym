from debug_gym.gym.entities import Observation
from debug_gym.gym.tools.tool import EnvironmentTool
from debug_gym.gym.tools.toolbox import Toolbox


@Toolbox.register()
class EvalTool(EnvironmentTool):
    name: str = "eval"
    description = "Evaluate the current code against pre-defined test cases."
    arguments = {}

    def use(self, environment) -> Observation:
        eval_output = environment.eval()
        return Observation(self.name, eval_output.output)

    # def on_env_reset(self, environment, **kwargs):
    #     super().on_env_reset(environment, **kwargs)
    #     return self(environment)

    def on_rewrite_success(self, environment, **kwargs):
        if environment.auto_eval_on_rewrite:
            return self(environment)
        return None
