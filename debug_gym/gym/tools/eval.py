from debug_gym.gym.entities import Observation
from debug_gym.gym.tools.tool import EnvironmentTool
from debug_gym.gym.tools.toolbox import Toolbox


@Toolbox.register()
class EvalTool(EnvironmentTool):
    name: str = "eval"
    instructions = {
        "template": "```eval```",
        "description": "Evaluate the current code against pre-defined test cases.",
    }

    def use(self, tool_args, **kwargs) -> Observation:
        eval_output = self.environment.eval(**kwargs)
        return Observation(self.name, eval_output.output)

    def on_env_reset(self, **kwargs):
        super().on_env_reset(**kwargs)
        return self(**kwargs)

    def on_rewrite_success(self, **kwargs):
        if self.environment.run_on_rewrite:
            return self(**kwargs)
        return None
