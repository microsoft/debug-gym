from froggy.entities import Observation
from froggy.tools.tool import EnvironmentTool
from froggy.tools.toolbox import Toolbox


@Toolbox.register()
class EvalTool(EnvironmentTool):
    name: str = "eval"
    action: str = "```eval"
    instructions = {
        "template": "```eval```",
        "description": "Evaluate the current code against pre-defined test cases.",
    }

    def use(self, action, **kwargs) -> Observation:
        obs = self.environment.eval(**kwargs)
        return Observation(self.name, obs)

    def on_env_reset(self, **kwargs):
        super().on_env_reset(**kwargs)
        return self(**kwargs)

    def on_rewrite_success(self, **kwargs):
        if self.environment.run_on_rewrite:
            return self(**kwargs)
        return None
