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

    def use(self, **kwargs):
        observation = self.environment.eval(**kwargs)
        return [{self.name: observation}]

    def on_env_reset(self, **kwargs):
        return self.use(**kwargs)

    def on_rewrite_success(self, **kwargs):
        observation = []
        if self.environment.run_on_rewrite:
            observation += self.use(**kwargs)
        return observation
