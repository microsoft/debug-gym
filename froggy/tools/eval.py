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

    def use(self, action=None, **kwargs):
        obs = self.environment.eval(**kwargs)
        return obs, [{self.name: obs}]

    def on_env_reset(self, **kwargs):
        super().on_env_reset(**kwargs)
        obs, chain_obs = self(**kwargs)
        return chain_obs

    def on_rewrite_success(self, **kwargs):
        if self.environment.run_on_rewrite:
            obs, chain_obs = self(**kwargs)
            return chain_obs
        return []
