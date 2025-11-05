import time

from debug_gym.gym.entities import Observation
from debug_gym.gym.tools.tool import EnvironmentTool
from debug_gym.gym.tools.toolbox import Toolbox


@Toolbox.register()
class EvalTool(EnvironmentTool):
    name: str = "eval"
    description = "Evaluate the current code against pre-defined test cases."
    arguments = {}

    def __init__(self, auto_eval_on_rewrite=None):
        super().__init__()
        self.auto_eval_on_rewrite = auto_eval_on_rewrite

    def use(self, environment) -> Observation:
        eval_output = environment.eval()
        return Observation(self.name, eval_output.output)

    def on_env_reset(self, environment, **kwargs):
        super().on_env_reset(environment, **kwargs)
        return self(environment)

    def on_rewrite_success(self, environment, **kwargs):
        # Determine whether to auto-evaluate on rewrite (tool overrides env if set)
        auto = (
            self.auto_eval_on_rewrite
            if self.auto_eval_on_rewrite is not None
            else environment.auto_eval_on_rewrite
        )
        if auto:
            return self(environment)
        return None
