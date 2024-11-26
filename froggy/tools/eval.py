from froggy.tools.tool import EnvironmentTool
from froggy.tools.toolbox import Toolbox


@Toolbox.register()
class EvalTool(EnvironmentTool):
    name: str = "eval"
    action: str = "```eval"
    description: str = "Evaluate the current code against pre-defined test cases."
    instructions = {
        "template": "```eval```",
        "description": "After debugging with pdb and fixing bug with rewrite command, one can use eval to evaluate the new code against pre-defined test cases.",
    }

    def register(self, environment):
        from froggy.envs.env import RepoEnv

        if not isinstance(environment, RepoEnv):
            raise ValueError("The environment must be a RepoEnv instance.")

        self.environment = environment

    def is_triggered(self, action):
        return action.startswith(self.action)

    def use(self, action):
        self.environment.run()
        return "Evaluation completed."
