from froggy.tools.tool import EnvironmentTool
from froggy.tools.toolbox import Toolbox


@Toolbox.register()
class EvalTool(EnvironmentTool):
    name: str = "eval"
    instructions = {
        "template": "eval()",
        "description": "Evaluate the current code against pre-defined test cases.",
    }

    def use(self):
        self.environment.run()
        return "Evaluation completed."
