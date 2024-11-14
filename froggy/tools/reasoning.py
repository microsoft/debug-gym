from froggy.tools.tool import EnvironmentTool
from froggy.tools.toolbox import Toolbox


@Toolbox.register()
class ReasoningTool(EnvironmentTool):
    name: str = "reasoning"
    action: str = "<reasoning>"
    description: str = "Preface any action with explicit reasoning tokens."

    @property
    def instructions(self):
        assert hasattr(self, "environment")
        instruction = {
            "template": "<reasoning> ... </reasoning>",
            "description": self.description,
            "examples": self.examples,
        }
        return instruction

    @property
    def examples(self):
        if self.allow_chain_action:
            ex = [
                "<reasoning> The execution trace points to line 43 in main.py, so I'll place a breakpoint there.</reasoning> ```pdb b 43",
                "<reasoning> There's a shape mismatch that corresponds to a matrix transpose, so I'll rewrite the function to account for the transpose. </reasoning> ```rewrite ....",
            ]
        else:
            ex = [
                "<reasoning> The execution trace points to line 43 in main.py, so I'll place a breakpoint there.</reasoning> ",
                "<reasoning> There's a shape mismatch that corresponds to a matrix transpose, so I'll rewrite the function to account for the transpose. </reasoning>",
            ]
        return ex

    @property
    def description(self):
        if self.allow_chain_action:
            desc = f"""You may explicitly reason about the current state and the best course of action before executing. You follow a particular reasoning style:
You break down complex problems into smaller parts and reason through them step by step, arriving at the best next action before then executing it. You should follow your reasoning with your next action. """
        else:
            desc = f"""You may explicitly reason about the current state and the best course of action. You follow a particular reasoning style:
You break down complex problems into smaller parts and reason through them step by step, arriving at the best next action(s). """
        return desc

    def __init__(self, allow_chain_action: bool = False):
        super().__init__()
        self.allow_chain_action = allow_chain_action

    def register(self, environment):
        from froggy.envs import RepoEnv

        if not isinstance(environment, RepoEnv):
            raise ValueError("The environment must be a RepoEnv instance.")

        self.environment = environment

    def is_triggered(self, action):
        return action.startswith(self.action)

    def use(self, action):
        if self.allow_chain_action:
            return self.use_with_chaining(action)
        else:
            return self.use_without_chaining()

    def use_with_chaining(self, action):
        """Reasoning tokens are only to benefit the model, so we strip them and then pass the remainder of the action
        as a free next action.
        """
        remaining_action = self.remove_reasoning(action)
        # now execute the next action
        next_action_obs = self.environment.step(remaining_action)
        if next_action_obs == f"Invalid action: {action}.":
            next_action_obs == f"You must provide a valid action after your reasoning. Found invalid action: {action}."
        return f"Reasoning text acknowledged. {next_action_obs[0]}"

    def use_without_chaining(self):
        return "Reasoning text acknowledged."

    def remove_reasoning(self, action):
        items = action.split("</reasoning>")
        return " ".join(items[1:]).lstrip()
